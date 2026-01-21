"""
gapfilling_utils.py

Utilities for EC flux gapfilling + uncertainty attachment.

Methodology (Modified for CV-Ensemble):
- Main Prediction: Model trained on 100% of data.
- Uncertainty Components:
    (a) Structural Uncertainty (sigma_ens): Standard deviation of predictions from K CV-fold models.
    (b) Residual Uncertainty (sigma_resid): Quantile-based scaling of OOF residuals.
- Total Uncertainty: sigma_gf(t) = sqrt(sigma_ens(t)^2 + sigma_resid(t)^2)

Notes
-----
- Removes block bootstrapping in favor of K-fold CV ensemble (Irvin et al. style).
- If you log-transform y, pass an inverse transform function `inv_y`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd

Idx = np.ndarray
Split = Tuple[Idx, Idx]


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------

def _as_1d_float(x: Any) -> np.ndarray:
    """Convert array-like to 1D float array (keeps NaNs)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return float("nan")
    y = y_true[m]
    yhat = y_pred[m]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _check_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {name}: {missing}")


# -----------------------------------------------------------------------------
# Transform helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Log1pShift:
    """log1p transform with optional shift if min < 0."""
    shift: float  # the min value if min<0 else 0

    def transform(self, x: Any) -> np.ndarray:
        x = _as_1d_float(x)
        if self.shift < 0:
            return np.log1p(x - self.shift)
        return np.log1p(x)

    def inverse(self, x: Any) -> np.ndarray:
        x = _as_1d_float(x)
        if self.shift < 0:
            return np.expm1(x) + self.shift
        return np.expm1(x)


def setup_log_transform(
    data: pd.DataFrame,
    target: str,
    *,
    apply: bool = False,
    plot: bool = False,
    bins: int = 20,
):
    _check_columns(data, [target], name="data")
    min_value = float(pd.to_numeric(data[target], errors="coerce").min())
    t = Log1pShift(shift=min_value if min_value < 0 else 0.0)

    def log_transform(x):
        return t.transform(x)

    def inverse_log_transform(x):
        return t.inverse(x)

    if plot:
        import matplotlib.pyplot as plt
        plt.hist(data[target].dropna(), bins=bins)
        plt.title("Original")
        plt.show()
        plt.hist(log_transform(data[target].dropna()), bins=bins)
        plt.title("Log-transformed")
        plt.show()

    data_out = data
    if apply:
        data_out = data.copy()
        data_out[target] = log_transform(data_out[target])

    return log_transform, inverse_log_transform, min_value, data_out


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def undersample_target(
    data: pd.DataFrame,
    target: str,
    *,
    quantile_cutoff: float = 0.8,
    fraction: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
):
    _check_columns(data, [target], name="data")

    if not (0.0 < quantile_cutoff < 1.0):
        raise ValueError("quantile_cutoff must be in (0, 1)")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    cutoff_value = float(pd.to_numeric(data[target], errors="coerce").quantile(quantile_cutoff))
    upper = data[data[target] > cutoff_value]
    lower = data[data[target] <= cutoff_value]

    lower_sampled = lower.sample(frac=fraction, random_state=random_state)
    out = pd.concat([upper, lower_sampled], axis=0).sort_index()

    if verbose:
        kept = len(out)
        total = len(data)
        print(
            f"Undersample {target}: cutoff q={quantile_cutoff:.2f} -> {cutoff_value:.6g}; "
            f"kept {kept}/{total} rows ({kept/total:.1%}); lower kept fraction={fraction:.2f}"
        )
    return out, cutoff_value


def infer_cv_block_size_from_gaps(
    s: pd.Series,
    *,
    quantile: float = 0.8,
    fallback: int = 6,
) -> int:
    if not (0.0 < quantile < 1.0):
        raise ValueError('quantile must be in (0, 1)')
    isna = s.isna()
    if not bool(isna.any()):
        return int(fallback)
    run_id = isna.ne(isna.shift()).cumsum()
    run_isna = isna.groupby(run_id).first()
    run_len = isna.groupby(run_id).size()
    gaps = run_len[run_isna]
    if gaps.empty:
        return int(fallback)
    qv = float(gaps.quantile(quantile))
    return int(max(1, int(np.ceil(qv))))


# -----------------------------------------------------------------------------
# Parcel / dataset assembly helpers
# -----------------------------------------------------------------------------

def build_df_for_parcel(data_main, target_flux, letter, selected_features, add_trt):
    if "parcel" not in data_main.columns:
        raise KeyError("data_main must contain a 'parcel' column for masking.")

    cols = set(data_main.columns)
    df = pd.DataFrame(index=data_main.index)

    for f in selected_features:
        if f == "trt":
            continue
        fp = f"{f}_parcel{letter}"
        if fp in cols:
            df[f] = data_main[fp]
        elif f in cols:
            df[f] = data_main[f]
        else:
            raise KeyError(f"Missing feature '{f}' (neither '{fp}' nor '{f}' exists).")

    if add_trt and "trt" in selected_features:
        df["trt"] = 0 if letter == "A" else 1

    mask = data_main["parcel"].eq(letter)
    flux_cols = [c for c in data_main.columns if c.startswith(target_flux)]
    if flux_cols:
        df[flux_cols] = data_main[flux_cols].where(mask)

    return df


# -----------------------------------------------------------------------------
# Time-series CV splits
# -----------------------------------------------------------------------------

def create_block_splits(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    split: float = 0.1,
    block_size: int = 12,
    random_state: int = 42,
    shuffle_blocks: bool = True,
    verbose: bool = True,
) -> List[Split]:
    n = len(X)
    if n == 0:
        raise ValueError("X is empty")
    idx = np.arange(n, dtype=int)

    n_folds = int(round(1.0 / split))
    n_folds = max(2, n_folds)

    block_id = idx // int(block_size)
    n_blocks = int(block_id.max() + 1)
    blocks = np.arange(n_blocks, dtype=int)

    rng = np.random.default_rng(random_state)
    if shuffle_blocks:
        rng.shuffle(blocks)

    fold_blocks = np.array_split(blocks, n_folds)
    splits: List[Split] = []
    achieved = []
    for fb in fold_blocks:
        test_mask = np.isin(block_id, fb)
        test_idx = idx[test_mask]
        train_idx = idx[~test_mask]
        splits.append((train_idx, test_idx))
        achieved.append(len(test_idx) / n)

    if verbose:
        print(
            f"Requested split={split:.2f}; n_folds={n_folds}; "
            f"achieved test fractions ~ {min(achieved):.3f}–{max(achieved):.3f}"
        )

    return splits


def plot_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    *,
    ncols: int = 2
):
    import math
    import matplotlib.pyplot as plt

    n_splits = len(splits)
    nrows = math.ceil(n_splits / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2), squeeze=False)
    ax_list = axes.flatten()

    for i, (train_idx, test_idx) in enumerate(splits):
        ax = ax_list[i]
        train_ix = X.iloc[train_idx].index
        test_ix = X.iloc[test_idx].index
        ax.plot(y.loc[train_ix].index, y.loc[train_ix], ".", label="Train")
        ax.plot(y.loc[test_ix].index, y.loc[test_ix], "x", label="Test")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Split {i+1}")
        ax.legend()

    for j in range(n_splits, len(ax_list)):
        fig.delaxes(ax_list[j])
    fig.tight_layout()
    plt.show()
    return fig, axes


# -----------------------------------------------------------------------------
# RFE Feature Selection
# -----------------------------------------------------------------------------

def _maybe_inverse(inv_y: Optional[Callable[[Any], Any]], x: Any) -> Any:
    if inv_y is None:
        return x
    return inv_y(x)

def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    *,
    model_factory: Callable[[], object],
    inv_y: Optional[Callable[[Any], Any]] = None,
    step: int = 1,
    min_features: int = 1,
    verbose: bool = True,
    score_mode: str = "composite",  # "rmse" or "composite"
    w_rmse: float = 0.5,
    w_r2: float = 0.5,
    w_penalty: float = 0.001,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Recursive Feature Elimination using mean feature_importances_ across folds.

    Scoring uses pooled Out-Of-Fold (OOF) predictions (overall RMSE/R²).
    Returns:
      best_features, feature_ranking, history_df
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if min_features < 1:
        raise ValueError("min_features must be >= 1")

    features = list(X.columns)
    history = []
    removal_order = []  # (feature, iteration)

    iteration = 0
    while len(features) > max(min_features, 1):
        iteration += 1

        y_true_oof = []
        y_pred_oof = []
        importances = []

        for tr, te in splits:
            X_tr = X[features].iloc[tr]
            y_tr = y.iloc[tr]
            X_te = X[features].iloc[te]
            y_te = y.iloc[te]

            m = model_factory()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)

            # Feature importance (simple + explicit as requested)
            if not hasattr(m, "feature_importances_"):
                raise AttributeError(
                    "Model has no feature_importances_. "
                    "Use a model that provides it or change the RFE strategy."
                )
            imp = np.asarray(m.feature_importances_, dtype=float)
            if imp.shape[0] != len(features):
                raise ValueError("feature_importances_ length does not match current feature list.")
            importances.append(imp)

            # Score on original scale if inv_y is provided
            y_te_s = _as_1d_float(_maybe_inverse(inv_y, y_te))
            pred_s = _as_1d_float(_maybe_inverse(inv_y, pred))

            y_true_oof.append(y_te_s)
            y_pred_oof.append(pred_s)

        y_true_oof = np.concatenate(y_true_oof)
        y_pred_oof = np.concatenate(y_pred_oof)

        rmse_oof = _rmse(y_true_oof, y_pred_oof)
        r2_oof = _r2(y_true_oof, y_pred_oof)

        history.append(
            {
                "iteration": iteration,
                "n_features": len(features),
                "features": features.copy(),
                "rmse_oof": rmse_oof,
                "r2_oof": r2_oof,
            }
        )

        mean_imp = np.mean(np.vstack(importances), axis=0)
        n_remove = min(step, len(features) - min_features)
        remove_idx = np.argsort(mean_imp)[:n_remove]  # lowest importance first
        remove_feats = [features[i] for i in remove_idx.tolist()]

        for f in remove_feats:
            features.remove(f)
            removal_order.append((f, iteration))

        if verbose:
            print(
                f"Iter {iteration}: kept={len(features)} removed={remove_feats} "
                f"RMSE_oof={rmse_oof:.4f} R2_oof={r2_oof:.4f}"
            )

    # add survivors to ranking (last survivors rank highest)
    for f in features:
        removal_order.append((f, iteration + 1))
    feature_ranking = [f for f, _ in sorted(removal_order, key=lambda x: x[1], reverse=True)]

    hist_df = pd.DataFrame(history)

    # choose best subset
    if hist_df.empty:
        return features.copy(), feature_ranking, hist_df

    if score_mode == "rmse":
        best_row = hist_df.sort_values(["rmse_oof", "n_features"], ascending=[True, True]).iloc[0]
        best_features = list(best_row["features"])
        return best_features, feature_ranking, hist_df

    if score_mode == "composite":
        rmse_vals = hist_df["rmse_oof"].to_numpy(float)
        r2_vals = hist_df["r2_oof"].to_numpy(float)
        nfeat = hist_df["n_features"].to_numpy(float)

        # safe min-max normalization
        def _minmax(a):
            a_min, a_max = float(np.min(a)), float(np.max(a))
            if np.isclose(a_min, a_max):
                return np.zeros_like(a, dtype=float)
            return (a - a_min) / (a_max - a_min)

        norm_rmse = _minmax(rmse_vals)
        norm_r2 = _minmax(r2_vals)

        comp = w_rmse * norm_rmse + w_r2 * (1.0 - norm_r2) + w_penalty * nfeat
        best_i = int(np.argmin(comp))
        best_features = list(hist_df.iloc[best_i]["features"])
        return best_features, feature_ranking, hist_df

    raise ValueError("score_mode must be 'rmse' or 'composite'")

# -----------------------------------------------------------------------------
# CV Model Fitting & OOF
# -----------------------------------------------------------------------------

def fit_cv_ensemble(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y_train: pd.Series,
    splits: List[Split],
    *,
    inv_y: Optional[Callable[[Any], Any]] = None,
) -> Tuple[pd.Series, List[object]]:
    """
    Trains one model per CV split and generates Out-Of-Fold predictions.
    
    Returns
    -------
    oof : pd.Series
        Aligned with X.index, containing predictions when that row was in 'test'.
    models : List[object]
        The list of trained models (one per split). 
        These form the 'Ensemble' for uncertainty estimation.
    """
    oof = pd.Series(index=X.index, dtype=float)
    models = []
    
    # Iterate through folds
    for tr, te in splits:
        # 1. Train on this fold's training set
        m = model_factory()
        m.fit(X.iloc[tr], y_train.iloc[tr])
        models.append(m)
        
        # 2. Predict on this fold's test set (OOF)
        pred = m.predict(X.iloc[te]).astype(float)
        pred = _maybe_inverse(inv_y, pred)
        oof.iloc[te] = _as_1d_float(pred)
        
    return oof, models


@dataclass
class ResidualScaleModel:
    """Maps a prediction yhat to a residual scale sigma_resid(yhat)."""
    method: str
    sigma_global: float
    bin_edges: Optional[np.ndarray] = None
    sigma_by_bin: Optional[np.ndarray] = None

    def sigma(self, yhat: Any) -> np.ndarray:
        yhat = _as_1d_float(yhat)
        if self.method == "global" or self.bin_edges is None or self.sigma_by_bin is None:
            return np.full_like(yhat, self.sigma_global, dtype=float)
        b = np.digitize(yhat, self.bin_edges, right=True) - 1
        b = np.clip(b, 0, len(self.sigma_by_bin) - 1)
        return self.sigma_by_bin[b].astype(float)


def fit_residual_scale(
    y_obs_raw: pd.Series,
    yhat_oof_raw: pd.Series,
    *,
    method: str = "by_pred_quantile",
    q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    min_per_bin: int = 200,
) -> ResidualScaleModel:
    df = pd.DataFrame({"y": y_obs_raw, "yhat": yhat_oof_raw}).dropna()
    resid = (df["y"] - df["yhat"]).to_numpy(dtype=float)

    sigma_global = float(np.nanstd(resid, ddof=1))

    if method == "global":
        return ResidualScaleModel(method="global", sigma_global=sigma_global)

    qs = np.unique(np.array(list(q), dtype=float))
    if qs[0] != 0.0: qs = np.insert(qs, 0, 0.0)
    if qs[-1] != 1.0: qs = np.append(qs, 1.0)

    edges = np.quantile(df["yhat"].to_numpy(dtype=float), qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_id = np.digitize(df["yhat"].to_numpy(dtype=float), edges, right=True) - 1
    n_bins = len(edges) - 1

    sigmas = np.full(n_bins, sigma_global, dtype=float)
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() < min_per_bin:
            continue
        r = resid[mask]
        sigmas[b] = float(np.nanstd(r, ddof=1))

    return ResidualScaleModel(
        method="by_pred_quantile",
        sigma_global=sigma_global,
        bin_edges=edges,
        sigma_by_bin=sigmas,
    )


def combine_gapfill_sigma(
    sigma_ens: Any,
    sigma_resid: Any,
    *,
    min_sigma: float = 0.0,
) -> np.ndarray:
    """sigma_gf = sqrt(sigma_ens^2 + sigma_resid^2), with optional floor."""
    sigma_ens = _as_1d_float(sigma_ens)
    sigma_resid = _as_1d_float(sigma_resid)
    sgf = np.sqrt(np.square(sigma_ens) + np.square(sigma_resid))
    if min_sigma > 0:
        sgf = np.maximum(sgf, float(min_sigma))
    return sgf


# -----------------------------------------------------------------------------
# Convenience pipeline: fit/apply gapfiller with uncertainty
# -----------------------------------------------------------------------------

@dataclass
class GapfillFit:
    target_col: str
    feature_cols: List[str]
    model_factory: Callable[[], object]
    model_final: object
    ensemble_models: List[object]   # Contains the K models from CV folds
    X_tr: pd.DataFrame
    y_tr: pd.Series
    y_raw: pd.Series
    inv_y: Optional[Callable[[Any], Any]]
    splits: List[Split]
    yhat_oof_raw: pd.Series
    resid_model: ResidualScaleModel
    random_state: int


# -----------------------------------------------------------------------------
# Internal Plotting Helper (Triggered inside fit_gapfill_ts)
# -----------------------------------------------------------------------------

def _plot_internal_diagnostics(
    y_obs: pd.Series, 
    y_pred: pd.Series, 
    ensemble_models: List[object], 
    model_final: object,         
    X: pd.DataFrame,             
    inv_y: Optional[Callable],    
    feature_cols: List[str], 
    target_name: str
):
    """
    Plots 2x2 Diagnostics:
    1. Obs vs Pred (Scatter)
    2. Feature Importance
    3. Ensemble Time Series (Instantaneous)
    4. Ensemble Time Series (Cumulative)
    """
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    # Create 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # --- Plot 1: Obs vs Pred (Global OOF) ---
    ax1 = axes[0, 0]
    df_plot = pd.DataFrame({"Obs": y_obs, "Pred": y_pred}).dropna()
    
    if len(df_plot) > 1:
        x, y = df_plot["Pred"], df_plot["Obs"]
        rmse = np.sqrt(np.mean((y - x)**2))
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        ax1.scatter(x, y, alpha=0.2, s=10, c='k', label='Data')
        min_v, max_v = min(x.min(), y.min()), max(x.max(), y.max())
        ax1.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1)
        ax1.plot(np.array([min_v, max_v]), slope * np.array([min_v, max_v]) + intercept, 'r-', lw=2, label=f"R²={r_val**2:.3f}")
        
        ax1.set_title(f"1. Global Performance (OOF)\nRMSE={rmse:.4f}")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Observed")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No Data", ha='center')

    # --- Plot 2: Feature Importance ---
    ax2 = axes[0, 1]
    importances = []
    for m in ensemble_models:
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
        elif hasattr(m, 'get_score'): # Native XGBoost
            scores = m.get_score(importance_type='gain')
            imp = np.zeros(len(feature_cols))
            for f, score in scores.items():
                if f in feature_cols: imp[feature_cols.index(f)] = score
            importances.append(imp)
            
    if importances:
        avg_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        # Top 10
        indices = np.argsort(avg_imp)[-10:] 
        ax2.barh(range(len(indices)), avg_imp[indices], xerr=std_imp[indices], align='center', capsize=3)
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels(np.array(feature_cols)[indices])
        ax2.set_title(f"2. Top 10 Features\n(Avg of {len(ensemble_models)} Folds)")
    else:
        ax2.text(0.5, 0.5, "Importance N/A", ha='center')

    # --- 3 & 4. Time Series Predictions ---
    # Sort X by time so lines connect correctly
    X_sorted = X.sort_index()
    # Generate Predictions 
    ens_preds = []
    for m in ensemble_models:
        p = m.predict(X_sorted)
        if inv_y: p = inv_y(p)
        ens_preds.append(_as_1d_float(p))
        
    p_final = model_final.predict(X_sorted)
    if inv_y: p_final = inv_y(p_final)
    p_final = _as_1d_float(p_final)
    x_axis = X_sorted.index
    # Plot 3: Instantaneous
    ax3 = axes[1, 0]
    for p in ens_preds:
        ax3.plot(x_axis, p, color='gray', alpha=0.5)
    ax3.plot(x_axis, p_final, color='red', alpha=1, label='Final Model')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_title("3. Ensemble Predictions (Training Data)")
    ax3.set_ylabel(f"{target_name}")
    # Plot 4: Cumulative
    ax4 = axes[1, 1]
    for p in ens_preds:
        ax4.plot(x_axis, np.cumsum(p), color='gray', alpha=0.5)
    ax4.plot(x_axis, np.cumsum(p_final), color='red', label='Final Model')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_title("4. Cumulative Sum Divergence (Training Data)")
    ax4.set_ylabel(f"Cum. {target_name}")
    ax4.legend()
    plt.tight_layout()
    plt.show()


def fit_gapfill_ts(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: List[str],
    model_factory: Callable[[], object],
    log_transform: bool = False,
    undersample: bool = False,
    undersample_quantile: float = 0.8,
    undersample_fraction: float = 0.5,
    cv_split: float = 0.1,
    cv_block_size: Optional[int] = None,
    cv_block_quantile: float = 0.8,
    cv_block_fallback: int = 6,
    random_state: int = 42,
    resid_method: str = "by_pred_quantile",
    resid_q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    resid_min_per_bin: int = 200,
    verbose: bool = True,
    plot: bool = True
) -> GapfillFit:
    """
    Fit a final model + uncertainty components using CV Ensemble.
    """
    _check_columns(df, [target_col] + list(feature_cols), name="df")

    out = df.copy()

    train_mask = out[target_col].notna() & out[feature_cols].notna().all(axis=1)
    if train_mask.sum() < 10:
        raise ValueError(f"Not enough complete training rows for {target_col}.")

    df_train = out.loc[train_mask, feature_cols + [target_col]].copy()
    df_train = df_train.rename(columns={target_col: "_y_raw"})
    y_raw = df_train["_y_raw"].astype(float).copy()

    if undersample:
        df_train, cutoff = undersample_target(
            df_train, "_y_raw",
            quantile_cutoff=undersample_quantile,
            fraction=undersample_fraction,
            random_state=random_state,
            verbose=verbose,
        )
        y_raw = df_train["_y_raw"].astype(float).copy()

    inv_y = None
    if log_transform:
        log_fn, inv_fn, min_value, _ = setup_log_transform(df_train, "_y_raw")
        df_train["_y_train"] = log_fn(df_train["_y_raw"].to_numpy(float))
        inv_y = inv_fn
    else:
        df_train["_y_train"] = df_train["_y_raw"].to_numpy(float)

    X_tr = df_train[feature_cols].copy()
    y_tr = pd.Series(df_train["_y_train"].to_numpy(float), index=df_train.index)

    # CV Splits & Block Size Calculation
    if cv_block_size is None:
        cv_block_size = infer_cv_block_size_from_gaps(
            out[target_col], quantile=cv_block_quantile, fallback=cv_block_fallback,
        )
        msg_source = f"Inferred from gaps (q={cv_block_quantile})"
    else:
        msg_source = "User defined"
        
    if verbose:
        print(f"CV Strategy: Block Size = {cv_block_size} timesteps ({msg_source})")

    splits = create_block_splits(
        X_tr, split=cv_split, block_size=cv_block_size, random_state=random_state, verbose=verbose,
    )

    # Plot CV Splits
    if plot:
        # Note: We pass y_raw so the plot shows the actual data structure
        plot_cv_splits(X_tr, y_raw, splits)

    # Fit CV Ensemble
    if verbose:
        print(f"Training CV Ensemble ({len(splits)} folds) & generating OOF predictions...")
        
    yhat_oof_raw, ensemble_models = fit_cv_ensemble(
        model_factory=model_factory, 
        X=X_tr, 
        y_train=y_tr, 
        splits=splits, 
        inv_y=inv_y
    )

    resid_model = fit_residual_scale(
        y_obs_raw=y_raw,
        yhat_oof_raw=yhat_oof_raw,
        method=resid_method,
        q=resid_q,
        min_per_bin=resid_min_per_bin,
    )

    # Final Model (Best Estimate trained on all data)
    if verbose:
        print("Fitting final model on full training set...")
    m_final = model_factory()
    m_final.fit(X_tr, y_tr)

    # Plots for diagnostics
    if plot:
        _plot_internal_diagnostics(
            y_obs=y_raw,
            y_pred=yhat_oof_raw,
            ensemble_models=ensemble_models,
            model_final=m_final,           
            X=X_tr,                        
            inv_y=inv_y,                   
            feature_cols=list(feature_cols),
            target_name=target_col
        )

    return GapfillFit(
        target_col=target_col,
        feature_cols=list(feature_cols),
        model_factory=model_factory,
        model_final=m_final,
        ensemble_models=ensemble_models, # CV models stored here
        X_tr=X_tr,
        y_tr=y_tr,
        y_raw=y_raw,
        inv_y=inv_y,
        splits=splits,
        yhat_oof_raw=yhat_oof_raw,
        resid_model=resid_model,
        random_state=int(random_state),
    )


def apply_gapfill_ts(
    df_pred: pd.DataFrame,
    fit: GapfillFit,
    *,
    prefix: str = "GF",
    min_sigma: float = 0.0,
) -> pd.DataFrame:
    """
    Apply the fitted model.
    - Prediction = fit.model_final
    - Sigma_Ens = StdDev of predictions from fit.ensemble_models (CV models)
    """
    _check_columns(df_pred, fit.feature_cols, name="df_pred")

    out = df_pred.copy()
    X_full = out[fit.feature_cols].copy()
    ok = X_full.notna().all(axis=1).to_numpy(bool)

    yhat = np.full(len(out), np.nan, dtype=float)
    sigma_ens = np.full(len(out), np.nan, dtype=float)
    sigma_resid = np.full(len(out), np.nan, dtype=float)
    sigma_gf = np.full(len(out), np.nan, dtype=float)

    if ok.sum() > 0:
        X_ok = X_full.loc[ok]

        # 1. Main Prediction
        yhat_ok = fit.model_final.predict(X_ok).astype(float)
        yhat_ok = _maybe_inverse(fit.inv_y, yhat_ok)
        yhat[ok] = _as_1d_float(yhat_ok)

        # 2. Ensemble Prediction (using CV models)
        if fit.ensemble_models:
            n_models = len(fit.ensemble_models)
            preds_ens = np.full((n_models, len(X_ok)), np.nan, dtype=float)
            
            for i, m in enumerate(fit.ensemble_models):
                p = m.predict(X_ok).astype(float)
                p = _maybe_inverse(fit.inv_y, p)
                preds_ens[i, :] = _as_1d_float(p)

            # Standard Deviation of the ensemble predictions
            sigma_ens_ok = np.nanstd(preds_ens, axis=0, ddof=1)
            sigma_ens[ok] = _as_1d_float(sigma_ens_ok)
        else:
            sigma_ens[ok] = 0.0

        # 3. Residual Uncertainty
        sigma_resid_ok = fit.resid_model.sigma(yhat_ok)
        sigma_resid[ok] = _as_1d_float(sigma_resid_ok)
        
        # 4. Combined
        sigma_gf[ok] = combine_gapfill_sigma(sigma_ens_ok, sigma_resid_ok, min_sigma=min_sigma)

    out[f"{prefix}_yhat"] = yhat
    out[f"{prefix}_sigmaEns"] = sigma_ens
    out[f"{prefix}_sigmaResid"] = sigma_resid
    out[f"{prefix}_sigmaGF"] = sigma_gf

    return out


# -----------------------------------------------------------------------------
# High-Level Reusable Workflows (Unchanged)
# -----------------------------------------------------------------------------
# ... (keep merge_gapfill_results and plot_gapfill_dashboard as they were) ...
# (You can paste the bottom section of your original file here if needed, 
#  but the logic changes above are self-contained)

def merge_gapfill_results(
    main_df: pd.DataFrame,
    views: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
    target_flux: str,
    target: str,
    model_type: str,
    ustar_cut: str,
    random_err_col: str,
    prefix: str = "GF",
    qc_levels: List[str] = ["QCF", "QCF0"],
) -> pd.DataFrame:
    """Merges gap-filling predictions into main dataframe."""
    df_final = main_df.copy()
    target_base_root = f"{target_flux}_L3.3_{ustar_cut}"
    
    for view_name, df_view, pred_df in views:
        base_name = f"{target}_{view_name}_gf{model_type}"
        col_pred_only = f"{base_name}_yhat"
        col_sigmaEns  = f"{base_name}_sigmaEns"
        col_sigmaRes  = f"{base_name}_sigmaResid"
        col_sigmaGF   = f"{base_name}_sigmaGF"
        
        df_final[col_pred_only] = pred_df[f"{prefix}_yhat"].reindex(df_final.index)
        df_final[col_sigmaEns]  = pred_df[f"{prefix}_sigmaEns"].reindex(df_final.index)
        df_final[col_sigmaRes]  = pred_df[f"{prefix}_sigmaResid"].reindex(df_final.index)
        df_final[col_sigmaGF]   = pred_df[f"{prefix}_sigmaGF"].reindex(df_final.index)
        
        for qc in qc_levels:
            obs_col = f"{target_base_root}_{qc}"
            y_obs = df_view[obs_col].astype(float)
            y_hat = pred_df[f"{prefix}_yhat"]
            is_gap = y_obs.isna() & y_hat.notna()
            
            obs_col_out   = f"{target_base_root}_{qc}_{view_name}"
            gf_base_name  = f"{target_base_root}_{qc}_{view_name}_gf{model_type}"
            col_filled    = gf_base_name
            col_isfilled  = f"{gf_base_name}_ISFILLED"
            col_total_unc = f"{gf_base_name}_total_unc"
            
            df_final[obs_col_out] = y_obs
            df_final[col_filled] = y_obs.where(~is_gap, y_hat)
            df_final[col_isfilled] = is_gap.astype(int)
            
            sigma_obs = df_view[random_err_col]
            sigma_gf = pred_df[f"{prefix}_sigmaGF"]
            df_final[col_total_unc] = sigma_obs.where(~is_gap, sigma_gf) 

    return df_final


def plot_gapfill_dashboard(
    df: pd.DataFrame,
    periods: List[Tuple[str, str, str]],
    target_flux: str,
    target,
    model_type: str,
    ustar_cut: str,
    qc_levels: List[str] = ["QCF", "QCF0"],
    parcels: List[str] = ["A", "B"]
):
    import matplotlib.pyplot as plt

    def _plot_flux_with_uncertainty(ax, data, col_val, col_unc=None, label=None, cumulative=False, sigma_scale=1.96):
        if col_val not in data.columns: return
        y = data[col_val]
        if label is None: label = col_val
        y_plot = y.cumsum() if cumulative else y
        line, = ax.plot(data.index, y_plot, label=label, alpha=0.8)
        
        if col_unc and col_unc in data.columns:
            sigma = data[col_unc].fillna(0.0)
            sigma_plot = np.sqrt((sigma**2).cumsum()) if cumulative else sigma
            ax.fill_between(data.index, y_plot - sigma_scale*sigma_plot, y_plot + sigma_scale*sigma_plot, color=line.get_color(), alpha=0.2, linewidth=0)

    def _tgt(qc): return f'{target_flux}_L3.3_{ustar_cut}_{qc}'
    def _get_sigma(c):
        if c.endswith("_yhat"): return c.replace("_yhat", "_sigmaGF")
        if "_gf" in c: return c + "_total_unc"
        return None

    for start, end, label in periods:
        period_df = df.loc[pd.to_datetime(start):pd.to_datetime(end)]
        if period_df.empty: continue
        
        rows = []
        cols = [f"{target}_parcel{p}_gf{model_type}_yhat" for p in parcels if f"{target}_parcel{p}_gf{model_type}_yhat" in period_df.columns]
        rows.append((cols, "Predicted (Model Only)"))

        for qc in qc_levels:
            _t = _tgt(qc)
            cols = [f"{_t}_parcel{p}" for p in parcels if f"{_t}_parcel{p}" in period_df.columns]
            rows.append((cols, f"Observed [{qc}]"))
            
        for qc in qc_levels:
            _t = _tgt(qc)
            cols = [f"{_t}_parcel{p}_gf{model_type}" for p in parcels if f"{_t}_parcel{p}_gf{model_type}" in period_df.columns]
            rows.append((cols, f"Gap-filled [{qc}]"))

        cols = [c for c in [f"{_tgt(qc)}_footprint_gf{model_type}" for qc in qc_levels] if c in period_df.columns]
        rows.append((cols, f"Gap-filled (Full-Footprint)"))

        nrows = len(rows)
        if nrows == 0: continue
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 3*nrows), sharex='col')
        if nrows == 1: axes = axes.reshape(1, -1)

        for r, (cols, title) in enumerate(rows):
            for col in cols:
                _plot_flux_with_uncertainty(axes[r, 0], period_df, col, _get_sigma(col), cumulative=False)
            axes[r, 0].set_title(title, fontsize=10, fontweight='bold')
            axes[r, 0].legend(fontsize=8, loc='upper right')
            axes[r, 0].grid(True, alpha=0.3)
            
            for col in cols:
                _plot_flux_with_uncertainty(axes[r, 1], period_df, col, _get_sigma(col), cumulative=True)
            axes[r, 1].set_title(f"{title} — Cumulative", fontsize=10, fontweight='bold')
            axes[r, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{label} ({start} → {end})", y=0.995, fontsize=14)
        plt.tight_layout()
        plt.show()


