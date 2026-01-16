"""
gapfilling_utils_v2.py

Utilities for EC flux gapfilling + uncertainty attachment.

Core uncertainty idea:
- Observed half-hours: use EddyPro random uncertainty (rand_err) as sigma_ec(t).
- Gapfilled half-hours: build sigma_gf(t) from
    (a) ensemble spread of the ML model (sigma_ens(t))
    (b) out-of-sample residual scale from time-series CV (sigma_resid(t))
  and combine as: sigma_gf(t) = sqrt(sigma_ens(t)^2 + sigma_resid(t)^2)

Notes
-----
- If you log-transform y, pass an inverse transform function `inv_y`.
- For time series, residual scale is best estimated from *out-of-fold* (OOF) predictions.
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


def _robust_sd(x: np.ndarray) -> float:
    """
    Robust SD estimate via MAD (1.4826 * median(|x - median(x)|)).

    Avoids SciPy dependency; handles heavy tails better than std.
    """
    x = _as_1d_float(x)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


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
    """
    Create a log1p transform + inverse for a target column, with an automatic
    shift if min<0.

    Returns
    -------
    log_transform : Callable
    inverse_log_transform : Callable
    min_value : float
        Minimum of the original target (used as shift if negative).
    data_out : DataFrame
        If apply=True, a copy with transformed target. Else original `data`.
    """
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

    # for backward compatibility: return "min_value" (original min), not the applied shift
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
    plot: bool = False,
    verbose: bool = True,
):
    """
    Undersample the *lower* part of the target distribution.

    Keeps all values > q(target, quantile_cutoff) and keeps only `fraction`
    of values <= cutoff. Returns a dataframe with the original index preserved,
    and sorted by index (chronological if your index is time).

    Returns
    -------
    out : DataFrame
    cutoff_value : float
    """
    _check_columns(data, [target], name="data")

    if not (0.0 < quantile_cutoff < 1.0):
        raise ValueError("quantile_cutoff must be in (0, 1)")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    cutoff_value = float(pd.to_numeric(data[target], errors="coerce").quantile(quantile_cutoff))

    upper = data[data[target] > cutoff_value]
    lower = data[data[target] <= cutoff_value]

    # keep index (and any other columns) intact
    lower_sampled = lower.sample(frac=fraction, random_state=random_state)

    # shuffle for model training convenience, then restore chronological order
    out = pd.concat([upper, lower_sampled], axis=0).sample(frac=1, random_state=random_state).sort_index()

    if verbose:
        kept = len(out)
        total = len(data)
        print(
            f"Undersample {target}: cutoff q={quantile_cutoff:.2f} -> {cutoff_value:.6g}; "
            f"kept {kept}/{total} rows ({kept/total:.1%}); lower kept fraction={fraction:.2f}"
        )

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        data[target].plot(x_compat=True, style=".", title="Before undersampling")
        plt.show()

        plt.figure()
        out[target].plot(x_compat=True, style=".", title="After undersampling")
        plt.show()

    return out, cutoff_value


def infer_cv_block_size_from_gaps(
    s: pd.Series,
    *,
    quantile: float = 0.8,
    fallback: int = 6,
) -> int:
    """
    Infer a reasonable CV block_size (in records) from the distribution of
    consecutive NaN gaps in the target series `s`.

    Uses the `quantile` of gap lengths
    If there are no gaps, returns `fallback`.
    """
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
    """
    Strict parcel dataframe builder (crashes on missing features).

    For each feature f in selected_features (except 'trt'):
      - use f_parcelA/B if present, else use shared f
      - if neither exists -> raise KeyError

    If add_trt and 'trt' in selected_features:
      - set trt = 0 for A, 1 for B

    Additionally, for any columns whose name starts with `target_flux`,
    keep values only in rows where parcel == letter (else NaN).
    """
    import pandas as pd

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
    """
    Block-based K-fold CV on row order.

    - Chop rows into contiguous blocks of length `block_size`
    - Assign each block to exactly one fold's TEST set
    - Approximate the requested test fraction by choosing n_folds ≈ 1/split

    Returns a list of (train_idx, test_idx) integer position arrays.
    """
    n = len(X)
    if n == 0:
        raise ValueError("X is empty")
    if not (0 < split < 1):
        raise ValueError("split must be between 0 and 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    idx = np.arange(n, dtype=int)

    n_folds = int(round(1.0 / split))
    n_folds = max(2, n_folds)  # avoid degenerate single-fold

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
            f"achieved test fractions ~ {min(achieved):.3f}–{max(achieved):.3f} "
            f"(block_size={block_size}, n_blocks={n_blocks}, shuffle_blocks={shuffle_blocks})"
        )

    return splits


def plot_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    *,
    ncols: int = 2,
    show: bool = True,
):
    """
    Plot train (.) and test (x) points for each CV split.

    Returns
    -------
    fig, axes
    """
    import math
    import matplotlib.pyplot as plt

    n_splits = len(splits)
    if n_splits == 0:
        raise ValueError("splits is empty")

    nrows = math.ceil(n_splits / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), squeeze=False)
    ax_list = axes.flatten()

    for i, (train_idx, test_idx) in enumerate(splits):
        ax = ax_list[i]
        train_ix = X.iloc[train_idx].index
        test_ix = X.iloc[test_idx].index

        ax.plot(y.loc[train_ix].index, y.loc[train_ix], ".", label="Train")
        ax.plot(y.loc[test_ix].index, y.loc[test_ix], "x", label="Test")
        ax.set_title(f"Split {i+1}")
        ax.legend()

    # remove extra axes
    for j in range(n_splits, len(ax_list)):
        fig.delaxes(ax_list[j])

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


# -----------------------------------------------------------------------------
# Cross-validation evaluation (metrics + optional plots)
# -----------------------------------------------------------------------------

def _maybe_inverse(inv_y: Optional[Callable[[Any], Any]], x: Any) -> Any:
    if inv_y is None:
        return x
    return inv_y(x)


def crossval_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    model_factory: Callable[[], object],
    *,
    inv_y: Optional[Callable[[Any], Any]] = None,
    collect_feature_importance: bool = True,
    plot: bool = False,
):
    """
    Cross-validate a model over precomputed splits.

    Returns a dict with:
      - metrics per fold (train/test): arrays for rmse and r2
      - concatenated obs/preds (train/test) for scatter plots
      - mean feature_importance (if available)
      - figures (if plot=True)
    """
    train_scores: List[Dict[str, float]] = []
    test_scores: List[Dict[str, float]] = []
    fi_folds: List[np.ndarray] = []

    y_train_all: List[float] = []
    y_train_pred_all: List[float] = []
    y_test_all: List[float] = []
    y_test_pred_all: List[float] = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        model = model_factory()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # back-transform for evaluation if needed
        y_train_eval = _maybe_inverse(inv_y, y_train)
        y_test_eval = _maybe_inverse(inv_y, y_test)
        y_train_pred_eval = _maybe_inverse(inv_y, y_train_pred)
        y_test_pred_eval = _maybe_inverse(inv_y, y_test_pred)

        train_scores.append({"rmse": _rmse(y_train_eval, y_train_pred_eval), "r2": _r2(y_train_eval, y_train_pred_eval)})
        test_scores.append({"rmse": _rmse(y_test_eval, y_test_pred_eval), "r2": _r2(y_test_eval, y_test_pred_eval)})

        if collect_feature_importance and hasattr(model, "feature_importances_"):
            fi_folds.append(np.asarray(model.feature_importances_, dtype=float))

        y_train_all.extend(_as_1d_float(y_train_eval).tolist())
        y_train_pred_all.extend(_as_1d_float(y_train_pred_eval).tolist())
        y_test_all.extend(_as_1d_float(y_test_eval).tolist())
        y_test_pred_all.extend(_as_1d_float(y_test_pred_eval).tolist())

    def _to_array(score_list: List[Dict[str, float]], key: str) -> np.ndarray:
        return np.array([d.get(key, np.nan) for d in score_list], dtype=float)

    results: Dict[str, Any] = {
        "train": {"rmse": _to_array(train_scores, "rmse"), "r2": _to_array(train_scores, "r2")},
        "test": {"rmse": _to_array(test_scores, "rmse"), "r2": _to_array(test_scores, "r2")},
        "y_train_all": np.array(y_train_all, dtype=float),
        "y_train_pred_all": np.array(y_train_pred_all, dtype=float),
        "y_test_all": np.array(y_test_all, dtype=float),
        "y_test_pred_all": np.array(y_test_pred_all, dtype=float),
    }

    if fi_folds:
        fi = np.nanmean(np.vstack(fi_folds), axis=0)
        results["feature_importance_mean"] = fi
        results["feature_names"] = np.array(getattr(X, "columns", np.arange(len(fi))), dtype=object)

    if plot:
        import matplotlib.pyplot as plt

        figs: Dict[str, Any] = {}

        if "feature_importance_mean" in results:
            fi = results["feature_importance_mean"]
            names = results["feature_names"]
            order = np.argsort(-fi)

            fig, ax = plt.subplots(figsize=(7, 10))
            ax.barh(names[order], fi[order])
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Features")
            ax.set_title("Mean Feature Importance Across Folds")
            fig.tight_layout()
            figs["feature_importance"] = fig

        # obs vs pred plots
        def _scatter(yobs, ypred, title):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(yobs, ypred, alpha=0.5)
            mn = np.nanmin([np.nanmin(yobs), np.nanmin(ypred)])
            mx = np.nanmax([np.nanmax(yobs), np.nanmax(ypred)])
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_title(title)
            fig.tight_layout()
            return fig

        ytr = results["y_train_all"]
        ytrp = results["y_train_pred_all"]
        figs["obs_vs_pred_train"] = _scatter(ytr, ytrp, f"Train: RMSE={_rmse(ytr, ytrp):.3g}, R²={_r2(ytr, ytrp):.3f}")

        yte = results["y_test_all"]
        ytep = results["y_test_pred_all"]
        figs["obs_vs_pred_test"] = _scatter(yte, ytep, f"Test: RMSE={_rmse(yte, ytep):.3g}, R²={_r2(yte, ytep):.3f}")

        results["figures"] = figs

    return results


# -----------------------------------------------------------------------------
# OOF prediction + residual-scale model (for sigma_resid)
# -----------------------------------------------------------------------------

def oof_predictions(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y_train: pd.Series,
    splits: List[Split],
    *,
    inv_y: Optional[Callable[[Any], Any]] = None,
) -> pd.Series:
    """
    Out-of-fold predictions aligned with X.index.

    y_train must be on the training scale (e.g. log-scale if you trained that way).
    inv_y maps predictions back to physical units if provided.
    """
    oof = pd.Series(index=X.index, dtype=float)
    for tr, te in splits:
        m = model_factory()
        m.fit(X.iloc[tr], y_train.iloc[tr])
        pred = m.predict(X.iloc[te]).astype(float)
        pred = _maybe_inverse(inv_y, pred)
        oof.iloc[te] = _as_1d_float(pred)
    return oof


@dataclass
class ResidualScaleModel:
    """Maps a prediction yhat to a residual scale sigma_resid(yhat)."""
    method: str  # 'global' or 'by_pred_quantile'
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
    use_robust: bool = True,
    min_per_bin: int = 200,
) -> ResidualScaleModel:
    """
    Fit sigma_resid from OOF residuals.

    method:
      - 'global': single sigma for all times
      - 'by_pred_quantile': sigma depends on predicted flux magnitude (useful for spiky gases)
    """
    df = pd.DataFrame({"y": y_obs_raw, "yhat": yhat_oof_raw}).dropna()
    resid = (df["y"] - df["yhat"]).to_numpy(dtype=float)

    sigma_global = _robust_sd(resid) if use_robust else float(np.nanstd(resid, ddof=1))

    if method == "global":
        return ResidualScaleModel(method="global", sigma_global=sigma_global)

    # quantile bin edges on yhat
    qs = np.unique(np.array(list(q), dtype=float))
    if qs.size < 2:
        raise ValueError("q must contain at least two quantiles")
    if qs[0] != 0.0:
        qs = np.insert(qs, 0, 0.0)
    if qs[-1] != 1.0:
        qs = np.append(qs, 1.0)

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
        sigmas[b] = _robust_sd(r) if use_robust else float(np.nanstd(r, ddof=1))

    return ResidualScaleModel(
        method="by_pred_quantile",
        sigma_global=sigma_global,
        bin_edges=edges,
        sigma_by_bin=sigmas,
    )


# -----------------------------------------------------------------------------
# Block bootstrap ensemble (for sigma_ens)
# -----------------------------------------------------------------------------

def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices of length n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    idx: List[int] = []
    for s in starts:
        idx.extend(((s + np.arange(block_len)) % n).tolist())
        if len(idx) >= n:
            break
    return np.asarray(idx[:n], dtype=int)


def ensemble_predict(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    *,
    n_models: int = 50,
    block_len: int = 48,
    random_state: int = 42,
    inv_y: Optional[Callable[[Any], Any]] = None,
) -> np.ndarray:
    """
    Block-bootstrap ensemble predictions.

    Returns array shape (n_models, n_times).
    """
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    preds = np.full((n_models, len(X_pred)), np.nan, dtype=float)

    for i in range(n_models):
        idx = block_bootstrap_indices(n=n, block_len=block_len, rng=rng)
        m = model_factory()
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        p = m.predict(X_pred).astype(float)
        p = _maybe_inverse(inv_y, p)
        preds[i, :] = _as_1d_float(p)

    return preds


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
    X_tr: pd.DataFrame
    y_tr: pd.Series                 # training scale (log or raw)
    y_raw: pd.Series                # raw physical units
    inv_y: Optional[Callable[[Any], Any]]
    splits: List[Split]
    yhat_oof_raw: pd.Series
    resid_model: ResidualScaleModel
    n_ens: int
    ens_block_len: int
    random_state: int


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
    n_ens: int = 50,
    ens_block_len: int = 48,
    resid_method: str = "by_pred_quantile",
    resid_q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    resid_min_per_bin: int = 200,
    use_robust_resid: bool = True,
    verbose: bool = True,
) -> GapfillFit:
    """
    Fit a final model + uncertainty components for time-series gapfilling.

    Important: training uses only rows where target and all predictors are present.
    """
    _check_columns(df, [target_col] + list(feature_cols), name="df")

    out = df.copy()

    train_mask = out[target_col].notna() & out[feature_cols].notna().all(axis=1)
    if train_mask.sum() < 10:
        raise ValueError(f"Not enough complete training rows for {target_col} ({train_mask.sum()}).")

    df_train = out.loc[train_mask, feature_cols + [target_col]].copy()
    df_train = df_train.rename(columns={target_col: "_y_raw"})
    y_raw = df_train["_y_raw"].astype(float).copy()

    if undersample:
        df_train, cutoff = undersample_target(
            df_train,
            "_y_raw",
            quantile_cutoff=undersample_quantile,
            fraction=undersample_fraction,
            random_state=random_state,
            plot=False,
            verbose=verbose,
        )
        y_raw = df_train["_y_raw"].astype(float).copy()
        if verbose:
            print(f"Undersampling cutoff (raw units): {cutoff:.6g}")

    inv_y = None
    if log_transform:
        log_fn, inv_fn, min_value, _ = setup_log_transform(df_train, "_y_raw", apply=False, plot=False)
        df_train["_y_train"] = log_fn(df_train["_y_raw"].to_numpy(float))
        inv_y = inv_fn
        if verbose:
            print(f"Log transform enabled (shift based on min={min_value:.6g}).")
    else:
        df_train["_y_train"] = df_train["_y_raw"].to_numpy(float)

    X_tr = df_train[feature_cols].copy()
    y_tr = pd.Series(df_train["_y_train"].to_numpy(float), index=df_train.index)

    if cv_block_size is None:
        cv_block_size = infer_cv_block_size_from_gaps(
            out[target_col],
            quantile=cv_block_quantile,
            fallback=cv_block_fallback,
        )
        if verbose:
            print(f"Inferred cv_block_size={cv_block_size} from gap lengths (q={cv_block_quantile:.2f}).")

    splits = create_block_splits(
        X_tr,
        split=cv_split,
        block_size=cv_block_size,
        random_state=random_state,
        verbose=verbose,
    )

    yhat_oof_raw = oof_predictions(model_factory=model_factory, X=X_tr, y_train=y_tr, splits=splits, inv_y=inv_y)

    resid_model = fit_residual_scale(
        y_obs_raw=y_raw,
        yhat_oof_raw=yhat_oof_raw,
        method=resid_method,
        q=resid_q,
        use_robust=use_robust_resid,
        min_per_bin=resid_min_per_bin,
    )

    m_final = model_factory()
    m_final.fit(X_tr, y_tr)

    return GapfillFit(
        target_col=target_col,
        feature_cols=list(feature_cols),
        model_factory=model_factory,
        model_final=m_final,
        X_tr=X_tr,
        y_tr=y_tr,
        y_raw=y_raw,
        inv_y=inv_y,
        splits=splits,
        yhat_oof_raw=yhat_oof_raw,
        resid_model=resid_model,
        n_ens=int(n_ens),
        ens_block_len=int(ens_block_len),
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
    Apply the fitted model to a full dataframe and attach uncertainty terms.

    Adds columns:
      - {prefix}_yhat
      - {prefix}_sigmaEns
      - {prefix}_sigmaResid
      - {prefix}_sigmaGF
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
        yhat_ok = fit.model_final.predict(X_full.loc[ok]).astype(float)
        yhat_ok = _maybe_inverse(fit.inv_y, yhat_ok)
        yhat[ok] = _as_1d_float(yhat_ok)

        preds_ens = ensemble_predict(
            model_factory=fit.model_factory,
            X_train=fit.X_tr,
            y_train=fit.y_tr,
            X_pred=X_full.loc[ok],
            n_models=fit.n_ens,
            block_len=fit.ens_block_len,
            random_state=fit.random_state,
            inv_y=fit.inv_y,
        )
        sigma_ens_ok = np.nanstd(preds_ens, axis=0, ddof=1)
        sigma_ens[ok] = _as_1d_float(sigma_ens_ok)

        sigma_resid_ok = fit.resid_model.sigma(yhat_ok)
        sigma_resid[ok] = _as_1d_float(sigma_resid_ok)

        sigma_gf[ok] = combine_gapfill_sigma(sigma_ens_ok, sigma_resid_ok, min_sigma=min_sigma)

    out[f"{prefix}_yhat"] = yhat
    out[f"{prefix}_sigmaEns"] = sigma_ens
    out[f"{prefix}_sigmaResid"] = sigma_resid
    out[f"{prefix}_sigmaGF"] = sigma_gf

    return out
