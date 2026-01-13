
"""
Utilities to attach uncertainty terms to EC flux gapfilling.

Core idea:
- Observed half-hours: use EddyPro random uncertainty (rand_err) as sigma_ec(t).
- Gapfilled half-hours: build sigma_gf(t) from
    (a) ensemble spread of the ML model (sigma_ens(t))
    (b) out-of-sample residual scale from time-series CV (sigma_resid(t))
  and combine as: sigma_gf(t) = sqrt(sigma_ens(t)^2 + sigma_resid(t)^2)

Notes:
- If you log-transform y, pass an inverse transform function `inv_y`.
- For time series, residual scale is best estimated from *out-of-fold* (OOF) predictions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Optional, Dict

try:
    from scipy.stats import median_abs_deviation
except Exception:  # scipy optional
    median_abs_deviation = None


Idx = np.ndarray
Split = Tuple[Idx, Idx]


def _robust_sd(x: np.ndarray) -> float:
    """Robust SD estimate from MAD (handles heavy tails better than sd)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    if median_abs_deviation is not None:
        return 1.4826 * float(median_abs_deviation(x, scale=1.0, nan_policy="omit"))
    # fallback: manual MAD
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * float(mad)


def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular block bootstrap indices of length n.
    Preserves short-range autocorrelation better than iid bootstrap.
    """
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    idx = []
    for s in starts:
        idx.extend(((s + np.arange(block_len)) % n).tolist())
        if len(idx) >= n:
            break
    return np.asarray(idx[:n], dtype=int)


def oof_predictions(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y_train: pd.Series,
    splits: List[Split],
    inv_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.Series:
    """
    Out-of-fold predictions aligned with X.index.
    y_train is on the scale you train on (e.g., log-scale if you transform).
    inv_y maps model output back to original physical units (optional).
    """
    oof = pd.Series(index=X.index, dtype=float)
    for (tr, te) in splits:
        m = model_factory()
        m.fit(X.iloc[tr], y_train.iloc[tr])
        pred = m.predict(X.iloc[te]).astype(float)
        if inv_y is not None:
            pred = inv_y(pred)
        oof.iloc[te] = pred
    return oof


@dataclass
class ResidualScaleModel:
    """Maps a prediction (or other key) to a residual scale sigma_resid."""
    method: str  # 'global' or 'by_pred_quantile'
    sigma_global: float
    bin_edges: Optional[np.ndarray] = None
    sigma_by_bin: Optional[np.ndarray] = None

    def sigma(self, yhat: np.ndarray) -> np.ndarray:
        if self.method == "global" or self.bin_edges is None or self.sigma_by_bin is None:
            return np.full_like(yhat, self.sigma_global, dtype=float)
        # digitize into bins
        b = np.digitize(yhat, self.bin_edges, right=True) - 1  # 0..n_bins-1
        b = np.clip(b, 0, len(self.sigma_by_bin) - 1)
        return self.sigma_by_bin[b].astype(float)


def fit_residual_scale(
    y_obs_raw: pd.Series,
    yhat_oof_raw: pd.Series,
    method: str = "by_pred_quantile",
    q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    use_robust: bool = True,
    min_per_bin: int = 200,
) -> ResidualScaleModel:
    """
    Fit sigma_resid based on OOF residuals.
    - global: single sigma for all times
    - by_pred_quantile: sigma depends on predicted flux magnitude (useful for spiky gases)
    """
    df = pd.DataFrame({"y": y_obs_raw, "yhat": yhat_oof_raw}).dropna()
    resid = (df["y"] - df["yhat"]).to_numpy(dtype=float)
    if use_robust:
        sigma_global = _robust_sd(resid)
    else:
        sigma_global = float(np.nanstd(resid, ddof=1))

    if method == "global":
        return ResidualScaleModel(method="global", sigma_global=sigma_global)

    # build quantile bins on yhat
    qs = np.array(list(q), dtype=float)
    qs = np.unique(qs)
    if qs[0] != 0.0:
        qs = np.insert(qs, 0, 0.0)
    if qs[-1] != 1.0:
        qs = np.append(qs, 1.0)

    edges = np.quantile(df["yhat"].to_numpy(dtype=float), qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # compute sigma per bin
    bin_id = np.digitize(df["yhat"].to_numpy(dtype=float), edges, right=True) - 1
    n_bins = len(edges) - 1
    sigmas = np.full(n_bins, sigma_global, dtype=float)  # fallback to global
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() < min_per_bin:
            continue
        r = resid[mask]
        sigmas[b] = _robust_sd(r) if use_robust else float(np.nanstd(r, ddof=1))

    return ResidualScaleModel(method="by_pred_quantile", sigma_global=sigma_global, bin_edges=edges, sigma_by_bin=sigmas)


def ensemble_predict(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    n_models: int = 50,
    block_len: int = 48,  # 48 half-hours = 1 day
    random_state: int = 42,
    inv_y: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Block-bootstrap ensemble predictions.
    Returns array shape (n_models, n_times).
    """
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    preds = np.zeros((n_models, len(X_pred)), dtype=float)

    for i in range(n_models):
        idx = block_bootstrap_indices(n=n, block_len=block_len, rng=rng)
        m = model_factory()
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        p = m.predict(X_pred).astype(float)
        if inv_y is not None:
            p = inv_y(p)
        preds[i, :] = p
    return preds


def combine_gapfill_sigma(
    sigma_ens: np.ndarray,
    sigma_resid: np.ndarray,
    min_sigma: float = 0.0,
) -> np.ndarray:
    """sigma_gf = sqrt(sigma_ens^2 + sigma_resid^2), with optional floor."""
    sgf = np.sqrt(np.square(sigma_ens) + np.square(sigma_resid))
    if min_sigma > 0:
        sgf = np.maximum(sgf, min_sigma)
    return sgf


def gapfill_with_uncertainty(
    df_obs: pd.DataFrame,
    df_full: pd.DataFrame,
    *,
    target_col: str,                 # observed flux column (physical units)
    feature_cols: list[str],         # model features
    model_factory: Callable[[], object],
    splits: List[Split],             # time-series CV splits on df_obs
    rand_err_col: Optional[str] = None,  # EddyPro rand_err column in df_full
    y_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
    y_inv: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_ens: int = 50,
    block_len: int = 48,
    resid_method: str = "by_pred_quantile",  # or "global"
    resid_q=(0, 0.5, 0.8, 0.95, 1.0),
    resid_min_per_bin: int = 200,
    use_robust_resid: bool = True,
    prefix: str = "GF",
    is_obs_mask: Optional[pd.Series] = None, # if you have a custom "kept" mask
) -> pd.DataFrame:
    """
    Returns df_full with:
      - {prefix}_yhat
      - {prefix}_sigmaEns
      - {prefix}_sigmaResid
      - {prefix}_sigmaGF
      - {prefix}_sigmaUsed (switches between rand_err and sigmaGF if rand_err_col provided)
    """
    # ---- training data ----
    y_raw = df_obs[target_col].astype(float).copy()
    X_tr  = df_obs[feature_cols].copy()

    if y_transform is None:
        y_tr = y_raw
    else:
        y_tr = y_transform(y_raw)

    # ---- OOF predictions (physical units) ----
    yhat_oof_raw = oof_predictions(
        model_factory=model_factory,
        X=X_tr,
        y_train=y_tr,
        splits=splits,
        inv_y=y_inv
    )

    # ---- residual-scale model sigma_resid(.) ----
    resid_model = fit_residual_scale(
        y_obs_raw=y_raw,
        yhat_oof_raw=yhat_oof_raw,
        method=resid_method,
        q=resid_q,
        use_robust=use_robust_resid,
        min_per_bin=resid_min_per_bin
    )

    # ---- fit on all obs ----
    m_final = model_factory()
    m_final.fit(X_tr, y_tr)

    X_full = df_full[feature_cols].copy()
    yhat = m_final.predict(X_full).astype(float)
    if y_inv is not None:
        yhat = y_inv(yhat)

    # ---- ensemble spread ----
    preds_ens = ensemble_predict(
        model_factory=model_factory,
        X_train=X_tr,
        y_train=y_tr,
        X_pred=X_full,
        n_models=n_ens,
        block_len=block_len,
        random_state=42,
        inv_y=y_inv
    )
    sigma_ens = preds_ens.std(axis=0, ddof=1)

    # ---- sigma_resid(t) from yhat(t) ----
    sigma_resid = resid_model.sigma(np.asarray(yhat, dtype=float))

    # ---- combine ----
    sigma_gf = combine_gapfill_sigma(sigma_ens=sigma_ens, sigma_resid=sigma_resid)

    out = df_full.copy()
    out[f"{prefix}_yhat"] = yhat
    out[f"{prefix}_sigmaEns"] = sigma_ens
    out[f"{prefix}_sigmaResid"] = sigma_resid
    out[f"{prefix}_sigmaGF"] = sigma_gf

    # ---- "used" sigma (piecewise) ----
    if is_obs_mask is None:
        # default: observed where target exists in df_full
        is_obs_mask = out[target_col].notna()

    if rand_err_col is not None and rand_err_col in out.columns:
        out[f"{prefix}_sigmaUsed"] = np.where(
            is_obs_mask.astype(bool),
            out[rand_err_col].astype(float),
            out[f"{prefix}_sigmaGF"].astype(float)
        )
    else:
        out[f"{prefix}_sigmaUsed"] = np.where(
            is_obs_mask.astype(bool),
            np.nan,
            out[f"{prefix}_sigmaGF"].astype(float)
        )

    return out
