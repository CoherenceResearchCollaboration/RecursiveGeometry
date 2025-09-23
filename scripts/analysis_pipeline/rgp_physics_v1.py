"""
# scripts/analysis_pipeline/rgp_physics_v1.py
# RGP Physics v1 — χ–β plane with optional curvature, torsion metrics, reliability flags,
# and optional p/q joins from the gamma-attractor affinity table.

Core model tested (at fixed tower (n_i, n_k)):
     # Thread frame in log10:
     #   log10(nu) = chi  + beta * gamma
     # where:
     #   beta = slope (tilt) with respect to gamma
     #   chi  = baseline (log-frequency at gamma=0)
    log10(nu) = χ + β * gamma

Outputs:
  1) tower-level fits with uncertainties and diagnostics,
  2) pairwise cross-ion mappings on the same tower (Δβ, Δχ, Δθ, s=10^Δβ),
  3) ion-level summary aggregates.

Notes
50-Pro. August 18, 2025
-----
- Weighted least squares (WLS): weights default to obs_hits-type counts if present,
  else uniform. WLS solves (X^T W X) β̂ = X^T W y with W=diag(w).
- Orientation angle: θ = atan(β) [radians and degrees].
# Arc length along the fitted sheet over observed span: L = Δγ * sqrt(1 + β^2).
- Mapping residuals are computed only when two ions on the same tower share γ points.

python -m scripts.analysis_pipeline.rgp_physics_v1 \
  --photon-dir data/meta/ion_photon_ladders_mu-1 \
  --out-dir    data/results/rgp_v1

python -m scripts.analysis_pipeline.rgp_physics_v1 \
  --photon-dir data/meta/ion_photon_ladders_mu-1 \
  --out-dir    data/results/rgp_v1 \
  --curvature \
  --affinity data/meta/gamma_attractor_affinity_mu-1.csv

"""

import argparse, os, math, warnings
import pandas as pd
import numpy as np
from itertools import combinations
import numpy as np
from math import erf as _scalar_erf

try:
    # Prefer SciPy if available (fast, vectorized)
    from scipy.special import erf as _erf
except Exception:
    # No SciPy? Make a vectorized shim from math.erf
    def _erf(x):
        x = np.asarray(x, dtype=float)
        # vectorize the scalar math.erf; returns a numpy array
        return np.vectorize(_scalar_erf)(x)

try:
    from numpy import trapezoid as _trap  # NumPy ≥2.0
except Exception:
    from numpy import trapz as _trap

# ---------- IO ----------

def normalize_pairs_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    cols = set(pairs.columns)

    # 1) Rotation alias
    if "delta_theta_deg" not in cols and "tilt_deg" in cols:
        pairs["delta_theta_deg"] = pairs["tilt_deg"]
    elif "delta_theta_deg" not in cols and {"chi_A", "chi_B"} <= cols:
        # compute from chis if neither present
        pairs["delta_theta_deg"] = (
            np.degrees(np.arctan(pairs["chi_B"].astype(float))) -
            np.degrees(np.arctan(pairs["chi_A"].astype(float)))
        )

    # 2) Scale alias
    if "scale_factor_10_pow_delta_beta" not in cols:
        if "scale_10dBeta" in cols:
            pairs["scale_factor_10_pow_delta_beta"] = pairs["scale_10dBeta"]
        elif "delta_beta" in cols:
            pairs["scale_factor_10_pow_delta_beta"] = 10.0 ** pairs["delta_beta"].astype(float)

    return pairs

def _read_table(path: str) -> pd.DataFrame:
    """
    Robust reader for your 'CSV' files that may have provenance headers and tab/space delims.
    - Skips lines beginning with '#'
    - Autodetects delimiter (comma, tab, or whitespace)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    sample = "".join(lines[:50])
    # heuristic: if tabs present, use '\t'; elif commas, use ','; else whitespace
    if "\t" in sample and "," not in sample:
        sep, engine = "\t", "python"
    elif "," in sample:
        sep, engine = ",", "python"
    else:
        sep, engine = r"\s+", "python"
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)), sep=sep, engine=engine)

def load_photon_tables(photon_dir: str) -> pd.DataFrame:
    files = [os.path.join(photon_dir, f) for f in os.listdir(photon_dir)
             if f.endswith("_photon_ladder.csv")]
    frames = []
    for p in files:
        try:
            df = _read_table(p)
            # normalize column names
            df.columns = [c.strip() for c in df.columns]
            # common schema expected:
            # ion, n_i, n_k, gamma_bin, frequency_hz, delta_e_ev, n_photons_matched, obs_hits_gamma, n_hits_gamma, obs_hits_gamma_context
            req = {"ion","n_i","n_k","gamma_bin","frequency_hz"}
            if not req.issubset(set(df.columns)):
                # minimal: ion n_i n_k gamma_bin frequency_hz must exist
                missing = req - set(df.columns)
                raise ValueError(f"Missing columns {missing} in {os.path.basename(p)}")
            frames.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read {p}: {e}")
    if not frames:
        raise RuntimeError(f"No photon ladders found in {photon_dir}")
    all_df = pd.concat(frames, ignore_index=True)
    # enforce dtypes
    for c in ["n_i","n_k","gamma_bin","frequency_hz"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    if "obs_hits_gamma" not in all_df:
        all_df["obs_hits_gamma"] = 1.0
    all_df["weight"] = all_df["obs_hits_gamma"].astype(float).clip(lower=0.0)
    all_df["log10_nu"] = np.log10(all_df["frequency_hz"].astype(float).clip(lower=1.0))
    return all_df.dropna(subset=["n_i","n_k","gamma_bin","log10_nu","weight"])

def load_affinity_table(path: str) -> pd.DataFrame:
    if not path:
        return None
    df = _read_table(path)
    # Expected columns per user:
    # ion, gamma_bin, obs_hits, n_hits, obs_hits_raw, n_i, n_k,
    # p_val, q_val, tol_meV, null_mean, null_sigma, z_score, direction,
    # frac_outward, source
    needed = {"ion","gamma_bin","n_i","n_k","p_val","q_val"}
    if not needed.issubset(df.columns):
        warnings.warn(f"Affinity table missing columns: {needed - set(df.columns)}; skipping affinity join.")
        return None
    for c in ["n_i","n_k","gamma_bin","p_val","q_val","obs_hits"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "obs_hits" not in df.columns:
        df["obs_hits"] = 1.0
    return df.dropna(subset=["ion","n_i","n_k","gamma_bin"])

# ---------- Weighted regression & geometry ----------

def wls_polyfit(x, y, w, degree=1):
    """
    Weighted LS polynomial fit of degree 1 or 2.

    Returns a dict:
      {
        # We expose 'chi' as intercept (baseline) and 'beta' as slope (tilt).
        "names": ["chi","beta"[,"c"]],
        "coef": {"chi":..., "beta":..., "c":...},
        "se":   {"beta":..., "chi":..., "c":...},
        "p":    {"beta":..., "chi":..., "c":...},   # two‑sided (normal approx)
        "rmse": <weighted RMSE in log10 Hz>,
        "r2_w": <weighted R^2>,
        "aic":  <AIC (Gaussian; robust)>,
        "resid": residuals (np.ndarray),
        "yhat":  fitted values (np.ndarray)
      }
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    # Design matrix
    if degree == 1:
        # intercept first (chi), then slope (beta)
        X = np.column_stack([np.ones_like(x), x]); names = ["chi","beta"]
    elif degree == 2:
        # [1, x, x^2] -> chi (intercept), beta (slope), c (quadratic)
        X = np.column_stack([np.ones_like(x), x, x*x]); names = ["chi","beta","c"]
    else:
        raise ValueError("degree must be 1 or 2")

    n, k = X.shape
    # Weights (non‑negative), build sqrt(W) once
    w_pos = np.clip(w, 0.0, None)
    sw = np.sqrt(w_pos)

    # Solve weighted LS via normal equations on sqrt‑weighted system
    Xw = X * sw[:, None]
    yw = y * sw
    coef_vec, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    # Predictions & residuals (unweighted space)
    yhat  = X @ coef_vec
    resid = y - yhat

    # Weighted SSE/SST about weighted mean
    wsum   = float(max(w_pos.sum(), 1e-12))
    sse    = float(np.sum(w_pos * resid**2))
    ybar_w = float(np.sum(w_pos * y) / wsum)
    sst    = float(np.sum(w_pos * (y - ybar_w)**2))
    r2_w   = 1.0 - (sse / sst) if sst > 0 else np.nan

    # Weighted RMSE
    rmse = math.sqrt(sse / wsum)

    # Covariance & SEs (use unbiased-ish variance scaled by dof)
    dof   = max(n - k, 1)
    s2    = (sse / wsum) * (n / dof)              # per-weight variance × (n/dof)
    XtWX  = X.T @ (w_pos[:, None] * X)
    cov   = s2 * np.linalg.pinv(XtWX)
    se_vec = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    # t-stats and two‑sided p-values (normal approx; vectorized _erf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = coef_vec / np.where(se_vec > 0, se_vec, np.inf)
    pvals = 2.0 * (1.0 - 0.5 * (1.0 + _erf(np.abs(tvals) / np.sqrt(2.0))))

    # Robust AIC (avoid log(0) on perfect fits)
    sigma2_hat = max(sse / wsum, 1e-300)
    aic = 2*k + n * (1.0 + math.log(2.0 * math.pi * sigma2_hat))

    return {
        "names": names,
        "coef": {names[i]: float(coef_vec[i]) for i in range(k)},
        "se":   {names[i]: float(se_vec[i])  for i in range(k)},
        "p":    {names[i]: float(pvals[i])   for i in range(k)},
        "rmse": float(rmse),
        "r2_w": float(r2_w),
        "aic":  float(aic),
        "resid": resid,
        "yhat":  yhat
    }

def arc_length_and_curvature(beta, chi, c, gmin, gmax):
    """
    Numeric arc length and mean curvature magnitude across [gmin, gmax].
    For linear (c=0) we use closed-form arc length: (gmax-gmin)*sqrt(1+beta^2),
    since y'(gamma) = beta (slope). With curvature, y'(gamma) = beta + 2*c*gamma.
    Otherwise numeric quadrature with 128 samples.
    """
    gmin, gmax = float(gmin), float(gmax)
    if gmax <= gmin:
        return 0.0, 0.0
    if abs(c) < 1e-12:
        L = (gmax - gmin) * math.sqrt(1.0 + beta*beta)
        mean_curv = 0.0
        return L, mean_curv
    # numeric
    N = 128
    gs = np.linspace(gmin, gmax, N)
    yprime = beta + 2.0*c*gs
    ypp = 2.0*c
    speed = np.sqrt(1.0 + yprime*yprime)
    L = float(_trap(speed, gs))
    curv = np.abs(ypp) / np.power(1.0 + yprime*yprime, 1.5)
    mean_curv = float(_trap(curv, gs) / (gmax - gmin))
    return L, mean_curv

# ---------- Fits per tower ----------

def fit_towers(ph_df, use_curvature=False, min_bins=6, rmse_thresh=0.025, min_weight=30.0, affinity=None):
    out_rows = []
    grouped = ph_df.groupby(["ion","n_i","n_k"], sort=False)
    for (ion, ni, nk), g in grouped:
        g = g.sort_values("gamma_bin")
        x = g["gamma_bin"].values
        y = g["log10_nu"].values
        w = g["weight"].values
        if len(np.unique(x)) < 2:
            continue
        # linear
        lin = wls_polyfit(x, y, w, degree=1)
        chi  = lin["coef"]["chi"]   # baseline (intercept)
        beta = lin["coef"]["beta"]  # slope (tilt)
        aic_lin = lin["aic"]
        # ----- curvature optional (safe defaults until/if quadratic runs) -----
        c_val = np.nan
        c_se = np.nan
        c_p = np.nan
        aic_quad = np.nan
        delta_aic = np.nan
        geom_c = 0.0
        if use_curvature and len(np.unique(x)) >= 3:
            quad = wls_polyfit(x, y, w, degree=2)
            aic_quad = quad["aic"]
            c_val = quad["coef"].get("c", np.nan)
            c_se = quad["se"].get("c", np.nan)
            c_p = quad["p"].get("c", np.nan)
            delta_aic = aic_quad - aic_lin
            # prefer quad only for reporting torsion metrics; χ,β remain the plane definition
            # torsion metrics computed with quad if it improves AIC by <= -2; otherwise linear curve.
            use_quad_for_geom = (delta_aic <= -2.0)
            geom_c = c_val if use_quad_for_geom and np.isfinite(c_val) else 0.0

        gmin, gmax = float(x.min()), float(x.max())
        L, mean_curv = arc_length_and_curvature(beta, chi, geom_c, gmin, gmax)
        torsion_index = L * mean_curv
        theta = math.degrees(math.atan(beta))  # orientation from slope (β)
        # reliability
        flags = {
            "fit_sparse_bins": int(len(np.unique(x)) < min_bins),
            "fit_low_weight": int(float(w.sum()) < float(min_weight)),
            "fit_high_rmse": int(lin["rmse"] > rmse_thresh),
            "fit_curvature_needed": int(use_curvature and np.isfinite(delta_aic) and delta_aic <= -2.0),
        }
        reliability_score = max(0.0, 1.0 - 0.25*flags["fit_sparse_bins"]
                                      - 0.25*flags["fit_low_weight"]
                                      - 0.35*flags["fit_high_rmse"]
                                      - 0.15*flags["fit_curvature_needed"])
        row = dict(
            ion=ion, n_i=float(ni), n_k=float(nk),
            beta=float(beta), chi=float(chi), theta_deg=float(theta),
            rmse_log10_hz=float(lin["rmse"]), r2_w=float(lin["r2_w"]),
            n_bins=int(len(x)), sum_weight=float(w.sum()),
            gamma_min=gmin, gamma_max=gmax,
            arc_len=float(L), mean_curv=float(mean_curv), torsion_index=float(torsion_index),
            aic_lin=float(aic_lin), aic_quad=float(aic_quad) if np.isfinite(aic_quad) else np.nan,
            delta_aic=float(delta_aic) if np.isfinite(delta_aic) else np.nan,
            c_quad=float(c_val) if np.isfinite(c_val) else np.nan,
            c_se=float(c_se) if np.isfinite(c_se) else np.nan,
            c_p=float(c_p) if np.isfinite(c_p) else np.nan,
            **flags, reliability_score=float(reliability_score),
        )
        # optional p/q join per tower
        if affinity is not None:
            af = affinity.merge(g[["ion","n_i","n_k","gamma_bin"]].drop_duplicates(),
                                on=["ion","n_i","n_k","gamma_bin"], how="inner")
            pq_w = float(af["obs_hits"].sum()) if "obs_hits" in af.columns else float(len(af))
            cov = float(len(af)) / float(len(np.unique(x))) if len(x) else 0.0
            for name, col in [("p","p_val"),("q","q_val")]:
                if col in af.columns and len(af):
                    weights = af["obs_hits"].astype(float) if "obs_hits" in af.columns else np.ones(len(af))
                    wsum = float(weights.sum()) if float(weights.sum())>0 else 1.0
                    wmean = float((weights * af[col].astype(float)).sum()/wsum)
                    row[f"{name}_mean"] = wmean
                    row[f"{name}_median"] = float(af[col].median())
                else:
                    row[f"{name}_mean"] = np.nan
                    row[f"{name}_median"] = np.nan
            row["p_q_weight"] = pq_w
            row["p_q_coverage"] = cov
        out_rows.append(row)
    return pd.DataFrame(out_rows)

# ---------- Pairwise maps (same (n_i,n_k)) ----------

def build_pairwise_maps(fits: pd.DataFrame, ph_df: pd.DataFrame, use_curvature=False,
                        min_overlap=4, map_rmse_thresh=0.030, affinity=None):
    out = []
    # Prepare per (ion,ni,nk) the available gamma bins and predictions:
    tower_groups = ph_df.groupby(["ion","n_i","n_k"])
    pred_cache = {}
    for key, g in tower_groups:
        ion, ni, nk = key
        fit = fits[(fits.ion==ion) & (fits.n_i==float(ni)) & (fits.n_k==float(nk))]
        if not len(fit):
            continue
        frow = fit.iloc[0]
        beta, chi = frow["beta"], frow["chi"]  # beta=slope, chi=baseline
        c = frow["c_quad"] if (use_curvature and pd.notna(frow["c_quad"])) else 0.0
        tmp = g[["gamma_bin","log10_nu","weight"]].copy()
        # Predict A's sheet at its own gamma grid: y = chi + beta*γ + c*γ^2
        tmp["yhat"] = chi + beta*tmp["gamma_bin"] + c*(tmp["gamma_bin"]**2)
        pred_cache[key] = tmp
    # Pairwise within each tower address:
    for (ni, nk), block in fits.groupby(["n_i","n_k"]):
        ions = list(block["ion"].unique())
        for A, B in combinations(ions, 2):
            kA = (A, ni, nk); kB = (B, ni, nk)
            if kA not in pred_cache or kB not in pred_cache:
                continue
            dA = pred_cache[kA]; dB = pred_cache[kB]
            # overlap on gamma
            overlap = dA.merge(dB, on="gamma_bin", suffixes=("_A","_B"))
            n_ov = int(len(overlap))
            if n_ov == 0:
                continue
            # Mapping from A to B via parameter deltas
            fA = fits[(fits.ion==A)&(fits.n_i==ni)&(fits.n_k==nk)].iloc[0]
            fB = fits[(fits.ion==B)&(fits.n_i==ni)&(fits.n_k==nk)].iloc[0]
            delta_beta = fB["beta"] - fA["beta"]  # slope difference
            delta_chi  = fB["chi"]  - fA["chi"]   # baseline difference
            cA = fA["c_quad"] if (use_curvature and pd.notna(fA["c_quad"])) else 0.0
            cB = fB["c_quad"] if (use_curvature and pd.notna(fB["c_quad"])) else 0.0
            delta_c    = cB - cA
            # Predict B from A at overlap γ:
            g = overlap["gamma_bin"].values
            yA = overlap["yhat_A"].values  # chi_A + beta_A*γ + c_A*γ^2
            yB = overlap["yhat_B"].values  # chi_B + beta_B*γ + c_B*γ^2
            # yB_pred = (chi_A + Δχ) + (beta_A + Δβ) * γ + (c_A + Δc) * γ^2
            yB_pred = yA + delta_chi + delta_beta*g + delta_c*(g**2)
            w  = overlap["weight_B"].values
            resid = yB - yB_pred
            w = np.asarray(w, float)
            sse = float((w * resid**2).sum()); wsum = float(max(w.sum(), 1e-12))
            rmse_map = math.sqrt(sse / wsum)
            # R2 (weighted) on overlap:
            ybar_w = float((w*yB).sum()/wsum)
            sst = float((w * (yB - ybar_w)**2).sum())
            r2_map = 1.0 - (sse/sst) if sst > 0 else np.nan
            tilt_deg = math.degrees(math.atan(fB["beta"])) - math.degrees(math.atan(fA["beta"]))
            # Reliability flags
            pair_flags = {
                "pair_low_overlap": int(n_ov < min_overlap),
                "pair_high_rmse_map": int(rmse_map > map_rmse_thresh),
            }
            pair_score = max(0.0, 1.0 - 0.6*pair_flags["pair_low_overlap"]
                                      - 0.4*pair_flags["pair_high_rmse_map"])
            row = dict(
                n_i=float(ni), n_k=float(nk),
                ion_A=A, ion_B=B,
                beta_A=float(fA["beta"]), chi_A=float(fA["chi"]),
                c_A=float(cA) if use_curvature else np.nan,
                beta_B=float(fB["beta"]), chi_B=float(fB["chi"]),
                c_B=float(cB) if use_curvature else np.nan,
                delta_beta=float(delta_beta), delta_chi=float(delta_chi),
                delta_c=float(delta_c) if use_curvature else np.nan,
                scale_factor_10_pow_delta_beta=float(10.0**delta_beta),
                tilt_deg=float(tilt_deg),
                rmse_map_B=float(rmse_map), r2_map_B=float(r2_map),
                n_overlap=int(n_ov), **pair_flags, reliability_score=float(pair_score)
            )
            # optional p/q aggregates for A & B across overlap bins
            if affinity is not None:
                for label, ion in [("A", A), ("B", B)]:
                    af = affinity[(affinity.ion==ion) &
                                  (affinity.n_i==float(ni)) &
                                  (affinity.n_k==float(nk))]
                    if len(af):
                        af = af.merge(overlap[["gamma_bin"]], on="gamma_bin", how="inner")
                    if len(af):
                        weights = af["obs_hits"].astype(float) if "obs_hits" in af.columns else np.ones(len(af))
                        wsum = float(weights.sum()) if float(weights.sum())>0 else 1.0
                        for name, col in [("p","p_val"),("q","q_val")]:
                            if col in af.columns:
                                wmean = float((weights * af[col].astype(float)).sum()/wsum)
                                row[f"{name}_mean_{label}"] = wmean
                                row[f"{name}_median_{label}"] = float(af[col].median())
                        row[f"p_q_weight_{label}"] = float(wsum)
                        row[f"p_q_coverage_{label}"] = float(len(af)) / float(max(n_ov,1))
                    else:
                        row[f"p_mean_{label}"] = np.nan; row[f"p_median_{label}"] = np.nan
                        row[f"q_mean_{label}"] = np.nan; row[f"q_median_{label}"] = np.nan
                        row[f"p_q_weight_{label}"] = 0.0; row[f"p_q_coverage_{label}"] = 0.0
            out.append(row)
    return pd.DataFrame(out)

# ---------- Summaries & coverage ----------

def summarize_ions(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ion, g in fits.groupby("ion"):
        w = g["sum_weight"].astype(float).values
        w = np.where(np.isfinite(w), w, 0.0)
        wsum = float(w.sum()) if w.sum()>0 else 1.0
        def wmean(col): 
            v = g[col].astype(float).values
            v = np.where(np.isfinite(v), v, np.nan)
            return float(np.nansum(w*v)/wsum)
        def wsumcol(col):
            return float(np.nansum(np.where(np.isfinite(g[col]), g[col].astype(float), 0.0)))
        rows.append(dict(
            ion=ion,
            n_towers=int(len(g)),
            mean_beta=wmean("beta"),   # slope (tilt)
            mean_chi=wmean("chi"),     # baseline (intercept)
            mean_theta_deg=wmean("theta_deg"),
            mean_c_quad=wmean("c_quad"),
            share_curvature_needed=float(np.nanmean(g["fit_curvature_needed"])) if "fit_curvature_needed" in g else np.nan,
            total_arc_len=wsumcol("arc_len"),
            total_torsion_index=wsumcol("torsion_index"),
            mean_rmse=wmean("rmse_log10_hz"),
            mean_reliability=wmean("reliability_score"),
            clean_towers=int((g["reliability_score"]>=0.7).sum())
        ))
    return pd.DataFrame(rows)

def coverage_table(ph_df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    cov = ph_df.groupby(["ion","n_i","n_k"], as_index=False)\
               .agg(n_bins=("gamma_bin","nunique"),
                    sum_weight=("weight","sum"))
    merged = cov.merge(fits[["ion","n_i","n_k","reliability_score"]],
                       on=["ion","n_i","n_k"], how="left")
    return merged

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="RGP Physics v1 — χβ-plane fits + torsion metrics")
    ap.add_argument("--res-dir", required=False, help="(unused in v1; kept for compatibility)")
    ap.add_argument("--photon-dir", required=True, help="Directory with *_photon_ladder.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--affinity", required=False, default=None,
                    help="Path to gamma_attractor_affinity_bio_vacuum_mu-1.csv to emit p/q stats")
    ap.add_argument("--curvature", action="store_true", help="Enable quadratic term in γ")
    ap.add_argument("--min-bins", type=int, default=6)
    ap.add_argument("--min-weight", type=float, default=30.0)
    ap.add_argument("--rmse-thresh", type=float, default=0.025)
    ap.add_argument("--min-overlap", type=int, default=4)
    ap.add_argument("--map-rmse-thresh", type=float, default=0.030)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ph_df = load_photon_tables(args.photon_dir)
    aff = load_affinity_table(args.affinity) if args.affinity else None

    fits = fit_towers(ph_df, use_curvature=args.curvature,
                      min_bins=args.min_bins, rmse_thresh=args.rmse_thresh,
                      min_weight=args.min_weight, affinity=aff)

    pairs = build_pairwise_maps(fits, ph_df, use_curvature=args.curvature,
                                min_overlap=args.min_overlap,
                                map_rmse_thresh=args.map_rmse_thresh,
                                affinity=aff)

    ions = summarize_ions(fits)
    cover = coverage_table(ph_df, fits)

    # Write
    fits.to_csv(os.path.join(args.out_dir, "rgp_v1_tower_fits.csv"), index=False)
    pairs.to_csv(os.path.join(args.out_dir, "rgp_v1_pairwise_maps.csv"), index=False)
    ions.to_csv(os.path.join(args.out_dir, "rgp_v1_ion_summary.csv"), index=False)
    cover.to_csv(os.path.join(args.out_dir, "rgp_v1_tower_coverage.csv"), index=False)

if __name__ == "__main__":
    main()
