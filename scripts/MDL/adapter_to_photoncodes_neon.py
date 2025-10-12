#!/usr/bin/env python3
# scripts/MDL/adapter_to_photoncodes_neon.py
"""
Adapter: NIST-style Ne I lines (ASCII/CSV) -> photons list CSV (emission)
BOING! Note: We use this script for lamps other than Ne I. The name carries it's original use, but it is now more general purpose.

Expected input file (flexible headers):
    data/raw/lines/Ne_I_lines_raw.txt
with columns including some combination of:
    - obs_wl_vac(nm), obs_wl_air(nm)
    - ritz_wl_vac(nm), ritz_wl_air(nm)
    - intens (or Intensity), Acc (accuracy grade), ...
The adapter auto-detects columns and lets you prefer Observed vs Ritz and Vacuum vs Air.

We:
  1) load the table (CSV or whitespace) and detect wavelength column
  2) pick wavelength_nm according to user prefs (observed/ritz, vacuum/air)
  3) convert nm -> Hz, log10 nu
  4) keep useful metadata (intensity, Acc)
  5) compute a lightweight emission weight (normalized intensity)
  6) write CSV: wavelength_nm, frequency_hz, log10_nu_hz, intensity, weight, acc

Run:

PYTHONPATH=. python -m scripts.MDL.adapter_to_photoncodes_neon \
  --in data/raw/lines/Hg_II_lines_raw.csv \
  --out-csv results/lamps/raw_csv/hg_II_ritz_vac.csv \
  --prefer ritz --medium vacuum --min-intens 0

Options:
  --prefer {observed, ritz}    (default: ritz)
  --medium  {vacuum, air}      (default: vacuum)
  --min-intens FLOAT           (drop lines with intensity below this; default: 0)
  --acc-min {A,B,C,D,E}        (optional: keep only rows with Acc <= this grade)
  --drop-na-wavelength         (default: True)
  --preview  N                 (print first N rows after parsing, then continue)

Notes
-----
* Matches pipeline conventions in your solar/Vega adapters (simple CLI, tidy CSV):contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.
* Outputs normalized 'weight' from intensity for emission-style MDL; no eventizer needed.
* If both observed and ritz are present, you can run MDL twice (observed vs ritz) to show robustness.
"""

import argparse
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

C_M_PER_S = 299_792_458.0  # speed of light


# --------------------------- helpers ---------------------------

def _read_table(path: str) -> pd.DataFrame:
    """
    Read a line list that might be CSV or whitespace-delimited.
    Try pandas read_csv with automatic delimiter inference; fall back to python engine.
    """
    try:
        # Try comma/semicolon/tab inference
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            # likely whitespace; re-read with delim whitespace
            df = pd.read_csv(path, delim_whitespace=True, engine="python", comment="#")
        return df
    except Exception:
        # robust fallback: flexible whitespace
        return pd.read_csv(path, delim_whitespace=True, engine="python", comment="#")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create canonical aliases so we can look up wavelength/intensity/accuracy reliably.
    """
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Known / likely wavelength column spellings
    aliases = {
        "obs_wl_vac(nm)": ["obs_wl_vac(nm)", "obs_wl_vac", "obs_wl_vac_nm", "obs_vac_nm", "wl_obs_vac", "w_obs_vac(nm)"],
        "obs_wl_air(nm)": ["obs_wl_air(nm)", "obs_wl_air", "obs_air_nm", "wl_obs_air"],
        "ritz_wl_vac(nm)": ["ritz_wl_vac(nm)", "ritz_wl_vac", "ritz_vac_nm", "wl_ritz_vac", "w_ritz_vac(nm)"],
        "ritz_wl_air(nm)": ["ritz_wl_air(nm)", "ritz_wl_air", "ritz_air_nm", "wl_ritz_air"],
        "intens": ["intens", "intensity", "int", "rel_int", "intensity_rel"],
        "acc": ["acc", "accuracy", "acc_grade"]
    }

    def first_present(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k in df.columns:
                return k
        return None

    # Build a small mapping of canonical -> actual names present
    present = {}
    for canon, variants in aliases.items():
        # lower-case variant lookup
        v_lower = [v.lower() for v in variants]
        hit = first_present(v_lower)
        if hit is not None:
            present[canon] = hit

    # Ensure at least one wavelength exists; we don't rename in place yet
    return df, present

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert a mixed/quoted series like '="25.635"' to float.
    Strips everything except digits, sign, decimal, and exponent.
    Returns float NaN where nothing numeric is found.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    ss = s.astype(str).str.replace(r"[^0-9eE\+\-\.]", "", regex=True)
    return pd.to_numeric(ss, errors="coerce")

def _intensity_series(df_norm: pd.DataFrame, intens_col: str, min_intens: float | None) -> pd.Series:
    """
    Build a usable intensity series.
    - If numeric present: use numeric; apply min_intens if provided.
    - Else map letters a..e → ranks (a=5,b=4,c=3,d=2,e=1); ignore min_intens.
    """
    s_raw = df_norm[intens_col]
    s_num = _coerce_numeric_series(s_raw)
    if s_num.notna().any():
        # normalize later after filters
        return s_num
    # letter grades / symbols → simple ordinal map (fallback)
    s_str = s_raw.astype(str).str.strip().str.lower().str.extract(r"([a-z])")[0]
    rank_map = {"a":5, "b":4, "c":3, "d":2, "e":1}
    return s_str.map(rank_map).fillna(1.0)

def _choose_wavelength_column(present_map: dict, prefer: str, medium: str) -> Optional[str]:
    """
    prefer in {observed, ritz}, medium in {vacuum, air}
    """
    # 1) exact match to user preference
    key = f"{prefer}_wl_{'vac' if medium=='vacuum' else 'air'}(nm)"
    if key in present_map:
        return present_map[key]

    # 2) fallbacks: same prefer, other medium
    alt_medium = "air" if medium == "vacuum" else "vac"
    alt_key = f"{prefer}_wl_{alt_medium}(nm)"
    if alt_key in present_map:
        return present_map[alt_key]

    # 3) switch prefer (ritz<->observed) with desired medium
    other_pref = "observed" if prefer == "ritz" else "ritz"
    key2 = f"{other_pref}_wl_{'vac' if medium=='vacuum' else 'air'}(nm)"
    if key2 in present_map:
        return present_map[key2]

    # 4) last resort: any wavelength column available in priority order
    fallback_order = ["ritz_wl_vac(nm)", "obs_wl_vac(nm)", "ritz_wl_air(nm)", "obs_wl_air(nm)"]
    for k in fallback_order:
        if k in present_map:
            return present_map[k]

    return None


def _acc_ok(acc_val: Optional[str], acc_min: Optional[str]) -> bool:
    """
    Keep only rows with Acc <= acc_min by grade ordering: A < B < C < D < E.
    If acc_min is None, accept all.
    """
    if acc_min is None:
        return True
    if acc_val is None or (isinstance(acc_val, float) and np.isnan(acc_val)):
        return False
    order = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    x = str(acc_val).strip().lower()[:1]
    y = str(acc_min).strip().lower()[:1]
    if x not in order or y not in order:
        return False
    return order[x] <= order[y]


def _nm_to_hz(lambda_nm: np.ndarray) -> np.ndarray:
    lam_m = lambda_nm * 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = C_M_PER_S / lam_m
    return nu


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Adapter: Ne I line list -> photons CSV (emission)")
    ap.add_argument("--in", dest="infile", required=True, help="Input Ne I lines file (CSV or whitespace)")
    ap.add_argument("--out-csv", dest="out_csv", required=True, help="Output CSV path")
    ap.add_argument("--prefer", choices=["observed", "ritz"], default="ritz",
                    help="Prefer observed vs ritz wavelength when both present")
    ap.add_argument("--medium", choices=["vacuum", "air"], default="vacuum",
                    help="Prefer vacuum vs air wavelength when both present")
    ap.add_argument("--min-intens", type=float, default=0.0,
                    help="Drop lines with intensity < this threshold (default 0)")
    ap.add_argument("--acc-min", type=str, default=None,
                    help="Keep only rows with Acc <= this grade (A,B,C,...)")
    ap.add_argument("--drop-na-wavelength", action="store_true", default=True,
                    help="Drop rows with missing wavelength (default True)")
    ap.add_argument("--preview", type=int, default=0, help="Print first N rows after parsing")
    args = ap.parse_args()

    # load & normalize
    df_raw = _read_table(args.infile)
    if df_raw.empty:
        print(f"[neon] No rows read from {args.infile}", file=sys.stderr)
        sys.exit(2)

    df_norm, present = _normalize_columns(df_raw)

    # intensity & acc columns (optional)
    intens_col = present.get("intens", None)
    acc_col = present.get("acc", None)

    # choose wavelength column
    wl_col = _choose_wavelength_column(present, args.prefer, args.medium)
    if wl_col is None:
        print("[neon] Could not find wavelength column (observed/ritz × vacuum/air). "
              "Present keys: " + ", ".join(sorted(present.keys())), file=sys.stderr)
        sys.exit(2)

    # build working frame
    out = pd.DataFrame()
    out["wavelength_nm"] = _coerce_numeric_series(df_norm[wl_col])
    if args.drop_na_wavelength:
        out = out.dropna(subset=["wavelength_nm"])

    if intens_col is not None:
        out["intensity"] = _intensity_series(df_norm, intens_col, args.min_intens)
        # Apply --min-intens ONLY if we actually have numeric intensities
        if args.min_intens and pd.api.types.is_numeric_dtype(out["intensity"]):
            out = out[out["intensity"] >= float(args.min_intens)]
    else:
        out["intensity"] = 1.0

    if acc_col is not None:
        acc_vals = df_norm[acc_col].astype(str)
        if args.acc_min:
            mask_acc = acc_vals.apply(lambda s: _acc_ok(s, args.acc_min))
            out = out.loc[mask_acc]
        out["acc"] = acc_vals
    else:
        out["acc"] = ""

    # compute frequencies and weights
    nu = _nm_to_hz(out["wavelength_nm"].to_numpy(dtype=float))
    out["frequency_hz"] = nu
    with np.errstate(divide="ignore"):
        out["log10_nu_hz"] = np.log10(out["frequency_hz"])

    # emission weight: normalize intensity to [0,1]
    if "intensity" in out.columns and len(out):
        imax = float(out["intensity"].max()) if np.isfinite(out["intensity"].max()) else 1.0
        out["weight"] = (out["intensity"] / imax) if imax > 0 else 1.0
    else:
        out["weight"] = 1.0

    # emission weight: normalize intensity to [0,1]
    if "intensity" in out.columns and len(out):
        imax = float(np.nanmax(out["intensity"].to_numpy())) if np.isfinite(np.nanmax(out["intensity"].to_numpy())) else 1.0
        out["weight"] = (out["intensity"] / imax) if imax > 0 else 1.0
    else:
        out["weight"] = 1.0

    # ensure weights are usable even if intensity was missing
    out["intensity"] = out["intensity"].fillna(1.0)
    out["weight"]    = out["weight"].fillna(1.0)

    # clean, sort, and save
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["wavelength_nm", "frequency_hz", "log10_nu_hz"])
    out = out.sort_values("wavelength_nm").reset_index(drop=True)

    if args.preview and args.preview > 0:
        print(out.head(int(args.preview)).to_string(index=False))

    out.to_csv(args.out_csv, index=False)
    print(f"[adapter] Wrote {len(out):,} lines -> {args.out_csv}")

    if len(out):
        lam_min = float(out["wavelength_nm"].min())
        lam_max = float(out["wavelength_nm"].max())
        # use an en dash or plain hyphen; both are fine
        print(f"[adapter] wavelength range: {lam_min:.3f}–{lam_max:.3f} nm")
    else:
        print("[adapter] wavelength range: n/a")

if __name__ == "__main__":
    main()
