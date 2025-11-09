# scripts/analysis_pipeline/radio_delay_analysis.py
# GRAPE Gen-1 (2019) ionospheric dispersion analysis
# Author: Coherence Research Collaboration (Kelly B. Heaton & ChatGPT)
#
# What this does (high level):
# - Parse GRAPE V1 station CSVs under data/Grape_V1/2019/
# - Extract metadata (station node, callsign, lat/lon, beacon)
# - Build per-station, per-frequency time series of Freq (Hz) and Vpk
# - Compute S(t) = -(Freq - nu_nom)/nu_nom and resample at 60s medians
# - At each time with >=3 bands present, fit S vs X=1/nu^2 via OLS; save slope, intercept, R^2
# - Optionally integrate S over a window to obtain relative delay Δτ(ν) vs 1/ν^2
# - Write results & plots to data/results/radio_delay/Grape_V1/
"""
Version 1.6

python -m scripts.analysis_pipeline.radio_delay_analysis \
  --data-root data/Grape_V1/2019/ data/Grape_V1/2020/ \
  --out-dir  data/results/radio_delay/Grape_V1/ \
  --pool-by-grid \
  --grid-prefix 2 \
  --region-prefix EN \
  --ref-freq-hz 10000000 \
  --window-minutes 30


"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_DATA_ROOT = "data/Grape_V1/2019"
DEFAULT_OUT_DIR = "data/results/radio_delay/Grape_V1"
RESAMPLE_RULE = "60S"      # median each 60s
MIN_BANDS = 3              # need at least 3 frequencies at a timestamp to fit S vs 1/nu^2
REF_FREQ_HZ = 10_000_000   # default reference for relative delay integration
WINDOW_MINUTES = 30        # integration window for relative delay visualization
MIN_STATIONS_PER_FREQ = 1  # for pooling: how many stations per frequency per minute before we accept the median

# -----------------------------
# Utilities: metadata parsing
# -----------------------------
BEACON_RE = re.compile(r"(WWV|WWVH|CHU)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_beacon_to_hz(beacon_str: str) -> Optional[float]:
    """
    Map a beacon descriptor like 'WWV5', 'WWV 2.5', 'WWVH10', 'CHU7' -> nominal frequency in Hz.
    Returns None if not recognized.
    """
    if not beacon_str:
        return None
    m = BEACON_RE.search(beacon_str)
    if not m:
        # Some files use compact tokens like 'WWV5' or 'CHU7' without space:
        compact = beacon_str.strip().upper()
        if compact.startswith(("WWVH", "WWV", "CHU")):
            if compact.startswith("CHU"):
                digits = compact.replace("CHU", "")
                # Choose nearest of CHU's canonical MHz: 3.33, 7.85, 14.67
                try:
                    val = float(digits) if digits else np.nan
                except:
                    val = np.nan
                candidates = np.array([3.33, 7.85, 14.67])
                target = candidates[np.argmin(np.abs(candidates - val))] if np.isfinite(val) else 7.85
                return float(target) * 1e6
            else:
                digits = compact.replace("WWVH", "").replace("WWV", "")
                try:
                    return float(digits) * 1e6
                except:
                    return None
        return None
    svc = m.group(1).upper()
    freq_mhz = float(m.group(2))
    if svc == "CHU":
        # Map CHU labels (e.g., 'CHU7') to canonical MHz
        candidates = np.array([3.33, 7.85, 14.67])
        target = candidates[np.argmin(np.abs(candidates - freq_mhz))]
        return float(target) * 1e6
    # WWV/WWVH: numbers are already in MHz
    return freq_mhz * 1e6

def read_grape_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read a GRAPE Gen-1 CSV with header comments and 3 data columns: UTC, Freq, Vpk.
    Returns (df, meta) where:
      df has columns ['UTC','Freq','Vpk'] with UTC tz-aware datetime index,
      meta is a dict with fields: node, callsign, grid, lat, lon, city_state, radio_id, beacon, nu_nom_hz.
    """
    meta = {
        "node": None, "callsign": None, "grid": None,
        "lat": None, "lon": None, "elev": None,
        "city_state": None, "radio_id": None,
        "beacon": None, "nu_nom_hz": None,
        "file": str(path)
    }
    # Read header and peek the first non-comment line to detect delimiter
    first_non_comment = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header_lines = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
                continue
            first_non_comment = line.strip()
            break
    # Detect delimiter: some files are "UTC,Freq,Vpk" (commas), others "UTC\tFreq\tVpk" (tabs/whitespace).
    # See ESSD Appendix C example file. :contentReference[oaicite:2]{index=2}
    if first_non_comment and "," in first_non_comment:
        sep_arg = ","
    else:
        sep_arg = r"\s+"
    # Extract metadata
    for h in header_lines:
        # Node number
        if "Station Node Number" in h:
            m = re.search(r"Station Node Number\s+([A-Z0-9]+)", h)
            if m: meta["node"] = m.group(1)
        # Callsign
        if "Callsign" in h and "Station Node" not in h:
            m = re.search(r"Callsign\s+([A-Z0-9\-]+)", h)
            if m: meta["callsign"] = m.group(1)
        # Grid
        if "Grid Square" in h:
            m = re.search(r"Grid Square\s+([A-R]{2}[0-9]{2}[a-x]{2})", h, re.IGNORECASE)
            if m: meta["grid"] = m.group(1).upper()
        # Lat Lon Elev
        if re.search(r"Lat\s+Long\s+Elv", h):
            nums = re.findall(r"(-?\d+\.\d+)", h)
            if len(nums) >= 2:
                meta["lat"] = float(nums[0]); meta["lon"] = float(nums[1])
            if len(nums) >= 3:
                meta["elev"] = float(nums[2])
        # City State
        if "City State" in h:
            # e.g., "City State               Macedonia Ohio"
            parts = h.split("City State")
            if len(parts) > 1:
                meta["city_state"] = parts[1].strip()
        # Radio ID
        if "Radio1ID" in h:
            m = re.search(r"Radio1ID\s+([A-Za-z0-9]+)", h)
            if m: meta["radio_id"] = m.group(1)
        # Beacon
        if "Beacon Now Decoded" in h:
            # e.g., "Beacon Now Decoded       WWV5"
            parts = h.split("Decoded")
            beacon = parts[-1].strip() if parts else None
            meta["beacon"] = beacon
            meta["nu_nom_hz"] = parse_beacon_to_hz(beacon)

    # Read with detected delimiter; allow spaces after commas
    df = pd.read_csv(
        path,
        comment="#",
        sep=sep_arg,
        engine="python",
        skipinitialspace=True
    )
    # Fallback: if the header came in as a single column "UTC,Freq,Vpk", re-read as CSV
    if list(df.columns) == ["UTC,Freq,Vpk"]:
        df = pd.read_csv(
            path,
            comment="#",
            sep=",",
            engine="python",
            skipinitialspace=True
        )
    # Expect columns: UTC, Freq, Vpk
    # Parse time
    # Normalize column names (strip whitespace/BOM)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "UTC" not in df.columns or "Freq" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns}")
    # Parse time. Files may use full ISO times ("2019-06-09T00:00:00Z") OR time-of-day ("00:29:42").
    # See ESSD Appendix C. :contentReference[oaicite:3]{index=3}
    raw_utc = df["UTC"].astype(str).str.strip()
    # Look at a few non-null entries to detect format
    sample = raw_utc[raw_utc.notna()].head(12)
    iso_like = sample.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$").mean() > 0.5
    tod_like = sample.str.match(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$").mean() > 0.5
    if iso_like:
        # Exact ISO Z format (fast, no warnings)
        df["UTC"] = pd.to_datetime(raw_utc, utc=True, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
    elif tod_like:
        # Time-of-day only; combine with start date extracted from header (Appendix C format)
        start_iso = None
        for h in header_lines:
            m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", h)
            if m:
                start_iso = m.group(1)
                break
        if start_iso:
            start_date = pd.to_datetime(start_iso, utc=True).normalize()
            tod = pd.to_timedelta(raw_utc, errors="coerce")
            df["UTC"] = start_date + tod
        else:
            # Fallback if header date is missing: last-resort parser (may emit NaT)
            df["UTC"] = pd.to_datetime(raw_utc, utc=True, errors="coerce")
    else:
        # Mixed or unexpected; fall back once without format (may be slower)
        df["UTC"] = pd.to_datetime(raw_utc, utc=True, errors="coerce")
    df = df.dropna(subset=["UTC", "Freq"]).copy()
    df = df.set_index("UTC").sort_index()

    # Clean numeric
    for col in ("Freq", "Vpk"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Freq"])

    # Attach metadata for reference
    for k, v in meta.items():
        df.attrs[k] = v

    return df, meta

# -----------------------------
# Aggregation
# -----------------------------
def load_dataset(roots) -> Dict[Tuple[str, str], Dict[float, pd.DataFrame]]:
    """
    Walk one or more root directories and load all CSVs.
    Returns nested dict: {(node, callsign): {nu_nom_hz: df, ...}, ...}
    """
    result: Dict[Tuple[str, str], Dict[float, pd.DataFrame]] = {}
    if isinstance(roots, (str, Path)):
        roots = [roots]
    files: List[Path] = []
    for r in roots:
        files.extend(sorted(Path(r).rglob("*.csv")))
    for fp in files:
        try:
            df, meta = read_grape_csv(fp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")
            continue
        node = meta.get("node") or "UNKNOWN"
        callsign = meta.get("callsign") or "UNKNOWN"
        nu_nom = meta.get("nu_nom_hz")
        if nu_nom is None:
            print(f"[WARN] Unknown beacon in {fp} (meta: {meta.get('beacon')}); skipping.")
            continue
        key = (node, callsign)
        if key not in result:
            result[key] = {}
        # Keep only core columns
        keep = df[["Freq"]].copy()
        keep.rename(columns={"Freq": f"Freq_{int(nu_nom)}"}, inplace=True)
        # Store with attrs
        keep.attrs.update(df.attrs)

        # Add a helpful region tag from Maidenhead grid (first N chars, computed later)
        result[key][nu_nom] = keep
    return result

def resample_and_merge(station_data: Dict[float, pd.DataFrame],
                       rule: str = RESAMPLE_RULE) -> pd.DataFrame:
    """
    For a single station (dict of {nu_nom: df}), resample each df to 'rule' (median),
    then outer-join on time. Returns wide DF with columns per band.
    """
    frames = []
    for nu, df in station_data.items():
        # resample by median to suppress outliers
        r = df.resample(rule).median()
        r.columns = [f"Freq_{int(nu)}"]
        frames.append(r)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1, join="outer").sort_index()
    # drop rows with all NaN
    wide = wide.dropna(how="all")
    return wide

# -----------------------------
# Estimators & fits
# -----------------------------
def compute_S_series(wide_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Given wide DF with columns 'Freq_{nu}', compute:
      S_{nu}(t) = - (Freq_{nu}(t) - nu)/nu  = -Δf/nu
    Returns (S_df, nu_map) where S_df columns are 'S_{nu}' and nu_map maps col->nu.
    """
    S_cols = {}
    nu_map = {}
    for col in wide_df.columns:
        if col.startswith("Freq_"):
            nu = float(col.split("_")[1])  # nominal in Hz as string
            # exact nu from name; guard against low decimals
            freq = wide_df[col]
            S = - (freq - nu) / nu
            sname = f"S_{int(nu)}"
            S_cols[sname] = S
            nu_map[sname] = nu
    S_df = pd.DataFrame(S_cols, index=wide_df.index).dropna(how="all")
    return S_df, nu_map

def ols_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Simple OLS for y = a + b x with R^2 and stderr on slope b.
    """
    n = len(x)
    X = np.vstack([np.ones(n), x]).T
    # beta = (X'X)^-1 X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    a, b = beta[0], beta[1]
    yhat = a + b * x
    resid = y - yhat
    sse = float(resid.T @ resid)
    sst = float(((y - y.mean())**2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    # stderr of slope
    dof = max(n - 2, 1)
    sigma2 = sse / dof
    var_beta = sigma2 * XtX_inv
    se_b = float(np.sqrt(var_beta[1, 1]))
    return {"a": float(a), "b": float(b), "r2": float(r2), "n": int(n), "se_b": se_b}

def fit_S_vs_invnu2(S_df: pd.DataFrame, nu_map: Dict[str, float],
                    min_bands: int = MIN_BANDS) -> pd.DataFrame:
    """
    At each timestamp, fit S(ν,t) against X=1/ν^2 across the available bands.
    Returns a time series DF with columns ['a','b','r2','n'].
    """
    # Build per-time vectors
    # columns like S_2500000, S_5000000, ...
    out_rows = []
    for t, row in S_df.iterrows():
        xs = []
        ys = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            nu = nu_map[col]  # Hz
            xs.append(1.0 / (nu**2))
            ys.append(val)
        if len(xs) >= min_bands:
            x = np.array(xs, dtype=float)
            y = np.array(ys, dtype=float)
            fit = ols_fit(x, y)
            out_rows.append({"UTC": t, **fit})
    if not out_rows:
        return pd.DataFrame()
    out = pd.DataFrame(out_rows).set_index("UTC").sort_index()
    return out

def integrate_relative_delay(S_df: pd.DataFrame, nu_map: Dict[str, float],
                             ref_freq_hz: float = REF_FREQ_HZ,
                             window_minutes: int = WINDOW_MINUTES) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Over a sliding window, integrate S(t) to form Δτ(ν) relative to ref band:
        Δτ_rel(ν; window) = ∑_window [ S(ν,t) - S(ref,t) ] Δt.
    With 60s resampling, Δt=60 s.
    For each window center, fit Δτ_rel across ν vs (1/ν^2 - 1/ref^2).
    Returns:
      (dtau_df, fit_df)
      dtau_df: one row per window with columns 'DTau_{nu}' in seconds,
      fit_df: per-window fit of Δτ_rel vs invnu2_diff (a,b,r2,n).
    """
    if S_df.empty:
        return None
    # Check ref column exists
    ref_col = f"S_{int(ref_freq_hz)}"
    if ref_col not in S_df.columns:
        # choose closest available band as ref
        available = [(abs(nu - ref_freq_hz), col) for col, nu in nu_map.items()]
        if not available:
            return None
        ref_col = min(available)[1]
        print(f"[INFO] Reference band {ref_freq_hz} Hz not found; using {ref_col} instead.")
    # seconds per sample after resampling
    # infer frequency from index; assume uniform 60s
    dt_s = 60.0
    win = f"{window_minutes}min"
    half_win = f"{window_minutes//2}min"

    # Rolling integration (centered window)
    # Build Δτ_rel(ν) time series
    dtau_cols = {}
    for col in S_df.columns:
        if not col.startswith("S_") or col == ref_col:
            continue
        sdiff = (S_df[col] - S_df[ref_col])
        # remove the window-mean so constant band offsets don't integrate to large baselines
        mu = sdiff.rolling(win, center=True, min_periods=3).mean()
        centered = sdiff - mu
        dtau = centered.rolling(win, center=True, min_periods=3).sum() * dt_s
        dtau_cols[col.replace("S_", "DTau_")] = dtau
    if not dtau_cols:
        return None
    dtau_df = pd.DataFrame(dtau_cols, index=S_df.index).dropna(how="all")

    # For each window center (timestamps where we have dtau values), fit across ν
    out_rows = []
    for t, row in dtau_df.iterrows():
        xs = []
        ys = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            nu = nu_map[col.replace("DTau_", "S_")]  # Hz
            invnu2_diff = (1.0/(nu**2)) - (1.0/(nu_map[ref_col]**2))
            xs.append(invnu2_diff)
            ys.append(val)  # seconds
        if len(xs) >= MIN_BANDS - 1:  # ref removed
            x = np.array(xs, dtype=float)
            y = np.array(ys, dtype=float)
            fit = ols_fit(x, y)
            out_rows.append({"UTC": t, **fit})
    if not out_rows:
        fit_df = pd.DataFrame()
    else:
        fit_df = pd.DataFrame(out_rows).set_index("UTC").sort_index()
    return dtau_df, fit_df

# -----------------------------
# Plot helpers
# -----------------------------
def plot_dt_vs_invnu2(ax, X, Y, a, b, r2, title, xlabel, ylabel):
    ax.scatter(X, Y, s=20)
    # Fit line
    xline = np.linspace(min(X), max(X), 100)
    yline = a + b * xline
    ax.plot(xline, yline)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    txt = f"y = {a:.3e} + {b:.3e} x\nR² = {r2:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left")

# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    data_roots = args.data_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_roots)
    if not dataset:
        print(f"[ERROR] No usable GRAPE CSVs found under {data_roots}")
        return

    # Keep S-series for optional cross-station pooling
    pooled_entries = []   # list of dicts: {'grid':..., 'grid_prefix':..., 'station':..., 'S_df':..., 'nu_map':...}

    for (node, callsign), station_data in dataset.items():
        station_tag = f"{node}_{callsign}"
        band_list = sorted({round(nu/1e6, 2) for nu in station_data.keys()})
        print(f"[INFO] Station: {station_tag} | bands: {band_list} MHz")

        wide = resample_and_merge(station_data, RESAMPLE_RULE)
        if wide.empty:
            print(f"[WARN] No data after resampling for {station_tag}")
            continue

        S_df, nu_map = compute_S_series(wide)
        # Save S time series
        S_out = out_dir / f"{station_tag}_S_timeseries.csv"
        S_df.to_csv(S_out, index=True)
        print(f"[OK] Wrote {S_out}")

        # Per-time fits: S vs 1/nu^2
        fit_ts = fit_S_vs_invnu2(S_df, nu_map, MIN_BANDS)
        if not fit_ts.empty:
            fits_out = out_dir / f"{station_tag}_S_vs_invnu2_timeseries.csv"
            fit_ts.to_csv(fits_out, index=True)
            print(f"[OK] Wrote {fits_out}")

            # Plot slope vs time (to see day/night change)
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(fit_ts.index, fit_ts["b"])
            ax.set_title(f"{station_tag}: slope of S vs 1/nu^2 (dispersion strength)")
            ax.set_ylabel("slope b [s/s * Hz^2]")
            ax.set_xlabel("UTC")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_dir / f"{station_tag}_dispersion_slope_timeseries.png", dpi=160)
            plt.close(fig)

        # Window-integrated relative delay visualization (Δτ vs 1/ν^2)
        rel = integrate_relative_delay(S_df, nu_map, args.ref_freq_hz, args.window_minutes)
        if rel is not None:
            dtau_df, fit_df = rel
            # Save dtau window series and fits
            dtau_out = out_dir / f"{station_tag}_DTau_rel_windows.csv"
            dtau_df.to_csv(dtau_out, index=True)
            print(f"[OK] Wrote {dtau_out}")
            if not fit_df.empty:
                fitw_out = out_dir / f"{station_tag}_DTau_fit_windows.csv"
                fit_df.to_csv(fitw_out, index=True)
                print(f"[OK] Wrote {fitw_out}")

                # Pick one representative window center (highest R^2)
                pick = fit_df.sort_values("r2", ascending=False).head(1)
                if not pick.empty:
                    t0 = pick.index[0]
                    row = dtau_df.loc[t0]
                    # Build X,Y for plot
                    X = []
                    Y = []
                    for col, val in row.items():
                        if pd.isna(val): continue
                        nu = nu_map[col.replace("DTau_", "S_")]
                        invnu2_diff = (1.0/(nu**2)) - (1.0/(args.ref_freq_hz**2))
                        X.append(invnu2_diff); Y.append(val)
                    fit = pick.iloc[0]
                    fig, ax = plt.subplots(figsize=(5, 4))
                    plot_dt_vs_invnu2(
                        ax, np.array(X), np.array(Y),
                        fit["a"], fit["b"], fit["r2"],
                        title=f"{station_tag} | Δτ vs 1/ν² (window centered {t0})",
                        xlabel="1/ν² − 1/ν_ref² [Hz⁻²]",
                        ylabel="Δτ_rel [s]"
                    )
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{station_tag}_DTau_vs_invnu2_window_example.png", dpi=160)
                    plt.close(fig)

        # Preserve station grid even if attrs were dropped by concat
        try:
            any_df = next(iter(station_data.values()))
            grid_full = (S_df.attrs.get("grid")
                         or any_df.attrs.get("grid")
                         or "")
        except StopIteration:
            grid_full = ""
        grid_full = str(grid_full).strip().upper()
        grid_prefix = grid_full[: args.grid_prefix] if grid_full else "UNK"
        pooled_entries.append({
            "grid": grid_full,
            "grid_prefix": grid_prefix,
            "station": station_tag,
            "S_df": S_df,
            "nu_map": nu_map
        })

    # -------------------------
    # Cross-station pooling (optional)
    # -------------------------
    if args.pool_by_grid:
        pooled_dir = out_dir / "pooled"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        run_pooled_grid_analysis(
            entries=pooled_entries,
            out_dir=pooled_dir,
            grid_prefix_filter=args.region_prefix,
            min_bands=MIN_BANDS,
            min_stations_per_freq=args.min_stations_per_freq,
            ref_freq_hz=args.ref_freq_hz,
            window_minutes=args.window_minutes
        )


    print("[DONE] Analysis complete.")

# ------------------------------------------------------------
# Cross-station pooling helpers
# ------------------------------------------------------------
def entries_to_long(entries: List[Dict], region_filter: Optional[str],
                    min_stations_per_freq: int, grid_prefix_len: int) -> pd.DataFrame:
    """
    Convert per-station S_df into one long DataFrame with columns:
      ['UTC','grid_prefix','nu','S','n'] (n = station count)
    Aggregate median S per (UTC, grid_prefix, nu).
    """
    long_frames = []
    for e in entries:
        grid_prefix = e.get("grid_prefix", "UNK")
        if region_filter and not str(grid_prefix).startswith(region_filter.upper()):
            continue
        S_df: pd.DataFrame = e["S_df"]
        if S_df.empty:
            continue
        tmp = S_df.copy()
        tmp["UTC"] = tmp.index
        m = tmp.melt(id_vars=["UTC"], var_name="var", value_name="S")
        m = m[m["var"].str.startswith("S_")]
        m["nu"] = m["var"].str.replace("S_", "", regex=False).astype(float)
        m["grid_prefix"] = grid_prefix
        long_frames.append(m[["UTC", "grid_prefix", "nu", "S"]])
    if not long_frames:
        return pd.DataFrame()
    long_all = pd.concat(long_frames, ignore_index=True).dropna(subset=["UTC", "nu", "S"])
    grouped = long_all.groupby(["UTC", "grid_prefix", "nu"]).agg(S=("S", "median"),
                                                                 n=("S", "count")).reset_index()
    grouped = grouped[grouped["n"] >= min_stations_per_freq]
    return grouped


def pooled_fit_time_series(grouped: pd.DataFrame, min_bands: int) -> Dict[str, pd.DataFrame]:
    """
    For each grid_prefix and UTC minute, fit S vs 1/nu^2.
    Return dict: grid_prefix -> DataFrame(index=UTC, cols=[a,b,r2,n,se_b]).
    """
    out: Dict[str, List[Dict]] = {}
    if grouped.empty:
        return {}
    for (gp, t), sub in grouped.groupby(["grid_prefix", "UTC"]):
        xs = (1.0 / (sub["nu"].values ** 2)).astype(float)
        ys = sub["S"].values.astype(float)
        if len(xs) >= min_bands:
            fit = ols_fit(xs, ys)
            out.setdefault(gp, []).append({"UTC": t, **fit})
    return {gp: pd.DataFrame(rows).set_index("UTC").sort_index()
            for gp, rows in out.items() if rows}


def run_pooled_grid_analysis(entries: List[Dict],
                             out_dir: Path,
                             grid_prefix_filter: Optional[str],
                             min_bands: int,
                             min_stations_per_freq: int,
                             ref_freq_hz: float,
                             window_minutes: int):
    """
    Pool stations by grid prefix; per-minute median by frequency;
    fit S vs 1/nu^2 when >= min_bands; write CSVs and a Δτ vs 1/ν² figure.
    """
    grouped = entries_to_long(entries, region_filter=grid_prefix_filter,
                              min_stations_per_freq=min_stations_per_freq,
                              grid_prefix_len=0)
    if grouped.empty:
        print("[WARN] Pooling: no data matched the region/prefix filter.")
        return

    avail = grouped.groupby(["grid_prefix", "UTC"]).agg(nu_count=("nu", "nunique")).reset_index()
    winners = (avail[avail["nu_count"] >= min_bands]
               .groupby("grid_prefix")["UTC"].count().sort_values(ascending=False))
    if winners.empty:
        print("[WARN] Pooling: found no timestamps with >=3 distinct frequencies. "
              "Try a broader region-prefix (e.g., 'EN').")
        return
    print("[INFO] Pooling candidates (grid_prefix -> minutes with >=3 freqs):")
    for gp, cnt in winners.head(8).items():
        print(f"   {gp}: {cnt} minutes")

    fits_by_gp = pooled_fit_time_series(grouped, min_bands=min_bands)
    for gp, fit_ts in fits_by_gp.items():
        gp_dir = out_dir / gp
        gp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = gp_dir / f"pooled_{gp}_S_vs_invnu2_timeseries.csv"
        fit_ts.to_csv(csv_path, index=True)
        print(f"[OK] Pooled fit written: {csv_path}")

        # Build a pivot S_df (median per nu) to reuse Δτ plotting
        sub = grouped[grouped["grid_prefix"] == gp]
        pivot = sub.pivot(index="UTC", columns="nu", values="S").sort_index()
        pivot.columns = [f"S_{int(nu)}" for nu in pivot.columns]
        nu_map = {col: float(col.replace("S_", "")) for col in pivot.columns}

        rel = integrate_relative_delay(pivot, nu_map,
                                       ref_freq_hz=ref_freq_hz,
                                       window_minutes=window_minutes)
        if rel is None:
            continue
        dtau_df, fit_df = rel
        dtau_df.to_csv(gp_dir / f"pooled_{gp}_DTau_rel_windows.csv", index=True)
        if not fit_df.empty:
            # add apparent ΔTEC per window from slope b = (40.3/c)*ΔTEC
            fit_df = fit_df.copy()
            fit_df["delta_TECU"] = (fit_df["b"] * 299792458.0) / (40.3 * 1.0e16)
            fit_df.to_csv(gp_dir / f"pooled_{gp}_DTau_fit_windows.csv", index=True)
            pick = fit_df.sort_values("r2", ascending=False).head(1)
            if not pick.empty:
                t0 = pick.index[0]
                row = dtau_df.loc[t0]
                X, Y = [], []
                for col, val in row.items():
                    if pd.isna(val): 
                        continue
                    nu = nu_map[col.replace("DTau_", "S_")]
                    X.append((1.0 / (nu ** 2)) - (1.0 / (ref_freq_hz ** 2)))
                    Y.append(val)
                fit = pick.iloc[0]
                fig, ax = plt.subplots(figsize=(5, 4))
                plot_dt_vs_invnu2(ax, np.array(X), np.array(Y),
                                  fit["a"], fit["b"], fit["r2"],
                                  title=f"Grid {gp} | Δτ vs 1/ν² (window {t0})",
                                  xlabel="1/ν² − 1/ν_ref² [Hz⁻²]",
                                  ylabel="Δτ_rel [s]")
                fig.tight_layout()
                fig.savefig(gp_dir / f"pooled_{gp}_DTau_vs_invnu2_window_example.png", dpi=160)
                plt.close(fig)

# ------------------------------------------------------------
# __main__
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRAPE Gen-1 radio delay (dispersion) analysis.")
    p.add_argument("--data-root", nargs="+", default=[DEFAULT_DATA_ROOT],
                   help="One or more roots with GRAPE CSVs")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where to write results")
    p.add_argument("--ref-freq-hz", type=float, default=REF_FREQ_HZ,
                   help="Reference band for Δτ integration (default 10 MHz)")
    p.add_argument("--window-minutes", type=int, default=WINDOW_MINUTES,
                   help="Integration window (minutes) for Δτ plots")
    p.add_argument("--pool-by-grid", action="store_true",
                   help="Enable cross-station pooling by Maidenhead grid prefix")
    p.add_argument("--grid-prefix", type=int, default=4,
                   help="Use first N chars of Maidenhead grid for pooling (e.g., 4 -> EN91)")
    p.add_argument("--region-prefix", type=str, default=None,
                   help="If set, only pool this region code (e.g., EN91 or EN)")
    p.add_argument("--min-stations-per-freq", type=int, default=MIN_STATIONS_PER_FREQ,
                   help="Min station count per freq per minute for pooled median")
    args = p.parse_args()
    main(args)
