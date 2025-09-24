# -*- coding: utf-8 -*-
"""
viz_all-ion_photon-decay_no-tower.py

Cross-ion visualization of photon decay in the γ frame *without* tower grouping.

Outputs:
  1) A global cross-ion plot:
       data/results/plots/viz_all-ion_photon-decay_no-tower/all_ions_photon_decay_no_tower.png
     showing pooled photons (optional) and one OLS line per ion (color-coded).
  2) One per-ion plot (pooled photons + OLS line) in the same directory.

Usage:
python -m scripts.views.cross_ion.viz_all-ion_photon-decay_no-tower \
  --input-dir data/results/photon_matched_resonant_pairs_mu-1/ \
  --output-dir data/results/plots/viz_all-ion_photon-decay_no-tower/ \
  --export-csv

python -m scripts.views.cross_ion.viz_all-ion_photon-decay_no-tower \
  --input-dir data/results/photon_matched_resonant_pairs_mu-1/ \
  --output-dir data/results/plots/viz_all-ion_photon-decay_no-tower/ \
  --hydrogenic-only \
  --hydrogenic-csv

  --no-scatter-global \ (optional to remove the photon "dots" and just see the trend lines)

Notes:
- We do NOT group by (n_i, n_k). Each ion is pooled to show that the band (β≈log10 α)
  appears without tower labels. The per-ion OLS intercept (baseline at γ=0) is an
  *aggregate* and will mix tower/site factors; tower partitioning is needed to interpret χ.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

C = 299_792_458.0  # m/s

# ---------- wavelength helpers (robust to column variety) ----------

PREF_NUMERIC_WAVELEN_COLS = [
    "lambda_nist_match_nm_best",
    "λ_NIST_match_nm_best",
    "lambda_best_nm",
]
FALLBACK_NUMERIC_WAVELEN_COLS = [
    "λ_photon_nm", "lambda_photon_nm", "lambda_nm",
]
LIST_WAVELEN_COLS = [
    "λ_NIST_match_nm", "lambda_nist_match_nm", "λ_NIST_match_list",
]

def pick_wavelength_nm(row: pd.Series) -> float:
    # 1) preferred numeric
    for col in PREF_NUMERIC_WAVELEN_COLS:
        if col in row and pd.notnull(row[col]):
            try:
                return float(row[col])
            except Exception:
                pass
    # 2) fallback numeric
    for col in FALLBACK_NUMERIC_WAVELEN_COLS:
        if col in row and pd.notnull(row[col]):
            try:
                return float(row[col])
            except Exception:
                pass
    # 3) parse first numeric from list
    for col in LIST_WAVELEN_COLS:
        if col in row and isinstance(row[col], str) and row[col].strip():
            for tok in map(str.strip, row[col].split(";")):
                try:
                    return float(tok)
                except Exception:
                    continue
    return np.nan

def fit_line(x: np.ndarray, y: np.ndarray):
    """OLS fit y = a + b x; return (a, b, r2, n)."""
    m = np.isfinite(x) & np.isfinite(y)
    x_fit, y_fit = x[m], y[m]
    n = len(x_fit)
    if n < 3:
        return (np.nan, np.nan, np.nan, n)
    A = np.vstack([np.ones_like(x_fit), x_fit]).T
    a, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    yhat = a + b * x_fit
    ss_res = np.sum((y_fit - yhat)**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (a, b, r2, n)

# ---------- load all *_resonant_transitions.csv ----------

def load_all_points(input_dir: Path) -> pd.DataFrame:
    """Return DataFrame with columns: ion, gamma, lambda_nm."""
    recs = []
    for csv in sorted(input_dir.glob("*_resonant_transitions.csv")):
        # load with comma, fallback to tab if needed
        try:
            df = pd.read_csv(csv)
        except Exception:
            df = pd.read_csv(csv, sep="\t")
        # detect gamma column
        gamma_col = None
        for c in ("gamma_bin", "γ", "gamma", "power"):
            if c in df.columns:
                gamma_col = c
                break
        if gamma_col is None:
            continue

        # detect ion label
        if "ion" in df.columns and df["ion"].notna().any():
            vals = df["ion"].dropna().astype(str)
            ion_name = vals.mode().iat[0] if len(vals) else csv.stem.split("_")[0]
        else:
            ion_name = csv.stem.split("_")[0]

        wl = df.apply(pick_wavelength_nm, axis=1)
        gam = pd.to_numeric(df[gamma_col], errors="coerce")

        m = np.isfinite(wl) & (wl > 0) & np.isfinite(gam)
        if not np.any(m):
            continue

        recs.append(pd.DataFrame({
            "ion": ion_name,
            "gamma": gam[m].astype(float).values,
            "lambda_nm": wl[m].astype(float).values
        }))
    if not recs:
        return pd.DataFrame(columns=["ion", "gamma", "lambda_nm"])
    return pd.concat(recs, ignore_index=True)

# ---------- plotting ----------

def make_per_ion_plot(ion: str, gamma: np.ndarray, lambda_nm: np.ndarray,
                      outpath: Path, dpi=160, annotate=True):
    """Pooled photons for one ion + OLS line."""
    nu = C / (lambda_nm * 1e-9)
    ylog = np.log10(nu)

    a, b, r2, n = fit_line(gamma, ylog)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=dpi)
    ax.scatter(gamma, ylog, s=12, alpha=0.45)
    ax.set_xlabel("γ (bin)")
    ax.set_ylabel("log$_{10}$(Photon frequency [Hz])")
    ax.set_title(f"{ion} — pooled photons (no towers)")

    if np.isfinite(b):
        xline = np.linspace(np.nanmin(gamma), np.nanmax(gamma), 200)
        yline = a + b * xline
        ax.plot(xline, yline, ls="--")
        if annotate:
            ax.text(0.02, 0.02,
                    f"β ≈ {b:.3f} | χ ≈ {a:.3f} | R² ≈ {r2:.3f} | n={n}",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)

    return a, b, r2, n

def make_global_plot(per_ion_stats, scatter_by_ion, outpath: Path,
                     dpi=160, no_scatter_global=False):
    """
    Global plot with optional pooled scatter and per-ion OLS lines (color-coded).
    per_ion_stats: dict ion -> dict with keys {gamma, ylog, a, b, r2, n}
    This function assigns a color to each ion and writes a PNG to `outpath`.
    """
    # choose a simple rotating palette and assign colors
    palette = plt.cm.tab20.colors
    for i, ion in enumerate(sorted(per_ion_stats.keys())):
        per_ion_stats[ion]["color"] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(10.5, 7), dpi=dpi)

    # pooled scatter per ion (light alpha), unless suppressed
    if not no_scatter_global and scatter_by_ion:
        for ion, d in per_ion_stats.items():
            gamma = d["gamma"]; ylog = d["ylog"]
            ax.scatter(gamma, ylog, s=6, alpha=0.15, color=d["color"])

    # per-ion fitted lines (color-coded with legend)
    handles, labels = [], []
    for ion, d in per_ion_stats.items():
        a, b, col = d["a"], d["b"], d["color"]
        if np.isfinite(b):
            xline = np.linspace(np.nanmin(d["gamma"]), np.nanmax(d["gamma"]), 2)
            yline = a + b * xline
            h, = ax.plot(xline, yline, lw=2.0, color=col,
                         label=f"{ion}: β={b:.3f}, χ={a:.2f}")
            handles.append(h); labels.append(h.get_label())

    ax.set_xlabel("γ (bin)")
    ax.set_ylabel("log$_{10}$(Photon frequency [Hz])")
    ax.set_title("Cross-ion photon decay in the γ frame (no towers)")

    if handles:
        ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.9)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)

def make_hydrogenic_plot(per_ion_stats: dict, outpath: Path, dpi=160,
                         export_csv: bool=False, csv_path: Path | None=None):
    # include D_I here
    hydro = ["H_I", "D_I", "He_II", "Li_III", "O_VIII"]
    present = [ion for ion in hydro if ion in per_ion_stats]
    if not present:
        print("[viz] No hydrogenic ions present; skipping hydrogenic-only figure.")
        return

    palette = plt.cm.tab10.colors
    color_map = {ion: palette[i % len(palette)] for i, ion in enumerate(present)}

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=dpi)

    # scatter + line per ion
    for ion in present:
        d = per_ion_stats[ion]
        col = color_map[ion]
        ax.scatter(d["gamma"], d["ylog"], s=20, alpha=0.35, color=col, label=f"{ion} (n={d['n']})")
        if np.isfinite(d["b"]):
            xline = np.linspace(np.nanmin(d["gamma"]), np.nanmax(d["gamma"]), 100)
            yline = d["a"] + d["b"] * xline
            ax.plot(xline, yline, lw=2.25, color=col,
                    label=f"{ion}: β={d['b']:.3f}, χ={d['a']:.2f}, R²={d['r2']:.3f}")

    # If both H_I and D_I are present, report Δχ in legend text box
    if "H_I" in present and "D_I" in present:
        a_H = per_ion_stats["H_I"]["a"]
        a_D = per_ion_stats["D_I"]["a"]
        dchi = a_D - a_H
        ax.text(0.02, 0.02, rf"$\Delta\hat\chi(\mathrm{{D}}-\mathrm{{H}}) \approx {dchi:.3e}\ \mathrm{{dex}}$",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xlabel("γ (bin)")
    ax.set_ylabel(r"log$_{10}$(Photon frequency [Hz])")
    ax.set_title("Hydrogenic subset in the γ frame (no towers)")
    ax.legend(fontsize=8, framealpha=0.9, ncols=1, loc="best")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[viz] wrote hydrogenic-only figure → {outpath}")

    # optional CSV export (now including D_I if present)
    if export_csv and csv_path is not None:
        import csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ion", "beta_hat", "chi_hat", "R2", "n"])
            for ion in present:
                d = per_ion_stats[ion]
                w.writerow([ion, f"{d['b']:.6f}", f"{d['a']:.6f}", f"{d['r2']:.6f}", d['n']])
        print(f"[viz] wrote hydrogenic CSV → {csv_path}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-ion photon decay in γ frame (no towers): global and per-ion plots.")
    ap.add_argument("--input-dir", type=str,
                    default="data/results/photon_matched_resonant_pairs_mu-1/",
                    help="Directory containing *_resonant_transitions.csv")
    ap.add_argument("--output-dir", type=str,
                    default="data/results/plots/viz_all-ion_photon-decay_no-tower/",
                    help="Output directory for plots")
    ap.add_argument("--min-points", type=int, default=50,
                    help="Minimum photon count per ion to be included")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--no-scatter-global", action="store_true",
                    help="If set, global plot shows only per-ion OLS lines (no pooled scatter)")
    ap.add_argument("--export-csv", action="store_true",
                    help="Export per-ion OLS summary (ion, beta_hat, chi_hat, R2, n) to CSV.")
    ap.add_argument("--csv-name", type=str, default="pooled_per_ion_ols_summary.csv",
                    help="Filename for the per-ion OLS summary CSV (written into --output-dir).")
    ap.add_argument("--hydrogenic-inset", action="store_true",
                    help="Add an inset for {H I, He II, Li III, O VIII} pooled bands.")
    ap.add_argument("--hydrogenic-only", action="store_true",
                    help="Write a dedicated hydrogenic subset figure (H_I, He_II, Li_III, O_VIII) with scatter + fits.")
    ap.add_argument("--hydrogenic-csv", action="store_true",
                    help="Also export a CSV of hydrogenic per-ion OLS (beta_hat, chi_hat, R2, n).")

    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    df = load_all_points(inp)
    if df.empty:
        print(f"[viz] No data found under {inp}")
        return

    per_ion_stats = {}
    hydrogenic_set = {"H_I", "D_I", "He_II", "Li_III", "O_VIII"}

    for ion, sub in df.groupby("ion"):
        # keep D_I only for the hydrogenic-only outputs
        if ion.strip() == "D_I" and not args.hydrogenic_only:
            print(f"[viz] Skipping isotope {ion} for global/all-ion view")
            continue

        sub = sub.dropna()
        # allow hydrogenic ions a lighter threshold only when making the hydrogenic-only figure
        if len(sub) < args.min_points:
            allow_hydro = args.hydrogenic_only and ion.strip() in hydrogenic_set and len(sub) >= 10
            if not allow_hydro:
                print(f"[viz] Skip {ion}: only {len(sub)} photons (< {args.min_points})")
                continue

        gamma = sub["gamma"].to_numpy(dtype=float)
        lambda_nm = sub["lambda_nm"].to_numpy(dtype=float)
        nu = C / (lambda_nm * 1e-9)
        ylog = np.log10(nu)

        # per-ion plot
        per_ion_path = out / f"{ion}_photon_decay_no_tower.png"
        a, b, r2, n = fit_line(gamma, ylog)
        # make the plot file
        make_per_ion_plot(ion, gamma, lambda_nm, per_ion_path, dpi=args.dpi, annotate=True)

        # stash for global
        per_ion_stats[ion] = dict(gamma=gamma, ylog=ylog, a=a, b=b, r2=r2, n=n)

        print(f"[viz] {ion}: β={b:.5f}, χ={a:.3f}, R²={r2:.4f}, n={n}, wrote {per_ion_path}")

    if not per_ion_stats:
        print("[viz] No ions passed min-points threshold.")
        return

    # ----- optional CSV export -----
    if args.export_csv and per_ion_stats:
        import csv
        csv_path = out / args.csv_name
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ion", "beta_hat", "chi_hat", "R2", "n"])
            for ion, d in sorted(per_ion_stats.items()):
                if ion.strip() == "D_I":  # exclude D I from the global CSV
                    continue
                writer.writerow([ion, f"{d['b']:.6f}", f"{d['a']:.6f}", f"{d['r2']:.6f}", d['n']])
        print(f"[viz] wrote per-ion OLS summary CSV → {csv_path}")

    # global plot (pooled scatter + lines or lines only)
    global_path = out / "all_ions_photon_decay_no_tower.png"

    # global plot (pooled scatter + lines or lines only)
    stats_for_global = {ion: d for ion, d in per_ion_stats.items() if ion.strip() != "D_I"}
    global_path = out / "all_ions_photon_decay_no_tower.png"
    make_global_plot(stats_for_global, True, global_path,
                     dpi=args.dpi, no_scatter_global=args.no_scatter_global)
    print(f"[viz] wrote global plot → {global_path}")

    make_global_plot(per_ion_stats, True, global_path,
                     dpi=args.dpi, no_scatter_global=args.no_scatter_global)
    print(f"[viz] wrote global plot → {global_path}")

    # dedicated hydrogenic subset figure (scatter + lines)
    if args.hydrogenic_only:
        hydro_png = out / "hydrogenic_subset_no_tower.png"
        hydro_csv = out / "hydrogenic_subset_ols.csv" if args.hydrogenic_csv else None
        # pass only hydrogenic ions (this includes D_I)
        stats_for_hydro = {ion: d for ion, d in per_ion_stats.items() if ion.strip() in hydrogenic_set}
        make_hydrogenic_plot(stats_for_hydro, hydro_png, dpi=args.dpi,
                             export_csv=args.hydrogenic_csv, csv_path=hydro_csv)

    # ----- optional hydrogenic inset -----
    if args.hydrogenic_inset:
        import matplotlib.pyplot as plt
        # reopen the figure we just saved
        fig = plt.figure(figsize=(10.5, 7), dpi=args.dpi)
        # redraw the global (lines only) quickly
        ax = fig.add_subplot(111)
        # quick re-plot lines
        for ion, d in per_ion_stats.items():
            if np.isfinite(d["b"]):
                xline = np.linspace(np.nanmin(d["gamma"]), np.nanmax(d["gamma"]), 2)
                yline = d["a"] + d["b"] * xline
                ax.plot(xline, yline, lw=1.2, color=d["color"])
        ax.set_xlabel("γ (bin)"); ax.set_ylabel("log$_{10}$(Photon frequency [Hz])")
        ax.set_title("Cross-ion photon decay in the γ frame (no towers)")
        # inset axes
        inset = ax.inset_axes([0.60, 0.10, 0.35, 0.35])  # [left, bottom, width, height] in axes coordinates
        hydro = {"H_I", "He_II", "Li_III", "O_VIII"}
        for ion in sorted(set(per_ion_stats.keys()) & hydro):
            d = per_ion_stats[ion]
            # points
            inset.scatter(d["gamma"], d["ylog"], s=6, alpha=0.15, color=d["color"])
            # line
            if np.isfinite(d["b"]):
                xline = np.linspace(np.nanmin(d["gamma"]), np.nanmax(d["gamma"]), 2)
                yline = d["a"] + d["b"] * xline
                inset.plot(xline, yline, lw=2.0, color=d["color"], label=ion)
        inset.legend(fontsize=7, framealpha=0.9, loc="upper right")
        inset.set_title("Hydrogenic subset", fontsize=9)
        inset.set_xlabel("γ", fontsize=8); inset.set_ylabel("log$_{10}\,\\nu$", fontsize=8)
        # save as a separate _inset.png to avoid overwriting your main global
        global_inset_path = out / "all_ions_photon_decay_no_tower_inset.png"
        fig.tight_layout()
        fig.savefig(global_inset_path)
        plt.close(fig)
        print(f"[viz] wrote hydrogenic inset → {global_inset_path}")

if __name__ == "__main__":
    main()
