#!/usr/bin/env python3
# Pretty re-draw of pooled Δτ vs 1/ν² for a chosen (or best-R²) window.
# Usage example:
#   python -m scripts.analysis_pipeline.pretty_dtau_plot \
#     --rel-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_rel_windows.csv \
#     --fit-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_fit_windows.csv \
#     --ref-freq-hz 10000000 \
#     --out data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_vs_invnu2_pretty.png
"""

python -m scripts.views.radio.pretty_dtau_plot \
  --rel-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_rel_windows.csv \
  --fit-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_fit_windows.csv \
  --ref-freq-hz 10000000 \
  --out data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_vs_invnu2_pretty.png

"""


import argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patheffects as path_effects

def parse_args():
    ap = argparse.ArgumentParser(description="Pretty plot for pooled Δτ vs 1/ν².")
    ap.add_argument("--rel-csv", required=True, help="pooled_*_DTau_rel_windows.csv")
    ap.add_argument("--fit-csv", required=True, help="pooled_*_DTau_fit_windows.csv")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--timestamp", default=None,
                    help="ISO timestamp (UTC) of the window center to plot. If omitted, best R² is used.")
    ap.add_argument("--ref-freq-hz", type=float, default=10_000_000.0,
                    help="Reference frequency used in your analysis (default 10 MHz).")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    return ap.parse_args()

def main():
    args = parse_args()
    fit_df = pd.read_csv(args.fit_csv, index_col=0, parse_dates=True)
    rel_df = pd.read_csv(args.rel_csv, index_col=0, parse_dates=True)

    # Choose window: user-specified timestamp or best R²
    if args.timestamp:
        t0 = pd.to_datetime(args.timestamp, utc=True)
        if t0 not in rel_df.index or t0 not in fit_df.index:
            raise SystemExit(f"[ERROR] Timestamp {t0} not found in both CSVs.")
    else:
        pick = fit_df.sort_values("r2", ascending=False).head(1)
        if pick.empty:
            raise SystemExit("[ERROR] No window fits found.")
        t0 = pick.index[0]

    # Build X, Y from the chosen window row
    row = rel_df.loc[t0]
    X, Y, labels = [], [], []
    for col, val in row.items():
        if not col.startswith("DTau_") or pd.isna(val):
            continue
        try:
            nu = float(col.replace("DTau_", ""))  # Hz
        except ValueError:
            # fallback: pull digits from the column name
            m = re.search(r"DTau_(\d+)", col)
            if not m: 
                continue
            nu = float(m.group(1))
        x = (1.0/(nu**2)) - (1.0/(args.ref_freq_hz**2))
        X.append(x); Y.append(val); labels.append(f"{nu/1e6:.2f} MHz")

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    if len(X) < 2:
        raise SystemExit("[ERROR] Need at least 2 bands in that window to draw a line.")

    # Grab the per-window fit for annotation
    fit = fit_df.loc[t0]
    a = float(fit["a"]); b = float(fit["b"]); r2 = float(fit["r2"])

    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=args.dpi)
    ax.scatter(X, Y, s=50, color="#0072B2", edgecolor="black", linewidth=0.5, zorder=3)

    # Add labels slightly offset and with halo for clarity
    for xi, yi, lab in zip(X, Y, labels):
        ax.annotate(
            lab, (xi, yi),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            weight="medium",
            color="black",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
            zorder=4,
        )

    # Fit line overlay
    xline = np.linspace(X.min(), X.max(), 100)
    yline = a + b * xline
    ax.plot(xline, yline, linewidth=1.6)

    # Axes labels (ASCII-safe)
    ax.set_xlabel("1/nu^2 - 1/nu_ref^2 [Hz^-2]")
    ax.set_ylabel("Delta tau_rel [s]")

    # Title with timestamp
    ax.set_title(
        f"Grid EN | Δτ vs 1/ν² (window {t0.strftime('%Y-%m-%d %H:%M UTC')})",
        pad=12, fontsize=12, weight="semibold"
    )

    # Scientific notation formatting and margins to avoid cropping
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    ax.margins(x=0.08, y=0.12)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.16)

    # Inset text with white background for readability
    txt = f"y = {a:.3e} + {b:.3e}·x\nR² = {r2:.3f}"
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#444", alpha=0.85),
        zorder=5
    )

    ax.grid(True, linewidth=0.6, alpha=0.45, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save without cropping right-side labels
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.15)
    print(f"[OK] Wrote {args.out}")

if __name__ == "__main__":
    main()
