#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate γ-ladder portraits for fixed n_i slices in two camera angles:

A) Rotated view (what you asked for):
   x = n_k, y = gamma_bin     → "looking into" the sheet

B) Original view (for reference):
   x = gamma_bin, y = n_k     → the usual portrait

Marker semantics:
- color = normalized intensity per point (see VALUE COLUMN below)
- size   = normalized intensity (same quantity as color)
- marker = "x" by default (configurable)

VALUE COLUMN (auto-detected, in this priority order):
  1) obs_hits_gamma
  2) n_photons_matched
  3) obs_hits
  4) n_hits
  … else → 1.0 for all points

Usage example (update with ion and tower)
-------------
python -m scripts.views.ion_identity.plot_gamma_ladder_views \
  --ladders H_I:data/meta/ion_photon_ladders_mu-1/H_I_photon_ladder.csv \
  --ni 4 \
  --out  data/results/plots/gamma_levels_by-tower/H_I_4_gamma_vs_nk_grid.png \
  --marker x --cmap viridis

"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- IO helpers ----------

def load_ladder(csv_path: Path) -> pd.DataFrame:
    """
    Read a ladder CSV, skipping provenance/comment lines beginning with '#'.
    Converts key columns to numeric where possible.
    """
    df = pd.read_csv(csv_path, comment="#")
    # normalize expected columns if present
    for col in ("n_i", "n_k", "gamma_bin", "frequency_hz"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def pick_value_column(df: pd.DataFrame) -> str | None:
    """
    Choose a column to represent intensity (both color and size).
    Priority: obs_hits_gamma, n_photons_matched, obs_hits, n_hits.
    Returns None if nothing suitable found.
    """
    for cand in ("obs_hits_gamma", "n_photons_matched", "obs_hits", "n_hits"):
        if cand in df.columns:
            return cand
    return None

def normalize_series(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Min–max normalize to [0,1]."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < eps:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


# ---------- plotting helpers ----------

def scatter_slice(ax, sub, xcol, ycol, vcol, marker, cmap,
                  s_min=20, s_max=140, alpha=0.95, title=None):
    """
    Scatter for one (ion, n_i) slice.
    - xcol/ycol: which axes to use (e.g., n_k vs gamma_bin OR gamma_bin vs n_k)
    - vcol: value column used for color+size (normalized)
    """
    if sub.empty:
        ax.set_title((title or "") + "\n(no data)")
        ax.axis("off")
        return

    # choose intensity (color & size)
    if vcol is None:
        val = pd.Series(np.ones(len(sub)), index=sub.index)
    else:
        val = normalize_series(sub[vcol])

    # color & size mapping
    sizes = (s_min + (s_max - s_min) * val).clip(s_min, s_max)
    colors = val.values  # matplotlib colormap expects [0,1]

    ax.scatter(sub[xcol], sub[ycol],
               c=colors, s=sizes, cmap=cmap,
               marker=marker, linewidths=1.0, alpha=alpha)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    if title:
        ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.3)


def grid_plot(ladder_map, ions, ni_list, *, rotated=True,
              marker="x", cmap="viridis", dpi=180, out_path="out.png"):
    """
    Build a grid of plots: rows = n_i slices, cols = ions.
    rotated=True:  x = n_k,       y = gamma_bin
    rotated=False: x = gamma_bin, y = n_k
    """
    nrows, ncols = len(ni_list), len(ions)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(3.8*ncols, 3.2*nrows), squeeze=False)

    for r, ni in enumerate(ni_list):
        for c, ion in enumerate(ions):
            ax = axes[r, c]
            if ion not in ladder_map:
                ax.set_title(f"{ion}  n_i={ni}\n(missing file)")
                ax.axis("off")
                continue

            df = ladder_map[ion].copy()
            # take slice by n_i
            sub = df[df.get("n_i") == ni].copy()

            # pick value column
            vcol = pick_value_column(sub)

            # drop rows missing core axes
            if rotated:
                needed = ("n_k", "gamma_bin")
            else:
                needed = ("gamma_bin", "n_k")
            for col in needed:
                if col not in sub.columns:
                    sub = pd.DataFrame(columns=sub.columns)  # empty
                    break

            # scatter with chosen axes
            xcol, ycol = (("n_k", "gamma_bin") if rotated else ("gamma_bin", "n_k"))
            scatter_slice(ax, sub, xcol, ycol, vcol, marker, cmap,
                          title=f"{ion}  n_i={ni}")

            # keep y within observed band if rotated (helps visibility)
            try:
                yvals = pd.to_numeric(sub[ycol], errors="coerce")
                if yvals.notna().any():
                    ymin, ymax = float(yvals.min()), float(yvals.max())
                    pad = 0.05*(ymax - ymin + 1e-6)
                    ax.set_ylim(ymin - pad, ymax + pad)
            except Exception:
                pass

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------- CLI ----------

def parse_ladders(ladder_args):
    """
    --ladders takes pairs of 'ION_LABEL:/path/to/ladder.csv'
    Returns dict { 'ION_LABEL' : DataFrame }
    """
    ladder_map = {}
    for item in ladder_args:
        try:
            label, path = item.split(":", 1)
        except ValueError:
            raise SystemExit(f"Bad --ladders item (expect LABEL:PATH): {item}")
        path = Path(path).expanduser()
        if not path.exists():
            print(f"[warn] ladder file not found for {label}: {path}")
            continue
        try:
            ladder_map[label] = load_ladder(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
    return ladder_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", nargs="+", required=True,
                    help="Pairs like H_I:/path/H_I_photon_ladder.csv (space separated)")
    ap.add_argument("--ni", nargs="+", type=int, default=[2,3,4],
                    help="List of n_i slices to plot (default: 2 3 4)")
    ap.add_argument("--marker", default="x", help="Matplotlib marker (default: x)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis)")
    ap.add_argument("--out", default="rotated_gamma_vs_nk_grid.png",
                    help="PNG for rotated view (x=n_k, y=gamma)")
    ap.add_argument("--out2", default="original_gamma_vs_nk_grid.png",
                    help="PNG for original view (x=gamma, y=n_k)")
    args = ap.parse_args()

    ladder_map = parse_ladders(args.ladders)
    ions = list(ladder_map.keys())
    if not ions:
        raise SystemExit("No valid ladders loaded. Check --ladders paths.")

    # A) Rotated view (your "orthogonal" camera)
    grid_plot(ladder_map, ions, args.ni, rotated=True,
              marker=args.marker, cmap=args.cmap, out_path=args.out)

    # B) Original view (for reference)
    grid_plot(ladder_map, ions, args.ni, rotated=False,
              marker=args.marker, cmap=args.cmap, out_path=args.out2)

if __name__ == "__main__":
    main()
