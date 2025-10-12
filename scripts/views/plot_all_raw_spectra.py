#!/usr/bin/env python3
"""
Plot raw spectra for all datasets in a standard time-series way.

Outputs:
  results/raw_plots/
    sun_fluxatl_grid.png            # montage of lm####.txt files
    sun_photatl_grid.png            # montage of wn####.txt files (solar component)
    vega_grid.png                   # montage of ELODIE txt files
    lamps_grid.png                  # montage of line-list stick plots (Ne/Na/Hg)
    molecule_grid.png               # montage of C2 Swan stick plots (if multiple)
    per-file/*.png                  # (optional) one image per file

Examples:
  python -m scripts.views.ion_identity.plot_all_raw_spectra \
  --fluxatl-glob   "data/solar/fluxatl/lm0296.txt" \
  --photatl-glob   "data/solar/photatl/wn1850.txt" \
  --vega-glob      "data/stars/vega_elodie/elodie_19940830_0009.txt" \
  --lamps-csv-glob "results/lamps/raw_csv/*_ritz_vac.csv" \
  --molecule-csv   "results/molecules/raw_csv/C2_Swan_visible.csv" \
  --max-files 6 --grid 2x3 --normalize --downsample 2 \
  --outdir results/raw_plots --save-per-file


Notes:
- We plot Sun/Vega as continuous flux-vs-λ curves.
- For line lists (lamps/molecule), we draw *sticks* at line centers with heights ~ intensity/weight.
"""

import argparse, re
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Loaders -----------------------------

def load_fluxatl(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    waves, rflux, irrad = [], [], []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 2: continue
            try:
                lam = float(parts[0])
                rf  = float(parts[1])
                waves.append(lam); rflux.append(rf)
                if len(parts) >= 3:
                    irrad.append(float(parts[2]))
            except Exception:
                continue
    lam = np.asarray(waves, float)
    rfl = np.asarray(rflux, float)
    irr = np.asarray(irrad, float) if len(irrad) == len(lam) else None
    return lam, rfl, irr

def load_photatl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    wn, sol = [], []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 2: continue
            try:
                wn_val = float(parts[0])
                sol_val = float(parts[1])
                wn.append(wn_val); sol.append(sol_val)
            except Exception:
                continue
    wn = np.asarray(wn, float)
    lam_nm = 1e7 / wn
    return lam_nm, np.asarray(sol, float)

def load_vega_elodie(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lam, flx = [], []
    with open(path, "r", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 2: continue
            try:
                lam.append(float(parts[0]))
                flx.append(float(parts[1]))
            except Exception:
                continue
    return np.asarray(lam, float), np.asarray(flx, float)

def load_line_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    wl = (cols.get("wavelength_nm") or cols.get("wavelength") or cols.get("lambda_nm"))
    if not wl:
        raise ValueError(f"No wavelength column found in {path.name}")
    out = pd.DataFrame({"wavelength_nm": pd.to_numeric(df[wl], errors="coerce")})

    # --- auto-convert if wavelengths look like wavenumbers (cm⁻¹)
    if out["wavelength_nm"].median() > 5000:           # clearly not nm
        out["wavelength_nm"] = 1e7 / out["wavelength_nm"]

    if "intensity" in cols:
        out["intensity"] = pd.to_numeric(df[cols["intensity"]], errors="coerce")
    if "weight" in cols:
        out["weight"] = pd.to_numeric(df[cols["weight"]], errors="coerce")
    return out.dropna(subset=["wavelength_nm"])

# ---------------------------- Plot helpers ----------------------------

def normalize_series(y: np.ndarray) -> np.ndarray:
    if y.size == 0: return y
    y = y.astype(float)
    p98 = np.nanpercentile(y, 98)
    if p98 > 0:
        return y / p98
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    rng = (ymax - ymin) if ymax > ymin else 1.0
    return (y - ymin) / rng

def downsample_xy(x: np.ndarray, y: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    if step <= 1 or x.size <= step:
        return x, y
    return x[::step], y[::step]

def plot_curve(ax, lam_nm: np.ndarray, y: np.ndarray, title: str,
               xlim: Optional[Tuple[float,float]] = None, ylabel: str = "flux (arb.)"):
    if lam_nm.size == 0 or y.size == 0:
        ax.set_axis_off(); ax.set_title(title); return
    idx = np.argsort(lam_nm)
    ax.plot(lam_nm[idx], y[idx], lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wavelength (nm)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if xlim: ax.set_xlim(*xlim)

def plot_sticks(ax, lam_nm: np.ndarray, h: np.ndarray, title: str,
                xlim: Optional[Tuple[float,float]] = None, color="tab:blue"):
    if lam_nm.size == 0:
        ax.set_axis_off(); ax.set_title(title); return
    idx = np.argsort(lam_nm)
    ax.vlines(lam_nm[idx], 0, h[idx], lw=0.7, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wavelength (nm)", fontsize=9)
    ax.set_ylabel("intensity (arb.)", fontsize=9)
    if xlim: ax.set_xlim(*xlim)

def make_grid(n: int, grid_spec: str) -> Tuple[int,int]:
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", grid_spec or "")
    if not m:
        cols = int(np.ceil(np.sqrt(n))) or 1
        rows = int(np.ceil(n / cols)) or 1
        return rows, cols
    r, c = int(m.group(1)), int(m.group(2))
    return r, c

# ----------------------------- Main driver -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluxatl-glob", type=str, default=None)
    ap.add_argument("--photatl-glob", type=str, default=None)
    ap.add_argument("--vega-glob",    type=str, default=None)
    ap.add_argument("--lamps-csv-glob", type=str, default=None)
    ap.add_argument("--molecule-csv", type=str, default=None)
    ap.add_argument("--max-files", type=int, default=6)
    ap.add_argument("--grid", type=str, default="2x3")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--xlim", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--save-per-file", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print matches and output paths")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    perdir = outdir / "per-file"
    if args.save_per_file: perdir.mkdir(parents=True, exist_ok=True)

    xlim = None
    if args.xlim:
        try:
            lo, hi = args.xlim.split(","); xlim = (float(lo), float(hi))
        except Exception:
            xlim = None

    def info(msg: str):
        if args.verbose: print(msg)

    # ------------ FluxAtlas grid ------------
    if args.fluxatl_glob:
        flux_files = sorted(list(Path().glob(args.fluxatl_glob)))[: args.max_files]
        info(f"[fluxatl] matched {len(flux_files)} file(s) for pattern: {args.fluxatl_glob}")
        if flux_files:
            rows, cols = make_grid(len(flux_files), args.grid)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*3.2))
            axes = np.atleast_1d(axes).ravel()
            for ax, p in zip(axes, flux_files):
                lam, rfl, irr = load_fluxatl(p)
                if args.downsample > 1:
                    lam, rfl = downsample_xy(lam, rfl, args.downsample)
                y = normalize_series(rfl) if args.normalize else rfl
                plot_curve(ax, lam, y, f"FluxAtlas: {p.name}", xlim=xlim, ylabel="residual flux")
                if args.save_per_file:
                    fig1, ax1 = plt.subplots(figsize=(6,3))
                    plot_curve(ax1, lam, y, f"{p.name}", xlim=xlim, ylabel="residual flux")
                    fig1.tight_layout(); fp = perdir / f"sun_fluxatl_{p.stem}.png"
                    fig1.savefig(fp, dpi=170); plt.close(fig1); info(f"[write] {fp}")
            for ax in axes[len(flux_files):]: ax.set_axis_off()
            fig.tight_layout(); fp = outdir / "sun_fluxatl_grid.png"
            fig.savefig(fp, dpi=190); plt.close(fig); info(f"[write] {fp}")

    # ------------ PhotAtlas grid ------------
    if args.photatl_glob:
        phot_files = sorted(list(Path().glob(args.photatl_glob)))[: args.max_files]
        info(f"[photatl] matched {len(phot_files)} file(s) for pattern: {args.photatl_glob}")
        if phot_files:
            rows, cols = make_grid(len(phot_files), args.grid)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*3.2))
            axes = np.atleast_1d(axes).ravel()
            for ax, p in zip(axes, phot_files):
                lam, sol = load_photatl(p)
                if args.downsample > 1:
                    lam, sol = downsample_xy(lam, sol, args.downsample)
                y = normalize_series(sol) if args.normalize else sol
                plot_curve(ax, lam, y, f"PhotAtlas: {p.name}", xlim=xlim, ylabel="solar intensity")
                if args.save_per_file:
                    fig1, ax1 = plt.subplots(figsize=(6,3))
                    plot_curve(ax1, lam, y, f"{p.name}", xlim=xlim, ylabel="solar intensity")
                    fig1.tight_layout(); fp = perdir / f"sun_photatl_{p.stem}.png"
                    fig1.savefig(fp, dpi=170); plt.close(fig1); info(f"[write] {fp}")
            for ax in axes[len(phot_files):]: ax.set_axis_off()
            fig.tight_layout(); fp = outdir / "sun_photatl_grid.png"
            fig.savefig(fp, dpi=190); plt.close(fig); info(f"[write] {fp}")

    # ------------ Vega grid ------------
    if args.vega_glob:
        vega_files = sorted(list(Path().glob(args.vega_glob)))[: args.max_files]
        info(f"[vega] matched {len(vega_files)} file(s) for pattern: {args.vega_glob}")
        if vega_files:
            rows, cols = make_grid(len(vega_files), args.grid)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*3.2))
            axes = np.atleast_1d(axes).ravel()
            for ax, p in zip(axes, vega_files):
                lam, flx = load_vega_elodie(p)
                if args.downsample > 1:
                    lam, flx = downsample_xy(lam, flx, args.downsample)
                y = normalize_series(flx) if args.normalize else flx
                plot_curve(ax, lam, y, f"Vega: {p.name}", xlim=xlim, ylabel="flux (arb.)")
                if args.save_per_file:
                    fig1, ax1 = plt.subplots(figsize=(6,3))
                    plot_curve(ax1, lam, y, f"{p.name}", xlim=xlim, ylabel="flux (arb.)")
                    fig1.tight_layout(); fp = perdir / f"vega_{p.stem}.png"
                    fig1.savefig(fp, dpi=170); plt.close(fig1); info(f"[write] {fp}")
            for ax in axes[len(vega_files):]: ax.set_axis_off()
            fig.tight_layout(); fp = outdir / "vega_grid.png"
            fig.savefig(fp, dpi=190); plt.close(fig); info(f"[write] {fp}")

    # ------------ Lamps (line lists) grid ------------
    if args.lamps_csv_glob:
        lamp_files = sorted(list(Path().glob(args.lamps_csv_glob)))[: args.max_files]
        info(f"[lamps] matched {len(lamp_files)} file(s) for pattern: {args.lamps_csv_glob}")
        if lamp_files:
            rows, cols = make_grid(len(lamp_files), args.grid)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*3.2))
            axes = np.atleast_1d(axes).ravel()
            for ax, p in zip(axes, lamp_files):
                df = load_line_csv(p)
                lam = df["wavelength_nm"].to_numpy()
                if "weight" in df.columns:
                    h = df["weight"].to_numpy()
                elif "intensity" in df.columns:
                    h = df["intensity"].to_numpy()
                else:
                    h = np.ones_like(lam)
                h = h / (np.nanpercentile(h, 98) or (np.nanmax(h) or 1.0))
                plot_sticks(ax, lam, h, f"Lamp: {p.stem}", xlim=xlim)
                if args.save_per_file:
                    fig1, ax1 = plt.subplots(figsize=(6,3))
                    plot_sticks(ax1, lam, h, f"{p.stem}", xlim=xlim)
                    fig1.tight_layout(); fp = perdir / f"lamp_{p.stem}.png"
                    fig1.savefig(fp, dpi=170); plt.close(fig1); info(f"[write] {fp}")
            for ax in axes[len(lamp_files):]: ax.set_axis_off()
            fig.tight_layout(); fp = outdir / "lamps_grid.png"
            fig.savefig(fp, dpi=190); plt.close(fig); info(f"[write] {fp}")

    # ------------ Molecule (C2 Swan) ------------
    if args.molecule_csv:
        mol_paths = list(Path().glob(args.molecule_csv)) if any(ch in args.molecule_csv for ch in "*?[]") \
                     else [Path(args.molecule_csv)]
        mol_files = sorted(mol_paths)[: args.max_files]
        info(f"[molecule] matched {len(mol_files)} file(s) for: {args.molecule_csv}")
        if mol_files:
            rows, cols = make_grid(len(mol_files), args.grid)
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*3.2))
            axes = np.atleast_1d(axes).ravel()
            for ax, p in zip(axes, mol_files):
                df = load_line_csv(p)
                lam = df["wavelength_nm"].to_numpy()
                h = df["weight"].to_numpy() if "weight" in df.columns else \
                    (df["intensity"].to_numpy() if "intensity" in df.columns else np.ones_like(lam))
                h = h / (np.nanpercentile(h, 98) or (np.nanmax(h) or 1.0))
                plot_sticks(ax, lam, h, f"Molecule: {p.stem}", xlim=xlim)
                if args.save_per_file:
                    fig1, ax1 = plt.subplots(figsize=(6,3))
                    plot_sticks(ax1, lam, h, f"{p.stem}", xlim=xlim, color="tab:red")
                    fig1.tight_layout(); fp = perdir / f"molecule_{p.stem}.png"
                    fig1.savefig(fp, dpi=170); plt.close(fig1); info(f"[write] {fp}")
            for ax in axes[len(mol_files):]: ax.set_axis_off()
            fig.tight_layout(); fp = outdir / "molecule_grid.png"
            fig.savefig(fp, dpi=190); plt.close(fig); info(f"[write] {fp}")

if __name__ == "__main__":
    main()
