#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threadlaw_photoncode.py
-------------------
Build a *whole‑spectrum* thread‑law photoncode from photons only.
Works for:
  • Molecules: JCAMP‑DX infrared spectra (peaks automatically found)
  • Ions (or anything): CSV listing photon frequencies

Minimal principle (slope‑first):
  1) y = log10 ν (dex), sort descending
  2) κ = (y - y0)/β with β = log10 α  (default)
  3) Bin κ on a fixed Δγ lattice → unitless binary photoncode

Outputs (under OUTDIR/LABEL/): NOTE! Some of these names are now deprecated. "Barcode" is "photon code," for example.

  - peaks.csv                 (if JCAMP: detected peaks)
  - y_sorted.csv              (descending y list)
  - kappa_series.csv          (κ and Δκ)
  - barcode_dense.csv         (γ_int_rel, γ_rel, present)
  - barcode_bw.png            (clean black/white barcode)
  - delta_kappa_hist.png      (Δκ histogram)
  - summary.json              (grid, counts, simple stats)

Usage examples
--------------
Process ions from their gamma photon ladders files; molecules use the NIST gas IR spectra.
Make sure to apply a fine grid because photon frequencies are highly specific.

# Molecules (all JCAMP IR), Δκ=0.002
python -m scripts.MDL.threadlaw_photoncode \
  --batch-molecules-glob 'data/perfume/*/*-IR.jdx' \
  --outdir data/meta/whole_barcodes_mu1_dk0002 \
  --grid 0.002 --k-max 4.14 \
  --k-max-zoom 1.60 \
  --min-prom 0.02 --min-dist-px 6 \
  --tick-step 0.10
  --skip-existing

To make the plots:

python -m scripts.MDL.threadlaw_photoncode \
  --batch-ion-ladders-glob 'data/meta/ion_photon_ladders_mu-1/*_photon_ladder.csv' \
  --outdir data/meta/whole_barcodes_mu1_dk0002 \
  --grid 0.002 --k-max 4.14 --k-max-zoom 1.60 \
  --tick-step 0.10

Optional:
  --skip-existing

# Ions (processed photon ladders → whole-ion κ-photoncodes), Δκ=0.002
python -m scripts.MDL.threadlaw_photoncode \
  --batch-ion-ladders-glob 'data/meta/ion_photon_ladders_mu-1/*_photon_ladder.csv' \
  --outdir data/meta/whole_barcodes_mu1_dk0002 \
  --grid 0.002 --k-max 4.14 \
  --skip-existing

# 532 nm (ROD 3500025) — black & white κ-barcode + zoom
python -m scripts.MDL.threadlaw_photoncode \
  --label quartz_ROD_532 \
  --jdx phonons/3500025_quartz.txt \
  --outdir phonons/photoncodes_quartz/3500025_quartz \
  --grid 0.002 \
  --k-max 4.14 \
  --k-max-zoom 1.70 \
  --tick-step 0.10

# 633 nm (ROD 3500272)
python -m scripts.MDL.threadlaw_photoncode \
  --label quartz_ROD_633 \
  --jdx phonons/3500272_quartz.txt \
  --outdir phonons/photoncodes_quartz/3500272_quartz \
  --grid 0.002 \
  --k-max 4.14 \
  --k-max-zoom 1.70 \
  --tick-step 0.10


Author: BOING! 2025
License: TBD (we are filing an invention disclosure for this body of work
"""
from __future__ import annotations
import argparse, json, math, re, sys, os, glob

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- constants ----------
C_MS = 299_792_458.0
ALPHA = 1.0/137.035999084
LOG10_ALPHA = math.log10(ALPHA)   # β ≈ log10 α  (negative)
EPS = 1e-12

# ---------- JCAMP-DX (molecule) ----------
def parse_jcamp_dx(path: str):
    """Return x (cm^-1), scaled y in [0,1] from a simple JCAMP-DX (XYDATA)."""
    with open(path, 'r', encoding='latin-1', errors='ignore') as f:
        text = f.read()
    lines = [ln.strip() for ln in text.replace('\r','\n').split('\n') if ln.strip()!='']

    firstx=lastx=deltax=None; npoints=None; xfactor=yfactor=1.0
    meta={}
    def kv(ln):
        m=re.match(r"^##\s*([A-Za-z0-9\-\_]+)\s*=\s*(.*)$", ln)
        return (m.group(1).upper(), m.group(2).strip()) if m else (None,None)
    xy_idx=None
    for i,ln in enumerate(lines):
        if ln.startswith("##"):
            k,v=kv(ln); 
            if not k: continue
            meta[k]=v
            if k=="FIRSTX":   firstx=float(v) if re.match(r"^-?\d",v) else firstx
            if k=="LASTX":    lastx =float(v) if re.match(r"^-?\d",v) else lastx
            if k=="NPOINTS":  npoints=int(float(v)) if re.match(r"^-?\d",v) else npoints
            if k=="DELTAX":   deltax=float(v) if re.match(r"^-?\d",v) else deltax
            if k=="XFACTOR":  xfactor=float(v) if re.match(r"^-?\d",v) else xfactor
            if k=="YFACTOR":  yfactor=float(v) if re.match(r"^-?\d",v) else yfactor
        s = ln.upper()
        if s.startswith("##XYDATA") or s.startswith("##PEAK TABLE"):
            xy_idx = i+1; break
    if xy_idx is None:
        raise ValueError("No XYDATA/PEAK TABLE in JCAMP")

    if deltax is None and (firstx is not None and lastx is not None and npoints and npoints>1):
        deltax = (lastx-firstx)/(npoints-1)

    xs, ys = [], []
    for ln in lines[xy_idx:]:
        if ln.startswith("##"): break
        parts = [p for p in ln.replace(',', ' ').split() if p]
        if not parts: continue
        try:
            first = float(parts[0]) * xfactor
            rest  = [float(t)*yfactor for t in parts[1:]]
            if len(rest)>=1 and deltax is not None:
                x0=first
                for k,yy in enumerate(rest):
                    xs.append(x0 + k*deltax); ys.append(yy)
            elif len(parts)==2:
                xs.append(first); ys.append(rest[0])
        except:
            continue
    if not xs: raise ValueError("Failed to parse XY points")
    x = np.asarray(xs,float); y = np.asarray(ys,float)
    # Normalize y to [0,1]
    y2 = (y - y.min()) / (y.max() - y.min() + EPS)
    # Sort x ascending (common convention); caller will handle
    order = np.argsort(x); x = x[order]; y2 = y2[order]
    return x, y2

def smooth(y, win=7):
    if win<=1: return y.copy()
    pad=win//2
    ypad=np.pad(y,(pad,pad),mode='edge')
    k=np.ones(win)/win
    return np.convolve(ypad,k,mode='valid')

def pick_peaks(x, y, min_prom=0.05, min_dist_px=8):
    """Simple peak picker; returns x_peak (cm^-1) and normalized peak heights."""
    ys = smooth(y,7)
    dy = np.diff(ys); sgn=np.sign(dy)
    cand = np.where((sgn[:-1]>0) & (sgn[1:]<0))[0] + 1
    if len(cand)==0: return np.array([]), np.array([])
    yr = (ys.max()-ys.min()) or 1.0
    peaks=[]; inten=[]
    last=-min_dist_px
    for idx in cand:
        if idx - last < min_dist_px: continue
        w=20; lo=max(0,idx-w); hi=min(len(ys), idx+w+1)
        base = (np.min(ys[lo:idx]) + np.min(ys[idx:hi]))/2.0
        prom = ys[idx]-base
        if prom >= min_prom*yr:
            peaks.append(idx); inten.append(ys[idx]); last=idx
    return x[peaks], np.asarray(inten,float)

# ---------- utilities ----------
def hz_from_csv_row(row: pd.Series) -> Optional[float]:
    # Accepted columns: frequency_hz, log10_nu_hz, wavenumber_cm1, wavelength_nm
    if "frequency_hz" in row and pd.notna(row["frequency_hz"]):
        return float(row["frequency_hz"])
    if "log10_nu_hz" in row and pd.notna(row["log10_nu_hz"]):
        return 10.0**float(row["log10_nu_hz"])
    if "wavenumber_cm1" in row and pd.notna(row["wavenumber_cm1"]):
        return float(row["wavenumber_cm1"]) * 100.0 * C_MS
    if "wavelength_nm" in row and pd.notna(row["wavelength_nm"]):
        lam_m = float(row["wavelength_nm"]) * 1e-9
        return C_MS / lam_m
    return None

def build_kappa_series(y_log10: np.ndarray, beta: float,
                       y0_anchor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return κ and Δκ using optional global anchor y0_anchor (log10 ν₀)."""
    if y_log10.size == 0:
        return np.array([]), np.array([])
    y_sorted = np.sort(y_log10)[::-1]
    # choose anchor: global if supplied, else local top
    anchor = float(y0_anchor) if (y0_anchor is not None) else y_sorted[0]
    kappa = (y_sorted - anchor) / beta     # β < 0 ⇒ κ increases downward
    dkap  = np.diff(kappa) if kappa.size > 1 else np.array([])
    return kappa, dkap

def build_barcode_from_kappa(kappa: np.ndarray, grid: float) -> Tuple[np.ndarray, int, int]:
    """Return dense bits over [gmin..gmax] and gmin,gmax for κ on Δ grid (relative axis)."""
    if kappa.size == 0:
        return np.zeros(0,np.uint8), 0, -1
    gamma_int = np.rint(kappa / grid).astype(int)
    gmin, gmax = int(gamma_int.min()), int(gamma_int.max())
    L = gmax - gmin + 1
    bits = np.zeros(L, np.uint8)
    for gi in gamma_int:
        bits[int(gi - gmin)] = 1
    return bits, gmin, gmax

def plot_barcode(bits: np.ndarray, gmin: int, grid: float, out_png: Path, title: str,
                 kmax: float, tick_step: Optional[float] = None):
    if bits.size == 0:
        fig=plt.figure(figsize=(8,1.6)); ax=plt.gca()
        ax.text(0.5,0.5,"empty",ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off(); fig.tight_layout(); fig.savefig(out_png,dpi=150); plt.close(fig); return

    # bins actually present
    x = np.arange(bits.size, dtype=int)  # relative 0..L-1
    # global axis extent in integer bins
    xmax_bins = int(round(kmax / grid))
    # figure width based on fixed axis extent
    fig_width = max(10.0, xmax_bins/25.0)
    fig = plt.figure(figsize=(fig_width, 1.6)); ax = plt.gca()
    for xi,b in zip(x,bits):
        if b: ax.plot([xi,xi],[0,1],color="black",linewidth=1.2)
    ax.set_yticks([])

    # choose tick positions in BIN units, then label them in κ units
    if tick_step is not None:
        # tick_step is in κ units; convert to bins
        stride_bins = max(1, int(round(float(tick_step) / grid)))
        xticks = np.arange(0, xmax_bins + 1, stride_bins)
    else:
        # simple "nice" auto stride targeting ~10 ticks across the axis
        target_ticks = 10
        stride_bins = max(1, int(round(xmax_bins / target_ticks)))
        xticks = np.arange(0, xmax_bins + 1, stride_bins)

    # apply ticks (positions are bin indices; labels are κ = bin * Δ)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{i * grid:g}" for i in xticks])

    # force fixed extent 0..κmax  (xmax_bins bins of width Δ => κmax = xmax_bins * Δ)
    # If you plot bars centered at integers, prefer (-0.5, xmax_bins-0.5) instead.
    ax.set_xlim(0, xmax_bins)

    ax.set_xlabel(f"κ (Δ={grid:g})  [relative]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def plot_delta_kappa_hist(dkap: np.ndarray, out_png: Path, title: str):
    fig=plt.figure(figsize=(6.4,3.2)); ax=plt.gca()
    if dkap.size>0:
        ax.hist(dkap, bins=40, range=(0, max(1.0, float(dkap.max()))), density=True, alpha=0.85)
        ax.set_xlabel("Δκ"); ax.set_ylabel("density")
    else:
        ax.text(0.5,0.5,"no Δκ (n<2)",ha="center",va="center",transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_png,dpi=160); plt.close(fig)

def plot_barcode_compact(bits, gmin, grid, outpath, label, kmax_zoom):
    """
    Minimal, tile-friendly zoom panel:
      - fixed κ range [0, kmax_zoom] (identical across all items)
      - no axes or ticks
      - small top-left label: "<name> (Δκ=<grid>)"
    """
    k = np.arange(len(bits))*grid + gmin
    mask = (k >= 0.0) & (k <= kmax_zoom + 1e-12)
    k = k[mask]; b = bits[mask]

    fig_h = 1.0  # short strip
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=250)
    ax.set_xlim(0, kmax_zoom)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # draw thin baseline
    ax.add_line(plt.Line2D([0, kmax_zoom], [0.5, 0.5], lw=0.6, color='0.65'))

    # draw bars where b==1
    if b.size:
        dk = grid
        y0, y1 = 0.2, 0.8
        for xi in k[b.astype(bool)]:
            ax.add_patch(Rectangle((xi, y0), dk, y1-y0, facecolor="black", edgecolor="black", lw=0.0))

    # compact top-left label
    ax.text(0.01, 0.95, f"{label} (Δκ={grid:g})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="black")

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------- main pipeline ----------
def process_jdx(label: str, jdx_path: Path, outdir: Path, grid: float, beta: float,
                kmax: float, kmax_zoom: float,
                min_prom: float, min_dist_px: int, tick_step: Optional[float]) -> dict:
    x_cm1, y_scaled = parse_jcamp_dx(str(jdx_path))
    px_cm1, p_int = pick_peaks(x_cm1, y_scaled, min_prom=min_prom, min_dist_px=min_dist_px)

    if px_cm1.size == 0:
        return {"label": label, "n_peaks": 0, "error": "no peaks detected"}

    nu_hz = px_cm1 * 100.0 * C_MS
    ylog  = np.log10(nu_hz)

    # Save detected peaks
    df_peaks = pd.DataFrame({"wavenumber_cm1": px_cm1, "intensity": p_int, "log10_nu_hz": ylog})
    df_peaks.to_csv(outdir/"peaks.csv", index=False)

    y0_anchor = float(np.max(ylog))

    return build_outputs(label, ylog, outdir, grid, beta, kmax, kmax_zoom, tick_step, y0_anchor=y0_anchor)

def _norm(s: str) -> str:
    # normalize column names: lowercase alnum only
    return "".join(ch for ch in s.lower() if ch.isalnum())

def build_outputs(label: str, ylog: np.ndarray, outdir: Path, grid: float, beta: float,
                  kmax: float, kmax_zoom: float, tick_step: Optional[float],
                  y0_anchor: Optional[float] = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # Sort y and build κ series using global anchor if provided
    y_sorted = np.sort(ylog)[::-1]
    pd.DataFrame({"log10_nu_hz_sorted": y_sorted}).to_csv(outdir/"y_sorted.csv", index=False)

    if y0_anchor is None:
        y0_anchor = float(y_sorted[0])

    kappa, dkap = build_kappa_series(y_sorted, beta=beta, y0_anchor=y0_anchor)
    pd.DataFrame({"kappa": kappa, "delta_kappa_next": np.append(dkap, np.nan)}).to_csv(outdir/"kappa_series.csv", index=False)

    # Build barcode
    bits, gmin, gmax = build_barcode_from_kappa(kappa, grid=grid)
    rows=[{"kappa_int_rel": gmin+i, "kappa_rel": (gmin+i)*grid, "present": int(b)} for i,b in enumerate(bits)]
    pd.DataFrame(rows).to_csv(outdir/"barcode_dense.csv", index=False)

    # Summary JSON
    exceeds = bool(kappa.size and (kappa.max() > kmax + 1e-9))
    # record dataset span in y and κ
    y_span = float(y_sorted[0] - y_sorted[-1]) if y_sorted.size>1 else 0.0
    k_span = float((y_sorted[0] - y_sorted[-1]) / abs(beta)) if y_sorted.size>1 else 0.0
    summary = {
        "label": label,
        "beta": beta,
        "grid": grid,
        "span": {"delta_y": y_span, "delta_kappa": k_span},
        "kmax_fixed": kmax,
        "kmax_zoom": kmax_zoom,
        "n_peaks": int(ylog.size),
        "kappa": {"min": float(kappa.min()) if kappa.size else None,
                  "max": float(kappa.max()) if kappa.size else None},
        "extent_note": ("WARNING: κ exceeds kmax; plot is clipped" if exceeds else "OK"),
        "barcode": {"gmin_rel": int(gmin), "gmax_rel": int(gmax),
                    "n_bins_total": int(bits.size), "n_bins_present": int(int(bits.sum()))}
    }
    with open(outdir/"summary.json","w") as f: json.dump(summary, f, indent=2)

    # Plots
    plot_barcode(bits, gmin, grid, outdir/"barcode_bw.png",
                 title=f"{label}: sheet-law barcode (Δ={grid:g})",
                 kmax=kmax, tick_step=tick_step)
    # Uniform zoom panel with the SAME style as the original (ticks, no midline)
    plot_barcode(bits, gmin, grid, outdir/"barcode_bw_zoom.png",
                 title=f"{label} (Δκ={grid:g})",
                 kmax=kmax_zoom, tick_step=tick_step)
    plot_delta_kappa_hist(dkap, outdir/"delta_kappa_hist.png",
                          title=f"{label}: Δκ histogram")
    return summary

def process_csv(label: str, csv_path: Path, outdir: Path, grid: float, beta: float,
                kmax: float, kmax_zoom: float,
                freq_col: str=None, log10_col: str=None, wn_col: str=None, wl_nm_col: str=None,
                tick_step: Optional[float] = None,
                weight_col: Optional[str] = None, min_weight_quantile: Optional[float] = None) -> dict:

    # Flexible parser: infer delimiter; ignore comments; allow tabs/spaces/commas.
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")
    except Exception as e:
        # Fallback to comma
        df = pd.read_csv(csv_path, comment="#")
    if df.empty:
        return {"label": label, "n_peaks": 0, "error": "empty CSV"}

    # Build a map from normalized name -> real column
    colmap = { _norm(c): c for c in df.columns }

    # If the user specified explicit columns, use them (case-insensitive)
    def pick(name):
        if not name: return None
        key = _norm(name)
        return colmap.get(key, None)
    freq_col = pick(freq_col)
    log10_col = pick(log10_col)
    wn_col    = pick(wn_col)
    wl_nm_col = pick(wl_nm_col)

    # If not specified, try auto-detection among common names
    def pick_auto(cands):
        for cand in cands:
            if _norm(cand) in colmap:
                return colmap[_norm(cand)]
        return None

    if not any([freq_col, log10_col, wn_col, wl_nm_col]):
        freq_col  = pick_auto(["frequency_hz","nu_hz","freq_hz","freq"])
        log10_col = pick_auto(["log10_nu_hz","log10nu","y_log10","log10nu_hz"])
        wn_col    = pick_auto(["wavenumber_cm1","wn_cm1","wavenumber","wn"])
        wl_nm_col = pick_auto(["wavelength_nm","vacuum_wavelength_nm","air_wavelength_nm","lambda_nm"])

    # Extract usable log10(nu) values
    ys = []
    for _, row in df.iterrows():
        hz = None
        try:
            if freq_col and pd.notna(row[freq_col]):
                hz = float(row[freq_col])
            elif log10_col and pd.notna(row[log10_col]):
                ys.append(float(row[log10_col])); continue
            elif wn_col and pd.notna(row[wn_col]):
                hz = float(row[wn_col]) * 100.0 * C_MS
            elif wl_nm_col and pd.notna(row[wl_nm_col]):
                lam_m = float(row[wl_nm_col]) * 1e-9
                if lam_m > 0: hz = C_MS / lam_m
        except Exception:
            hz = None
        if hz is not None and hz > 0:
            ys.append(math.log10(hz))

    # Optional weight-based filtering BEFORE extracting ylog
    if weight_col and (weight_col in df.columns):
        w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy()
        if (min_weight_quantile is not None) and np.isfinite(w).any():
            thr = np.nanquantile(w[np.isfinite(w)], float(min_weight_quantile))
            df = df.loc[w >= thr].reset_index(drop=True)

    ylog = np.asarray(ys, float)
    if ylog.size == 0:
        return {"label": label, "n_peaks": 0,
                "error": f"no usable frequency columns. Available: {list(df.columns)}"}
    return build_outputs(label, ylog, outdir, grid, beta, kmax, kmax_zoom, tick_step)

def main():
    ap = argparse.ArgumentParser(description="Whole‑spectrum sheet‑law barcode (photons‑only)")
    # In single-file mode, --label is required. In batch mode it is ignored.
    ap.add_argument("--label", required=False, help="Short label for outputs (folder name)")
    ap.add_argument("--outdir", required=True, help="Base output directory")
    ap.add_argument("--grid", type=float, default=0.002, help="κ lattice step Δ (default 0.002)")
    ap.add_argument("--tick-step", type=float, default=None,
                    help="force κ tick spacing (e.g., 0.20); auto if omitted")
    ap.add_argument("--beta", type=float, default=LOG10_ALPHA, help="Slope β (default log10 α)")
    ap.add_argument("--k-max", type=float, default=4.14,
                    help="Maximum κ extent for plots (default 4.14)")
    ap.add_argument("--k-max-zoom", type=float, default=1.70, help="Uniform zoom view κ max for visuals (no effect on stats)")

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--jdx",  help="JCAMP‑DX IR file (molecule)")
    src.add_argument("--csv",  help="CSV with photon frequencies (ions or molecules)")

    ap.add_argument("--min-prom", type=float, default=0.02, help="Peak prominence (JCAMP only)")
    ap.add_argument("--min-dist-px", type=int, default=6, help="Min distance (index) between peaks (JCAMP only)")
    # Optional explicit column names (case-insensitive). If omitted, we auto-detect.
    ap.add_argument("--freq-col", default=None, help="column with frequency in Hz")
    ap.add_argument("--log10-col", default=None, help="column with log10(nu_hz)")
    ap.add_argument("--wn-col", default=None, help="column with wavenumber (cm^-1)")
    ap.add_argument("--wl-nm-col", default=None, help="column with wavelength (nm)")

    # -------- Batch mode --------
    ap.add_argument("--batch-molecules-glob", default=None,
                    help="Glob of JCAMP‑DX files for molecules (labels = file stem)")
    ap.add_argument("--batch-ion-ladders-glob", default=None,
                    help="Glob of processed ion photon‑ladder CSVs (labels = <ION>_whole)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip labels that already have barcode_dense.csv")
    ap.add_argument("--weight-col", default=None,
                    help="Name of weight column in CSV (e.g., 'weight'); if set, enables thinning")
    ap.add_argument("--min-weight-quantile", type=float, default=None,
                    help="Keep rows with weight >= this quantile (e.g., 0.85)")
    ap.add_argument("--max-per-kappa", type=int, default=None,
                    help="Keep at most N lines per κ-bin (e.g., 1)")
    ap.add_argument("--beta-override", type=float, default=None,
                    help="If set, use this β (log10 sigma) instead of log10 alpha")

    args = ap.parse_args()

    # --- β override logic ---
    if args.beta_override is not None:
        args.beta = float(args.beta_override)

    # -------- Helper: derive ion whole‑label from ladder csv --------
    def _ion_label_from_ladder(path: Path) -> str:
        base = path.stem  # e.g., "Al_I_photon_ladder"
        for suf in ["_photon_ladder", "_ladder", "_photons", "_photonlist"]:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return f"{base}_whole"

    # -------- Batch driver --------
    def _run_batch() -> None:
        tasks = []
        if args.batch_molecules_glob:
            for p in sorted(set(glob.glob(args.batch_molecules_glob))):
                pth = Path(p)
                label = pth.stem
                tasks.append(("molecule", label, pth))
        if args.batch_ion_ladders_glob:
            for p in sorted(set(glob.glob(args.batch_ion_ladders_glob))):
                pth = Path(p)
                label = _ion_label_from_ladder(pth)
                tasks.append(("ion_ladder_csv", label, pth))

        if not tasks:
            print("[ERROR] Batch mode selected but no files matched.", file=sys.stderr)
            sys.exit(2)

        batch_ix = []
        for kind, label, in_path in tasks:
            outdir_label = Path(args.outdir) / label
            outdir_label.mkdir(parents=True, exist_ok=True)
            dense = outdir_label / "barcode_dense.csv"
            if args.skip_existing and dense.exists():
                batch_ix.append({"label": label, "kind": kind, "path": str(in_path), "skipped": True})
                continue
            if kind == "molecule":
                summary = process_jdx(label, in_path, outdir_label,
                                      grid=args.grid, beta=args.beta,
                                      kmax=args.k_max, kmax_zoom=args.k_max_zoom,
                                      min_prom=args.min_prom, min_dist_px=args.min_dist_px,
                                      tick_step=args.tick_step)
            else:  # ion ladder CSV → whole-ion κ-barcode
                summary = process_csv(label, in_path, outdir_label,
                                      grid=args.grid, beta=args.beta,
                                      kmax=args.k_max, kmax_zoom=args.k_max_zoom,
                                      freq_col=args.freq_col, log10_col=args.log10_col,
                                      wn_col=args.wn_col, wl_nm_col=args.wl_nm_col,
                                      tick_step=args.tick_step,
                                      weight_col=args.weight_col, min_weight_quantile=args.min_weight_quantile)

            batch_ix.append({"label": label, "kind": kind, "path": str(in_path),
                             "skipped": False, "summary": summary})
        # Write a tiny batch index next to outdir root (helpful for later steps)
        ix_path = Path(args.outdir) / "_barcode_batch_index.json"
        with open(ix_path, "w") as f:
            json.dump(batch_ix, f, indent=2)
        print(json.dumps({"processed": len([r for r in batch_ix if not r.get("skipped")]),
                          "skipped": len([r for r in batch_ix if r.get("skipped")]),
                          "index": str(ix_path)}, indent=2))

    # -------- Mode selection --------
    if args.batch_molecules_glob or args.batch_ion_ladders_glob:
        _run_batch()
        return

    # Single‑file mode (original behavior)
    if not args.label:
        print("[ERROR] --label is required in single‑file mode.", file=sys.stderr)
        sys.exit(2)
    outdir = Path(args.outdir) / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    if args.jdx:
        summary = process_jdx(args.label, Path(args.jdx), outdir,
                              grid=args.grid, beta=args.beta,
                              kmax=args.k_max, kmax_zoom=args.k_max_zoom,
                              min_prom=args.min_prom, min_dist_px=args.min_dist_px,
                              tick_step=args.tick_step)
    elif args.csv:
        summary = process_csv(args.label, Path(args.csv), outdir,
                              grid=args.grid, beta=args.beta,
                              kmax=args.k_max, kmax_zoom=args.k_max_zoom,
                              freq_col=args.freq_col, log10_col=args.log10_col,
                              wn_col=args.wn_col, wl_nm_col=args.wl_nm_col,
                              weight_col=args.weight_col, min_weight_quantile=args.min_weight_quantile)

    else:
        print("[ERROR] Provide either --jdx or --csv, or use batch arguments.", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
