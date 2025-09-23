#!/usr/bin/env python3
"""
nist_levels_parser_v13.py — Upgraded NIST *levels* → tidy levels

────────────────────────────────────────────────────────────────
RGP pipeline (Kelly Heaton & The Coherence Research Collective)

Purpose
-------
Produce per-ion *tidy levels* that are physically truthful, reproducible,
and audit-friendly for resonance pairing and γ-sweep analysis.

Key properties (contract):
- One canonical energy per row `Level_eV` with provenance.
- Strictly energy-sorted output; `Level_ID` assigned *after* sorting.
- Dense/discontinuous regions flagged (`dense_zone`, `energy_band_id`).
- Honest labeling: inferred fields (like n) stamp `n_inferred`/`n_source`.
- Series scaffolding: LS-term series and outer-electron series (best-effort).
- Adjacency edges emitted for three views: energy, series_ls, series_outer.
- JSON sidecar with thresholds, header row, file hash, parser version, etc.
- Markdown QA report + tiny spacing histograms (saved under results/reports).

CLI
---
python -m scripts.preprocess.nist_levels_parser_v13 [--ion H_I|Fe_II|...] [--rebuild]
    If --ion omitted, processes all RAW_LEVELS_DIR/*_levels_raw.csv

python -m scripts.preprocess.nist_levels_parser_v13 --ion O_III

Outputs
-------
- data/tidy/levels/{Ion}_levels.csv
- data/tidy/levels/{Ion}_levels_adjacency.parquet
- data/results/reports/levels/{Ion}.md
- data/results/plots/levels/{Ion}_spacing_hist.png
- data/tidy/levels/{Ion}_levels.meta.json

Notes
-----
- We treat NIST **levels** energy as 'Level (cm-1)' by default; convert to eV.
- We do *not* scale for medium here (medium="vacuum" in metadata).

"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project utilities (paths, constants, provenance)
from scripts.utils.path_config import (
    RAW_LEVELS_DIR, TIDY_LEVELS_DIR, RESULTS_DIR, PLOTS_DIR, ensure_dir
)
from scripts.utils.constants import CM2EV, alpha2_target, principal_qn
from scripts.utils.provenance import (
    write_csv_with_provenance,
    write_markdown_with_provenance,
    stamp_plot_provenance,
)

PARSER_VERSION = "v1.3"

# Pattern: e.g., "Cu II (3d10 1S<0>)" — appears in Configuration on ionization limit rows
_species_cfg_pattern = re.compile(r"^[A-Z][a-z]?\s+[IVX]+\s*\(")

# Periodic table lookup: atomic symbol → Z (subset sufficient for project)
Z_LOOKUP = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63, "Gd": 64,
    "Dy": 66, "Er": 68, "Yb": 70, "Hg": 80, "Pb": 82, "Bi": 83, "Rn": 86
}

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def ion_to_symbol_Z(ion: str) -> Tuple[str, Optional[int]]:
    """
    'Fe_II' -> ('Fe', 26)
    """
    el = ion.split("_")[0]
    return el, Z_LOOKUP.get(el)

def hash_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def strip_excel_wrappers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove Excel-style wrappers =\"...\" and stray spaces/commas from string cells
    that might interfere with numeric parsing.
    """
    def _clean(val):
        if isinstance(val, str):
            s = val.strip()
            if s.startswith('="') and s.endswith('"'):
                s = s[2:-1]
            # remove thousands separators if any
            s = s.replace(",", "")
            return s
        return val
    return df.applymap(_clean)

def detect_header_row(raw: pd.DataFrame) -> int:
    """
    Find header row containing both 'Configuration' and 'Term' (case-insensitive).
    If multiple rows match, choose the latest prior to numeric body.
    """
    hits = []
    for i in range(min(len(raw), 200)):  # search first 200 rows
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains("configuration").any() and row.str.contains("term").any():
            hits.append(i)
    if not hits:
        raise ValueError("Header row with 'Configuration' and 'Term' not found")
    return hits[-1]

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize header names to simpler tokens.
    """
    out = df.copy()
    out.columns = (
        out.columns
           .str.strip()
           .str.replace(r"\s+", " ", regex=True)
           .str.replace("(", "", regex=False)
           .str.replace(")", "", regex=False)
    )
    return out

_energy_cm_patterns = re.compile(r"(Level|Wavenumber|Energy|Ritz)\s*\[?cm-?1\]?", re.IGNORECASE)
_unc_cm_patterns    = re.compile(r"(Unc|Uncertainty)\s*\[?cm-?1\]?", re.IGNORECASE)

def choose_energy_cm_column(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Choose primary energy (cm^-1) column. Prefer 'Level (cm-1)'.
    Return (energy_col, unc_col or None).
    """
    cols = {c: c for c in df.columns}
    # Exact preference list
    pref = [c for c in cols if c.lower().startswith("level") and "cm-1" in c.lower()]
    if not pref:
        pref = [c for c in cols if _energy_cm_patterns.search(c)]
    if not pref:
        raise ValueError("No energy cm^-1 column found")
    energy_col = pref[0]

    unc_candidates = [c for c in cols if _unc_cm_patterns.search(c)]
    unc_col = unc_candidates[0] if unc_candidates else None
    return energy_col, unc_col

def is_centroid_like_row(row: pd.Series) -> bool:
    """
    Heuristic: placeholder/centroid-like if Configuration is numeric-like only,
    or Term/J missing while energy present.
    """
    cfg = str(row.get("Configuration", "")).strip()
    term = str(row.get("Term", "")).strip()
    jval = str(row.get("J", "")).strip()
    # numeric-only config like "2" or "4" etc.
    numeric_cfg = bool(re.fullmatch(r"\d+", cfg))
    missing_labels = (term == "" and jval == "" and cfg == "")
    return numeric_cfg or missing_labels

def ls_series_id(term: str, parity: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Build LS-term+parity series id, e.g., '3P*' or '1D'.
    Confidence: 'high' when term present; 'low' otherwise.
    """
    t = (term or "").strip()
    if not t:
        return None, "low"
    # Parity star is often encoded directly in Term as a '*'
    has_star = "*" in t
    if not has_star and parity:
        # Append star for odd parity if not already included
        if parity.strip().lower().startswith("odd"):
            t = t + "*"
    return t, "high"

_outer_electron_re = re.compile(r"(\d+\s*[spdfghijk])", re.IGNORECASE)

def outer_electron_series(cfg: str) -> Tuple[Optional[str], str]:
    """
    Extract the outermost electron (e.g., '4p', '5s') from Configuration.
    Confidence high if a clear trailing subshell is found.
    """
    if not isinstance(cfg, str):
        return None, "low"
    tokens = _outer_electron_re.findall(cfg.replace(".", " ").replace("^", ""))
    if not tokens:
        return None, "low"
    # take the last token as outer electron
    token = tokens[-1].replace(" ", "")
    return token.lower(), "high"

def build_adjacency_edges(levels: pd.DataFrame, ion: str) -> pd.DataFrame:
    """
    Build adjacency edge table for three views: energy, series_ls, series_outer.
    Returns a Parquet-friendly DataFrame.
    """
    rows = []

    def add_edge(mode, lo, hi, extra: dict):
        rows.append({
            "Ion": ion,
            "adjacency_mode": mode,
            "Level_ID_lo": int(lo.Level_ID),
            "Level_ID_hi": int(hi.Level_ID),
            "Level_eV_lo": float(lo.Level_eV),
            "Level_eV_hi": float(hi.Level_eV),
            "dE_eV": float(hi.Level_eV - lo.Level_eV),
            "dense_zone_lo": bool(lo.dense_zone),
            "dense_zone_hi": bool(hi.dense_zone),
            "energy_band_id_lo": int(lo.energy_band_id),
            "energy_band_id_hi": int(hi.energy_band_id),
            **extra
        })

    # Energy-adjacent (global spine)
    for i in range(len(levels) - 1):
        lo = levels.iloc[i]
        hi = levels.iloc[i + 1]
        if lo.is_limit or hi.is_limit or lo.incomplete or hi.incomplete:
            continue
        add_edge("energy", lo, hi, {"series_id": None, "series_confidence": None})

    # LS-series adjacent
    for sid, grp in levels.dropna(subset=["series_id_ls"]).groupby("series_id_ls", sort=False):
        g = grp.sort_values("Level_eV")
        conf = grp["series_confidence_ls"].mode(dropna=True)
        conf = None if conf.empty else conf.iloc[0]
        for i in range(len(g) - 1):
            lo = g.iloc[i]; hi = g.iloc[i + 1]
            if lo.is_limit or hi.is_limit or lo.incomplete or hi.incomplete:
                continue
            add_edge("series_ls", lo, hi, {"series_id": sid, "series_confidence": conf})

    # Outer-electron series adjacent (prefer n order when safe)
    for sid, grp in levels.dropna(subset=["series_id_outer"]).groupby("series_id_outer", sort=False):
        # If both rows have n, order by n; else by energy
        have_n = grp["n"].notna().sum() >= 2
        if have_n:
            g = grp.sort_values(["n", "Level_eV"])
        else:
            g = grp.sort_values("Level_eV")
        conf = grp["series_confidence_outer"].mode(dropna=True)
        conf = None if conf.empty else conf.iloc[0]
        for i in range(len(g) - 1):
            lo = g.iloc[i]; hi = g.iloc[i + 1]
            if lo.is_limit or hi.is_limit or lo.incomplete or hi.incomplete:
                continue
            add_edge("series_outer", lo, hi, {"series_id": sid, "series_confidence": conf})

    return pd.DataFrame(rows)

def spacing_histogram(levels: pd.DataFrame, out_png: Path):
    # Build and save a simple histogram plot of dE_min_eV (nonzero only)
    vals = levels["dE_min_eV"].replace(0, np.nan).dropna()
    if len(vals) == 0:
        return
    plt.figure()
    plt.hist(vals.values, bins=60)
    plt.xlabel("Nearest-neighbor min spacing dE_min (eV)")
    plt.ylabel("Count")
    plt.title("Nearest-neighbor spacing histogram")
    stamp_plot_provenance()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def detect_limit_row(row: pd.Series) -> bool:
    """
    Heuristic: flag as 'limit' if:
    - Term or Configuration equals 'limit'
    - Term or Configuration mentions 'ioniz' or ends with 'limit'
    - Configuration matches species-label pattern, e.g., 'Fe III (3d6...)'
    """
    for col in ("Term", "Configuration"):
        s = str(row.get(col, "")).strip().lower()
        if s == "limit" or "ioniz" in s or s.endswith("limit"):
            return True
    cfg = str(row.get("Configuration", "")).strip()
    if _species_cfg_pattern.match(cfg):
        return True
    return False

def parse_one_raw_levels(raw_path: Path, gamma_bin: float = 0.02, mu: float = 1.0) -> Tuple[pd.DataFrame, dict, str]:
    """
    Load, clean, and annotate a single raw NIST levels file.
    Returns (levels_df, metadata_dict, ion).
    """
    ion = raw_path.stem.replace("_levels_raw", "")
    el, Z = ion_to_symbol_Z(ion)

    # Load raw with minimal assumptions; header unknown
    raw0 = pd.read_csv(raw_path, header=None, dtype=str, engine="python", comment="#")
    raw0 = strip_excel_wrappers(raw0)

    # Header detection
    hdr_idx = detect_header_row(raw0)
    header = list(raw0.loc[hdr_idx].astype(str))
    body = raw0.drop(index=range(hdr_idx + 1)).reset_index(drop=True)
    body.columns = header

    # Normalize column names for convenience, keep original for provenance
    body = canonicalize_columns(body)

    # Choose energy and uncertainty columns
    energy_col, unc_col = choose_energy_cm_column(body)

    # Parse numeric energy / uncertainty
    energy_cm = pd.to_numeric(body[energy_col], errors="coerce")
    unc_cm = pd.to_numeric(body[unc_col], errors="coerce") if unc_col and unc_col in body.columns else np.nan

    # Build working table
    levels = pd.DataFrame({
        "Ion": ion,
        "Configuration": body.get("Configuration"),
        "Term": body.get("Term"),
        "J": body.get("J"),
        "Parity": body.get("Parity"),
        "g": pd.to_numeric(body.get("g"), errors="coerce"),
        "Level_cm1": energy_cm,
    })
    levels["Level_eV"] = levels["Level_cm1"] * CM2EV
    levels["energy_source"] = "Level_cm1"
    levels["energy_sigma_eV"] = pd.to_numeric(unc_cm, errors="coerce") * CM2EV if not isinstance(unc_cm, float) else np.nan
    levels["orig_row"] = np.arange(len(levels))

    # Tag special rows
    levels["is_centroid_like"] = levels.apply(is_centroid_like_row, axis=1)
    levels["is_limit"] = levels.apply(detect_limit_row, axis=1)
    # QA column: detect config pattern like 'Fe III (...)'
    levels["config_contains_species_label"] = levels["Configuration"].astype(str).str.match(_species_cfg_pattern)
    levels["incomplete"] = levels["is_centroid_like"]  # conservative

    # n parsing (honest): prefer Term, else Configuration; mark inferred
    n_from_term = levels["Term"].apply(principal_qn)
    n_from_cfg  = levels["Configuration"].apply(principal_qn)
    levels["n"] = n_from_term.where(~n_from_term.isna(), n_from_cfg)
    levels["n_inferred"] = ~levels["n"].isna()
    levels["n_source"] = np.where(~n_from_term.isna(), "term",
                           np.where(~n_from_cfg.isna(), "config", "none"))
    levels.loc[levels["n_source"] == "none", "n_inferred"] = False

    # Sort by energy (stable) and assign Level_ID
    levels = levels.dropna(subset=["Level_eV"]).copy()
    levels = levels.sort_values("Level_eV", kind="mergesort").reset_index(drop=True)
    levels["Level_ID"] = np.arange(len(levels))

    # Neighbor spacings
    levels["dE_prev_eV"] = levels["Level_eV"].diff()
    levels["dE_next_eV"] = levels["Level_eV"].shift(-1) - levels["Level_eV"]
    levels["dE_min_eV"]  = np.where(
        levels["dE_prev_eV"].abs().isna(), levels["dE_next_eV"].abs(),
        np.where(levels["dE_next_eV"].abs().isna(), levels["dE_prev_eV"].abs(),
                 np.minimum(levels["dE_prev_eV"].abs(), levels["dE_next_eV"].abs()))
    )

    # Duplicate clusters (within epsilon or uncertainty overlap)
    eps = 1e-9
    same_as_prev = levels["dE_prev_eV"].abs() < eps
    overlap_prev = (levels["energy_sigma_eV"].fillna(0) + levels["energy_sigma_eV"].shift(1).fillna(0)) >= levels["dE_prev_eV"].abs()
    levels["is_duplicate_energy"] = same_as_prev | overlap_prev

    # Dense threshold (γ-aware): combine local stats, γ bin width, uncertainty
    spacings = levels["dE_min_eV"].replace(0, np.nan).dropna()
    Q1 = float(spacings.quantile(0.25)) if len(spacings) else math.nan
    median_sigma = float(levels["energy_sigma_eV"].median(skipna=True)) if "energy_sigma_eV" in levels else math.nan
    # γ bin width in energy (approximate) using α² target spacing for this Z, μ
    dE_gamma_bin = None
    if Z is not None:
        dE_gamma_bin = gamma_bin * alpha2_target(Z, mu)
    # threshold: min(0.25*Q1, max(1e-6, 0.5*dE_gamma_bin, 5*median_sigma))
    floors = [1e-6]
    if dE_gamma_bin is not None:
        floors.append(0.5 * dE_gamma_bin)
    if not math.isnan(median_sigma):
        floors.append(5.0 * median_sigma)
    dense_floor = max(floors) if floors else 1e-6
    dense_thresh = min(0.25 * Q1 if not math.isnan(Q1) else float("inf"), dense_floor)
    levels["dense_zone"] = levels["dE_min_eV"] < dense_thresh

    # Discontinuity bands (large gaps: median + 5*IQR of positive dE_prev)
    pos_prev = levels["dE_prev_eV"].dropna()
    med = float(pos_prev.median()) if len(pos_prev) else math.nan
    iqr = float(pos_prev.quantile(0.75) - pos_prev.quantile(0.25)) if len(pos_prev) else math.nan
    gap_thresh = (med + 5*iqr) if (not math.isnan(med) and not math.isnan(iqr)) else 0.1
    band_break = levels["dE_prev_eV"] > gap_thresh
    levels["energy_band_id"] = band_break.cumsum().fillna(0).astype(int)

    # Series labels
    sid_ls, conf_ls = [], []
    sid_outer, conf_outer = [], []
    for _, r in levels.iterrows():
        sid, conf = ls_series_id(str(r.get("Term", "")), r.get("Parity"))
        sid_ls.append(sid); conf_ls.append(conf)
        s2, c2 = outer_electron_series(str(r.get("Configuration", "")))
        sid_outer.append(s2); conf_outer.append(c2)
    levels["series_id_ls"] = sid_ls
    levels["series_confidence_ls"] = conf_ls
    levels["series_id_outer"] = sid_outer
    levels["series_confidence_outer"] = conf_outer

    # Metadata
    meta = {
        "parser_version": PARSER_VERSION,
        "ion": ion,
        "element_symbol": el,
        "Z": Z,
        "gamma_bin": gamma_bin,
        "mu": mu,
        "energy_conv_eV_per_cm1": CM2EV,
        "energy_source": "Level_cm1",
        "dense_rule": "min(0.25*Q1, max(1e-6, 0.5*gamma_bin*alpha2_target(Z,μ), 5*median_sigma))",
        "dense_thresh_eV": dense_thresh,
        "gap_rule": "median(dE_prev) + 5*IQR",
        "gap_thresh_eV": gap_thresh,
        "medium": "vacuum",
        "header_row_index": int(hdr_idx),
        # Provenance: hash of the exact raw file used to produce this tidy
        "raw_file_hash_sha256": hash_file(raw_path),
        "raw_file_name": raw_path.name
    }

    return levels, meta, ion

def write_outputs(levels: pd.DataFrame, meta: dict, ion: str):
    # Paths
    tidy_csv = TIDY_LEVELS_DIR / f"{ion}_levels.csv"
    adjacency_parquet = TIDY_LEVELS_DIR / f"{ion}_levels_adjacency.parquet"
    report_md = RESULTS_DIR / "reports" / "levels" / f"{ion}.md"
    plot_png = PLOTS_DIR / "levels" / f"{ion}_spacing_hist.png"
    sidecar_json = TIDY_LEVELS_DIR / f"{ion}_levels.meta.json"

    ensure_dir(tidy_csv.parent)
    ensure_dir(report_md.parent)
    ensure_dir(plot_png.parent)

    # Adjacency edges
    edges = build_adjacency_edges(levels, ion)
    edges.to_parquet(adjacency_parquet, index=False)

    # Tiny spacing histogram
    spacing_histogram(levels, plot_png)

    # Write tidy CSV with provenance header
    write_csv_with_provenance(levels, tidy_csv)

    # Write JSON sidecar
    with open(sidecar_json, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # Markdown report
    dense_pct = 100.0 * float(levels["dense_zone"].mean()) if len(levels) else 0.0
    bands = levels["energy_band_id"].nunique()
    n_dupes = int(levels["is_duplicate_energy"].sum())
    n_centroid = int(levels["is_centroid_like"].sum())
    n_limit = int(levels["is_limit"].sum())
    n_cfg_species = int(levels["config_contains_species_label"].sum())

    md = [
        f"# Levels QA — {ion}",
        "",
        f"- Parser version: **{meta['parser_version']}**",
        f"- Z: **{meta['Z']}**, gamma_bin: **{meta['gamma_bin']}**, mu: **{meta['mu']}**",
        f"- Rows: **{len(levels)}**; dense: **{dense_pct:.2f}%**; bands: **{bands}**; duplicates: **{n_dupes}**",
        f"- Centroid-like rows: **{n_centroid}**; limit rows: **{n_limit}**",
        "",
        "## Thresholds",
        f"- Dense rule: `{meta['dense_rule']}`",
        f"- Dense threshold (eV): **{meta['dense_thresh_eV']:.6e}**",
        f"- Gap rule: `{meta['gap_rule']}`",
        f"- Gap threshold (eV): **{meta['gap_thresh_eV']:.6e}**",
        "",
        "## Columns (excerpt)",
        "`Ion, Level_ID, orig_row, energy_band_id, Level_eV, energy_source, energy_sigma_eV, "
        "dE_prev_eV, dE_next_eV, dE_min_eV, dense_zone, is_duplicate_energy, Configuration, Term, "
        "J, Parity, g, n, n_inferred, n_source, is_centroid_like, incomplete, is_limit, "
        "series_id_ls, series_confidence_ls, series_id_outer, series_confidence_outer`",
        "",
        f"![Spacing histogram]({plot_png})",
        f"# Levels QA — {ion}",
        "",
        f"- Parser version: **{meta['parser_version']}**",
        f"- Z: **{meta['Z']}**, gamma_bin: **{meta['gamma_bin']}**, mu: **{meta['mu']}**",
        f"- Rows: **{len(levels)}**; dense: **{dense_pct:.2f}%**; bands: **{bands}**; duplicates: **{n_dupes}**",
        f"- Centroid-like rows: **{n_centroid}**; limit rows: **{n_limit}**",
        f"- Species-pattern in config: **{n_cfg_species}**",
    ]
    write_markdown_with_provenance(md, report_md)

def process_all(only_ion: Optional[str] = None, gamma_bin: float = 0.02, mu: float = 1.0, rebuild: bool = False):
    raw_dir = RAW_LEVELS_DIR
    files = sorted(raw_dir.glob("*_levels_raw.csv"))
    if only_ion:
        files = [p for p in files if p.stem.replace('_levels_raw','') == only_ion]
        if not files:
            print(f"[skip] No raw file for ion={only_ion} at {raw_dir}")
            return

    for raw_fp in files:
        ion = raw_fp.stem.replace("_levels_raw", "")
        tidy_csv = TIDY_LEVELS_DIR / f"{ion}_levels.csv"
        if tidy_csv.exists() and not rebuild:
            print(f"[skip] {ion}: tidy exists ({tidy_csv.name}); use --rebuild to overwrite")
            continue
        try:
            levels, meta, ion_actual = parse_one_raw_levels(raw_fp, gamma_bin=gamma_bin, mu=mu)
            write_outputs(levels, meta, ion_actual)
            print(f"[✓] {ion:>6} → tidy levels ({len(levels)} rows)")
        except Exception as e:
            print(f"[FAIL] {ion}: {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ion", type=str, default=None, help="e.g., H_I, Fe_II; default=all")
    ap.add_argument("--gamma", type=float, default=0.02, help="gamma bin width for dense threshold (default 0.02)")
    ap.add_argument("--mu", type=float, default=1.0, help="reduced mass ratio (default 1.0)")
    ap.add_argument("--rebuild", action="store_true", help="overwrite tidy outputs even if present")
    args = ap.parse_args()

    process_all(only_ion=args.ion, gamma_bin=args.gamma, mu=args.mu, rebuild=args.rebuild)