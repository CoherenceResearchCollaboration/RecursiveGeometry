# scripts/analysis_pipeline/rgp_mass_estimator.py
# -*- coding: utf-8 -*-
"""
RGP Mass Estimator (Photons-Only, Non-Circular)
===============================================

Purpose
-------
Estimate isotope mass *ratios* from photon spectra alone by fitting the
"Thread frame" per tower:

    log10(nu) = beta * gamma + chi_t,

with:
  - nu       : photon frequency (from measured vacuum wavelength, nu = c / lambda)
  - gamma    : dimensionless energy coordinate (from LEVELS ΔE only; NOT from photons)
  - beta     : universal slope ~ log10(alpha) (estimated globally, then locked)
  - chi_t    : per-tower intercept (ion/tower scale)

For two isotopes A and B of the *same species* (e.g., H I vs D I),
the theory predicts:

    Delta_chi = chi_B - chi_A  ≈  log10( mu_B / mu_A )

where mu is the electron–nucleus *reduced mass*.

This script:
  1) Builds photons-only ladders for each ion from tidy LINES + LEVELS CSVs,
  2) Fits per-tower planes, extracts global beta, then computes tower intercepts with beta locked,
  3) Aggregates per-isotope intercepts (robust median) + bootstrap CIs,
  4) Computes Delta_chi_observed and compares to Delta_chi_predicted (from atomic masses),
  5) Runs a permutation null (shuffle gamma within towers) to show Delta_chi → ~0.

Inputs (expected)
-----------------
data/tidy/lines/<ION>_lines.csv    # vacuum wavelengths; columns are flexible
data/tidy/levels/<ION>_levels.csv  # upper/lower level energies; columns are flexible

Robust parsing is used; provenance headers at top of CSVs are handled.

Outputs
-------
<out-dir>/per_tower_fits__<ION>.csv
<out-dir>/per_tower_intercepts_locked__<ION>.csv
<out-dir>/mass_summary__H_I__D_I.json (if two ions provided)
<out-dir>/perm_null__H_I__D_I.csv
Console summary with beta, k=beta/log10(alpha), per-isotope chi, bootstrap CIs, observed vs predicted Delta_chi, and null p-value.

Run
---

We ran our hsweep first to obtain the value sigma 0.0072973848 and then:

# H I with your levels-derived σ⋆
python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions H_I \
  --out-dir  data/results/mass_estimator_HI_mp6_sigma_star \
  --gamma-bin 0.02 --u-gamma 13.6 \
  --min-points 6 \
  --bootstrap 5000 --perm-null 0 \
  --beta-fallback alpha \
  --sigma 0.0072973848 \
  --ledger data/results/mass_estimator_HI_mp6_sigma_star/ledger.json \
  --ledger-md data/results/mass_estimator_HI_mp6_sigma_star/ledger.md

(Optional control with CODATA α — omit --sigma):

python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions H_I \
  --out-dir  data/results/mass_estimator_HI_mp6_raw \
  --gamma-bin 0.02 --u-gamma 13.6 \
  --min-points 6 \
  --bootstrap 5000 --perm-null 0 \
  --beta-fallback alpha \
  --ledger data/results/mass_estimator_HI_mp6_raw/ledger.json \
  --ledger-md data/results/mass_estimator_HI_mp6_raw/ledger.md

B) Hydrogenic set — H I, He II, Li III, O VIII. There aren't as many points to try as with H I.

# With σ⋆
python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions He_II Li_III O_VIII \
  --out-dir  data/results/mass_estimator_hydrogenic_sigma_star_mp2 \
  --gamma-bin 0.02 --u-gamma 13.6 \
  --min-points 2 \
  --bootstrap 5000 --perm-null 0 \
  --beta-fallback alpha \
  --sigma 0.0072973848 \
  --normalize-Z hydrogenic \
  --gamma-mode levels \
  --ledger data/results/mass_estimator_hydrogenic_sigma_star_mp2/ledger.json \
  --ledger-md data/results/mass_estimator_hydrogenic_sigma_star_mp2/ledger.md


(Optional control with CODATA α):

python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions H_I He_II Li_III O_VIII \
  --out-dir  data/results/mass_estimator_hydrogenic_raw \
  --gamma-bin 0.02 --u-gamma 13.6 \
  --min-points 3 \
  --bootstrap 5000 --perm-null 0 \
  --beta-fallback alpha \
  --normalize-Z hydrogenic \
  --gamma-mode levels \
  --ledger data/results/mass_estimator_hydrogenic_raw/ledger.json \
  --ledger-md data/results/mass_estimator_hydrogenic_raw/ledger.md

For the H I / D I comparison used in our paper:

A) Mass estimator with your levels-derived σ⋆ (primary). Note that D I has sparse data points in its levels file, so the script will not find results with min points greater than 3.

python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions H_I D_I \
  --out-dir    data/results/mass_estimator_sigma_star \
  --gamma-bin  0.02 --u-gamma 13.6 \
  --min-points 3 \
  --bootstrap  5000 --perm-null 400 --null both \
  --beta-fallback alpha \
  --sigma 0.0072973848 \
  --ledger data/results/mass_estimator_sigma_star/ledger.json \
  --ledger-md data/results/mass_estimator_sigma_star/ledger.md

Then run:

# α̂ from H I; Δχ(H→D) with low-N D I
python -m scripts.analysis_pipeline.hsweep.hs_alpha_from_intercepts \
  --H data/results/mass_estimator_HI_mp6_sigma_star/per_tower_intercepts_locked__H_I.csv \
  --D data/results/mass_estimator_sigma_star/per_tower_intercepts_locked__D_I.csv \
  --out data/results/alpha_from_intercepts_sigma_star \
  --n-bootstrap 5000 \
  --sigma-star 0.0072973848 \
  --sigma-star-ci "±5e-6"


B) Mass estimator baseline with CODATA α (control)

python -m scripts.analysis_pipeline.rgp_mass_estimator \
  --lines-dir  data/tidy/lines \
  --levels-dir data/tidy/levels \
  --ions H_I D_I \
  --out-dir  data/results/mass_estimator_RAW \
  --gamma-bin 0.02 --u-gamma 13.6 \
  --min-points 6 \
  --bootstrap 5000 --perm-null 400 --null both \
  --beta-fallback alpha \
  --ledger data/results/mass_estimator_RAW/ledger.json \
  --ledger-md data/results/mass_estimator_RAW/ledger.md

"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import hashlib, json, os, sys
from datetime import datetime, timezone

# ----------------------------- Constants ------------------------------------

C_M_S = 299_792_458.0                  # speed of light [m/s]
HC_eVnm = 1239.841984                  # h*c in [eV*nm]
CMINV_TO_eV = HC_eVnm * 1e-7           # 1 cm^-1 = (h c)/1cm = 1.239841984e-4 eV
ELECTRON_MASS_U = 5.48579909065e-4     # electron mass in atomic mass units (u) (CODATA-2018/2022)

# ── Reference CODATA α (do not use in σ/γ math; for labels/comparisons only) ─────────
ALPHA_CODATA_REF = 7.2973525693e-3     # reference α (CODATA-like)
LOG10_ALPHA_REF  = math.log10(ALPHA_CODATA_REF)

# ── Runtime spiral/phase scale σ (used everywhere the math needs α/σ) ──────
# NOTE: Final resolution now happens in main(), with precedence:
#       --sigma (CLI) > SIGMA (env) > ALPHA_CODATA_REF.
#       We set a bootstrap default here but allow main() to override & log.
ALPHA = ALPHA_CODATA_REF
LOG10_ALPHA = math.log10(ALPHA)


# ── Minimal element Z lookup (extend as needed) ─────────────────────────────
Z_LOOKUP = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20
}

# ── Helpers to map ion label → element symbol and Z ─────────────────────────
def ion_Z(ion_label: str) -> int:
    sym = isotope_family_symbol(element_symbol_from_ion_label(ion_label))
    return Z_LOOKUP.get(sym, 0)
# ── Nuclear masses (u) for μ/m_e (preferred for isotope physics) ───────────
# Use NUCLEAR masses (no bound electrons), not atomic, when computing μ.
NUCLEAR_MASS_U = {
    "H": 1.007276466621,   # proton (1H nucleus)
    "D": 2.013553212745,   # deuteron (2H nucleus)
    "He": 4.001506179127,  # 4He nucleus (alpha)
    # Extend as needed
}

# Optional atomic masses (for context / fallbacks only)
ATOMIC_MASS_U = {
    "H": 1.00782503223, "D": 2.01410177812, "He": 4.00260325413,
    "Li": 7.0160034366, "C": 12.0,          "O": 15.99491461957
}

def mu_over_me_from_symbol(sym: str) -> float:
    """
    Return μ/m_e for electron–nucleus reduced mass using **nuclear** mass for the nucleus:
        μ/m_e = (M_nuc/m_e) / (1 + M_nuc/m_e)
    Falls back to atomic mass if nuclear mass is unavailable.
    """
    ME_U = ELECTRON_MASS_U
    if sym in NUCLEAR_MASS_U:
        M_over_me = NUCLEAR_MASS_U[sym] / ME_U
    elif sym in ATOMIC_MASS_U:
        # Fallback: atomic mass → small bias at ~10^-4 level
        M_over_me = ATOMIC_MASS_U[sym] / ME_U
    else:
        # Heavy nucleus approximation (μ≈m_e): return 1.0 (neutral)
        return 1.0
    return M_over_me / (1.0 + M_over_me)

# ------------------------- Utility: Robust CSV Load --------------------------

def load_csv_with_header_detection(path: Path) -> pd.DataFrame:
    """
    Load a CSV that may have provenance lines before the header row.
    We scan first 40 lines; pick the first line that looks like a CSV header
    (i.e., contains multiple commas and at least one alphabetic column name).
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = []
        for _ in range(40):
            try:
                lines.append(next(f))
            except StopIteration:
                break
    header_idx = 0
    for i, line in enumerate(lines):
        if "," in line and len([tok for tok in line.strip().split(",") if tok.strip()]) > 1:
            header_idx = i
            break
    return pd.read_csv(path, skiprows=header_idx)


def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # allow case-insensitive (common in mixed exports)
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def percent_error(obs: float, ref: float) -> float | None:
    try:
        if ref == 0:
            return None
        return 100.0 * (obs - ref) / abs(ref)
    except Exception:
        return None

# ----------------------------- Parsing Helpers -------------------------------

def element_symbol_from_ion_label(ion_label: str) -> str:
    # "H_I" -> "H", "He_II" -> "He", "Li_III" -> "Li"
    return ion_label.split("_")[0]

# Isotope family mapping (so D, T are recognized as H-family, etc.)
ISOTOPE_FAMILY = {
    "D": "H",  # Deuterium -> Hydrogen family
    "T": "H",  # Tritium   -> Hydrogen family
    # extend as needed, e.g., "13C": "C" if you ever encode 13C_I style labels
}

def isotope_family_symbol(sym: str) -> str:
    """Return the family symbol that groups isotopes together (e.g., D -> H)."""
    return ISOTOPE_FAMILY.get(sym, sym)

# Place near Z_LOOKUP and helpers
ROMAN = {"I":1,"V":5,"X":10}
def roman_to_int(s: str) -> int:
    s = s.strip().upper()
    # minimal robust parse for I..XX
    vals = [ROMAN.get(ch,0) for ch in s if ch in ROMAN]
    if not vals: return 0
    total = 0
    for i,v in enumerate(vals):
        total += (-v if i+1<len(vals) and vals[i+1]>v else v)
    return total

def ion_stage(ion_label: str) -> int:
    # "He_II" -> 2 ; "Li_III" -> 3
    tail = ion_label.split("_",1)[1]
    return roman_to_int(tail)

def electron_count(ion_label: str) -> int:
    # Map isotopes (e.g., D -> H) before Z lookup so hydrogenic ions stay hydrogenic
    base = element_symbol_from_ion_label(ion_label)
    elem = isotope_family_symbol(base)
    Z = Z_LOOKUP.get(elem)
    if not Z: return -1
    stage = ion_stage(ion_label)  # I->1 = neutral
    charge = stage - 1
    return Z - charge

def choose_wavelength_nm(lines: pd.DataFrame) -> pd.Series:
    """
    Choose a vacuum wavelength column and return nm.
    Preference order: explicit 'vac' columns (nm or Å), then any 'wavelength'.
    """
    cand = [
        "vacuum_wavelength_nm", "lambda_vac_nm", "wavelength_vac_nm",
        "vacuum_wavelength_angstrom", "lambda_vac_A", "wavelength_vac_A",
        "wavelength_nm", "lambda_nm", "Wavelength (nm)", "Vacuum wavelength (nm)",
        "wavelength", "lambda", "Wavelength"
    ]
    name = first_present(lines, cand)
    if name is None:
        # Try any column containing 'wav' or 'lambda'
        for col in lines.columns:
            if re.search(r"lam|wav", col, re.IGNORECASE):
                name = col
                break
    if name is None:
        raise ValueError("No wavelength column found in lines CSV.")
    wav = pd.to_numeric(lines[name], errors="coerce")
    # Convert Å to nm if column name contains 'A' or looks like Angstrom
    if re.search(r"ang|Å|\(A\)", name, re.IGNORECASE):
        wav_nm = wav * 0.1
    else:
        wav_nm = wav
    return wav_nm


def extract_energy_eV(df: pd.DataFrame, upper: bool) -> Optional[pd.Series]:
    """
    Extract an upper or lower level energy column in eV from LINES table.
    Try eV first, then cm^-1, return None if not found.
    """
    base = "upper" if upper else "lower"
    # Common patterns:
    cand_eV = [
        f"E_{base}_eV", f"{base}_E_eV", f"{base}_level_eV", f"{base}_term_eV",
        f"{base}E (eV)", f"{base}_energy_eV"
    ]
    cand_cm = [
        f"E_{base}_cm-1", f"{base}_E_cm-1", f"{base}_level_cm-1",
        f"{base}E (cm-1)", f"{base}_energy_cm-1", f"{base}_level (cm-1)",
        f"{'Ei' if not upper else 'Ek'} (cm-1)"  # NIST often uses Ei, Ek
    ]
    name = first_present(df, cand_eV)
    if name:
        return pd.to_numeric(df[name], errors="coerce")
    name = first_present(df, cand_cm)
    if name:
        cm = pd.to_numeric(df[name], errors="coerce")
        return cm * CMINV_TO_eV
    return None


def parse_n_quantum(lines: pd.DataFrame, upper: bool) -> Optional[pd.Series]:
    """
    Extract principal quantum number n for upper/lower levels, if present. If not,
    attempt a regex on term/config labels like '2s', '3p', etc.
    """
    base = "upper" if upper else "lower"
    cand_n = [
        f"n_{base}", f"{base}_n", f"{base}_level_n", f"{base}_principal_n",
        f"n{base.capitalize()}", f"n{base[0].upper()}"
    ]
    name = first_present(lines, cand_n)
    if name:
        return pd.to_numeric(lines[name], errors="coerce")

    # Try regex on level label/config columns
    label_cands = [
        f"{base}_label", f"{base}_level_label", f"{base}_config",
        f"{base}_term", f"{base}_designation", f"{base}_level"
    ]
    lname = first_present(lines, label_cands)
    if lname:
        s = lines[lname].astype(str)
        # find a leading number, or digit before s/p/d/f
        n = s.str.extract(r"(^\d+)|(\d+)(?=\s*[spdfghijk])", expand=True)
        n = n[0].fillna(n[1])
        # keep dtype inference explicit to avoid future downcast warning
        try:
            n = n.infer_objects(copy=False)
        except Exception:
            pass
        return pd.to_numeric(n, errors="coerce")
    return None

def recover_n_from_levels(lines: pd.DataFrame, levels_df: Optional[pd.DataFrame], upper: bool) -> Optional[pd.Series]:
    """
    If principal quantum numbers are missing in the LINES table, try to recover them
    by joining to LEVELS on the explicit upper/lower level IDs and parsing the level label.
    We look for a leading integer or a digit before s/p/d/f in the level name.
    """
    if levels_df is None:
        return None
    base = "upper" if upper else "lower"
    up_id = first_present(lines, ["upper_id", "upper_level_id", "upper_index", "Upper level"])
    lo_id = first_present(lines, ["lower_id", "lower_level_id", "lower_index", "Lower level"])
    lvl_id_col = first_present(levels_df, ["level_id", "id", "index"])
    if upper and up_id and lvl_id_col and (lvl_id_col in levels_df.columns):
        # Join to bring the level 'name' we can parse n from
        # Try common label columns in LEVELS
        lbl_col = first_present(levels_df, ["level_label", "level", "label", "configuration", "term"])
        if lbl_col is None:
            return None
        df = lines[[up_id]].rename(columns={up_id: "_lid"}).copy()
        lcopy = levels_df[[lvl_id_col, lbl_col]].rename(columns={lvl_id_col: "_lid", lbl_col: "_lbl"})
        df = df.merge(lcopy, on="_lid", how="left")
        # regex: leading digits or digit before s/p/d/f
        n = df["_lbl"].astype(str).str.extract(r"(^\d+)|(\d+)(?=\s*[spdfghijk])", expand=True)
        n = n[0].fillna(n[1])
        return pd.to_numeric(n, errors="coerce")
    if (not upper) and lo_id and lvl_id_col and (lvl_id_col in levels_df.columns):
        lbl_col = first_present(levels_df, ["level_label", "level", "label", "configuration", "term"])
        if lbl_col is None:
            return None
        df = lines[[lo_id]].rename(columns={lo_id: "_lid"}).copy()
        lcopy = levels_df[[lvl_id_col, lbl_col]].rename(columns={lvl_id_col: "_lid", lbl_col: "_lbl"})
        df = df.merge(lcopy, on="_lid", how="left")
        n = df["_lbl"].astype(str).str.extract(r"(^\d+)|(\d+)(?=\s*[spdfghijk])", expand=True)
        n = n[0].fillna(n[1])
        return pd.to_numeric(n, errors="coerce")
    return None


def compute_gamma_from_levels(lines: pd.DataFrame, levels_df: Optional[pd.DataFrame], u_gamma_ev: float) -> pd.Series:
    """
    Compute gamma per line *from LEVELS data inside the LINES table* (preferred),
    falling back to LEVELS files if only level IDs are present; otherwise, as a last
    resort use upper/lower energies present in LINES.

    Returns gamma (dimensionless) = ΔE_levels / U_gamma.
    """
    # Preferred: upper/lower energies present in LINES
    e_up = extract_energy_eV(lines, upper=True)
    e_lo = extract_energy_eV(lines, upper=False)
    if e_up is not None and e_lo is not None:
        de = e_up - e_lo
        de_abs = np.abs(de)
        # Avoid log(0): drop non-positive rows upstream; here just guard:
        de_abs = np.where(de_abs > 0, de_abs, np.nan)
        gamma_alpha = np.log(de_abs / float(u_gamma_ev)) / np.log(ALPHA)
        return pd.Series(gamma_alpha, index=de.index if hasattr(de, "index") else None)

    # Fallback (join to LEVELS): attempt by level IDs if present
    up_id = first_present(lines, ["upper_id", "upper_level_id", "upper_index", "Upper level"])
    lo_id = first_present(lines, ["lower_id", "lower_level_id", "lower_index", "Lower level"])
    if levels_df is not None and up_id and lo_id:
        # try levels energy columns
        lev_e_cands_eV = ["energy_eV", "E_eV", "level_eV", "Level (eV)"]
        lev_e_cands_cm = ["energy_cm-1", "E_cm-1", "level_cm-1", "Level (cm-1)"]
        lev_e_col = first_present(levels_df, lev_e_cands_eV)
        conv = 1.0
        if lev_e_col is None:
            lev_e_col = first_present(levels_df, lev_e_cands_cm)
            conv = CMINV_TO_eV
        if lev_e_col is not None:
            # left-join upper/lower level energies by explicit level ID if present
            id_col = first_present(levels_df, ["level_id", "id", "index"])
            if id_col and id_col in levels_df.columns:
                lcopy = levels_df[[id_col, lev_e_col]].rename(columns={id_col: "_lvl_id", lev_e_col: "E_level"})
                df = lines.copy()
                df = df.rename(columns={up_id: "_up", lo_id: "_lo"})
                df = df.merge(lcopy.rename(columns={"_lvl_id": "_up", "E_level": "E_up"}), on="_up", how="left")
                df = df.merge(lcopy.rename(columns={"_lvl_id": "_lo", "E_level": "E_lo"}), on="_lo", how="left")
                de = (pd.to_numeric(df["E_up"], errors="coerce") - pd.to_numeric(df["E_lo"], errors="coerce")) * conv

                # α-exponent gamma (NON-CIRCULAR): γ = ln(|ΔE|/Uγ) / ln(α)
                de_abs = np.abs(de)
                de_abs = np.where(de_abs > 0, de_abs, np.nan)
                gamma_alpha = np.log(de_abs / float(u_gamma_ev)) / np.log(ALPHA)
                return pd.Series(gamma_alpha, index=df.index)

    # As a last structured fallback: sometimes only upper/lower wavenumbers are present in LINES
    # Already handled via extract_energy_eV above; if we reach here, nothing usable was found.
    raise ValueError("Unable to compute gamma from LEVELS: upper/lower level energies not found.")

def hydrogenic_deltaE_eV(n_i: np.ndarray, n_k: np.ndarray, Z: int = 1, mu_over_me: float = 1.0) -> np.ndarray:
    """
    Hydrogenic ΔE (eV) between n_i -> n_k using a single-electron model:
        ΔE = R_H * mu_over_me * Z^2 * (1/n_i^2 - 1/n_k^2)
    where R_H = 13.605693009 eV (Rydberg energy for hydrogen).
    """
    R_H_eV = 13.605693009
    ni = np.array(n_i, dtype=float)
    nk = np.array(n_k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return R_H_eV * mu_over_me * (Z**2) * (1.0/(ni**2) - 1.0/(nk**2))

def gamma_alpha_from_deltaE(de_eV: np.ndarray, U_gamma_eV: float, alpha: float | None = None) -> np.ndarray:
    """γ_ref = ln(|ΔE|/Uγ) / ln(α)"""
    de = np.abs(np.asarray(de_eV, dtype=float))
    de[de <= 0] = np.nan
    if alpha is None:
        alpha = ALPHA
    return np.log(de / float(U_gamma_eV)) / np.log(alpha)


# ----------------------------- Fitting Helpers -------------------------------

@dataclass
class TowerFit:
    ion: str
    n_i: Optional[int]
    n_k: Optional[int]
    N: int
    beta_lin: float
    chi_lin: float
    rss_lin: float


def ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Unweighted OLS: y = a + b x
    Returns (a, b, rss).
    """
    n = len(x)
    X = np.vstack([np.ones(n), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(np.sum((y - yhat) ** 2))
    a = float(beta[0])
    b = float(beta[1])
    return a, b, rss


def lock_intercept_given_beta(x: np.ndarray, y: np.ndarray, beta_locked: float) -> float:
    """
    With slope locked, the intercept that minimizes RSS is simply:
    a = mean(y - beta * x).
    """
    return float(np.mean(y - beta_locked * x))

# ----------------------------- Main Pipeline ---------------------------------

def build_ladder_for_ion(ion: str, lines_path: Path, levels_path: Optional[Path],
                         u_gamma_ev: float, gamma_bin: float) -> pd.DataFrame:
    """
    Build a photons-only ladder for one ion:
      - frequency_hz from vacuum wavelength
      - gamma from levels-only ΔE (independent of photons)
      - n_i, n_k from lines if present / parsed via regex
      - gamma_bin snapped to provided grid
    """
    lines = load_csv_with_header_detection(lines_path)
    levels_df = load_csv_with_header_detection(levels_path) if (levels_path and levels_path.exists()) else None

    wav_nm = choose_wavelength_nm(lines)
    freq_hz = C_M_S / (wav_nm * 1e-9)

    # gamma from levels energies present in LINES (preferred) or LEVELS fallback
    gamma = compute_gamma_from_levels(lines, levels_df, u_gamma_ev=u_gamma_ev)

    # principal quantum numbers if present or parse
    n_up = parse_n_quantum(lines, upper=True)
    n_lo = parse_n_quantum(lines, upper=False)

    # if missing/mostly NaN, recover from LEVELS by joining on level IDs
    if (n_up is None) or (pd.isna(n_up).mean() > 0.5):
        n_up_alt = recover_n_from_levels(lines, levels_df, upper=True)
        if n_up_alt is not None:
            n_up = n_up_alt if n_up is None else n_up.fillna(n_up_alt)
    if (n_lo is None) or (pd.isna(n_lo).mean() > 0.5):
        n_lo_alt = recover_n_from_levels(lines, levels_df, upper=False)
        if n_lo_alt is not None:
            n_lo = n_lo_alt if n_lo is None else n_lo.fillna(n_lo_alt)

    # Assign tower indices (n_i, n_k) as smaller/larger of (n_up, n_lo) if both available
    n_i = None
    n_k = None
    if n_up is not None and n_lo is not None:
        n_i = np.minimum(n_up, n_lo)
        n_k = np.maximum(n_up, n_lo)

    df = pd.DataFrame({
        "ion": ion,
        "wavelength_nm": wav_nm,
        "frequency_hz": freq_hz,
        "gamma": gamma,
        "n_i": n_i if n_i is not None else np.nan,
        "n_k": n_k if n_k is not None else np.nan
    })

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency_hz", "gamma"])
    # Snap gamma to a grid (bin center approximation for tower stacking)
    df["gamma_bin"] = (np.round(df["gamma"] / gamma_bin) * gamma_bin).astype(float)
    # Coerce n_i/n_k to Int64 if present
    for col in ("n_i", "n_k"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["log10_nu"] = np.log10(df["frequency_hz"])
    return df

def fit_towers(df: pd.DataFrame, ion: str, min_points: int = 3) -> Tuple[pd.DataFrame, List[TowerFit]]:
    """
    Fit per-tower linear law: log10(nu) = chi + beta * gamma_bin.
    Returns per-row tower keys and per-tower fits.
    """
    fits: List[TowerFit] = []
    groups = df.groupby(["n_i", "n_k", "gamma_bin"])  # gamma_bin isn't part of tower; regroup below
    # True tower is defined by (n_i, n_k). We fit across gamma for each tower.
    groups = df.groupby(["n_i", "n_k"], dropna=False)

    out_rows = []
    for (ni, nk), g in groups:
        g = g.dropna(subset=["gamma","log10_nu"])
        if len(g) < min_points:
            continue
        x = g["gamma"].to_numpy(float)
        y = g["log10_nu"].to_numpy(float)
        a, b, rss = ols_line(x, y)  # y = a + b x
        fits.append(TowerFit(ion=ion, n_i=int(ni) if pd.notna(ni) else None,
                             n_k=int(nk) if pd.notna(nk) else None,
                             N=len(g), beta_lin=b, chi_lin=a, rss_lin=rss))
    # Pack per-tower fits DataFrame
    fits_df = pd.DataFrame([f.__dict__ for f in fits])

    # safeguard: ensure columns exist even if no fits were recorded
    if fits_df.empty:
        print(f"[warn] No valid tower fits for ion {ion}. Using fallback method.")
        fits_df = pd.DataFrame(columns=[
            "ion","n_i","n_k","N","beta_lin","chi_lin","rss_lin"
        ])

    return fits_df, fits

def compute_global_beta(fits_list: List[pd.DataFrame], beta_fallback: str = "alpha") -> float:
    """Median slope across all provided per-tower fit tables.
    Falls back to theoretical LOG10_ALPHA or NaN if no valid fits are found.
    """
    def _fallback():
        if beta_fallback == "alpha":
            print("[warn] No valid slopes; falling back to LOG10_ALPHA.")
            return LOG10_ALPHA
        print("[warn] No valid slopes; returning NaN (beta-fallback=nan).")
        return float("nan")

    if not fits_list:
        print("[warn] No fit tables provided.")
        return _fallback()

    # Drop empties before concat to avoid future dtype warnings
    non_empty = [df for df in fits_list if isinstance(df, pd.DataFrame) and not df.empty]
    if not non_empty:
        return _fallback()
    try:
        allfits = (pd.concat(non_empty, ignore_index=True)
                   if len(non_empty) > 1 else non_empty[0].copy())
    except ValueError:
        # e.g., concatenating only empty DataFrames
        return _fallback()

    if "beta_lin" not in allfits.columns:
        return _fallback()

    b = pd.to_numeric(allfits["beta_lin"], errors="coerce").dropna()
    if b.empty:
        return _fallback()

    return float(b.median())

def compute_locked_intercepts(df: pd.DataFrame, beta_locked: float, ion: str, min_points: int = 3) -> pd.DataFrame:
    """
    With beta locked, compute per-tower intercepts by chi = mean(log10_nu - beta * gamma_bin).
    Returns one row per tower with chi_locked and N.
    """
    rows = []
    for (ni, nk), g in df.groupby(["n_i", "n_k"], dropna=False):
        g = g.dropna(subset=["gamma","log10_nu"])
        if len(g) < min_points:
            continue
        chi = lock_intercept_given_beta(g["gamma"].to_numpy(float),
                                        g["log10_nu"].to_numpy(float),
                                        beta_locked)
        rows.append({"ion": ion,
                     "n_i": int(ni) if pd.notna(ni) else None,
                     "n_k": int(nk) if pd.notna(nk) else None,
                     "N": len(g),
                     "chi_locked": chi})
    if not rows:
        return pd.DataFrame(columns=["ion","n_i","n_k","N","chi_locked"])

    return pd.DataFrame(rows)


def robust_median_ci(values: np.ndarray, n_boot: int = 2000, rng: Optional[np.random.Generator] = None) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap median and 68% CI (you can adjust to 95% if desired).
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    med = float(np.median(v))
    if n_boot <= 0 or len(v) < 3:
        return med, (med, med)
    rng = rng or np.random.default_rng(123)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(v, size=len(v), replace=True)
        boots.append(np.median(sample))
    boots = np.sort(np.array(boots))
    lo = float(np.percentile(boots, 16))
    hi = float(np.percentile(boots, 84))
    return med, (lo, hi)

def reduced_mass_ratio_log10(ionA: str, ionB: str) -> float:
    def base_symbol(ion: str) -> str:  # "H_I" -> "H"
        return ion.split("_")[0]
    A, B = base_symbol(ionA), base_symbol(ionB)
    me = ELECTRON_MASS_U
    # Prefer nuclear mass; fall back to atomic if needed
    def pick_M(sym: str) -> float:
        if sym in NUCLEAR_MASS_U:
            return NUCLEAR_MASS_U[sym]
        if sym in ATOMIC_MASS_U:
            return ATOMIC_MASS_U[sym]
        raise ValueError(f"Unknown mass for {sym}")
    mA, mB = pick_M(A), pick_M(B)
    muA = (me * mA) / (me + mA)
    muB = (me * mB) / (me + mB)
    return math.log10(muB / muA)

def permutation_null_delta_chi(intercepts_A: np.ndarray, intercepts_B: np.ndarray,
                               gamma_by_tower_A: List[np.ndarray], gamma_by_tower_B: List[np.ndarray],
                               log10_nu_by_tower_A: List[np.ndarray], log10_nu_by_tower_B: List[np.ndarray],
                               beta_locked: float, n_perm: int = 400, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Build a permutation null for Delta_chi by shuffling gamma within each tower,
    recomputing chi per tower (with beta locked), aggregating medians per isotope,
    and taking their difference. Returns (null_distribution, p_value_two_sided).

    The idea: if the gamma structure is destroyed, the observed intercept difference should collapse.
    """
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_perm):
        chis_A = []
        chis_B = []
        # A isotope
        for x, y in zip(gamma_by_tower_A, log10_nu_by_tower_A):
            if len(x) < 3:
                continue
            xp = rng.permutation(x)
            chis_A.append(np.mean(y - beta_locked * xp))
        # B isotope
        for x, y in zip(gamma_by_tower_B, log10_nu_by_tower_B):
            if len(x) < 3:
                continue
            xp = rng.permutation(x)
            chis_B.append(np.mean(y - beta_locked * xp))
        if len(chis_A) == 0 or len(chis_B) == 0:
            continue
        da = np.median(chis_B) - np.median(chis_A)
        v.append(da)
    null = np.array(v)
    return null, float("nan") if len(null) == 0 else None  # p-value computed later where observed is known

def write_ledger_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[ok] Research ledger (JSON) → {path}")

def write_ledger_markdown(path: str, payload: dict):
    lines = []
    L = lines.append
    meta = payload.get("run_metadata", {})
    res  = payload.get("results", {})

    # Header
    L(f"# Research Ledger")
    L("")
    L(f"**Timestamp:** {meta.get('timestamp_utc','')}")
    L(f"**Script:** {meta.get('script','')}")
    L(f"**Args:** `{meta.get('argv','')}`")
    L(f"**beta-fallback:** {meta.get('beta_fallback','')}")
    L("")

    # Inputs
    L("## Inputs")
    for ion, files in payload.get("inputs", {}).items():
        L(f"- **{ion}**")
        L(f"  - tidy lines: `{files.get('lines','')}` (sha256={files.get('lines_sha256','')})")
        L(f"  - tidy levels: `{files.get('levels','')}` (sha256={files.get('levels_sha256','')})")
    L("")

    # Global slope
    gs = res.get("global_slope", {})
    L("## Global Slope")
    L(f"- beta_global = {gs.get('beta_global')}")
    L(f"- k = beta/log10(alpha) = {gs.get('k')}")
    if gs.get("k_percent_error_vs_1") is not None:
        L(f"- % error vs 1.0 = {gs.get('k_percent_error_vs_1'):.6f}%")
    L("")

    # Per-ion Z-norm medians
    L("## Per-ion Z-normalized Medians (chi_norm)")
    L("| Ion | median | CI68 | n_towers | used_fallback |")
    L("|---|---:|---|---:|:--:|")
    for ion, d in res.get("per_ion", {}).items():
        ci = d.get("chi_norm_CI68")
        ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "NA"
        L(f"| {ion} | {d.get('chi_norm_median','NA')} | {ci_str} | {d.get('n_towers',0)} | {'yes' if d.get('used_fallback') else 'no'} |")
    L("")

    # Cross-ion Z-scaling and/or collapse checks
    if "z_scaling_checks" in res:
        L("## Cross-ion Z-scaling Checks")
        L("| pair | Δ median(chi_norm) | expected | % error |")
        L("|---|---:|---:|---:|")
        for row in res["z_scaling_checks"]:
            L(f"| {row['pair']} | {row['delta_observed']} | {row['delta_expected']} | {row.get('percent_error','NA')} |")
        L("")

    # Isotope test (if present)
    if "isotope_test" in res:
        iso = res["isotope_test"]
        L("## Isotope Δχ Test")
        L(f"- pair: {iso.get('pair')}")
        L(f"- Δχ_obs = {iso.get('delta_chi_observed')}")
        L(f"- Δχ_pred = {iso.get('delta_chi_predicted')}")
        pe = iso.get("percent_error")
        if pe is not None:
            L(f"- % error = {pe:.6f}%")
    L("")
    # Save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[ok] Research ledger (Markdown) → {path}")

# ----------------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Photons-only isotope mass estimator (non-circular).")
    ap.add_argument("--lines-dir", required=True, help="Directory containing <ION>_lines.csv")
    ap.add_argument("--levels-dir", required=True, help="Directory containing <ION>_levels.csv")
    ap.add_argument("--ions", nargs="+", required=True, help="Ions to process, e.g. H_I D_I")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--gamma-bin", type=float, default=0.02, help="Gamma bin width (default 0.02)")
    ap.add_argument("--u-gamma", type=float, default=13.6, help="U_gamma in eV for gamma = ΔE/U_gamma (default 13.6 eV)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CI (default 2000)")
    ap.add_argument("--perm-null", type=int, default=400, help="Permutation/swap iterations (default 400)")
    ap.add_argument(
        "--null", choices=["gamma-permute","label-permute","both"],
        default="gamma-permute",
        help="Null construction: permute γ within towers (alignment null); paired label permutation (mass-law); or both."
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Override sigma (α base). Precedence: --sigma > SIGMA env > CODATA reference."
    )
    ap.add_argument("--normalize-Z",
                    choices=["none","hydrogenic"],
                    default="none",
                    help="Apply a simple Z^2 Coulomb-scale correction to intercepts: "
                         "chi_norm = chi - 2*log10(Z). Use only for hydrogenic ions (one electron).")
    ap.add_argument(
        "--isotopes",
        action="store_true",
        help="Force isotope-matched Δχ test even if leading element symbols differ "
             "(e.g., H_I vs D_I). Also useful when symbols are isotope tags."
    )
    ap.add_argument(
        "--beta-fallback",
        choices=["alpha", "nan"],
        default="alpha",
        help="Fallback behavior when no valid tower slopes are found. "
             "'alpha' uses theoretical LOG10_ALPHA (default). 'nan' returns NaN."
    )
    ap.add_argument(
        "--ledger",
        default=None,
        help="Path to write a JSON research ledger (e.g., data/results/mass_estimator/ledger.json)"
    )
    ap.add_argument(
        "--ledger-md",
        default=None,
        help="Optional Markdown companion to summarize the ledger for reviewers."
    )
    ap.add_argument("--min-points", type=int, default=3, help="Minimum lines per (n_i,n_k) tower fit (default 3)")
 
    ap.add_argument("--gamma-mode",
                    choices=["levels","site"], default="levels",
                    help="Gamma gauge: 'levels' uses γ=log_alpha(ΔE/E0) (default); "
                         "'site' uses γ_site=log_alpha(ΔE/(E0*Z^2)) so χ carries μ & Z^2.")
    ap.add_argument("--ladders-dir",
                    help="Optional directory with <ION>_photon_ladder.csv; if present for an ion, "
                         "ingest ladder directly (comment='#' supported).")

    args = ap.parse_args()

    # ── Resolve σ with clear precedence and log it ──────────────────────────
    global ALPHA, LOG10_ALPHA
    sigma_env = os.environ.get("SIGMA", None)
    if args.sigma is not None:
        ALPHA = float(args.sigma)
        sigma_source = "--sigma"
    elif sigma_env is not None:
        ALPHA = float(sigma_env)
        sigma_source = "env(SIGMA)"
    else:
        ALPHA = ALPHA_CODATA_REF
        sigma_source = "CODATA"
    LOG10_ALPHA = math.log10(ALPHA)
    print(f"[info] sigma_used = {ALPHA} ; log10_sigma_used = {LOG10_ALPHA} ; source = {sigma_source}")


    lines_dir = Path(args.lines_dir)
    levels_dir = Path(args.levels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ions = args.ions
    gamma_bin = float(args.gamma_bin)
    u_gamma_ev = float(args.u_gamma)

    ladders: Dict[str, pd.DataFrame] = {}
    fits_tables: Dict[str, pd.DataFrame] = {}
    intercepts_tables: Dict[str, pd.DataFrame] = {}

    # ---- Run metadata & input provenance (for ledger) ----
    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.abspath(sys.argv[0]),
        "argv": " ".join(sys.argv[1:]),
        "beta_fallback": args.beta_fallback,
        "gamma_bin": args.gamma_bin,
        "u_gamma": args.u_gamma,
        "bootstrap": args.bootstrap,
        "perm_null": args.perm_null,
        "normalize_Z": args.normalize_Z,
        "sigma_used": ALPHA,
        "log10_sigma_used": LOG10_ALPHA,
        "sigma_source": sigma_source,
    }

    inputs = {}
    for _ion in args.ions:
        _lp = (Path(args.lines_dir) / f"{_ion}_lines.csv").as_posix()
        _lv = (Path(args.levels_dir) / f"{_ion}_levels.csv").as_posix()
        inputs[_ion] = {
            "lines": _lp,
            "lines_sha256": sha256_file(_lp),
            "levels": _lv,
            "levels_sha256": sha256_file(_lv),
        }

    # 1) Build ladders and per-tower fits for each ion
    for ion in ions:
        lp_lines = lines_dir / f"{ion}_lines.csv"
        lp_levels = levels_dir / f"{ion}_levels.csv"
        if not lp_lines.exists():
            raise FileNotFoundError(f"Missing lines file: {lp_lines}")

        # --- Try photon-ladder ingestion (optional) ---
        ladder = None
        if args.ladders_dir:
            lp_ladder = Path(args.ladders_dir) / f"{ion}_photon_ladder.csv"
            if lp_ladder.exists():
                # skip provenance lines beginning with '#'
                ladder_df = pd.read_csv(lp_ladder, comment="#")

                # keep gamma_bin; create gamma from it
                if "gamma" not in ladder_df.columns and "gamma_bin" in ladder_df.columns:
                    ladder_df["gamma"] = pd.to_numeric(ladder_df["gamma_bin"], errors="coerce")

                # if frequency_hz missing, derive from lambda
                if "frequency_hz" not in ladder_df.columns and "lambda_photon_nm" in ladder_df.columns:
                    lam = pd.to_numeric(ladder_df["lambda_photon_nm"], errors="coerce")
                    ladder_df["frequency_hz"] = C_M_S / (lam * 1e-9)

                ladder_df["log10_nu"] = np.log10(pd.to_numeric(ladder_df["frequency_hz"], errors="coerce"))

                # gauge (site) adjustment—apply to 'gamma' only
                if args.gamma_mode == "site":
                    Z = ion_Z(ion)
                    if Z > 0:
                        ladder_df["gamma"] = pd.to_numeric(ladder_df["gamma"], errors="coerce") \
                                           - (2.0 * np.log(float(Z)) / np.log(ALPHA))

                # keep BOTH gamma_bin and gamma so fit_towers finds gamma_bin, and chi uses gamma
                ladder = ladder_df[["ion","n_i","n_k","gamma_bin","gamma","log10_nu"]].copy()

        # --- Fallback: build ladder from tidy LINES/LEVELS ---
        if ladder is None:
            ladder = build_ladder_for_ion(
                ion,
                lp_lines,
                lp_levels if lp_levels.exists() else None,
                u_gamma_ev=u_gamma_ev,
                gamma_bin=gamma_bin
            )

            # apply site gauge here too (if not already done above)
            if args.gamma_mode == "site":
                Z = ion_Z(ion)
                if Z > 0:
                    ladder["gamma"] = pd.to_numeric(ladder["gamma"], errors="coerce") \
                                    - (2.0 * np.log(float(Z)) / np.log(ALPHA))

            # ensure the columns fit_towers expects
            ladder["log10_nu"] = pd.to_numeric(ladder.get("log10_nu", np.log10(ladder["frequency_hz"])), errors="coerce")

        # ---- record the ladder, then fit per-tower slopes and save ----
        ladders[ion] = ladder
        fits_df, _ = fit_towers(ladder, ion, min_points=int(args.min_points))
        # Stamp σ provenance into outputs
        if not fits_df.empty:
            fits_df["sigma_used"] = ALPHA
            fits_df["log10_sigma_used"] = LOG10_ALPHA
            fits_df["sigma_source"] = sigma_source
        fits_tables[ion] = fits_df
        fits_df.to_csv(out_dir / f"per_tower_fits__{ion}.csv", index=False)

    # 2) Global slope across all ions
    beta_global = compute_global_beta(list(fits_tables.values()), beta_fallback=args.beta_fallback)
    k_est = beta_global / LOG10_ALPHA
    print(f"[info] Global slope beta = {beta_global:.6f} ; k = beta/log10(alpha) = {k_est:.6f}")

    # ---- Global slope summary (for ledger) ----
    global_slope = {
        "beta_global": beta_global,
        "k": k_est,
        "k_percent_error_vs_1": percent_error(k_est, 1.0)
    }

    # 3) Locked intercepts per tower
    for ion in ions:
        inter_df = compute_locked_intercepts(ladders[ion], beta_locked=beta_global, ion=ion, min_points=int(args.min_points))
        # --- stamp physical context into the *raw* intercept file ---
        # Z from ion label; μ/m_e from isotope (use nuclear masses)
        sym_iso = element_symbol_from_ion_label(ion)   # "H_I" -> "H", "D_I" -> "D"
        inter_df["Z"] = ion_Z(ion)
        inter_df["mu_over_me"] = mu_over_me_from_symbol(sym_iso)
        inter_df["norm_applied"] = "none"  # important: helps downstream avoid double-subtracting μ

        if not inter_df.empty:
            inter_df["sigma_used"] = ALPHA
            inter_df["log10_sigma_used"] = LOG10_ALPHA
            inter_df["sigma_source"] = sigma_source
        intercepts_tables[ion] = inter_df
        inter_df.to_csv(out_dir / f"per_tower_intercepts_locked__{ion}.csv", index=False)

    # ---- Per-ion stats for ledger ----
    per_ion = {}
    # (we'll also fill z_norm_stats if you run hydrogenic normalization)
    z_norm_stats = {}

    # 3a) OPTIONAL hydrogenic Z-normalization: chi_norm = chi - 2*log10(Z)
    if args.normalize_Z == "hydrogenic":
        print("[info] Hydrogenic normalization: chi_norm = chi - 2*log10(Z) - log10(mu/me)")
        for ion in ions:
            if electron_count(ion) != 1:
                print(f"[warn] {ion}: not one-electron; skipping normalization.")
                continue
            inter_df = intercepts_tables[ion]
            if inter_df.empty:
                continue

            elem = isotope_family_symbol(element_symbol_from_ion_label(ion))
            Z = ion_Z(ion)
            if Z <= 0:
                print(f"[warn] {ion}: unknown Z; skipping normalization.")
                continue
            mu_over_me = mu_over_me_from_symbol(elem) if elem in ATOMIC_MASS_U else 1.0

            inter_df = inter_df.copy()
            inter_df["chi_norm"] = (pd.to_numeric(inter_df["chi_locked"], errors="coerce")
                                    - 2.0 * math.log10(float(Z))
                                    - math.log10(mu_over_me))
            intercepts_tables[ion] = inter_df
            inter_df.to_csv(out_dir / f"per_tower_intercepts_locked__{ion}__Znorm.csv", index=False)

            med_norm, (lo_norm, hi_norm) = robust_median_ci(inter_df["chi_norm"].to_numpy(float),
                                                    n_boot=args.bootstrap)
            print(f"[Z-norm] {ion}: median(chi_norm) = {med_norm:.9f} (68% CI [{lo_norm:.9f}, {hi_norm:.9f}])")

            # ---- record for ledger ----
            z_norm_stats[ion] = {"median": med_norm, "CI68": [lo_norm, hi_norm]}
            per_ion.setdefault(ion, {})
            per_ion[ion]["chi_norm_median"] = med_norm
            per_ion[ion]["chi_norm_CI68"]   = [lo_norm, hi_norm]

    # 3b) If multiple ions: quick cross-ion summary of normalized medians (hydrogenic only)
    if args.normalize_Z == "hydrogenic" and len(ions) >= 2:
        med_tbl = {}
        for ion in ions:
            idf = intercepts_tables[ion]
            if "chi_norm" not in idf.columns or idf.empty:
                continue
            med_norm, _ = robust_median_ci(idf["chi_norm"].to_numpy(float), n_boot=0)
            med_tbl[ion] = med_norm
        if len(med_tbl) >= 2:
            print("[Z-norm] Cross-ion normalized median differences (pairwise):")
            ions_list = list(med_tbl.keys())
            for i in range(len(ions_list)):
                for j in range(i+1, len(ions_list)):
                    a, b = ions_list[i], ions_list[j]
                    d = med_tbl[b] - med_tbl[a]
                    print(f"    Δ median(chi_norm): {b} - {a} = {d:+.9f}")

    # 4) If two ions provided, perform isotope mass test (MATCHED TOWERS ONLY)
    iso_test = None
    skip_iso_reason = None
    if len(ions) == 2:
        ionA, ionB = ions[0], ions[1]

        elemA = element_symbol_from_ion_label(ionA)
        elemB = element_symbol_from_ion_label(ionB)

        # Auto-group by isotope family (e.g., D -> H). Allow manual override via --isotopes
        famA = isotope_family_symbol(elemA)
        famB = isotope_family_symbol(elemB)

        # 4a) Require same family unless user forces with --isotopes
        if not args.isotopes and famA != famB:
            print(f"[info] {ionA} vs {ionB}: different element families ({famA} vs {famB}) → "
                  f"skipping isotope matched Δχ test. Use --isotopes to force.")
            skip_iso_reason = "different_element_families"

        # 4b) Hydrogenic guardrail: clean reduced-mass logic assumes one-electron ions
        if skip_iso_reason is None and (not args.isotopes) and (electron_count(ionA) != 1 or electron_count(ionB) != 1):
            print(f"[warn] {ionA} vs {ionB}: not hydrogenic (one-electron). "
                  "Reduced-mass Δχ prediction can be invalid for multi-electron ions. "
                  "Use --isotopes to run anyway (phenomenology only).")
            skip_iso_reason = "not_hydrogenic_without_override"

        # 4c) Prepare per-tower intercept tables (once)
        if skip_iso_reason is None:
            A = intercepts_tables[ionA].copy()
            B = intercepts_tables[ionB].copy()

        need = {"n_i","n_k","chi_locked"}
        if skip_iso_reason is None and (A.empty or B.empty or not need.issubset(A.columns) or not need.issubset(B.columns)):
            print("[warn] One or both intercept tables are empty or missing required columns; cannot compute matched Δχ.")
            skip_iso_reason = "missing_or_empty_intercepts"

        if skip_iso_reason is None:
            A = A.dropna(subset=["n_i","n_k","chi_locked"])
            B = B.dropna(subset=["n_i","n_k","chi_locked"])

        # (record per-ion tower counts using the already-built tables that
        # respect --min-points; do not recompute here)
        for ion in ions:
            inter_df = intercepts_tables[ion]
            n_towers = int(inter_df.shape[0])
            per_ion.setdefault(ion, {})
            per_ion[ion]["n_towers"] = n_towers
            per_ion[ion]["used_fallback"] = (n_towers == 0)

        # 4d) Inner-join on exact (n_i, n_k) towers present in BOTH isotopes
        if skip_iso_reason is None:
            key = ["n_i","n_k"]
            M = A.merge(B, on=key, suffixes=("_A","_B"))
            if M.empty:
                print("[warn] No matched towers between isotopes; cannot compute matched Δχ.")
                skip_iso_reason = "no_matched_towers"

        # --- Build a common reference γ for matched towers (H as reference) ---
        if skip_iso_reason is None:
            # reduced-mass ratio μ_H / m_e in electron-mass units
            me_u = ELECTRON_MASS_U
            mu_H_over_me = (me_u * NUCLEAR_MASS_U["H"]) / (me_u + NUCLEAR_MASS_U["H"]) / me_u
            # Hydrogenic ΔE for each matched (n_i, n_k), with Z=1
            de_ref = hydrogenic_deltaE_eV(M["n_i"].to_numpy(int),
                                          M["n_k"].to_numpy(int),
                                          Z=1, mu_over_me=mu_H_over_me)
            gamma_ref = gamma_alpha_from_deltaE(de_ref, U_gamma_eV=u_gamma_ev, alpha=ALPHA)
            M["gamma_ref"] = gamma_ref
            
            # Map (n_i, n_k) -> scalar gamma_ref for quick lookup
            gamma_ref_map = {(int(ni), int(nk)): float(gr)
                             for ni, nk, gr in zip(M["n_i"], M["n_k"], M["gamma_ref"])}

        # --- Observed statistics: (1) ref-based (mass-law), (2) rows-based (alignment) ---
        def chis_with_gamma_ref(ladder_df: pd.DataFrame,
                                gamma_ref_map: dict[Tuple[int,int], float],
                                beta_locked: float) -> dict[tuple[int,int], float]:
            out = {}
            for (ni, nk), g in ladder_df.groupby(["n_i","n_k"], dropna=False):
                key = (int(ni), int(nk))
                if key not in gamma_ref_map:
                    continue
                gr = gamma_ref_map[key]                 # scalar γ_ref for this tower
                gg = g.dropna(subset=["log10_nu"])
                if len(gg) < 1:
                    continue
                # χ_ref = mean( log10(ν) - β * γ_ref_scalar )
                chi_ref = float(np.mean(gg["log10_nu"].to_numpy(float) - beta_locked * gr))
                out[key] = chi_ref
            return out

        if skip_iso_reason is None:
            chisA = chis_with_gamma_ref(ladders[ionA], gamma_ref_map, beta_global)
            chisB = chis_with_gamma_ref(ladders[ionB], gamma_ref_map, beta_global)

        if skip_iso_reason is None:
            common_keys = sorted(set(chisA).intersection(chisB))
            # ref-based per-tower diffs (mass-law statistic)
            v_ref = np.array([chisB[k] - chisA[k] for k in common_keys], dtype=float)
            if v_ref.size == 0:
                print("[warn] No matched towers with usable γ_ref; cannot compute Δχ.")
                skip_iso_reason = "no_usable_gamma_ref"

        # Build per-row γ/ν arrays per matched tower (alignment statistic)
        if skip_iso_reason is None:
            gamma_by_tower_A, y_by_tower_A = [], []
            gamma_by_tower_B, y_by_tower_B = [], []
            chiA_rows, chiB_rows = [], []
            for key in common_keys:
                ni, nk = key
                gA = ladders[ionA][(ladders[ionA]["n_i"]==ni)&(ladders[ionA]["n_k"]==nk)]\
                        .dropna(subset=["gamma","log10_nu"])
                gB = ladders[ionB][(ladders[ionB]["n_i"]==ni)&(ladders[ionB]["n_k"]==nk)]\
                        .dropna(subset=["gamma","log10_nu"])
                if len(gA) >= 3 and len(gB) >= 3:
                    xA = gA["gamma"].to_numpy(float); yA = gA["log10_nu"].to_numpy(float)
                    xB = gB["gamma"].to_numpy(float); yB = gB["log10_nu"].to_numpy(float)
                    gamma_by_tower_A.append(xA); y_by_tower_A.append(yA)
                    gamma_by_tower_B.append(xB); y_by_tower_B.append(yB)
                    chiA_rows.append(float(np.mean(yA - beta_global * xA)))
                    chiB_rows.append(float(np.mean(yB - beta_global * xB)))
            v_rows = np.array(chiB_rows) - np.array(chiA_rows)

        # Observed medians & CIs for both statistics
        if skip_iso_reason is None:
            rng = np.random.default_rng(123)
            n_boot = int(args.bootstrap)
            # ref-based (mass-law)
            med_obs_ref = float(np.median(v_ref))
            if v_ref.size >= 3 and n_boot > 0:
                boots_ref = []
                for _ in range(n_boot):
                    sample = rng.choice(v_ref, size=v_ref.size, replace=True)
                    boots_ref.append(np.median(sample))
                boots_ref = np.sort(np.array(boots_ref))
                lo_ref = float(np.percentile(boots_ref, 16)); hi_ref = float(np.percentile(boots_ref, 84))
            else:
                lo_ref = hi_ref = med_obs_ref
            # rows-based (alignment; mass cancels → predicted ~0)
            med_obs_rows = float(np.median(v_rows)) if v_rows.size else float("nan")
            if v_rows.size >= 3 and n_boot > 0:
                boots_rows = []
                for _ in range(n_boot):
                    sample = rng.choice(v_rows, size=v_rows.size, replace=True)
                    boots_rows.append(np.median(sample))
                boots_rows = np.sort(np.array(boots_rows))
                lo_rows = float(np.percentile(boots_rows, 16)); hi_rows = float(np.percentile(boots_rows, 84))
            else:
                lo_rows = hi_rows = med_obs_rows
        # Predicted Δχ from reduced-mass ratio (for ref-based statistic)
        if skip_iso_reason is None:
            try:
                delta_pred = reduced_mass_ratio_log10(ionA, ionB)
            except Exception as e:
                delta_pred = float("nan")
                print(f"[warn] Could not compute reduced-mass prediction: {e}")

        # Nulls (match method to statistic)
        if skip_iso_reason is None:
            null_method = getattr(args, "null", "gamma-permute")
            pvals = {}
            # Null A: permute γ within towers (alignment null)
            if null_method in ("gamma-permute","both"):
                null_A, _ = permutation_null_delta_chi(
                    None, None,
                    gamma_by_tower_A, gamma_by_tower_B,
                    y_by_tower_A, y_by_tower_B,
                    beta_locked=beta_global,
                    n_perm=int(args.perm_null), seed=42
                )
                if null_A.size:
                    center = float(np.median(null_A))
                    p_A = (np.sum(np.abs(null_A - center) >= abs(med_obs_rows - center)) + 1) / (null_A.size + 1)
                else:
                    p_A = float("nan")
                pvals["gamma_permute"] = p_A
                # write for provenance
                if null_A.size:
                    (out_dir / f"perm_null_gamma__{ionA}___{ionB}.csv").write_text(
                        "\n".join(str(x) for x in null_A), encoding="utf-8"
                    )
            # Null B: paired label permutation / sign-flip across matched towers
            if null_method in ("label-permute","both"):
                v = v_ref  # label-permute tests the mass-law (ref) statistic
                K = v.size
                rng = np.random.default_rng(42)
                if K <= 14:
                    # exact enumeration
                    import itertools as _it
                    signs = np.array(list(_it.product([-1.0, 1.0], repeat=K)))
                    null_B = np.median(signs * v, axis=1)
                else:
                    null_B = []
                    for _ in range(int(args.perm_null)):
                        s = rng.choice([-1.0, 1.0], size=K)
                        null_B.append(np.median(s * v))
                    null_B = np.asarray(null_B, float)
                centerB = float(np.median(null_B))
                # label-permute tests the REF (mass-law) statistic → compare to med_obs_ref
                p_B = (np.sum(np.abs(null_B - centerB) >= abs(med_obs_ref - centerB)) + 1) / (null_B.size + 1)
                pvals["label_permute"] = p_B
                (out_dir / f"perm_null_label__{ionA}___{ionB}.csv").write_text(
                    "\n".join(str(x) for x in null_B), encoding="utf-8"
                )
            # choose primary p for console and JSON
            p_two_sided = pvals.get("gamma_permute") or pvals.get("label_permute") or float("nan")
            # record an appropriate null size for the summary (method-dependent)
            if "gamma_permute" in pvals and pvals["gamma_permute"] == p_two_sided:
                perm_null_size_for_summary = int(args.perm_null)
            elif "label_permute" in pvals and pvals["label_permute"] == p_two_sided:
                # exact enumeration produces size 2^K when K small; otherwise args.perm_null
                K = v_ref.size
                perm_null_size_for_summary = (1 << K) if K <= 14 else int(args.perm_null)
            else:
                perm_null_size_for_summary = 0

        if skip_iso_reason is None:
            iso_test = {
                "pair": f"{ionB} - {ionA}",
                "stat_rows": {
                    "Delta_chi_rows_observed": med_obs_rows,
                    "Delta_chi_rows_CI68": [lo_rows, hi_rows],
                    "Delta_chi_rows_predicted": 0.0,
                    "p_two_sided": pvals.get("gamma_permute")
                },
                "stat_ref": {
                    "Delta_chi_ref_observed": med_obs_ref,
                    "Delta_chi_ref_CI68": [lo_ref, hi_ref],
                    "Delta_chi_ref_predicted_log10_mu_ratio": delta_pred,
                    "Delta_chi_ref_obs_minus_pred": (med_obs_ref - delta_pred) if not math.isnan(delta_pred) else float("nan"),
                    "p_two_sided": pvals.get("label_permute")
                },
                "matched_towers": int(len(M)),
                "perm_null_method": getattr(args, "null", "gamma-permute"),
                "perm_null_size": perm_null_size_for_summary,
                "primary_p_two_sided": p_two_sided
            }

        # Always write a mass summary JSON for two-ion runs
        out_json = out_dir / f"mass_summary__matched__{ionA}__{ionB}.json"
        if skip_iso_reason is None:
            summary = {
                "ions": {"A": ionA, "B": ionB},
                "beta_global": beta_global,
                "k_est": k_est,
                "sigma_used": ALPHA,
                "log10_sigma_used": LOG10_ALPHA,
                "sigma_source": sigma_source,
                "matched_towers": int(len(M)),
                "stat_rows": {
                    "Delta_chi_rows_observed": med_obs_rows,
                    "Delta_chi_rows_CI68": [lo_rows, hi_rows],
                    "Delta_chi_rows_predicted": 0.0,
                    "p_two_sided": pvals.get("gamma_permute")
                },
                "stat_ref": {
                    "Delta_chi_ref_observed": med_obs_ref,
                    "Delta_chi_ref_CI68": [lo_ref, hi_ref],
                    "Delta_chi_ref_predicted_log10_mu_ratio": delta_pred,
                    "Delta_chi_ref_obs_minus_pred": (med_obs_ref - delta_pred) if not math.isnan(delta_pred) else float("nan"),
                    "p_two_sided": pvals.get("label_permute")
                },
                "perm_null_method": getattr(args, "null", "gamma-permute"),
                "perm_null_size": perm_null_size_for_summary,
                "primary_p_two_sided": p_two_sided
            }
            with out_json.open("w") as f:
                json.dump(summary, f, indent=2)
        # Only print the computed summary when not skipped
        if skip_iso_reason is None:
            print("\n=== ISOTOPE MASS (MATCHED TOWERS) ===")
            print(json.dumps(summary, indent=2))
        else:
            with out_json.open("w") as f:
                json.dump({
                    "ions": {"A": ionA, "B": ionB},
                    "beta_global": beta_global,
                    "k_est": k_est,
                    "skipped": True,
                    "reason": skip_iso_reason
                }, f, indent=2)
            print(f"[info] Wrote skipped summary → {out_json} (reason={skip_iso_reason})")

    # ---- Research ledger (optional) ----
    if args.ledger or args.ledger_md:
        payload = {
            "run_metadata": run_metadata,
            "inputs": inputs,
            "results": {
                "global_slope": global_slope,
                "per_ion": per_ion
            }
        }
        # tuck in isotope test if you ran exactly two ions
        if iso_test is not None:
            payload["results"]["isotope_test"] = iso_test

        # If you also computed a raw Z-scaling check, you can add it here similar to earlier guidance

        if args.ledger:
            write_ledger_json(args.ledger, payload)
        if args.ledger_md:
            write_ledger_markdown(args.ledger_md, payload)

    else:
        print("[info] Provide exactly two ions (e.g., H_I D_I) to run the isotope mass test.")

if __name__ == "__main__":
    main()
