#!/usr/bin/env python3
"""
nist_lines_parser_v1.py  —  Tidy NIST lines with Level_ID mapping + selection rules + provenance

Usage
-----
for f in data/raw/lines/*_lines_raw.csv; do
    ion=$(basename "$f" _lines_raw.csv)
    python -m scripts.preprocess.nist_lines_parser_v1 \
        --raw_lines "$f" \
        --tidy_levels "data/tidy/levels/${ion}_levels.csv" \
        --out_dir data/tidy/lines \
        --wavelength_medium vacuum \
        --energy_tol_meV 0.50
done

Single ion:

python -m scripts.preprocess.nist_lines_parser_v1 \
  --raw_lines data/raw/lines/O_III_lines_raw.csv \
  --tidy_levels data/tidy/levels/O_III_levels.csv \
  --out_dir data/tidy/lines \
  --wavelength_medium vacuum \
  --energy_tol_meV 0.50



Notes
-----
- Reads CSVs that may have provenance headers (lines starting with '#').
- Resolves Level_IDs by nearest energy (within tolerance derived from photon/level uncertainty; fallback to --energy_tol_meV).
- Emits:
    * data/tidy/lines/{ion}_lines.csv                         (with provenance header)
    * data/tidy/lines/{ion}_lines.meta.json                   (sidecar: inputs, hashes, thresholds)
- Key columns:
    line_id, wavelength_nm, wavelength_medium, frequency_THz, photon_energy_meV, photon_sigma_meV,
    upper_id, lower_id, match_method, residual_meV,
    E1_allowed, dJ, parity_change, Term_upper, Term_lower, J_upper, J_lower, Parity_upper, Parity_lower,
    Configuration_upper, Configuration_lower
"""

from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


CM2EV = 0.0001239841984  # eV per cm^-1 (matches your levels parser)
EV2MEV = 1e3

# --------------------------- IO helpers ---------------------------

def hash_file(p: Path) -> Optional[str]:
    try:
        h = sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def read_csv_provenance_aware(path: Path) -> pd.DataFrame:
    """Read CSV while ignoring provenance/comment lines."""
    try:
        return pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception:
        return pd.read_csv(path, comment="#", skip_blank_lines=True, engine="python")

def write_csv_with_provenance(df: pd.DataFrame, out_csv: Path, meta: dict) -> None:
    """Write a CSV with a human-readable provenance header."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# created_utc: {datetime.utcnow().isoformat()}Z",
        f"# generator: nist_lines_parser_v1",
        f"# ion: {meta.get('ion')}",
        f"# raw_lines: {meta.get('raw_lines')}",
        f"# raw_lines_sha256: {meta.get('raw_lines_sha256')}",
        f"# tidy_levels: {meta.get('tidy_levels')}",
        f"# tidy_levels_sha256: {meta.get('tidy_levels_sha256')}",
        f"# wavelength_medium: {meta.get('wavelength_medium')}",
        f"# energy_tol_meV: {meta.get('energy_tol_meV')}",
        f"# parser_version: v1.0",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False)

# --------------------------- Physics helpers ---------------------------

def nm_to_THz(wavelength_nm: float, medium: str = "vacuum") -> float:
    """
    Convert wavelength in nm to frequency in THz.
    For air vs vacuum, we let the catalog decide; this function assumes input is already tagged as such.
    c ≈ 299792458 m/s → in nm: 2.99792458e17 nm/s; THz = 1e12 Hz.
    """
    if wavelength_nm <= 0 or pd.isna(wavelength_nm):
        return np.nan
    c_nm_per_s = 2.99792458e17
    Hz = c_nm_per_s / wavelength_nm
    return Hz / 1e12  # THz

def photon_energy_meV_from_wavelength_nm(wavelength_nm: float, medium: str = "vacuum") -> float:
    # E = h c / λ ; using frequency conversion to stay consistent
    f_THz = nm_to_THz(wavelength_nm, medium)
    if pd.isna(f_THz):
        return np.nan
    # h = 4.135667696e-15 eV·s ; THz = 1e12 Hz, so E[eV] = h[eV·s] * f[THz] * 1e12
    E_eV = 4.135667696e-15 * (f_THz * 1e12)
    return E_eV * EV2MEV

def eV_to_meV(x: float) -> float:
    return x * EV2MEV

# --- gentle string→numeric cleaner for wavelength fields (handles trailing '+' etc.) ---
def _to_num_series(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    # Normalize to string, strip a trailing '+' used in NIST RITZ entries, then remove stray chars.
    ss = s.astype(str).str.replace(r"\s*\+$", "", regex=True)
    ss = ss.str.replace(r"[^0-9eE.\-+]", "", regex=True)
    return pd.to_numeric(ss, errors="coerce")

# Selection rules: E1 requires parity change and ΔJ ∈ {0, ±1}, but NOT 0↔0.
def parse_J(Jstr: str) -> Optional[float]:
    if pd.isna(Jstr):
        return None
    s = str(Jstr).strip()
    if not s:
        return None
    # Accept forms like "3/2", "1.5", "2"
    if "/" in s:
        try:
            n, d = s.split("/")
            return float(n) / float(d)
        except Exception:
            # Sometimes weird strings; try to strip non-digits
            pass
    try:
        return float(s)
    except Exception:
        # tokens like "1/23/2" → take last two as a rational if possible
        m = re.findall(r"(\d+)\s*/\s*(\d+)", s)
        if m:
            n, d = m[-1]
            return float(n) / float(d)
        return None

def parity_change(par_u: Optional[str], par_l: Optional[str]) -> Optional[bool]:
    if par_u is None or par_l is None or pd.isna(par_u) or pd.isna(par_l):
        return None
    su = str(par_u).strip().lower()
    sl = str(par_l).strip().lower()
    if not su or not sl:
        return None
    return su != sl

def e1_allowed(Ju: Optional[float], Jl: Optional[float], parity_changed: Optional[bool]) -> Optional[bool]:
    if Ju is None or Jl is None or parity_changed is None:
        return None
    if Ju == 0 and Jl == 0:
        return False
    dJ = abs(Ju - Jl)
    return parity_changed and (math.isclose(dJ, 0.0) or math.isclose(dJ, 1.0))

# --------------------------- Matching logic ---------------------------

@dataclass
class MatchResult:
    upper_id: Optional[int]
    lower_id: Optional[int]
    method: str
    residual_meV: Optional[float]

def resolve_level_id_by_energy(
    levels: pd.DataFrame,
    target_E_eV: float,
    sigma_eV: float,
    fallback_tol_meV: float,
) -> Tuple[Optional[int], str]:
    """Pick nearest Level_ID in energy, within tol; tol = max(3*sigma, fallback)."""
    if pd.isna(target_E_eV):
        return (None, "no_energy")
    tol_eV = max(3.0 * (sigma_eV or 0.0), fallback_tol_meV / EV2MEV)
    diffs = (levels["Level_eV"] - target_E_eV).abs()
    idx_min = int(diffs.values.argmin())
    if diffs.iloc[idx_min] <= tol_eV:
        return (int(levels.iloc[idx_min]["Level_ID"]), "energy_within_tol")
    return (None, "no_match_within_tol")

def best_effort_term_tiebreak(levels: pd.DataFrame, level_id: Optional[int], term: Optional[str], parity: Optional[str]) -> tuple[Optional[int], str]:
    """If multiple candidates, try to tiebreak on Term/Parity."""
    if level_id is None or term is None:
        return (level_id, "no_tiebreak")
    # If we had multiple candidates we would use them; in this nearest-neighbor approach,
    # we just annotate if Term/Parity agrees.
    try:
        row = levels.loc[levels["Level_ID"] == level_id].iloc[0]
        ok_term = (str(row.get("Term", "")).strip() == str(term).strip())
        ok_par = (str(row.get("Parity", "")).strip() == str(parity).strip()) if parity is not None else True
        method = "energy+term" if ok_term and ok_par else "energy_only"
        return (level_id, method)
    except Exception:
        return (level_id, "energy_only")

def attach_level_ids_for_line(
    levels: pd.DataFrame,
    E_photon_meV: float,
    upper_E_eV: Optional[float],
    lower_E_eV: Optional[float],
    sigma_upper_eV: float,
    sigma_lower_eV: float,
    fallback_tol_meV: float,
    term_u: Optional[str], term_l: Optional[str],
    par_u: Optional[str], par_l: Optional[str],
) -> MatchResult:
    """Resolve upper & lower Level_IDs and compute residual."""
    # If upper/lower level energies are present in the raw lines table, use them;
    # else fall back to energy difference matching later.
    uid, m_u = resolve_level_id_by_energy(levels, upper_E_eV, sigma_upper_eV, fallback_tol_meV) if upper_E_eV is not None else (None, "no_upper_energy")
    lid, m_l = resolve_level_id_by_energy(levels, lower_E_eV, sigma_lower_eV, fallback_tol_meV) if lower_E_eV is not None else (None, "no_lower_energy")

    # Best-effort tiebreak with Term/Parity (annotation only)
    uid, m_u = best_effort_term_tiebreak(levels, uid, term_u, par_u)
    lid, m_l = best_effort_term_tiebreak(levels, lid, term_l, par_l)

    method = f"{m_u}|{m_l}"
    residual_meV = None

    if uid is not None and lid is not None:
        # Compute residual using levels table energies
        try:
            Eu = float(levels.loc[levels["Level_ID"] == uid].iloc[0]["Level_eV"])
            El = float(levels.loc[levels["Level_ID"] == lid].iloc[0]["Level_eV"])
            residual_meV = abs((Eu - El) * EV2MEV - E_photon_meV)
        except Exception:
            residual_meV = None

    return MatchResult(uid, lid, method, residual_meV)

# --------------------------- Main pipeline ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse NIST *_lines_raw.csv, attach Level_IDs, selection rules, and provenance.")
    ap.add_argument("--ion", type=str, help="Ion label, e.g., Na_I. If omitted, inferred from raw_lines filename.")
    ap.add_argument("--raw_lines", required=True, type=Path, help="Path to raw lines CSV (pattern: *_lines_raw.csv)")
    ap.add_argument("--tidy_levels", required=True, type=Path, help="Path to tidy levels CSV (with provenance header)")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory for tidy lines")
    ap.add_argument("--wavelength_medium", choices=["air","vacuum"], default="vacuum")
    ap.add_argument("--energy_tol_meV", type=float, default=0.50, help="Fallback absolute tolerance for level-energy matching (meV)")
    args = ap.parse_args()

    # Derive ion from filename if not provided
    if args.ion:
        ion = args.ion
    else:
        stem = args.raw_lines.stem
        if stem.endswith("_lines_raw"):
            ion = stem.replace("_lines_raw", "")
        else:
            raise ValueError(f"Cannot infer ion from {args.raw_lines.name} — expected pattern *_lines_raw.csv")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read inputs
    lines = read_csv_provenance_aware(args.raw_lines)
    levels = read_csv_provenance_aware(args.tidy_levels).reset_index(drop=True)

    # Required columns check (be permissive but explicit)
    required_level_cols = {"Level_ID","Level_eV"}
    if not required_level_cols.issubset(set(levels.columns)):
        raise ValueError(f"{args.tidy_levels.name}: missing columns {required_level_cols - set(levels.columns)}")

    # Normalize raw line column names (include NIST ASD aliases)
    # Wavelength priority: RITZ vacuum > observed vacuum > observed air > generic wavelength
    colmap = {}
    for c in lines.columns:
        cl = c.strip().lower()
        if cl in {"ritz_wl_vac(nm)"}:
            colmap["wavelength_nm_ritz_vac"] = c
        elif cl in {"obs_wl_vac(nm)"}:
            colmap["wavelength_nm_obs_vac"] = c
        elif cl in {"obs_wl_air(nm)"}:
            colmap["wavelength_nm_obs_air"] = c
        elif cl in {"wavelength_nm","vacuum_wavelength_nm","air_wavelength_nm","wavelength"}:
            colmap["wavelength_nm"] = c
        elif cl in {"frequency_thz","nu_thz"}:
            colmap["frequency_THz"] = c
        elif cl in {"photon_energy_mev","deltae_mev","e_mev"}:
            colmap["photon_energy_meV"] = c
        # Level energies: prefer explicit eV, else cm^-1 (Ei/Ek)
        elif cl in {"upper_e_ev","upper_energy_ev","e_upper_ev"}:
            colmap["upper_E_eV"] = c
        elif cl in {"lower_e_ev","lower_energy_ev","e_lower_ev"}:
            colmap["lower_E_eV"] = c
        elif cl in {"ek(cm-1)"}:
            colmap["Ek_cm1"] = c
        elif cl in {"ei(cm-1)"}:
            colmap["Ei_cm1"] = c
        elif cl in {"upper_term","term_upper"}:
            colmap["Term_upper"] = c
        elif cl in {"lower_term","term_lower"}:
            colmap["Term_lower"] = c
        elif cl in {"term_k"}:
            colmap["Term_upper_from_k"] = c
        elif cl in {"term_i"}:
            colmap["Term_lower_from_i"] = c
        elif cl in {"upper_parity","parity_upper"}:
            colmap["Parity_upper"] = c
        elif cl in {"lower_parity","parity_lower"}:
            colmap["Parity_lower"] = c
        elif cl in {"upper_j","j_upper"}:
            colmap["J_upper"] = c
        elif cl in {"lower_j","j_lower"}:
            colmap["J_lower"] = c
        elif cl in {"j_k"}:
            colmap["J_upper_from_k"] = c
        elif cl in {"j_i"}:
            colmap["J_lower_from_i"] = c
        elif cl in {"upper_config","configuration_upper","upper_configuration"}:
            colmap["Configuration_upper"] = c
        elif cl in {"lower_config","configuration_lower","lower_configuration"}:
            colmap["Configuration_lower"] = c
        elif cl in {"conf_k"}:
            colmap["Configuration_upper_from_k"] = c
        elif cl in {"conf_i"}:
            colmap["Configuration_lower_from_i"] = c
        elif cl in {"a_ki","aki","intensity","branching_ratio"}:
            colmap["intensity_like"] = c
        elif cl in {"line_id","id","index"}:
            colmap["line_id"] = c

    # Build canonical fields with safe defaults
    df = pd.DataFrame()
    df["line_id"] = lines[colmap.get("line_id")] if "line_id" in colmap else np.arange(len(lines))

    # --- photon energy / wavelength (robust per-row fallback) ---
    # Priority: explicit photon_energy_meV > first non-NaN among [RITZ vac, obs vac, obs air, generic]
    wl_candidates: list[pd.Series] = []
    if "wavelength_nm_ritz_vac" in colmap:
        wl_candidates.append(_to_num_series(lines[colmap["wavelength_nm_ritz_vac"]]))
    if "wavelength_nm_obs_vac" in colmap:
        wl_candidates.append(_to_num_series(lines[colmap["wavelength_nm_obs_vac"]]))
    if "wavelength_nm_obs_air" in colmap:
        wl_candidates.append(_to_num_series(lines[colmap["wavelength_nm_obs_air"]]))
    if "wavelength_nm" in colmap:
        wl_candidates.append(_to_num_series(lines[colmap["wavelength_nm"]]))

    wl_nm = None
    if wl_candidates:
        wl_nm = wl_candidates[0].copy()
        for s in wl_candidates[1:]:
            wl_nm = wl_nm.combine_first(s)
    else:
        wl_nm = pd.Series([np.nan] * len(lines))

    # Set canonical wavelength fields
    df["wavelength_nm"]   = wl_nm
    df["lambda_photon_nm"] = wl_nm  # downstream alias
    df["wavelength_medium"] = args.wavelength_medium

    # Frequency / photon energy from the final wavelength unless explicit energy provided
    if "photon_energy_meV" in colmap:
        df["photon_energy_meV"] = pd.to_numeric(lines[colmap["photon_energy_meV"]], errors="coerce")
    else:
        df["photon_energy_meV"] = wl_nm.apply(lambda x: photon_energy_meV_from_wavelength_nm(x, args.wavelength_medium))

    if "frequency_THz" in colmap:
        df["frequency_THz"] = pd.to_numeric(lines[colmap["frequency_THz"]], errors="coerce")
    else:
        df["frequency_THz"] = wl_nm.apply(lambda x: nm_to_THz(x, args.wavelength_medium))

    # Optional: quick visibility into cleaning/fallback behavior
    cleaned_from_ritz = 0
    if "wavelength_nm_ritz_vac" in colmap:
        ritz_raw = lines[colmap["wavelength_nm_ritz_vac"]].astype(str)
        cleaned_from_ritz = int(ritz_raw.str.contains(r"\+$").sum())
    print(f"[wavelength] cleaned +'s in RITZ: {cleaned_from_ritz} ; finite λ rows: {int(np.isfinite(df['wavelength_nm']).sum())}/{len(df)}")

    # Optional line energy uncertainty (if present)
    # Accept columns like 'sigma_wavelength_nm', 'sigma_energy_meV'
    sigma_meV = None
    if "sigma_energy_meV" in lines.columns:
        sigma_meV = pd.to_numeric(lines["sigma_energy_meV"], errors="coerce")
    elif "sigma_wavelength_nm" in lines.columns:
        sigma_meV = lines["sigma_wavelength_nm"].apply(lambda x: photon_energy_meV_from_wavelength_nm(x, args.wavelength_medium))
    df["photon_sigma_meV"] = sigma_meV if sigma_meV is not None else np.nan

    # Upper/Lower level hints (if provided in raw lines)
    # Level energies: use eV if present; else convert Ei/Ek cm^-1 → eV
    df["upper_E_eV"] = pd.to_numeric(lines[colmap.get("upper_E_eV")], errors="coerce") if "upper_E_eV" in colmap else np.nan
    df["lower_E_eV"] = pd.to_numeric(lines[colmap.get("lower_E_eV")], errors="coerce") if "lower_E_eV" in colmap else np.nan
    if "Ek_cm1" in colmap:
        df["upper_E_eV"] = pd.to_numeric(lines[colmap["Ek_cm1"]], errors="coerce") * CM2EV
    if "Ei_cm1" in colmap:
        df["lower_E_eV"] = pd.to_numeric(lines[colmap["Ei_cm1"]], errors="coerce") * CM2EV

    # Spectroscopic labels (for later joins/diagnostics)
    for k in ["Term_upper","Term_lower","Parity_upper","Parity_lower","J_upper","J_lower","Configuration_upper","Configuration_lower"]:
        if k in colmap:
            df[k] = lines[colmap[k]]
        else:
            df[k] = np.nan

    # Fill from NIST i/k aliases if explicit fields were absent
    if df["Term_upper"].isna().all() and "Term_upper_from_k" in colmap:
        df["Term_upper"] = lines[colmap["Term_upper_from_k"]]
    if df["Term_lower"].isna().all() and "Term_lower_from_i" in colmap:
        df["Term_lower"] = lines[colmap["Term_lower_from_i"]]
    if df["J_upper"].isna().all() and "J_upper_from_k" in colmap:
        df["J_upper"] = lines[colmap["J_upper_from_k"]]
    if df["J_lower"].isna().all() and "J_lower_from_i" in colmap:
        df["J_lower"] = lines[colmap["J_lower_from_i"]]
    if df["Configuration_upper"].isna().all() and "Configuration_upper_from_k" in colmap:
        df["Configuration_upper"] = lines[colmap["Configuration_upper_from_k"]]
    if df["Configuration_lower"].isna().all() and "Configuration_lower_from_i" in colmap:
        df["Configuration_lower"] = lines[colmap["Configuration_lower_from_i"]]

    # Infer parity from '*' in Term if Parity not present
    def _infer_parity(term):
        if pd.isna(term): return np.nan
        return "odd" if "*" in str(term) else "even"
    if df["Parity_upper"].isna().all():
        df["Parity_upper"] = df["Term_upper"].apply(_infer_parity)
    if df["Parity_lower"].isna().all():
        df["Parity_lower"] = df["Term_lower"].apply(_infer_parity)

    # Resolve Level_IDs
    # Tolerance: use photon sigma if available, else fallback to --energy_tol_meV.
    fallback_tol_meV = float(args.energy_tol_meV)
    upper_ids, lower_ids, methods, residuals = [], [], [], []
    # Cache for quick Level_ID→energy lookups
    levels = levels.copy()
    levels["Level_ID"] = pd.to_numeric(levels["Level_ID"], errors="coerce")
    levels["Level_eV"] = pd.to_numeric(levels["Level_eV"], errors="coerce")

    for i in range(len(df)):
        E_meV = float(df.loc[i, "photon_energy_meV"]) if not pd.isna(df.loc[i, "photon_energy_meV"]) else np.nan
        uE = float(df.loc[i, "upper_E_eV"]) if not pd.isna(df.loc[i, "upper_E_eV"]) else None
        lE = float(df.loc[i, "lower_E_eV"]) if not pd.isna(df.loc[i, "lower_E_eV"]) else None

        sigma_u = 0.0
        sigma_l = 0.0
        res = attach_level_ids_for_line(
            levels=levels,
            E_photon_meV=E_meV,
            upper_E_eV=uE, lower_E_eV=lE,
            sigma_upper_eV=sigma_u, sigma_lower_eV=sigma_l,
            fallback_tol_meV=fallback_tol_meV,
            term_u=df.loc[i, "Term_upper"], term_l=df.loc[i, "Term_lower"],
            par_u=df.loc[i, "Parity_upper"], par_l=df.loc[i, "Parity_lower"],
        )
        upper_ids.append(res.upper_id)
        lower_ids.append(res.lower_id)
        methods.append(res.method)
        residuals.append(res.residual_meV)

    df["upper_id"] = upper_ids
    df["lower_id"] = lower_ids
    df["match_method"] = methods
    df["residual_meV"] = residuals

    # E1 selection rule tags (if J/parity known)
    Ju = df["J_upper"].apply(parse_J)
    Jl = df["J_lower"].apply(parse_J)
    pchg = [parity_change(u, l) for u, l in zip(df["Parity_upper"], df["Parity_lower"])]
    df["dJ"] = [abs((Ju.iloc[i] - Jl.iloc[i])) if Ju.iloc[i] is not None and Jl.iloc[i] is not None else np.nan for i in range(len(df))]
    df["parity_change"] = pchg
    df["E1_allowed"] = [e1_allowed(Ju.iloc[i], Jl.iloc[i], pchg[i]) for i in range(len(df))]

    # Reorder columns for sanity
    cols_order = [
        "line_id","wavelength_nm","wavelength_medium","frequency_THz",
        "photon_energy_meV","photon_sigma_meV",
        "upper_id","lower_id","match_method","residual_meV",
        "E1_allowed","dJ","parity_change",
        "Term_upper","Term_lower","J_upper","J_lower","Parity_upper","Parity_lower",
        "Configuration_upper","Configuration_lower",
    ]
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    # Write outputs
    out_csv = out_dir / f"{ion}_lines.csv"
    out_meta = out_dir / f"{ion}_lines.meta.json"

    meta = {
        "parser_version": "v1.0",
        "ion": ion,
        "raw_lines": str(args.raw_lines),
        "raw_lines_sha256": hash_file(args.raw_lines),
        "tidy_levels": str(args.tidy_levels),
        "tidy_levels_sha256": hash_file(args.tidy_levels),
        "wavelength_medium": args.wavelength_medium,
        "energy_tol_meV": fallback_tol_meV,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    write_csv_with_provenance(df, out_csv, meta)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[done] tidy lines → {out_csv}")
    print(f"[meta] {out_meta}")

if __name__ == "__main__":
    main()
