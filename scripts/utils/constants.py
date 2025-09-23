# constants.py — Recursive Geometric Physics Constants
# ────────────────────────────────────────────────────
# Kelly Heaton & The Coherence Research Collective
# August 2025
#
# Provides immutable physical constants plus helpers used
# throughout the resonance‑analysis pipeline.
# ────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import re

# --- Physical constants -------------------------------------------------
CM2EV   = 1.239841984e-4            # wavenumber → eV  (cm⁻¹ → eV)
ALPHA_FS = 1 / 137.035999084        # CODATA‑2018 fine‑structure constant (unitless)

# --- Hydrogenic α² target spacing (eV) ----------------------------------
def alpha2_target(Z: int, mu: float = 1.0) -> float:
    """
    Hydrogenic α²·E₀ spacing (in eV).

    Parameters
    ----------
    Z   : int
        Atomic number (proton count).
    mu  : float, optional
        Reduced‑mass ratio μ (μ=1 for infinite‑mass nucleus).

    Returns
    -------
    float
        Target spacing ΔE = α_FS² · 13.605693009 eV · Z² · μ
    """
    E0 = 13.605693009 * Z**2 * mu           # Rydberg energy scaled by Z² μ
    return ALPHA_FS**2 * E0                 # ✅ include α² factor

# --- Principal‑quantum‑number parser ------------------------------------
_qn_re = re.compile(r"^(\d+)")

def principal_qn(text: str) -> float:
    """
    Extract the leading integer (principal quantum number n) from a
    spectroscopic term label (e.g. '4s', '3p').  Returns NaN if absent.
    """
    m = _qn_re.match(str(text).strip())
    return float(m.group(1)) if m else np.nan

# ── Canonical column name mapping ─────────────────────────────────────

CANONICAL_COLUMNS = {
    # ── Identity ──────────────────────────────
    "Ion": "ion",
    "ION": "ion",
    "ion": "ion",

    # ── Energies ──────────────────────────────
    "ΔE_eV": "delta_e_ev",
    "delta_E_eV": "delta_e_ev",
    "Delta_E_eV": "delta_e_ev",       # case variant found in tidy_lines
    "E_lower_eV": "lower_e_ev",
    "E_upper_eV": "upper_e_ev",

    # ── Frequency ─────────────────────────────
    "ν_Hz": "frequency_hz",
    "nu_Hz": "frequency_hz",

    # ── Photon wavelength ─────────────────────
    "λ_photon_nm": "lambda_photon_nm",
    "photon_nm": "lambda_photon_nm",
    "Wavelength_nm": "lambda_photon_nm",  # variant found in tidy_lines

    # ── NIST match wavelength ─────────────────
    "λ_NIST_match_nm": "lambda_nist_match_nm",
    "nist_nm": "lambda_nist_match_nm",

    # ── Hit counts ────────────────────────────
    "obs_hits_raw": "obs_hits",
    "obs_hits": "obs_hits",
    "Hits": "obs_hits",

    # ── Matching metadata ─────────────────────
    "q_val": "q_val",
    "Qval": "q_val",

    # ── Ladder structure ──────────────────────
    "n_i": "n_i",
    "n_k": "n_k",
    "gamma": "gamma_bin",
    "gamma_bin": "gamma_bin",

    # ── Emission rate (new) ───────────────────
    "Aki": "aki",
}


def apply_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns using the canonical naming system.
    Keeps only known canonical mappings, leaves others untouched.
    """
    return df.rename(columns={k: v for k, v in CANONICAL_COLUMNS.items() if k in df.columns})
