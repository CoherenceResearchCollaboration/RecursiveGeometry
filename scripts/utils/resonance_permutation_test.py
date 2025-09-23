"""
# scripts/utils/resonance_permutation_test.py (called by run_resonance_sweep.py)
# formerly known as alpha_permtest_levels.py
# no direct user interface

# Kelly Heaton and The Coherence Research Collective (4o, 3o, 3o-Pro)
# August 2025
"""

import numpy as np
import pandas as pd
import itertools
import hashlib
from scripts.utils.constants import alpha2_target, ALPHA_FS
from scripts.utils.load_sigma import get_sigma


def get_rng_seed(ion: str, sigma: float, gamma: float) -> int:

    key = f"{ion}_{sigma:.9f}_{gamma:.3f}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)


def resonance_permutation_test(
    tidy: pd.DataFrame,
    Z: int,
    mu: float = 1.0,
    gamma_bin: float = 2.0,
    tol_meV: float = 0.05,
    n_iter: int = 2000,
    dedup: bool = True,
    consecutive: bool = False,
    return_summary: bool = False,
    null_mode: str = "uniform",            # NEW: "uniform" | "spacing"
    spacing_jitter_meV: float = 0.0        # NEW: optional tiny jitter (vacuum default = 0)
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Run a permutation test to find resonant level pairs near α^gamma_bin (using ALPHA_FS for physics and σ for RNG seeding).

    Parameters:
        tidy         – pd.DataFrame with 'Level_eV' and optional 'n'
        Z            – atomic number
        mu           – reduced mass
        gamma_bin        – alpha exponent
        tol_meV      – energy match tolerance
        n_iter       – number of permutations
        dedup        – whether to deduplicate hits
        consecutive  – whether to test only consecutive levels
        return_summary – return summary dict if True

    Returns:
        hit_df or (hit_df, summary_dict)
    """
    levels_eV = tidy.Level_eV.to_numpy()
    n_vals = tidy.get("n", pd.Series(np.nan, index=tidy.index)).to_numpy()
    N = len(levels_eV)

    target_meV = alpha2_target(Z, mu) * 1e3
    if gamma_bin != 2:
        target_meV *= ALPHA_FS ** (gamma_bin - 2)

    # ---------- hit‑count helpers -----------------
    def hits_all(levels, tol):
        d = np.abs(np.subtract.outer(levels, levels))
        np.fill_diagonal(d, np.inf)
        return int(np.sum(np.abs(1e3 * d - target_meV) < tol) // 2)

    def hits_consec(levels, tol):
        dE = np.diff(np.sort(levels))
        return int(np.sum(np.abs(1e3 * dE - target_meV) < tol))

    hits_fn = hits_consec if consecutive else hits_all
    obs_hits = hits_fn(levels_eV, tol_meV)

    # ---------- permutation null -----------------
    ion_name = tidy.attrs.get("ion_name", "UNKNOWN")
    sigma_val = get_sigma()
    gamma_val = gamma_bin
    seed = get_rng_seed(ion_name, sigma_val, gamma_val)
    rng = np.random.default_rng(seed)

    Emin, Emax = levels_eV.min(), levels_eV.max()
    levels_sorted = np.sort(levels_eV)
    spacings_eV = np.diff(levels_sorted)              # observed positive spacings

    def _null_levels_uniform():
        return np.sort(rng.uniform(Emin, Emax, size=N))

    def _null_levels_spacing():
        if len(spacings_eV) == 0:
            return _null_levels_uniform()
        # bootstrap spacings, cum-sum from Emin, then scale to [Emin, Emax]
        s = rng.choice(spacings_eV, size=N-1, replace=True)
        L = Emin + np.concatenate([[0.0], np.cumsum(s)])
        # normalize final span to (Emax - Emin)
        span = L[-1] - L[0]
        if span > 0:
            L = Emin + (Emax - Emin) * (L - L[0]) / span
        # optional tiny jitter (vacuum default = 0)
        if spacing_jitter_meV > 0:
            jitter_eV = spacing_jitter_meV * 1e-3
            L = L + rng.normal(0.0, jitter_eV, size=N)
            L = np.clip(L, Emin, Emax)
        return np.sort(L)

    if null_mode not in {"uniform", "spacing"}:
        raise ValueError(f"null_mode must be 'uniform' or 'spacing', got {null_mode!r}")

    _sampler = _null_levels_spacing if null_mode == "spacing" else _null_levels_uniform

    null_hits = np.asarray([hits_fn(_sampler(), tol_meV) for _ in range(n_iter)])

    p_val = (np.sum(null_hits >= obs_hits) + 1) / (n_iter + 1)

    # ---------- build resonant "hit‑pair" table --------------
    pairs = []
    if consecutive:
        for i in range(N - 1):
            E_i, E_k = levels_eV[i], levels_eV[i + 1]
            if abs((E_k - E_i) * 1e3 - target_meV) < tol_meV:
                pairs.append((i, i + 1, n_vals[i], n_vals[i + 1], E_i, E_k))
    else:
        for i, k in itertools.combinations(range(N), 2):
            E_i, E_k = levels_eV[i], levels_eV[k]
            if abs(abs(E_k - E_i) * 1e3 - target_meV) < tol_meV:
                pairs.append((i, k, n_vals[i], n_vals[k], E_i, E_k))

    if dedup:
        pairs = list({tuple(p) for p in pairs})

    # Create hit DataFrame
    hit_df = pd.DataFrame(
        pairs, columns=["idx_i", "idx_k", "n_i", "n_k", "E_i", "E_k"]
    ).dropna()

    # ✅ Force numeric types before computing κ (prevents .round() crash)
    hit_df["E_i"] = pd.to_numeric(hit_df["E_i"], errors="coerce")
    hit_df["E_k"] = pd.to_numeric(hit_df["E_k"], errors="coerce")

    # Label gamma attractor (e.g., "a1.08" for γ = 1.08)
    hit_df["attractor_type"] = f"a{gamma_bin}"

    # Update obs_hits if dedup changed size
    if dedup and len(hit_df) != obs_hits:
        obs_hits = len(hit_df)
        p_val = (np.sum(null_hits >= obs_hits) + 1) / (n_iter + 1)

    if return_summary:
        return hit_df, {
            "obs_hits": obs_hits,
            "null_mean": float(null_hits.mean()),
            "null_sigma": float(null_hits.std()),
            "p_val": p_val,
            "z_score": ((obs_hits - null_hits.mean()) / null_hits.std())
            if null_hits.std() > 0 else float("nan"),
            "target_meV": target_meV,
            "tol_meV": tol_meV,
            "gamma_bin": gamma_bin,
            "n_hits": len(hit_df),
            "null_mode": null_mode,
        }

    return hit_df