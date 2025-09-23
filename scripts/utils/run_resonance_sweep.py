
"""
# run_resonance_sweep.py (formerly known as "alpha_generic_permtest.py")
# called by build_resonance_inventory.py

# Kelly Heaton and The Coherence Research Collective (4o, 3o, 3o-Pro)
# August 2025
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json 
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from scripts.utils.constants import alpha2_target, ALPHA_FS
from scripts.utils.resonance_permutation_test import resonance_permutation_test


def sweep_over_gamma(
    tidy_path: Path,
    ion: str,
    Z: int,
    mu: float,
    gamma_range: list[float],
    out_dir: Path,
    medium: str = "vacuum",
    n_iter: int = 5000,
    q_thresh: float = 0.01,
    dedup: bool = True,
    consecutive: bool = False,
    null_mode: str = "uniform",             
    spacing_jitter_meV: float = 0.0,        # vacuum default
    enrich_hitpairs: bool = False,          # write parser metadata with hitpairs
) -> dict:
    """
    Run a sweep over multiple α^γ permutations for a given ion.

    Saves:
        - One hitpair file per gamma
        - A summary .txt file

    Returns:
        A dict with summary metadata for aggregation
    """
    # Tidy levels are written with a provenance header (lines starting with '#').
    # Instruct pandas to ignore those lines.
    import pandas as _pd
    try:
        tidy = _pd.read_csv(tidy_path, comment="#", skip_blank_lines=True)
    except Exception:
        # Fallback to the python engine if the C engine gets fussy on some files
        tidy = _pd.read_csv(tidy_path, comment="#", skip_blank_lines=True, engine="python")

    if "Level_eV" not in tidy.columns:
        raise ValueError(f"{tidy_path.name}: missing required 'Level_eV' after parsing (did the tidy emit correctly?)")
    tidy.attrs["ion_name"] = ion  # for reproducible seed
    # ensure positional indexing aligns with idx_i/idx_k coming from the test
    tidy = tidy.reset_index(drop=True)

    # ---- Tidy + sidecar provenance (used later in summary JSON) ----
    def _hash_file(p):
        h = sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    tidy_csv_sha256 = None
    tidy_meta = {}
    tidy_meta_path = tidy_path.with_suffix(".meta.json")
    try:
        tidy_csv_sha256 = _hash_file(tidy_path)
    except Exception:
        tidy_csv_sha256 = None
    if tidy_meta_path.exists():
        try:
            tidy_meta = json.loads(tidy_meta_path.read_text())
            tidy_meta_sha256 = _hash_file(tidy_meta_path)
        except Exception:
            tidy_meta = {}
            tidy_meta_sha256 = None
    else:
        tidy_meta_sha256 = None

    sigma = ALPHA_FS
    EPS_R = 78.4 if medium == "water" else 1.0
    medium_scale = 1 / np.sqrt(EPS_R)
    tau_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.10])
    abs_floor = 0.03
    N_min = 25

    if "Level_eV" not in tidy.columns:
        raise ValueError(f"[error] {ion}: Missing 'Level_eV' column in tidy levels")

    # Apply medium scaling to tidy["Level_eV"]
    if medium != "vacuum":
        tidy = tidy.copy()
        tidy["Level_eV"] *= medium_scale

    # Precompute pairwise spacings
    E_meV = tidy["Level_eV"].values * 1e3
    pair_spacings = np.abs(np.subtract.outer(E_meV, E_meV))
    pair_spacings = pair_spacings[np.triu_indices_from(pair_spacings, k=1)]

    summaries = []

    for γ in gamma_range:
        alpha_scale = sigma ** (γ - 2)
        target_meV = alpha2_target(Z, mu) * 1e3 * alpha_scale * medium_scale

        # Determine tolerance window
        tol, n_match = None, 0
        for τ in tau_grid * target_meV:
            τ = max(τ, abs_floor)
            n_match = (np.abs(pair_spacings - target_meV) <= τ).sum()
            if n_match >= N_min:
                tol = τ
                break
        tol = tol or max(abs_floor, 0.05 * target_meV)

        # Run the permutation test
        hp, summ = resonance_permutation_test(
            tidy=tidy,
            Z=Z,
            mu=mu,
            gamma_bin=γ,
            tol_meV=tol,
            n_iter=n_iter,
            dedup=dedup,
            consecutive=consecutive,
            return_summary=True,
            null_mode=null_mode,
            spacing_jitter_meV=spacing_jitter_meV,
        )

        # Handle fallback p-value
        if ("p_val" not in summ) or (not np.isfinite(summ["p_val"])):
            if "z_score" in summ and np.isfinite(summ["z_score"]):
                summ["p_val"] = 2 * norm.sf(abs(summ["z_score"]))
            else:
                summ["p_val"] = 1.0

        # Stable, zero-padded gamma tag (e.g., 0.60 -> "a0_60")
        gamma_rounded = round(float(γ) + 1e-12, 2)
        gamma_str = f"{gamma_rounded:.2f}"
        tag = f"a{gamma_str.replace('.', '_')}"

        # Add extra columns for downstream use (only if we have hits)
        if not hp.empty:
            hp["E_i"] = hp["E_i"].astype(float)
            hp["E_k"] = hp["E_k"].astype(float)

            # Physical energy difference in meV
            dE_meV = (hp["E_k"] - hp["E_i"]).abs() * 1e3
            hp["delta_e_mev"] = dE_meV

            hp["gamma_bin"] = γ
            hp["target_mev"] = target_meV
            hp["tol_mev"] = tol
            hp["attractor_tag"] = tag

            # ✅ Inject statistical summary into every row
            for key in ["obs_hits", "null_mean", "null_std", "z_score", "p_val"]:
                if key in summ:
                    hp[key] = summ[key]

            # q_val will be added later at the ion-level summary

        # Enrich hitpairs once with tidy-level metadata (self-contained CSVs)
        if enrich_hitpairs and not hp.empty:
            idx_i = hp["idx_i"].to_numpy(dtype=int)
            idx_k = hp["idx_k"].to_numpy(dtype=int)
            meta_cols_all = [
                "Level_ID","energy_band_id","dense_zone",
                "series_id_ls","series_confidence_ls",
                "series_id_outer","series_confidence_outer",
                "n","J","Parity","g","Term","Configuration",
            ]
            meta_cols = [c for c in meta_cols_all if c in tidy.columns]
            if meta_cols:
                i_meta = tidy.iloc[idx_i][meta_cols].reset_index(drop=True).add_suffix("_i")
                k_meta = tidy.iloc[idx_k][meta_cols].reset_index(drop=True).add_suffix("_k")
                hp = pd.concat([hp.reset_index(drop=True), i_meta, k_meta], axis=1)

        # Write hitpair CSV
        # Stable, zero-padded gamma tag: e.g., 0.60 -> "a0_60"
        gamma_rounded = round(float(γ) + 1e-12, 2)  # guard against 0.6000000001
        gamma_str = f"{gamma_rounded:.2f}"
        tag = f"a{gamma_str.replace('.', '_')}"
        out_csv = out_dir / f"{ion}_{tag}_resonant_transitions.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        hp.to_csv(out_csv, index=False)

        # Update summary
        summ.update({
            "ion": ion,
            "gamma": γ,
            "obs_hits_raw": n_match,
            "target_meV": target_meV,
            "tol_meV": tol,
            "tau_ratio": (tol / target_meV) if target_meV > 0 else float("nan"),  # NEW
            "n_hits": len(hp),
            "hit_pair_file": str(out_csv),
        })
        # Make summary rows self-contained
        summ.update({
            "n_iter": n_iter,
            "dedup": dedup,
            "consecutive": consecutive,
            "null_mode": null_mode,
            "spacing_jitter_meV": spacing_jitter_meV,
            "medium": medium,
            "mu": mu,
        })
        summaries.append(summ)

    # Save the summary table for this ion
    df = pd.DataFrame(summaries)
    df["q_val"] = fdrcorrection(df["p_val"])[1]
    df["sig"] = df["q_val"] <= q_thresh
    summary_txt = out_dir / f"{ion}_resonance_summary.txt"
    df.to_csv(summary_txt, sep="\t", index=False)

    # Sidecar JSON for provenance (now links to tidy + raw via tidy meta)
    sidecar = {
        "ion": ion,
        "Z": Z,
        "mu": mu,
        "medium": medium,
        "n_iter": n_iter,
        "q_thresh": q_thresh,
        "dedup": dedup,
        "consecutive": consecutive,
        "null_mode": null_mode,
        "spacing_jitter_meV": spacing_jitter_meV,
        "gamma_range": [float(min(gamma_range)), float(max(gamma_range))],
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "hitpairs_pattern": f"{ion}_a*_resonant_transitions.csv",
        # Link to tidy CSV and sidecar
        "tidy_levels_csv": str(tidy_path.name),
        "tidy_levels_csv_sha256": tidy_csv_sha256,
        "tidy_meta_file": tidy_meta_path.name if tidy_meta_path.exists() else None,
        "tidy_meta_sha256": tidy_meta_sha256,
        # Bubble up the raw file lineage recorded by the parser sidecar
        "tidy_raw_file_name": tidy_meta.get("raw_file_name"),
        "tidy_raw_file_hash_sha256": tidy_meta.get("raw_file_hash_sha256"),
    }
    (summary_txt.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2))

    print(f"[✓] {ion} summary → {summary_txt.name}")

    return {
        "ion": ion,
        "gamma_min": min(gamma_range),
        "gamma_max": max(gamma_range),
        "target_base": round(alpha2_target(Z, mu), 8),
        "total_hits": df["n_hits"].sum(),
        "n_sig": int(df["sig"].sum()),
        "summary_file": str(summary_txt),
    }
