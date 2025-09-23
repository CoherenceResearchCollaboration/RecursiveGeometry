#!/usr/bin/env python3
"""
process_photons.py

# Kelly Heaton and The Coherence Research Collective (4o, 3o, 3o-Pro)
# August 2025

photon_overlay builder for NIST-aligned emission analysis
────────────────────────────────────────────────────────────

This script builds a photon emission overlay for a specified γ‑hitpair directory,
matching ΔE-derived λ_photon values against NIST spectral lines using a
statistical + physical windowing model.

One call → writes:
    • Per‑ion match CSVs      → (--out_dir)
    • Full overlay (CSV/Parquet) → (--overlay_csv, --overlay_parquet)
    • Evaluation markdown     → (--report_md)
    • Δλ histograms           → (--hist_dir)

────────────────────────────────────────────────────────────
# ✳️ Example usage mu-1 tag:

python -m scripts.analysis_pipeline.process_photons \
  --sweep_tag mu-1 \
  --hitpair_dir data/results/resonance_inventory_mu-1 \
  --lines_dir data/tidy/lines \
  --overlay_csv data/meta/photon_overlay_mu-1.csv \
  --overlay_parquet data/meta/photon_overlay_mu-1.parquet \
  --out_dir data/results/photon_matched_resonant_pairs_mu-1 \
  --hist_dir data/results/plots \
  --report_md data/results/reports/photon_overlay_report_mu-1.md \
  --medium vacuum \
  --overwrite

python -m scripts.analysis_pipeline.process_photons \
  --hitpair_dir data/results/resonance_inventory_H_I \
  --lines_dir  data/tidy/lines \
  --out_dir    data/results/photon_matched_resonant_pairs_H_I \
  --sweep_tag  H_I \
  --overlay_csv     data/meta/photon_overlay_H_I.csv \
  --overlay_parquet data/meta/photon_overlay_H_I.parquet \
  --report_md       data/results/reports/photon_overlay_report_H_I.md \
  --hist_dir        data/results/reports/overlay_hists_H_I \
  --medium vacuum \
  --overwrite

python -m scripts.analysis_pipeline.process_photons --sweep_tag Fe_II --medium vacuum \
  --hitpair_dir data/results/resonance_inventory_Fe_II \
  --lines_dir data/tidy/lines \
  --out_dir data/results/photon_matched_resonant_pairs_Fe_II \
  --overlay_csv     data/meta/photon_overlay_FeII.csv \
  --overlay_parquet data/meta/photon_overlay_Fe_II.parquet --overwrite

python -m scripts.analysis_pipeline.process_photons \
  --sweep_tag D_I_micro \
  --hitpair_dir data/results/resonance_inventory_D_I_micro \
  --lines_dir data/tidy/lines \
  --overlay_csv     data/meta/photon_overlay_D_I_micro.csv \
  --overlay_parquet data/meta/photon_overlay_D_I_micro.parquet \
  --out_dir         data/results/photon_matched_resonant_pairs_D_I_micro \
  --hist_dir        data/results/plots \
  --report_md       data/results/reports/photon_overlay_report_D_I_micro.md \
  --medium vacuum \
  --overwrite

"""

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scripts.utils.path_config import get_paths
from scripts.utils.constants import apply_canonical_columns

HC_eV_nm = 1239.841984  # h·c
GRID_STEP_EV     = 0.02
k_sigma          = 2.0                # 95% confidence
U_NIST_PM_DEF    = 2.0                # pm
MIN_DE_eV        = 1e-3               # ignore ΔE < 1 meV
MAX_LAMBDA_NM    = 20_000             # ignore λ > 20 µm

# Adaptive floor control
MIN_FLOOR_NM     = 2.0                # 2 nm
MAX_FLOOR_NM     = 5.0                # 5 nm

# Water-specific λ-scaled cap
MAX_SHIFT_FRAC   = 0.03               # allow ±3% shift (e.g., ±15 nm at 500 nm)

# ── project dirs (auto‑detect) ───────────────────────────────────
def project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir():
            return p
    sys.exit("Cannot locate project root (no /data).")

ROOT = project_root(Path(__file__).resolve())
LINES_DIR   = ROOT / "data" / "tidy/lines"
RESULTS_DIR = ROOT / "data" / "results"
META_DIR    = ROOT / "data" / "meta"

# ── helpers ──────────────────────────────────────────────────────
def n_water(λ_nm: pd.Series | np.ndarray) -> np.ndarray:
    """Ciddor‑like dispersion for liquid water 200–1300 nm."""
    λ_um = λ_nm / 1000.0
    n2 = 1.7686 + 0.0094/λ_um**2 + 0.0001/λ_um**4
    return np.sqrt(n2)

def adaptive_window(hit: pd.DataFrame,
                    lines: pd.DataFrame,
                    med: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Δλ match window between photons and NIST lines.

    Returns:
        diff_nm   – absolute Δλ difference matrix (n_hits × n_lines)
        window_nm – matching window for each photon-line pair
        idx       – boolean mask of hits within allowed window # relaxed to see edge cases
    """
    hits_arr = hit["λ_photon_nm"].to_numpy()
    line_arr = lines["λ_NIST_nm"].to_numpy()

    # Uncertainty model
    σ_grid_nm = (HC_eV_nm * GRID_STEP_EV) / (hit["ΔE_eV"]**2) / 2
    σ_nist_nm = (
        lines["Unc_nm"].to_numpy()
        if "Unc_nm" in lines.columns
        else np.full_like(line_arr, U_NIST_PM_DEF * 1e-3)
    )
    σ_tot_nm = np.sqrt(σ_grid_nm.to_numpy()[:, None]**2 + σ_nist_nm**2)

    # Adaptive floor: clip one full grid step to 2–5 nm
    grid_step_nm = (HC_eV_nm * GRID_STEP_EV) / (hit["ΔE_eV"]**2)
    floor_nm = np.clip(grid_step_nm.to_numpy(), MIN_FLOOR_NM, MAX_FLOOR_NM)[:, None]

    # Start with stat-based window
    window_nm = np.maximum(k_sigma * σ_tot_nm, floor_nm)

    # Water-specific cap: fractional window scaling
    if med == "water":
        λ_nm = hit["λ_photon_nm"].to_numpy()[:, None]
        frac_cap_nm = λ_nm * MAX_SHIFT_FRAC
        window_nm = np.minimum(window_nm, frac_cap_nm)

    # Compute Δλ and apply mask
    diff_nm = np.abs(hits_arr[:, None] - line_arr)
    idx = diff_nm <= window_nm  # include boundary
    return diff_nm, window_nm, idx

# ── main processing loop ─────────────────────────────────────────
def build_overlay(med: str, hp_dir: Path, lines_dir: Path, out_dir: Path,
                  overlay_csv: Path, overlay_parquet: Path,
                  affinity_df: pd.DataFrame, sweep_tag: str) -> pd.DataFrame | None:

    hp_dir = Path(hp_dir)
    out_dir = Path(out_dir) if out_dir else hp_dir.parent / "photon_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hp_dir.is_dir():
        print(f"[skip] {med}: hit‑pair directory missing.")
        return None

    tidy_cache: dict[str, pd.DataFrame] = {}
    rows = []
    rex = re.compile(r"^([A-Za-z0-9_]+)_a([\d_]+)_resonant_transitions\.csv$")

    for hp_csv in sorted(hp_dir.glob("*_resonant_transitions.csv")):
        ion_match = rex.match(hp_csv.name)
        if not ion_match:
            continue

        ion = ion_match.group(1)
        # Optional whitelist
        if args.only_ions and ion not in set(args.only_ions):
            continue

        gamma_str = ion_match.group(2)
        gamma = float(gamma_str.replace("_", "."))

        hit = pd.read_csv(hp_csv)
        hit = apply_canonical_columns(hit)

        # After: gamma = float(gamma_str.replace("_", "."))
        # Ensure every row carries a numeric gamma_bin
        hit["gamma_bin"] = gamma

        # Optional but helpful: if attractor tags aren’t present, synthesize a stable tag
        if "attractor_tag" not in hit.columns or hit["attractor_tag"].isna().all():
            hit["attractor_tag"] = f"a{gamma_str}"  # e.g., a1_30
        if "attractor_type" not in hit.columns or hit["attractor_type"].isna().all():
            # Keep a coarse label (a<integer>.<two-digit>)
            parts = gamma_str.split("_")
            if len(parts) >= 2:
                hit["attractor_type"] = f"a{parts[0]}.{parts[1]}"
            else:
                hit["attractor_type"] = f"a{gamma:.2f}"

        if not {"E_i", "E_k"}.issubset(hit.columns):
            continue

        # ── Attach hit count from affinity summary ──
        if "gamma_bin" in affinity_df.columns:
            mask = (affinity_df["ion"] == ion) & (np.abs(affinity_df["gamma_bin"] - gamma) < 1e-8)
        else:
            mask = pd.Series(False, index=affinity_df.index)
        match = affinity_df.loc[mask]

        if not match.empty and "obs_hits" in match.columns:
            # Canonical names for clarity in the overlay
            hit["obs_hits_gamma"] = int(match["obs_hits"].values[0])
            hit["n_hits_gamma"]   = int(match["n_hits"].values[0]) if "n_hits" in match.columns else np.nan
            # Back-compat alias so nothing else breaks (optional to keep for now)
            hit["obs_hits"] = hit["obs_hits_gamma"]
            # Diagnostic alignment
            hit["delta_gamma"] = np.abs(gamma - match["gamma_bin"].values[0])
        else:
            # Conservative fallbacks
            hit["obs_hits_gamma"] = 1
            hit["n_hits_gamma"]   = np.nan
            hit["obs_hits"]       = hit["obs_hits_gamma"]  # legacy alias
            hit["delta_gamma"]    = np.nan

        # Sanity: n_hits should never exceed obs_hits
        if "n_hits_gamma" in hit.columns:
            bad = (hit["n_hits_gamma"] > hit["obs_hits_gamma"]).sum()
            if bad:
                print(f"[warn] {ion} γ={gamma}: {bad} rows with n_hits_gamma > obs_hits_gamma")

        # Compute photon energy and wavelength
        hit["ΔE_eV"] = (hit["E_k"] - hit["E_i"]).abs()
        hit = hit[hit["ΔE_eV"] > MIN_DE_eV].copy()
        hit["λ_photon_nm"] = HC_eV_nm / hit["ΔE_eV"]

        # ── Add recursion direction metadata ───────────────────────────
        # Sign of energy change: +1 (E_k > E_i), -1 (E_k < E_i)
        hit["delta_e_sign"] = np.sign(hit["E_k"] - hit["E_i"]).astype(int)

        # n_step: change in quantum number (if columns present)
        if {"n_i", "n_k"}.issubset(hit.columns):
            hit["n_step"] = hit["n_k"] - hit["n_i"]
        else:
            hit["n_step"] = np.nan

        # Recursion direction: outward if ΔE > 0, inward if < 0
        hit["recursion_direction"] = hit["delta_e_sign"].map({1: "outward", -1: "inward", 0: "flat"})

        if med == "water":
            hit["λ_photon_nm"] /= n_water(hit["λ_photon_nm"])

        hit = hit[hit["λ_photon_nm"] < MAX_LAMBDA_NM]
        if hit.empty:
            continue

        # Load and cache tidy NIST lines
        if ion not in tidy_cache:
            cand = [
                lines_dir / f"{ion}_lines.csv",
                lines_dir / f"{ion}_lines_annotated.csv",
            ]
            line_file = next((p for p in cand if p.exists()), None)
            if line_file is None:
                # last-ditch: any file that starts with the ion and ends with _lines*.csv
                globbed = sorted(lines_dir.glob(f"{ion}_lines*.csv"))
                if globbed:
                    line_file = globbed[0]
            if line_file is None:
                print(f"[!] Skipping {ion}: tidy lines file not found in {lines_dir}")
                continue

            lines = pd.read_csv(
                line_file,
                comment="#",
                skip_blank_lines=True,
                usecols=lambda c: c != "Unnamed: 0"
            )

            lines = apply_canonical_columns(lines)

            if med == "water":
                lines["λ_NIST_nm"] = lines["lambda_photon_nm"] / n_water(lines["lambda_photon_nm"])

            else:
                lines["λ_NIST_nm"] = lines["lambda_photon_nm"]

            tidy_cache[ion] = lines

        # Compute match window and matches
        diff_nm, window_nm, idx = adaptive_window(hit, tidy_cache[ion], med)
        # (keep your existing “all matches” string)
        match_strings = [
            ";".join(f"{tidy_cache[ion]['λ_NIST_nm'].iloc[j]:.4f}" for j in np.where(row)[0])
            for row in idx
        ]
        hit["λ_NIST_match_nm"] = match_strings

        # ---- Best (nearest) NIST match per photon ----
        mins     = np.where(idx, diff_nm, np.inf)           # residuals per candidate; inf if not in window
        best_j   = np.argmin(mins, axis=1)                  # index of nearest candidate in each row
        has_best = np.isfinite(mins[np.arange(len(mins)), best_j])  # photons that actually had a match

        line_nm_vec   = tidy_cache[ion]["λ_NIST_nm"].to_numpy()
        photon_nm_vec = hit["λ_photon_nm"].to_numpy()
        best_nm = np.full(len(hit), np.nan, dtype=float)
        best_dp = np.full(len(hit), np.nan, dtype=float)
        best_nm[has_best] = line_nm_vec[best_j[has_best]]
        best_dp[has_best] = (best_nm[has_best] - photon_nm_vec[has_best]) * 1e3  # nm → pm
        hit["lambda_nist_match_nm_best"] = best_nm
        hit["delta_lambda_pm_best"]      = best_dp

        # ---- Carry a stable line identifier if present ----
        if "line_id" in tidy_cache[ion].columns:
            # keep dtype=object so string IDs stay intact
            best_ids = np.empty(len(hit), dtype=object)
            best_ids[:] = None
            ids_vec = tidy_cache[ion]["line_id"].to_numpy()
            best_ids[has_best] = ids_vec[best_j[has_best]]
            hit["line_id_best"] = best_ids

        # Now drop photons with no matches (unchanged behavior)
        hit = hit[hit["λ_NIST_match_nm"] != ""]
        if hit.empty:
            continue


        # Tag metadata
        hit["ion"] = ion
        hit["medium"] = med
        hit["file_source"] = hp_csv.name
        hit["hitpair_set"] = hp_dir.name  # optional, but useful

        rows.append(hit)
        hit.to_csv(out_dir / hp_csv.name, index=False)

    # Finalize overlay
    if not rows:
        print(f"[{med}] 0 total matches.")
        return None

    else:
        unique_ions = sorted({df["ion"].iloc[0] for df in rows})
        print(f"[✓] Photon matches completed: {len(unique_ions)} ions → {overlay_csv.name}")

    overlay = pd.concat(rows, ignore_index=True)

    # Reorder metadata columns to front
    meta_cols = ["ion", "medium", "file_source", "hitpair_set"]
    meta_cols = [col for col in meta_cols if col in overlay.columns]
    ordered_cols = meta_cols + [col for col in overlay.columns if col not in meta_cols]
    overlay = overlay[ordered_cols]

    # --- Ensure canonical column names before saving ---
    overlay = apply_canonical_columns(overlay)

    # --- Sanitize dtypes for Parquet (Arrow) ---
    # Columns like J_i/J_k can be "3/2" or numeric; Terms/Parity/Config/IDs are strings.
    def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        to_string_exact = []
        for c in df.columns:
            lc = c.lower()
            if (
                lc.startswith(("term_", "configuration_", "parity_"))
                or lc.startswith(("series_id", "line_id"))
                or lc.endswith(("_id", "_tag"))
                or lc in {"j_i", "j_k", "j_upper", "j_lower"}
            ):
                to_string_exact.append(c)
        # Cast selected columns explicitly to string; keep numerics numeric
        for c in to_string_exact:
            if c in df.columns:
                df[c] = df[c].astype("string")
        return df

    overlay = _sanitize_for_parquet(overlay)

    # (optional but nice) deterministic column order for diffs
    # overlay = overlay.reindex(sorted(overlay.columns), axis=1)

    # --- Guard: ensure gamma_bin is present and numeric ---
    if "gamma_bin" not in overlay.columns:
        raise RuntimeError("overlay is missing gamma_bin column entirely.")

    # Coerce to numeric and check for NaNs
    overlay["gamma_bin"] = pd.to_numeric(overlay["gamma_bin"], errors="coerce")
    n_missing = overlay["gamma_bin"].isna().sum()
    if n_missing > 0:
        raise RuntimeError(
            f"overlay has {n_missing} rows with non-numeric or missing gamma_bin "
            "(photon labeling incomplete)"
        )

    # Make sure output dirs exist
    overlay_csv.parent.mkdir(parents=True, exist_ok=True)
    overlay_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    overlay.to_csv(overlay_csv, index=False)
    overlay.to_parquet(overlay_parquet, index=False)

    # Save copy into hitpair directory

    tag = sweep_tag
    overlay.to_csv(out_dir / f"photon_overlay_from_{tag}.csv", index=False)

    print(f"[✓] Overlay saved → {overlay_csv.name} ({len(overlay):,} photon matches)")
    return overlay

# ── evaluation report ────────────────────────────────────────────
from scripts.utils.constants import apply_canonical_columns

def evaluate(df_v: pd.DataFrame | None, df_w: pd.DataFrame | None) -> None:
    # Canonicalize columns before any operations
    if df_v is not None:
        df_v = apply_canonical_columns(df_v)
    if df_w is not None:
        df_w = apply_canonical_columns(df_w)

    # Combine available dataframes
    frames = [x for x in (df_v, df_w) if x is not None]
    per_ion = (
        pd.concat(frames)
          .groupby(["medium", "ion"]).size()
          .unstack(fill_value=0).T
    )

    # Sort by most active medium (usually vacuum)
    sort_key = "vacuum" if "vacuum" in per_ion.columns else per_ion.columns[0]
    per_ion = per_ion.sort_values(by=sort_key, ascending=False)

    # ── histogram saving helper ─────────────────────────────────────
    def save_hist(df: pd.DataFrame | None, title: str, fname: str, tag: str, max_pm: float = 20000):
        if df is None or df.empty:
            print(f"[skip] histogram: no data for {title}")
            return

        Path(args.hist_dir).mkdir(parents=True, exist_ok=True)

        try:
            if "lambda_nist_match_nm_best" in df.columns:
                lam_match = pd.to_numeric(df["lambda_nist_match_nm_best"], errors="coerce")
            else:
                lam_match = pd.to_numeric(
                    df["lambda_nist_match_nm"].astype(str).str.split(";").str[0],
                    errors="coerce"
            )
            lam_phot = pd.to_numeric(df["lambda_photon_nm"], errors="coerce")
            delta_pm = (lam_match - lam_phot) * 1e3
            clipped = delta_pm[(delta_pm >= -max_pm) & (delta_pm <= max_pm)].dropna()

            # 🔍 Outlier info
            print(f"[info] {title}")
            print("  max Δλ (pm):", delta_pm.max())
            print("  min Δλ (pm):", delta_pm.min())
            print("  N(|Δλ| > 100,000 pm):", (np.abs(delta_pm) > 100_000).sum())

            # Clip for visualization
            clipped = delta_pm[(-max_pm <= delta_pm) & (delta_pm <= max_pm)]

            plt.figure(figsize=(6, 4))
            plt.hist(clipped, bins=120, color="#004488", edgecolor="white")
            plt.xlabel("Δλ (pm)")
            plt.ylabel("Number of photon–line matches")
            plt.title(title)
            plt.tight_layout()

            tagged_fname = fname.replace(".png", f"_{tag}.png")
            outpath = Path(args.hist_dir) / tagged_fname
            plt.savefig(outpath)
            plt.close()
            print(f"[✓] histogram saved: {outpath.name} ({len(clipped)} matches shown)")
        except Exception as e:
            print(f"[!] failed to generate histogram {fname}: {e}")

    # ── generate plots ──────────────────────────────────────────────
    save_hist(df_v, "Photon – NIST wavelength mismatch (vacuum)", "delta_lambda_hist_vacuum.png", args.sweep_tag)
    save_hist(df_w, "Photon – NIST wavelength mismatch (water only)", "delta_lambda_hist_water.png", args.sweep_tag)

    if df_v is not None and df_w is not None:
        save_hist(
            pd.concat([df_v, df_w]),
            "Photon – NIST wavelength mismatch (combined)",
            "delta_lambda_hist_combined.png",
            args.sweep_tag
        )

    # ── write markdown report to reports/ folder ────────────────────
    report_path = Path(args.report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Photon Overlay Evaluation\n\n")

        if df_v is not None:
            fh.write(f"**Vacuum total matches:** {len(df_v):,}\n\n")

        if df_w is not None:
            fh.write(f"**Water total matches:** {len(df_w):,}\n\n")

        try:
            fh.write("## Top 20 ions by matches\n\n")
            fh.write(per_ion.head(20).to_markdown() + "\n\n")
        except ImportError:
            fh.write(per_ion.head(20).to_string() + "\n\n")

        fh.write(f"![Δλ vacuum](delta_lambda_hist_vacuum_{args.sweep_tag}.png)\n\n")
        if df_w is not None:
            fh.write(f"![Δλ water](delta_lambda_hist_water_{args.sweep_tag}.png)\n\n")
            fh.write(f"![Δλ combined](delta_lambda_hist_combined_{args.sweep_tag}.png)\n")

# ── CLI entrypoint ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photon match + overlay builder")

    parser.add_argument("--hitpair_dir", required=True,
        help="Directory containing *_resonant_transitions.csv")
    parser.add_argument("--lines_dir", default="data/tidy/lines",
                        help="Directory of tidy NIST line CSVs")
    parser.add_argument("--out_dir", default=None,
                        help="Directory to save per-ion matched CSVs (default: <hitpair_dir>/photon_overlay)")
    parser.add_argument("--sweep_tag", required=True,
                        help="Sweep tag (e.g., beta, mu-1, H_I, Fe_II_micro)")
    parser.add_argument("--only_ions", nargs="*", default=None,
                        help="If set, only process these ions (e.g., H_I Fe_II)")
    parser.add_argument("--overlay_csv", required=True,
                        help="Output path for full photon overlay (CSV)")
    parser.add_argument("--overlay_parquet", required=True,
                        help="Output path for full photon overlay (Parquet)")
    parser.add_argument("--report_md", default="data/meta/photon_overlay_report.md",
                        help="Output path for evaluation markdown report")
    parser.add_argument("--hist_dir", default="data/meta",
                        help="Directory to write wavelength mismatch histograms")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing overlay/report files if present")
    parser.add_argument("--medium", choices=["vacuum", "water"],
                        required=True, help="Which medium to process: vacuum or water")

    args = parser.parse_args()

    # ── Load sweep-specific paths ──
    from scripts.utils.path_config import get_paths
    paths = get_paths(args.sweep_tag)

    # ── Load affinity summary (only once!) ──
    affinity_df = pd.read_csv(paths["affinity"])
    affinity_df = apply_canonical_columns(affinity_df)

    # 1) Drop duplicate-named columns (keep the first)
    affinity_df = affinity_df.loc[:, ~affinity_df.columns.duplicated(keep="first")]

    # 2) Ensure we have a numeric gamma column named gamma_bin
    if "gamma_bin" not in affinity_df.columns and "gamma" in affinity_df.columns:
        affinity_df = affinity_df.rename(columns={"gamma": "gamma_bin"})

    affinity_df["gamma_bin"] = pd.to_numeric(affinity_df["gamma_bin"], errors="coerce")

    # 3) Coerce counts to clean integers if present
    for c in ("obs_hits", "n_hits", "obs_hits_raw"):
        if c in affinity_df.columns:
            affinity_df[c] = pd.to_numeric(affinity_df[c], errors="coerce").astype("Int64")

    if args.overwrite:
        for f in [args.overlay_csv, args.overlay_parquet, args.report_md]:
            Path(f).unlink(missing_ok=True)

    df = build_overlay(
        med=args.medium,
        hp_dir=Path(args.hitpair_dir),
        lines_dir=Path(args.lines_dir),
        out_dir=Path(args.out_dir) if args.out_dir else Path(args.hitpair_dir) / "photon_overlay",
        overlay_csv=Path(args.overlay_csv),
        overlay_parquet=Path(args.overlay_parquet),
        affinity_df=affinity_df,
        sweep_tag=args.sweep_tag,
    )

    if df is not None:
        evaluate(df, None)
