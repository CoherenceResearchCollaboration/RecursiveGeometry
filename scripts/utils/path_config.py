# path_config.py – central path registry for the RGP (Recursive Geometric Physics) project
# ─────────────────────────────────────────────────────────────────────────────
# Author: The Coherence Research Collective
# August 2025 – BOINGified
#
# Supports full multi-sweep reproducibility (e.g., beta vs mu‑1)
# by dynamically mapping output folders, overlays, and metadata by tag.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Base directories ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
TIDY_DIR     = DATA_DIR / "tidy"
META_DIR     = DATA_DIR / "meta"
RESULTS_DIR  = DATA_DIR / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"

# ── Raw and tidy data ────────────────────────────────────────────────────
RAW_LEVELS_DIR = RAW_DIR / "levels"
RAW_LINES_DIR  = RAW_DIR / "lines"
TIDY_LEVELS_DIR = TIDY_DIR / "levels"
TIDY_LINES_DIR  = TIDY_DIR / "lines"

# ── Static metadata ──────────────────────────────────────────────────────
ION_MASS_CSV     = META_DIR / "ion_physical_masses.csv"
NUCLEAR_MASS_CSV = META_DIR / "nuclear_masses.csv"
KAPPA_CONSTANT_TXT = META_DIR / "kappa_constant.txt"

# ── Sweep Tags ───────────────────────────────────────────────────────────
SWEEP_TAGS = ["beta", "mu-1"]

# ── Sweep-specific paths ─────────────────────────────────────────────────
RESONANCE_DIRS = {
    "beta": RESULTS_DIR / "resonance_inventory_beta",
    "mu-1": RESULTS_DIR / "resonance_inventory_mu-1",
}

ATTRACTOR_AFFINITY = {
    "beta": META_DIR / "gamma_attractor_affinity_bio_vacuum_beta.csv",
    "mu-1": META_DIR / "gamma_attractor_affinity_bio_vacuum_mu-1.csv",
}

PHOTON_OVERLAY_CSVS = {
    "beta": META_DIR / "photon_overlay_beta.csv",
    "mu-1": META_DIR / "photon_overlay_mu-1.csv",
}

PHOTON_OVERLAY_PARQUETS = {
    "beta": META_DIR / "photon_overlay_beta.parquet",
    "mu-1": META_DIR / "photon_overlay_mu-1.parquet",
}

PHOTON_OVERLAY_REPORTS = {
    "beta": RESULTS_DIR / "reports" / "photon_overlay_report_beta.md",
    "mu-1": RESULTS_DIR / "reports" / "photon_overlay_report_mu-1.md",
}

PHOTON_MATCHED_HITPAIR_DIRS = {
    "beta": RESULTS_DIR / "photon_matched_resonant_pairs_beta",
    "mu-1": RESULTS_DIR / "photon_matched_resonant_pairs_mu-1",
}

LADDER_DIRS = {
    "beta": META_DIR / "ion_photon_ladders_beta",
    "mu-1": META_DIR / "ion_photon_ladders_mu-1",
}

LOCKIN_DIRS = {
    "beta":  Path("data/results/photon_matched_resonant_pairs_beta/lock-in"),
    "mu-1":  Path("data/results/photon_matched_resonant_pairs_mu-1/lock-in"),
}

ION_CONFIG_YAMLS = {
    "beta": META_DIR / "bio_vacuum_beta.yaml",
    "mu-1": META_DIR / "bio_vacuum_mu-1.yaml",
}

INVENTORY_CSVS = {
    "beta": META_DIR / "inventory_bio_vacuum_beta.csv",
    "mu-1": META_DIR / "inventory_bio_vacuum_mu-1.csv",
}

LOCKIN_SUMMARY_CSVS = {
    "beta": RESULTS_DIR / "photon_matched_resonant_pairs_beta" / "lockin_summary__ALL.csv",
    "mu-1": RESULTS_DIR / "photon_matched_resonant_pairs_mu-1" / "lockin_summary__ALL.csv",
}

SKELETON_SUMMARY_CSVS = {
    "beta": META_DIR / "skeleton_summary_beta.csv",
    "mu-1": META_DIR / "skeleton_summary_mu-1.csv",
}

CONTAINMENT_SUMMARY_CSVS = {
    "beta": META_DIR / "gamma_photon_containment_summary_beta.csv",
    "mu-1": META_DIR / "gamma_photon_containment_summary_mu-1.csv",
}

K_EXPLAIN_TABLES = {
    "beta": META_DIR / "k_explain_table_bio_vacuum_beta.csv",
    "mu-1": META_DIR / "k_explain_table_bio_vacuum_mu-1.csv",
}

PHOTON_SLOPE_CSVS = {
    "beta": META_DIR / "photon_ladder_slope_beta.csv",
    "mu-1": META_DIR / "photon_ladder_slope_mu-1.csv",
}

ION_IDENTITY_SUMMARY_CSVS = {
    "beta": META_DIR / "ion_identity_summary_beta.csv",
    "mu-1": META_DIR / "ion_identity_summary_mu-1.csv",
}

# ── Utility: Fetch all paths for a sweep tag ─────────────────────────────
def get_paths(tag: str) -> dict[str, Path]:
    if tag in RESONANCE_DIRS:
        return {
            "resonance": RESONANCE_DIRS[tag],
            "affinity": ATTRACTOR_AFFINITY[tag],
            "overlay_csv": PHOTON_OVERLAY_CSVS[tag],
            "overlay_parquet": PHOTON_OVERLAY_PARQUETS[tag],
            "overlay_report": PHOTON_OVERLAY_REPORTS[tag],
            "matched_dir": PHOTON_MATCHED_HITPAIR_DIRS[tag],
            "config_yaml": ION_CONFIG_YAMLS[tag],
            "inventory_csv": INVENTORY_CSVS[tag],
            "k_explain_table": K_EXPLAIN_TABLES[tag],
            "lockin_summary_csv": LOCKIN_SUMMARY_CSVS[tag],
            "skeleton_summary_csv": SKELETON_SUMMARY_CSVS[tag],
            "containment_summary_csv": CONTAINMENT_SUMMARY_CSVS[tag],
            "photon_slope_csv": PHOTON_SLOPE_CSVS[tag],
            "ion_identity_summary_csv": ION_IDENTITY_SUMMARY_CSVS[tag],
            "lock_in": LOCKIN_DIRS[tag],
        }
    # ── Dynamic fallback for custom tags (e.g., "H_I", "Fe_II_micro") ──
    res_dir = RESULTS_DIR / f"resonance_inventory_{tag}"
    matched_dir = RESULTS_DIR / f"photon_matched_resonant_pairs_{tag}"
    return {
        "resonance": res_dir,
        "affinity": META_DIR / f"gamma_attractor_affinity_{tag}.csv",
        "overlay_csv": META_DIR / f"photon_overlay_{tag}.csv",
        "overlay_parquet": META_DIR / f"photon_overlay_{tag}.parquet",
        "overlay_report": RESULTS_DIR / "reports" / f"photon_overlay_report_{tag}.md",
        "matched_dir": matched_dir,
        "config_yaml": META_DIR / f"{tag}.yaml",
        "inventory_csv": META_DIR / f"inventory_{tag}.csv",
        "k_explain_table": META_DIR / f"k_explain_table_{tag}.csv",
        "lockin_summary_csv": matched_dir / "lockin_summary__ALL.csv",
        "skeleton_summary_csv": META_DIR / f"skeleton_summary_{tag}.csv",
        "containment_summary_csv": META_DIR / f"gamma_photon_containment_summary_{tag}.csv",
        "photon_slope_csv": META_DIR / f"photon_ladder_slope_{tag}.csv",
        "ion_identity_summary_csv": META_DIR / f"ion_identity_summary_{tag}.csv",
        "lock_in": matched_dir / "lock-in",
    }

# ── Ensure folders exist ─────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

for d in [RAW_LEVELS_DIR, RAW_LINES_DIR, TIDY_LEVELS_DIR, TIDY_LINES_DIR, META_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sweep_path in RESONANCE_DIRS.values():
    sweep_path.mkdir(parents=True, exist_ok=True)
