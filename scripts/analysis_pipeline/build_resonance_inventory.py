# build_resonance_inventory.py
"""
Updated August 9, 2025
# ------------------------------------------------------------
# Entry-point script: batch sweep of alpha^gamma spacings
# across all ions listed in a YAML config.
#
# Output: per-ion resonance CSVs and summaries, plus
# inventory and affinity heatmaps.
# ------------------------------------------------------------

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/bio_vacuum_beta.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 5000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/bio_vacuum_mu-1.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 5000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs


Use this tag to continue without re-processing existing ion data:

--continue_

He specific micro-sweep:

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/He_sweep.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 5000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs

D I specific micro-sweep:

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/D_I.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 10000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs

"""


import pandas as pd
import argparse, sys, glob
from pathlib import Path
from scripts.utils import run_resonance_sweep, path_config
from scripts.utils import provenance as _prov
from scripts.utils.path_config import ensure_dir
import yaml
from scripts.utils.constants import ALPHA_FS

# ------------------------------------------------------------
def load_config(path: Path) -> tuple[list[dict], str]:
    """Load and normalize ion YAML into a list of ion configs, and return sweep tag.
    Expects YAML with an optional 'paths: { tag: "beta" | "mu-1" }'.
    If absent, derive tag from the filename.
    """
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    defaults = cfg.get("defaults", {})
    tag = (cfg.get("paths", {}) or {}).get("tag")
    if not tag:
        # Fallback: infer from filename (e.g., ...beta.yaml or ...mu-1.yaml)
        name = Path(path).name
        tag = "mu-1" if "mu-1" in name else ("beta" if "beta" in name else "beta")
    ion_dict = {k: v for k, v in cfg.items() if k not in {"defaults", "paths"}}

    records = []
    for ion, params in ion_dict.items():
        merged = dict(defaults)
        merged.update(params or {})
        merged["ion"] = ion
        # --- Harmonize gamma specification ---
        gamma_src = None
        for key in ("gamma_bin", "gamma_bins", "gamma_range", "power", "powers"):
            if key in merged:
                gamma_src = merged.pop(key)
                break
        if gamma_src is None:
            raise KeyError(f"{ion}: missing 'gamma_bins'/'gamma_bin' (or legacy 'power(s)') in YAML")
        if isinstance(gamma_src, (int, float)):
            merged["gamma_range"] = [float(gamma_src)]
        elif isinstance(gamma_src, (list, tuple)):
            merged["gamma_range"] = [float(x) for x in gamma_src]
        else:
            merged["gamma_range"] = [float(x) for x in str(gamma_src).split(",")]
        records.append(merged)

    return records, tag

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run alpha^gamma resonance inventory across all configured ions."
    )
    parser.add_argument("--cfg", required=True, help="Path to ion config YAML (e.g. bio_vacuum.yaml)")
    parser.add_argument("--null_mode", choices=["uniform","spacing"], default="spacing")
    parser.add_argument("--spacing_jitter_meV", type=float, default=0.0)
    # --- new controls passed into the sweep ---
    parser.add_argument("--n_iter", type=int, default=5000,
                        help="Permutation iterations per gamma (default 5000).")
    parser.add_argument("--q_thresh", type=float, default=0.01,
                        help="FDR q-value threshold to flag significance (default 0.01).")
    parser.add_argument("--dedup", action="store_true",
                        help="Deduplicate hitpairs before counting (recommended).")
    parser.add_argument("--consecutive", action="store_true",
                        help="Restrict hits to consecutive levels only (off by default).")
    parser.add_argument("--enrich_hitpairs", action="store_true",
                        help="Write hitpair CSVs with parser metadata columns (series, bands, dense flags).")

    parser.add_argument("--resume", action="store_true",
                        help="Do NOT recompute sweeps; rebuild inventory from existing per-ion summaries.")
    parser.add_argument("--continue_", action="store_true",
                        help="Skip ions that already have a per-ion summary file; compute the rest.")

    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.cfg)
    ions, tag = load_config(cfg_path)
    paths = path_config.get_paths(tag)

    out_dir = ensure_dir(paths["resonance"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fast path: resume without recompute ---
    if args.resume:
        print(f"[resume] Rebuilding inventory from existing summaries in {out_dir} ...")
        rows = []
        for summ_path in sorted(out_dir.glob("*_resonance_summary.txt")):
            try:
                df = pd.read_csv(summ_path, sep="\t")
                if df.empty:
                    continue
                ion = summ_path.name.split("_resonance_summary.txt")[0]
                gamma_min = float(df["gamma"].min())
                gamma_max = float(df["gamma"].max())
                # Recover alpha2_target(Z,mu) in eV from any row: target_meV / (1e3 * ALPHA_FS**(gamma-2))
                t0 = df.iloc[0]
                alpha_scale = ALPHA_FS ** (float(t0["gamma"]) - 2.0)
                target_base = float(t0["target_meV"]) / (1e3 * alpha_scale)
                total_hits = int(df["n_hits"].sum())
                # Use the same q-threshold that would be passed to the sweep
                n_sig = int((df.get("q_val", pd.Series([])) <= args.q_thresh).sum())
                rows.append({
                    "ion": ion,
                    "gamma_min": gamma_min,
                    "gamma_max": gamma_max,
                    "target_base": round(target_base, 8),
                    "total_hits": total_hits,
                    "n_sig": n_sig,
                    "summary_file": str(summ_path),
                })
            except Exception as e:
                print(f"[warn] could not parse {summ_path.name}: {e}")
        inv_df = pd.DataFrame(rows).sort_values("ion")
        _prov.INVENTORY_CSV = paths["inventory_csv"]
        from scripts.utils.provenance import write_csv_with_provenance
        write_csv_with_provenance(inv_df, paths["inventory_csv"])
        print("[done] Inventory rebuilt →", paths["inventory_csv"])
        return


    all_records = []

    for ion_spec in ions:
        ion = ion_spec['ion']
        Z = ion_spec['Z']
        mu = ion_spec.get('mu', 1.0)
        gamma_range = ion_spec['gamma_range']

        tidy_path = path_config.TIDY_LEVELS_DIR / f"{ion}_levels.csv"
        if not tidy_path.exists():
            print(f"[skip] {ion}: tidy levels file missing → {tidy_path}")
            continue

        # Skip ions we already finished (idempotent continuation)
        summ_path = out_dir / f"{ion}_resonance_summary.txt"
        if args.continue_ and summ_path.exists():
            print(f"[skip-existing] {ion}: {summ_path.name} already present")
            continue

        print(f"[run] {ion}")
        sweep_result = run_resonance_sweep.sweep_over_gamma(
            tidy_path=tidy_path,
            ion=ion,
            Z=Z,
            mu=mu,
            gamma_range=gamma_range,
            out_dir=out_dir,
            null_mode=args.null_mode,
            spacing_jitter_meV=args.spacing_jitter_meV,
            n_iter=args.n_iter,
            q_thresh=args.q_thresh,
            dedup=args.dedup,
            consecutive=args.consecutive,
            enrich_hitpairs=args.enrich_hitpairs,
        )

        if sweep_result is not None:
            all_records.append(sweep_result)

    # Save inventory summary table
    summary_df = pd.DataFrame(all_records)
    _prov.INVENTORY_CSV = paths["inventory_csv"]
    from scripts.utils.provenance import write_csv_with_provenance
    # Let the provenance helper control CSV kwargs (avoids duplicate 'index=').
    if summary_df.empty:
        print("[warn] sweep produced empty inventory; attempting resume-style rebuild from summaries ...")
        rows = []
        for summ_path in sorted(out_dir.glob("*_resonance_summary.txt")):
            try:
                df = pd.read_csv(summ_path, sep="\t")
                if df.empty: continue
                ion = summ_path.name.split("_resonance_summary.txt")[0]
                gamma_min = float(df["gamma"].min())
                gamma_max = float(df["gamma"].max())
                t0 = df.iloc[0]
                alpha_scale = ALPHA_FS ** (float(t0["gamma"]) - 2.0)
                target_base = float(t0["target_meV"]) / (1e3 * alpha_scale)
                total_hits = int(df["n_hits"].sum())
                n_sig = int((df.get("q_val", pd.Series([])) <= args.q_thresh).sum())
                rows.append({
                    "ion": ion,
                    "gamma_min": gamma_min,
                    "gamma_max": gamma_max,
                    "target_base": round(target_base, 8),
                    "total_hits": total_hits,
                    "n_sig": n_sig,
                    "summary_file": str(summ_path),
                })
            except Exception as e:
                print(f"[warn] could not parse {summ_path.name}: {e}")
        summary_df = pd.DataFrame(rows).sort_values("ion")
    write_csv_with_provenance(summary_df, paths["inventory_csv"])
    print("[done] Inventory saved →", paths["inventory_csv"])

if __name__ == "__main__":
    main()
