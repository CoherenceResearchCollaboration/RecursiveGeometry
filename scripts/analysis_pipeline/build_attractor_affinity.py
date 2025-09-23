#!/usr/bin/env python3
"""
build_attractor_affinity.py – γ‑Attractor Affinity Matrix
───────────────────────────────────────────────────────────
Aggregates all per‑ion γ‑sweep results into a tallied matrix showing
resonant activity per (ion, gamma_bin), including statistical metrics.

Usage:

python -m scripts.analysis_pipeline.build_attractor_affinity \
  --tag mu-1

python -m scripts.analysis_pipeline.build_attractor_affinity \
  --tag O_III

python -m scripts.analysis_pipeline.build_attractor_affinity \
  --tag D_I_micro \
  --hitpair_dir data/results/resonance_inventory_D_I_micro \
  --out_csv    data/meta/gamma_attractor_affinity_D_I_micro.csv

"""

import argparse, re
from pathlib import Path
import pandas as pd
from scripts.utils.path_config import ensure_dir
from scripts.utils import path_config

# ─────────────────────────────────────────────────────────────
# Patterns for matching filenames
HP_PATTERN  = re.compile(r"(.*)_a([\d_]+)_resonant_transitions\.csv")
SUM_PATTERN = re.compile(r"(.*)_resonance_summary\.txt")

# ─────────────────────────────────────────────────────────────
def main(hitpair_dir: str, out_csv: str) -> None:
    hitpair_dir = Path(hitpair_dir)
    out_csv     = Path(out_csv)
    ensure_dir(out_csv.parent)

    rows = []

    # ── 1. Parse per-ion *_resonance_summary.txt files ───────────────
    for f in hitpair_dir.glob("*_resonance_summary.txt"):
        m = SUM_PATTERN.match(f.name)
        if not m:
            continue
        ion = m.group(1)
        try:
            df = pd.read_csv(f, sep="\t", comment="#")
            # support either 'gamma_bin' or 'gamma'
            gcol = "gamma_bin" if "gamma_bin" in df.columns else ("gamma" if "gamma" in df.columns else None)
            if gcol is None:
                raise ValueError("No 'gamma' or 'gamma_bin' column found.")

            for row in df.itertuples(index=False):
                r = row._asdict()
                # define gamma_val once, from the detected column
                try:
                    gamma_val = float(r.get(gcol))
                except Exception:
                    # skip bad/blank rows instead of crashing
                    continue

                rows.append({
                    "ion"         : ion,
                    # keep BOTH names to avoid breaking other scripts
                    "gamma_bin"   : gamma_val,      # the gamma match, aka "power"
                    "obs_hits"    : r.get("obs_hits"),  # total observed hits (with recurrence)
                    "n_hits"      : r.get("n_hits"),    # unique transitions (recursion depth)
                    "obs_hits_raw": r.get("obs_hits_raw"),
                    "p_val"       : r.get("p_val"),
                    "q_val"       : r.get("q_val"),
                    "tol_meV"     : r.get("tol_meV", r.get("tol_mev")),
                    "null_mean"   : r.get("null_mean"),
                    "null_sigma"  : r.get("null_sigma"),
                    "z_score"     : r.get("z_score"),
                    "source"      : f.name,
                    "n_i"         : r.get("n_i"),
                    "n_k"         : r.get("n_k"),
                })

        except Exception as exc:
            print(f"[!] Failed to parse {f.name}: {exc}")

    # ── 2. Infer hitpair directionality (aggregate over ALL rows) ─────────
    for hp in hitpair_dir.glob("*_a*_resonant_transitions.csv"):
        m = HP_PATTERN.match(hp.name)
        if not m:
            continue
        ion_hp, gstr = m.groups()
        gamma_hp = float(gstr.replace("_", "."))
        try:
            df_hp = pd.read_csv(hp)  # read ALL rows
            if {"E_i", "E_k"}.issubset(df_hp.columns) and not df_hp.empty:
                frac_out = float(((df_hp["E_k"] - df_hp["E_i"]) > 0).mean())
                if   frac_out >= 0.60: direction = "+"
                elif frac_out <= 0.40: direction = "-"
                else:                  direction = "mixed"
                # attach to the matching row in `rows`
                for r in rows:
                    if r["ion"] == ion_hp and abs(r["gamma_bin"] - gamma_hp) < 1e-9:
                        r["direction"] = direction
                        r["frac_outward"] = round(frac_out, 4)
                        if {"n_i", "n_k"}.issubset(df_hp.columns):
                            r["n_i"] = int(df_hp["n_i"].iloc[0])
                            r["n_k"] = int(df_hp["n_k"].iloc[0])

                        break
        except Exception as exc:
            print(f"[!] Failed to infer direction from {hp.name}: {exc}")

    # ── 3. Write output affinity CSV ────────────────────────────────────
    columns = [
        "ion","gamma_bin",
        "obs_hits","n_hits","obs_hits_raw", "n_i", "n_k",
        "p_val","q_val","tol_meV",
        "null_mean","null_sigma","z_score",
        "direction","frac_outward","source"
    ]

    sample = rows[0] if rows else {}
    missing = [k for k in sample.keys() if k not in columns]
    if missing:
        print("[warn] columns list is missing:", missing)

    df_out = pd.DataFrame(rows, columns=columns).sort_values(["ion", "gamma_bin"])
    df_out.to_csv(out_csv, index=False)
    print(f"[✓] γ-attractor affinity map written to → {out_csv}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from scripts.utils import path_config

    p = argparse.ArgumentParser(
        description="Build γ-attractor affinity matrix from *_resonance_summary.txt"
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Sweep tag (e.g., beta, mu-1, H_I, Fe_II_micro). If unknown, use --hitpair_dir/--out_csv.",
    )
    p.add_argument(
        "--hitpair_dir",
        default=None,
        help="Override directory containing *_resonant_transitions.csv and *_resonance_summary.txt",
    )
    p.add_argument(
        "--out_csv",
        default=None,
        help="Override output CSV path for the attractor affinity table",
    )
    args = p.parse_args()

    # Resolve paths from tag if known; otherwise require explicit overrides
    try:
        paths = path_config.get_paths(args.tag)
        hitpair_dir = Path(args.hitpair_dir) if args.hitpair_dir else paths["resonance"]
        out_csv     = Path(args.out_csv)     if args.out_csv     else paths["affinity"]
    except Exception:
        if not (args.hitpair_dir and args.out_csv):
            raise SystemExit(
                f"[error] Unknown tag '{args.tag}' and no overrides given. "
                f"Pass --hitpair_dir and --out_csv."
            )
        hitpair_dir = Path(args.hitpair_dir)
        out_csv     = Path(args.out_csv)

    print(f"[tag] {args.tag}")
    print(f"[in ] hitpairs  : {hitpair_dir}")
    print(f"[out] affinity  : {out_csv}")
    main(hitpair_dir, out_csv)
