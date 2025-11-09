#!/usr/bin/env python3
"""
🌞 For the Sun (FluxAtlas) dataset
python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/solar/raw_csv/fluxatl_all_events.csv \
  --outdir MDLA_results/solar/mdla \
  --label sun_fluxatl_all \
  --series-name "Sun (FluxAtlas, photons)" \
  --grid 0.002 \
  --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'

🌟 For the Vega (ELODIE) dataset
python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/stars/raw_csv/vega_all_photons.csv \
  --outdir MDLA_results/stars/mdla \
  --label vega_elodie_all \
  --series-name "Vega (ELODIE, photons)" \
  --grid 0.002 \
  --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'upper right'

Neon (lamp):

python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/lamps/raw_csv/neon_neI_ritz_vac.csv \
  --outdir      MDLA_results/lamps/mdla/neon_neI_ritz_vac \
  --label       neon_neI_ritz_vac \
  --series-name "Ne I (Ritz, vacuum)" \
  --grid 0.002 \
  --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'


# Na I (Ritz, vacuum)
python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/lamps/raw_csv/na_I_ritz_vac.csv \
  --outdir      MDLA_results/lamps/mdla/na_I_ritz_vac \
  --label       na_I_ritz_vac \
  --series-name "Na I (Ritz, vacuum)" \
  --grid 0.002 --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'

# Hg II (Ritz, vacuum)
python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/lamps/raw_csv/hg_II_ritz_vac.csv \
  --outdir      MDLA_results/lamps/mdla/hg_II_ritz_vac \
  --label       hg_II_ritz_vac \
  --series-name "Hg II (Ritz, vacuum)" \
  --grid 0.002 --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'

Molecule: C2 swan (no adapter step required)

python -m scripts.analysis_pipeline.mdla_sweep \
  --events-csv MDLA_results/molecules/raw_csv/C2_Swan_visible.csv \
  --outdir      MDLA_results/molecules/mdla/C2_Swan_visible \
  --label       C2_Swan_visible \
  --series-name "C₂ Swan bands (visible)" \
  --grid 0.002 --kmax 1.70 \
  --two-stage \
  --beta-min -2.45 --beta-max -1.95 --beta-step 0.01 \
  --refine-win 0.06 --refine-step 0.001
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'


"""

import argparse, json, math, os, subprocess, tempfile, csv
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, numpy as np

def code_length_bernoulli(n, k):
    # Bernoulli plug-in code (bits), as in paper Methods §4.4
    k = max(0, min(int(k), int(n)))
    if n <= 0:
        return float("inf")
    p = max(1e-12, min(1 - 1e-12, k / float(n)))
    return (-k * math.log2(p) - (n - k) * math.log2(1 - p))

def code_length_enumerative(n, k):
    from math import comb, log2
    if not (0 <= k <= n):
        return float("inf")
    return log2(comb(int(n), int(k)))

# Helper
def _count_support_unique(dfb: pd.DataFrame, col: str, N: int):
    present = dfb.loc[dfb.get("present",1)>0, col].astype(int).to_numpy()
    sel = present[(present >= 0) & (present <= N)]
    if sel.size == 0:
        return 0, -1, -1, 0
    u = np.unique(sel)
    first_bin = int(u.min())
    last_bin  = int(u.max())
    N_eff     = max(0, min(N, last_bin - first_bin + 1))
    K_uniq    = int(u.size)
    return N_eff, first_bin, last_bin, K_uniq

def _sweep_beta_list(beta_values, label_tag, args, outdir, N):
    rows = []
    for b in beta_values:
        tmpdir = Path(tempfile.mkdtemp())
        subprocess.check_call([
            "python", "-m", "scripts.analysis_pipeline.threadlaw_photoncode",
            "--label", args.label,
            "--csv", args.events_csv,
            "--outdir", str(tmpdir),
            "--grid", str(args.grid),
            "--beta-override", str(b)
        ])
        bc = (tmpdir / args.label / "barcode_dense.csv")
        if not bc.exists():
            cand = list((tmpdir/args.label).glob("*barcode_dense*.csv"))
            if not cand:
                raise SystemExit(f"barcode_dense not found under {tmpdir/args.label}")
            bc = cand[0]

        dfb = pd.read_csv(bc)
        col = ("kappa_int_rel" if "kappa_int_rel" in dfb.columns
               else "kappa_int" if "kappa_int" in dfb.columns
               else None)
        if col is None:
            raise SystemExit("no kappa_int/_rel column in barcode_dense")

        # Paper-consistent occupancy: K out of fixed N bins in [0..N]
        present = dfb.loc[dfb.get("present", 1) > 0, col].astype(int)
        sel = present[(present >= 0) & (present <= N)].to_numpy()
        K = int(np.unique(sel).size)  # unique occupied bins
        L_bits = code_length_bernoulli(N, K)

        rows.append({
            "beta": b, "N": N, "K": K, "L_bits": L_bits
        })

    # write stage csv (MUST include the new columns)
    tag = f"mdl_sweep_{label_tag}.csv"
    with open(outdir / tag, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["beta","N","N_eff","first_bin","last_bin","K","L_bits"])
        w.writeheader(); w.writerows(rows)

    # choose best, skipping any +inf rows
    finite_rows = [r for r in rows if math.isfinite(r["L_bits"])]
    best = min(finite_rows, key=lambda r: r["L_bits"]) if finite_rows else {"beta": None}
    Path(outdir / f"mdl_best_{label_tag}.json").write_text(json.dumps(best, indent=2))
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-csv", required=True, help="Merged events (e.g., fluxatl_all_events.csv)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid", type=float, default=0.002)
    ap.add_argument("--kmax", type=float, default=1.70, help="κ span for MDL (0..kmax)")
    ap.add_argument("--beta-list", type=str, default="alpha,log10e,log10phi,log10(1/128)",
                    help="Comma list: alpha or numeric values; e.g. 'alpha,-2.1368,-2.0,-1.8'")
    ap.add_argument("--label", default="sun_fluxatl_all",
                    help="Short identifier used in outputs and suptitle (e.g., 'vega_elodie_all').")
    ap.add_argument("--series-name", default=None,
                    help="Legend label for the measured L(β) curve. Defaults to --label if not set.")
    ap.add_argument("--legend-loc", default="upper right",
                    help="Matplotlib legend location for the left panel.")
    ap.add_argument("--windows", default="0-0.2,0.2-0.4,0.4-1.0",
                    help="κ-window fractions within 0..κmax for windowed MDL, e.g. '0-0.2,0.2-0.4,0.4-1.0'")
    ap.add_argument("--bandsplit", action="store_true",
                    help="If events CSV has 'source_path', compute L(β) separately for fluxatl and photatl.")
    # --- Null controls (optional): when >0 we also compute a density-preserving null line
    ap.add_argument("--nulls", type=int, default=0,
                    help="If >0, also compute a density-preserving null curve (rep count reserved for future).")
    ap.add_argument("--null-beta", type=str, default="alpha",
                    help="Reference β used to fix per-window density for the null (default: alpha). "
                         "Accepts tokens like 'alpha', 'log10(1/128)', 'log10(1/e)'.")
    ap.add_argument("--null-windows", type=int, default=50,
                    help="Number of κ windows for density-preserving bookkeeping (kept for provenance).")

    ap.add_argument("--two-stage", action="store_true",
                    help="Use a two-stage uniform sweep: global coarse then fixed local refinement.")
    ap.add_argument("--beta-min", type=float, default=-2.40,
                    help="Global sweep: min beta (default -2.40).")
    ap.add_argument("--beta-max", type=float, default=-2.00,
                    help="Global sweep: max beta (default -2.00).")
    ap.add_argument("--beta-step", type=float, default=0.01,
                    help="Global sweep: step (default 0.01).")
    ap.add_argument("--refine-win", type=float, default=0.05,
                    help="Refine sweep: half-width window around beta*_coarse (default 0.05).")
    ap.add_argument("--refine-step", type=float, default=0.001,
                    help="Refine sweep: step (default 0.001).")

    args = ap.parse_args()

    # ---------- utilities ----------
    def longest_run_ones(bits: np.ndarray) -> int:
        # bits is 0/1 vector over 0..N
        if bits.size == 0: return 0
        # trick: differences at zeros break runs
        z = np.where(bits==0)[0]
        if z.size == 0: return bits.size
        runs = np.diff(np.concatenate(([-1], z, [bits.size-1])))
        # runs counts gaps between zeros; longest ones-run = runs.max()-1
        return int(max(0, runs.max()-1))

    def edge_concentration(present_idx: np.ndarray, N: int, tail_bins: int = 10) -> float:
        if N <= 0: return 0.0
        tail_lo = max(0, N - tail_bins + 1)
        in_tail = ((present_idx >= tail_lo) & (present_idx <= N)).sum()
        total   = ((present_idx >= 0) & (present_idx <= N)).sum()
        return float(in_tail) / float(total or 1)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Resolve beta tokens (reusable)
    def _resolve_beta_list(spec: str):
        vals=[]
        phi = (1+5**0.5)/2
        _ctx = {"e": math.e, "phi": phi}  # names allowed in log10(...)
        for tok in [t.strip() for t in spec.split(",") if t.strip()]:
            t = tok.lower()
            if t == "alpha":
                vals.append(math.log10(1/137.035999084))
            elif t == "log10e":
                vals.append(math.log10(1/math.e))   # 1/e so β<0
            elif t == "log10phi":
                vals.append(math.log10(1/phi))      # 1/φ so β<0
            elif t.startswith("log10(") and t.endswith(")"):
                inner = tok[6:-1]
                val = eval(inner, {"__builtins__": {}}, _ctx)  # e.g. "1/128", "1/e", "1/phi"
                vals.append(math.log10(float(val)))
            else:
                vals.append(float(tok))
        return vals

    # Resolve beta candidates (force β<0; physical frame uses β≈log10 α < 0)
    betas = _resolve_beta_list(args.beta_list)

    # also add a fine grid around log10 alpha
    beta0 = math.log10(1/137.035999084)
    fine = [beta0 + d for d in [i*1e-3 for i in range(-12,13)]]
    # de-dup preserving order
    seen=set(); beta_list=[]
    for b in betas+fine:
        key=round(b,6)
        if key not in seen:
            seen.add(key); beta_list.append(b)
    # Force physical sign: if someone passed a positive β, flip to its negative magnitude
    beta_list = [(-abs(b)) for b in beta_list]

    # ===== Two-stage sweep (global coarse -> fixed local refine), or legacy single-stage =====
    rows = []
    N = int(round(args.kmax/args.grid))  # bins in 0..kmax

    if args.two_stage:
        # Stage 1: global coarse grid [-2.40, -2.00] with step 0.01
        coarse = np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step)
        coarse = [(-abs(b)) for b in coarse]  # enforce β<0
        rows_coarse, best_coarse = _sweep_beta_list(coarse, "coarse", args, outdir, N)

        # Stage 2: symmetric refinement ±refine_win around coarse best, step refine_step
        b0 = float(best_coarse["beta"])
        lo_nom = max(args.beta_min, b0 - args.refine_win)
        hi_nom = min(args.beta_max, b0 + args.refine_win)
        refine = np.arange(lo_nom, hi_nom + 1e-12, args.refine_step)

        # Always include a dense micro-grid around log10(1/α)
        beta_alpha = math.log10(1/137.035999084)
        micro_lo = max(args.beta_min,  beta_alpha - 0.03)
        micro_hi = min(args.beta_max,  beta_alpha + 0.03)
        micro    = np.arange(micro_lo, micro_hi + 1e-12, args.refine_step)

        # Union, sort, and quantize to a stable key to avoid float-dup glitches
        refine = np.unique(np.concatenate([refine, micro]))
        refine = np.round(refine, 9)  # 9–12 dp is plenty for 0.001 steps

        rows_refine, best_refine = _sweep_beta_list(refine, "refine", args, outdir, N)

        # Merge unique betas and write the canonical sweep CSV
        df_coarse  = pd.DataFrame(rows_coarse)
        df_refine  = pd.DataFrame(rows_refine)

        df_merged = pd.concat([df_coarse, df_refine], ignore_index=True)
        df_merged["beta_key"] = df_merged["beta"].round(9)  # stable duplicate key
        df_merged = (df_merged
                     .sort_values("beta_key")
                     .drop_duplicates(subset=["beta_key"], keep="last")
                     .drop(columns=["beta_key"])
                     .sort_values("beta")
                    )

        df_merged.to_csv(outdir/"mdl_sweep.csv", index=False)
        Path(outdir/"mdl_best.json").write_text(json.dumps(best_refine, indent=2))

        # Make rows available to downstream windows/bandsplit blocks
        rows = df_merged.to_dict("records")

        # Flag boundary-limited refinement (for reproducibility reports)
        boundary_limited = (abs(b0 - lo_nom) <= 2*args.refine_step) or (abs(hi_nom - b0) <= 2*args.refine_step)
        if boundary_limited:
            Path(outdir/"mdl_flags.json").write_text(json.dumps({"boundary_limited": True}, indent=2))

        # Use the merged β grid later (nulls, plotting)
        beta_list_for_null = df_merged["beta"].tolist()

        print(json.dumps({"best": best_refine, "out": str(outdir/"mdl_sweep.csv")}, indent=2))

    else:
        # Legacy single-stage path: use the explicit beta_list you resolved earlier
        rows_legacy, best_legacy = _sweep_beta_list(beta_list, "legacy", args, outdir, N)
        df_legacy = pd.DataFrame(rows_legacy).sort_values("beta")
        df_legacy.to_csv(outdir/"mdl_sweep.csv", index=False)
        Path(outdir/"mdl_best.json").write_text(json.dumps(best_legacy, indent=2))

        rows = df_legacy.to_dict("records")
        beta_list_for_null = df_legacy["beta"].tolist()

        print(json.dumps({"best": best_legacy, "out": str(outdir/"mdl_sweep.csv")}, indent=2))


    # ---------- simple outlier report (2nd–98th percentile band) ----------
    try:
        df_all = pd.read_csv(outdir/"mdl_sweep.csv").sort_values("beta")
        y = df_all["L_bits"].to_numpy()
        if y.size:
            ylo, yhi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
            mask_out = (y < ylo) | (y > yhi)
            if mask_out.any():
                cols = [c for c in ["beta","N","N_eff","K","L_bits"] if c in df_all.columns]
                df_all.loc[mask_out, cols].to_csv(outdir/"mdl_outliers.csv", index=False)

    except Exception:
        pass

    # ---------- Windowed MDL over κ-windows ----------
    try:
        # parse window spec like "0-0.2,0.2-0.4,0.4-1.0"
        win_specs = []
        for tok in args.windows.split(","):
            a,b = tok.strip().split("-")
            fa, fb = max(0.0,float(a)), min(1.0,float(b))
            if fa < fb: win_specs.append((fa, fb))

        out_rows = []
        for r in rows:
            b = r["beta"]
            # rebuild barcode at this beta (cheap + robust provenance)
            tmpdir = Path(tempfile.mkdtemp())
            subprocess.check_call([
                "python","-m","scripts.analysis_pipeline.threadlaw_photoncode",
                "--label", args.label, "--csv", args.events_csv, "--outdir", str(tmpdir),
                "--grid", str(args.grid), "--beta-override", str(b)
            ])
            bc = (tmpdir / args.label / "barcode_dense.csv")
            dfb = pd.read_csv(bc)
            col = ("kappa_int_rel" if "kappa_int_rel" in dfb.columns
                   else "kappa_int" if "kappa_int" in dfb.columns else None)
            present = dfb.loc[dfb.get("present",1)>0, col].astype(int).to_numpy()

            N_full = int(round(args.kmax/args.grid))
            for (fa, fb) in win_specs:
                i0 = int(round(fa * N_full))
                i1 = int(round(fb * N_full))

                idx = present[(present >= i0) & (present <= i1)]
                Nw  = max(0, i1 - i0 + 1)

                if idx.size == 0:
                    firstw, lastw, Kw_uniq = -1, -1, 0
                else:
                    u       = np.unique(idx)
                    firstw  = int(u.min())
                    lastw   = int(u.max())
                    Kw_uniq = int(u.size)

                # Paper-consistent: encode K unique bins in a fixed window of size Nw
                Lw     = code_length_bernoulli(Nw, Kw_uniq) if Nw > 0 else 0.0
                Nw_eff = Nw  # keep the column for continuity; equal to fixed window size

                out_rows.append({
                    "beta": b, "win": f"{fa:.2f}-{fb:.2f}",
                    "N_win": Nw, "N_win_eff": Nw_eff,
                    "first_win_bin": firstw, "last_win_bin": lastw,
                    "K_win": Kw_uniq, "L_bits_win": Lw
                })

        with open(outdir/"mdl_windowed.csv","w",newline="") as f:
            cols = sorted({k for r in out_rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=sorted({k for r in out_rows for k in r.keys()}), extrasaction="ignore")
            w.writeheader(); w.writerows(out_rows)

    except Exception as e:
        print(f"[WARN] windowed MDL skipped: {e}")

    # ---------- Band-split MDL (fluxatl vs photatl) ----------
    try:
        ev = pd.read_csv(args.events_csv)
        if "source_path" in ev.columns and args.bandsplit:
            flux_m = ev["source_path"].str.contains("/fluxatl/", na=False)
            phot_m = ev["source_path"].str.contains("/photatl/", na=False)

            def _subset_L(beta_val, mask):
                tmp = Path(tempfile.mkdtemp())
                sub = ev.loc[mask].copy()
                spath = tmp/"events_subset.csv"; sub.to_csv(spath, index=False)
                subprocess.check_call([
                    "python","-m","scripts.analysis_pipeline.threadlaw_photoncode",
                    "--label", args.label, "--csv", str(spath), "--outdir", str(tmp),
                    "--grid", str(args.grid), "--beta-override", str(beta_val)
                ])
                bc = (tmp / args.label / "barcode_dense.csv")
                dfb = pd.read_csv(bc)
                col = ("kappa_int_rel" if "kappa_int_rel" in dfb.columns else
                       "kappa_int"     if "kappa_int"     in dfb.columns else None)
                if col is None: raise SystemExit("no kappa_int/_rel column in barcode_dense")
                Nloc = int(round(args.kmax/args.grid))
                N_eff, first_bin, last_bin, K_uniq = _count_support_unique(dfb, col, Nloc)
                L_bits = code_length_bernoulli(N_eff, K_uniq) if N_eff > 0 else 0.0
                return Nloc, N_eff, first_bin, last_bin, K_uniq, L_bits

            out_rows = []
            # iterate the merged β grid already computed above
            for r in rows:
                b = r["beta"]
                Nf, Nf_eff, firstf, lastf, Kf, Lf = _subset_L(b, flux_m)
                Np, Np_eff, firstp, lastp, Kp, Lp = _subset_L(b, phot_m)
                out_rows.append({
                    "beta": b,
                    "N_flux": Nf,  "N_eff_flux": Nf_eff,  "first_flux_bin": firstf, "last_flux_bin": lastf,
                    "K_flux": Kf,  "L_bits_flux": Lf,
                    "N_phot": Np,  "N_eff_phot": Np_eff,  "first_phot_bin": firstp, "last_phot_bin": lastp,
                    "K_phot": Kp,  "L_bits_phot": Lp
                })

            with open(outdir/"mdl_bandsplit.csv","w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "beta",
                    "N_flux","N_eff_flux","first_flux_bin","last_flux_bin","K_flux","L_bits_flux",
                    "N_phot","N_eff_phot","first_phot_bin","last_phot_bin","K_phot","L_bits_phot"
                ])
                w.writeheader(); w.writerows(out_rows)
    except Exception as e:
        print(f"[WARN] band-split MDL skipped: {e}")

    # ---------- Density-preserving null (flat-in-beta Bernoulli MDL) ----------
    # We fix global K (and, by provenance, per-window counts) at a reference β (default alpha),
    # then evaluate L_null = L(K_ref) for all β – flat baseline for comparison.
    if args.nulls and args.nulls > 0:
        # 1) resolve reference β for null
        null_betas = _resolve_beta_list(args.null_beta)
        null_beta = -abs(null_betas[0] if null_betas else beta0)

        # 2) build barcode at null_beta to get K_ref within 0..kmax
        tmpdir = Path(tempfile.mkdtemp())
        subprocess.check_call([
            "python","-m","scripts.analysis_pipeline.threadlaw_photoncode",
            "--label", args.label, "--csv", args.events_csv, "--outdir", str(tmpdir),
            "--grid", str(args.grid), "--beta-override", str(null_beta)
        ])
        bc = tmpdir / args.label / "barcode_dense.csv"
        if not bc.exists():
            cand = list((tmpdir/args.label).glob("*barcode_dense*.csv"))
            if not cand:
                raise SystemExit(f"[null] barcode_dense not found under {tmpdir/args.label}")
            bc = cand[0]
        df_ref = pd.read_csv(bc)
        col = ("kappa_int_rel" if "kappa_int_rel" in df_ref.columns
               else "kappa_int" if "kappa_int" in df_ref.columns else None)
        if col is None:
            raise SystemExit("[null] no kappa_int/_rel column in barcode_dense")
        bins_ref = df_ref.loc[df_ref.get("present",1)>0, col].astype(int)
        N = int(round(args.kmax/args.grid))
        K_ref = int(bins_ref[bins_ref.between(0, N)].count())
        L_ref = code_length_bernoulli(N, K_ref)

        # 3) write a null CSV mirroring mdl_sweep but flat in β
        with open(outdir/"mdl_null.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=["beta","N","K_ref","L_bits_null","null_beta","null_windows"])
            w.writeheader()
            for b in beta_list_for_null:
                w.writerow({"beta": b, "N": N, "K_ref": K_ref, "L_bits_null": L_ref,
                            "null_beta": null_beta, "null_windows": int(args.null_windows)})

    # ---------- Δbits summary (effect size near α, ignoring shallow-slope artifact) ----------
    # we report two numbers: best L in a window around α, and its Δ to the next-best inside that window
    try:
        df_all = pd.read_csv(outdir/"mdl_sweep.csv")
        # keep only physical region (β <= -0.8) to avoid the shallow-slope collapse artifact
        df_phys = df_all.loc[df_all["beta"] <= -0.8].copy()
        # tight window around α for the headline effect (you can widen later)
        beta_alpha = math.log10(1/137.035999084)
        df_win = df_phys.loc[
            (df_phys["beta"] >= beta_alpha - 0.03) &
            (df_phys["beta"] <= beta_alpha + 0.03)
        ].copy()
        df_win.sort_values("L_bits", inplace=True)
        if not df_win.empty:
            best_row = df_win.iloc[0]
            runner = df_win.iloc[1] if len(df_win) > 1 else None
            delta_bits = (runner["L_bits"] - best_row["L_bits"]) if runner is not None else float("nan")
            summary = {
                "beta_alpha": beta_alpha,
                "best_beta_near_alpha": float(best_row["beta"]),
                "L_best": float(best_row["L_bits"]),
                "delta_bits_to_next": (float(delta_bits) if pd.notna(delta_bits) else None)
            }
            (outdir/"mdl_effect_near_alpha.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps({"effect_near_alpha": summary}, indent=2))

    except Exception as e:
        print(f"[WARN] Δbits summary skipped: {e}")

    # ---- Plotting (publication-friendly ΔL with physical filter) ----
    try:
        from matplotlib.ticker import MaxNLocator
        plt.rcParams.update({
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8
        })
        beta_alpha = math.log10(1/137.035999084)
        df_all = pd.read_csv(outdir/"mdl_sweep.csv").sort_values("beta")
        df = df_all.copy()
        Lmin = float(df["L_bits"].min())
        df["dL_bits"] = df["L_bits"] - Lmin

        # --- structure gain: sum_windowed_MDLS - global_MDL at each β ---
        # load windowed MDLs and sum across windows for each β
        win_path = outdir / "mdl_windowed.csv"
        if not win_path.exists():
            raise RuntimeError("mdl_windowed.csv not found — run with --windows enabled to compute structure gain.")

        win = pd.read_csv(win_path)
        # keep only betas that are in the physical df
        win = win.loc[win["beta"].isin(df["beta"])].copy()
        sum_win = (win.groupby("beta", as_index=False)["L_bits_win"]
                      .sum()
                      .rename(columns={"L_bits_win":"L_bits_win_sum"}))

        # merge with global MDL
        df = df.merge(sum_win, on="beta", how="left")

        # structure gain (more negative => more structure)
        df["G_struct"] = df["L_bits_win_sum"] - df["L_bits"]

        # Replace +/-inf with NaN, then drop any NaNs in the columns we plot
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["beta", "G_struct", "L_bits", "L_bits_win_sum"]
).copy()

        # Robust inlier band on G_struct (2nd–98th percentile) for aesthetics
        y_all = df["G_struct"].to_numpy()
        if y_all.size >= 3 and np.isfinite(y_all).any():
            ylo, yhi = np.nanpercentile(y_all, 2), np.nanpercentile(y_all, 98)
            mask_in  = (y_all >= ylo) & (y_all <= yhi)
        else:
            mask_in  = np.ones_like(y_all, dtype=bool)
        mask_out = ~mask_in

        def decorate(ax, zoom=False):
            ax.axvline(beta_alpha, ls="--", color="tab:blue", lw=1.0, alpha=0.9)
            # shaded band ±0.010 around log10 α (edit width as desired)
            ax.axvspan(beta_alpha - 0.010, beta_alpha + 0.010,
                       color="tab:blue", alpha=0.10, lw=0)
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$G(\beta) = \sum_w L_w - L_{\mathrm{global}}\;\;(\mathrm{bits})$")
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_title("zoom near log$_{10}\\,\\alpha$" if zoom else "full span (physical region)")

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,3.8), gridspec_kw={"wspace":0.28})

        # y-limits for the left panel based on G_struct
        y = df["G_struct"].to_numpy()
        if y.size and np.isfinite(y).any():
            ylo, yhi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
            if np.isfinite(ylo) and np.isfinite(yhi):
                pad = 0.06 * max(1e-9, (yhi - ylo))
                ax1.set_ylim(ylo - pad, yhi + pad)

        # LEFT: physical full span, ΔL (single stats box; no legend)
        series_label = (args.series_name or args.label)

        # main curve (inliers)
        ax1.plot(
            df["beta"][mask_in], df["G_struct"][mask_in],
            marker="o", ms=2.4, lw=1.0, color="tab:blue", alpha=0.95
        )
        if mask_out.any():
            ax1.scatter(
                df["beta"][mask_out], df["G_struct"][mask_out],
                marker="x", s=18, lw=0.8, color="0.55", alpha=0.70
            )

        decorate(ax1, zoom=False)
        if len(y):
            if len(y):
                pad = 0.06 * (yhi - ylo)
                ax1.set_ylim(ylo - pad, yhi + pad)

        # build compact stats box (acts as the only "legend")
        try:
            N = int(df["N"].iloc[0]) if "N" in df.columns else None
        except Exception:
            N = None
        try:
            dn0  = pd.read_csv(outdir/"mdl_null.csv") if (outdir/"mdl_null.csv").exists() else None
            Kref = int(dn0["K_ref"].iloc[0]) if dn0 is not None else None
        except Exception:
            Kref = None

        stats_lines = [series_label]
        if N is not None:    stats_lines.append(f"N bins (0..κmax): {N}")
        if Kref is not None: stats_lines.append(f"K_ref (null @ α): {Kref}")
        stats_lines.append(f"grid Δκ: {args.grid:g}, κmax: {args.kmax:g}")
        if mask_out.any():   stats_lines.append(f"outliers: {int(mask_out.sum())}")

        ax1.text(
            0.02, 0.98, "\n".join(stats_lines),
            transform=ax1.transAxes, ha="left", va="top",
            fontsize=7.5, color="0.25",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8")
        )

        # RIGHT: tight zoom around α, structure gain G(β) (no legend)
        z_lo, z_hi = beta_alpha - 0.024, beta_alpha + 0.024
        dfz = df[(df["beta"] >= z_lo) & (df["beta"] <= z_hi)].copy()

        # plot G_struct in the zoom window
        ax2.plot(
            dfz["beta"], dfz["G_struct"],
            marker="o", ms=2.4, lw=1.0, color="tab:blue"
        )

        # vertical α line and a light band for the zoom window
        ax2.axvline(beta_alpha, ls="--", lw=1.0, color="tab:blue", alpha=0.7)
        ax2.axvspan(z_lo, z_hi, color="tab:blue", alpha=0.08)

        # y-limits sized to the local range of G_struct
        if not dfz.empty and np.isfinite(dfz["G_struct"]).any():
            y_min = float(np.nanmin(dfz["G_struct"]))
            y_max = float(np.nanmax(dfz["G_struct"]))
            if np.isfinite(y_min) and np.isfinite(y_max):
                ypad  = (y_max - y_min) * 0.15 + 0.1
                if ypad <= 0: ypad = 0.5
                ax2.set_ylim(y_min - ypad, y_max + ypad)

        # OPTIONAL: label the interior minimum β* of G(β) if present
        if not dfz.empty:
            idx_min = dfz["G_struct"].idxmin()
            bx, by = float(dfz.at[idx_min, "beta"]), float(dfz.at[idx_min, "G_struct"])
            ax2.annotate(
                r"$\beta^\ast$", xy=(bx, by), xytext=(0.02, 0.10),
                textcoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"),
                arrowprops=dict(arrowstyle="->", color="0.4"),
                fontsize=8
            )

        decorate(ax2, zoom=True)

        # annotate β* near α if available
        eff_path = outdir / "mdl_effect_near_alpha.json"
        if eff_path.exists():
            eff = json.loads(eff_path.read_text())
            ax2.annotate(
                (r"$\beta^{\ast}$ = %.6f\n$\Delta L_{\min}$ = %.1f" % (eff["best_beta_near_alpha"], 0.0)),
                xy=(eff["best_beta_near_alpha"], 0.0),
                xytext=(0.02, 0.08), textcoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"),
                arrowprops=dict(arrowstyle="->", color="0.4"),
                fontsize=7.5, ha="left", va="bottom", color="0.2"
            )

        fig.suptitle(f"{args.label}: structure gain  G(β) = Σ_w L_w − L_global", y=0.98, fontsize=11.5)
        fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
        fig.savefig(outdir / "L_vs_beta_full+zoom.png", dpi=220, bbox_inches="tight")
        fig.savefig(outdir / "L_vs_beta_full+zoom.svg", bbox_inches="tight")

    except Exception as e:
        print(f"[WARN] plot skipped: {e}")

if __name__ == "__main__":
    main()