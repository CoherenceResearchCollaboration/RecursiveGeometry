#!/usr/bin/env python3
"""
🌞 For the Sun (FluxAtlas) dataset
PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --outdir results/solar/mdla \
  --label sun_fluxatl_all \
  --series-name "Sun (FluxAtlas, photons)" \
  --grid 0.002 --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50

🌟 For the Vega (ELODIE) dataset
PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/stars/raw_csv/vega_all_photons.csv \
  --outdir results/stars/mdla \
  --label vega_elodie_all \
  --series-name "Vega (ELODIE, photons)" \
  --grid 0.002 \
  --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'upper right'

Neon (lamp):

PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/lamps/raw_csv/neon_neI_ritz_vac.csv \
  --outdir      results/lamps/mdla/neon_neI_ritz_vac \
  --label       neon_neI_ritz_vac \
  --series-name "Ne I (Ritz, vacuum)" \
  --grid 0.002 \
  --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'


# Na I (Ritz, vacuum)
PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/lamps/raw_csv/na_I_ritz_vac.csv \
  --outdir      results/lamps/mdla/na_I_ritz_vac \
  --label       na_I_ritz_vac \
  --series-name "Na I (Ritz, vacuum)" \
  --grid 0.002 --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'

# Hg II (Ritz, vacuum)
PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/lamps/raw_csv/hg_II_ritz_vac.csv \
  --outdir      results/lamps/mdla/hg_II_ritz_vac \
  --label       hg_II_ritz_vac \
  --series-name "Hg II (Ritz, vacuum)" \
  --grid 0.002 --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'

Molecule: C2 swan (no adapter step required)

PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv data/molecules/C2_Swan_visible.csv \
  --outdir      results/molecules/mdla/C2_Swan_visible \
  --label       C2_Swan_visible \
  --series-name "C₂ Swan bands (visible)" \
  --grid 0.002 --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50 \
  --legend-loc 'center right'


"""

import argparse, json, math, os, subprocess, tempfile, csv
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, numpy as np

def code_length_bernoulli(n, k):
    # fast approx: Stirling for log2(n choose k) + Bernoulli part
    # exact is fine too for n<=5000; here we do exact via Python math.comb if available.
    from math import comb, log2
    k = max(0, min(k, n))
    p = max(1e-6, min(1-1e-6, k/float(n)))
    logC = log2(comb(n,k)) if 0 <= k <= n else float("inf")
    return logC + (-k*math.log2(p) - (n-k)*math.log2(1-p))

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

    # MDL sweep
    rows=[]
    N = int(round(args.kmax/args.grid))  # bins in 0..kmax
    for b in beta_list:
        tmpdir = Path(tempfile.mkdtemp())
        # build barcode at beta b
        subprocess.check_call([
            "python","-m","scripts.MDL.threadlaw_photoncode",
            "--label", args.label,
            "--csv", args.events_csv,
            "--outdir", str(tmpdir),
            "--grid", str(args.grid),
            "--beta-override", str(b)
        ])
        # threadlaw_photoncode writes under <tmp>/<label>/barcode_dense.csv
        bc = (tmpdir / args.label / "barcode_dense.csv")
        if not bc.exists():
            # fallback: any *barcode_dense*.csv under the label dir
            cand = list((tmpdir / args.label).glob("*barcode_dense*.csv"))
            if not cand:
                raise SystemExit(f"barcode_dense not found under {tmpdir/args.label}")
            bc = cand[0]
        # count ones within 0..kmax
        import pandas as pd
        df = pd.read_csv(bc)
        col = ("kappa_int_rel" if "kappa_int_rel" in df.columns
               else "kappa_int" if "kappa_int" in df.columns
               else None)
        if col is None: raise SystemExit("no kappa_int/_rel column in barcode_dense")
        present = df.loc[df.get("present",1)>0, col].astype(int)
        sel = present[(present>=0) & (present<=N)].to_numpy()
        k = int(sel.size)
        L = code_length_bernoulli(N, k)
        # build a dense 0/1 vector over 0..N for diagnostics
        bits = np.zeros(N+1, dtype=int)
        bits[sel] = 1
        diag_edge = edge_concentration(sel, N, tail_bins=10)
        diag_run  = longest_run_ones(bits)
        rows.append({"beta": b, "N": N, "K": k, "L_bits": L,
                     "edge_conc_last10": diag_edge, "runlen_max": diag_run})
        # stash optional artifact
        (outdir / f"beta_{b:+.6f}_K{k}.txt").write_text(json.dumps({"beta":b,"K":k,"N":N,"L_bits":L}, indent=2))

    # write sweep CSV
    with open(outdir/"mdl_sweep.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["beta","N","K","L_bits","edge_conc_last10","runlen_max"])
        w.writeheader(); w.writerows(rows)

    # identify argmin
    best = min(rows, key=lambda r: r["L_bits"])
    Path(outdir/"mdl_best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps({"best":best, "out":str(outdir/"mdl_sweep.csv")}, indent=2))

    # ---------- simple outlier report (2nd–98th percentile band) ----------
    try:
        df_all = pd.read_csv(outdir/"mdl_sweep.csv").sort_values("beta")
        y = df_all["L_bits"].to_numpy()
        if y.size:
            ylo, yhi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
            mask_out = (y < ylo) | (y > yhi)
            if mask_out.any():
                df_all.loc[mask_out, ["beta","N","K","L_bits","edge_conc_last10","runlen_max"]].to_csv(
                    outdir/"mdl_outliers.csv", index=False)
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
                "python","-m","scripts.MDL.threadlaw_photoncode",
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
                idx = present[(present>=i0) & (present<=i1)]
                Nw  = max(0, i1 - i0 + 1)
                Kw  = int(idx.size)
                Lw  = code_length_bernoulli(Nw, Kw) if Nw>0 else float("nan")
                out_rows.append({"beta": b, "win": f"{fa:.2f}-{fb:.2f}",
                                 "N_win": Nw, "K_win": Kw, "L_bits_win": Lw})

        with open(outdir/"mdl_windowed.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=["beta","win","N_win","K_win","L_bits_win"])
            w.writeheader(); w.writerows(out_rows)
    except Exception as e:
        print(f"[WARN] windowed MDL skipped: {e}")

    # ---------- Band-split MDL (fluxatl vs photatl) ----------
    try:
        ev = pd.read_csv(args.events_csv)
        if "source_path" in ev.columns and args.bandsplit:
            def _subset_L(beta_val, mask):
                tmp = Path(tempfile.mkdtemp())
                sub = ev.loc[mask].copy()
                spath = tmp/"events_subset.csv"; sub.to_csv(spath, index=False)
                subprocess.check_call([
                    "python","-m","scripts.MDL.threadlaw_photoncode",
                    "--label", args.label, "--csv", str(spath), "--outdir", str(tmp),
                    "--grid", str(args.grid), "--beta-override", str(beta_val)
                ])
                bc = (tmp / args.label / "barcode_dense.csv")
                dfb = pd.read_csv(bc)
                col = ("kappa_int_rel" if "kappa_int_rel" in dfb.columns
                       else "kappa_int" if "kappa_int" in dfb.columns else None)
                present = dfb.loc[dfb.get("present",1)>0, col].astype(int)
                Nloc = int(round(args.kmax/args.grid))
                Kloc = int(present[(present>=0)&(present<=Nloc)].count())
                return Nloc, Kloc, code_length_bernoulli(Nloc, Kloc)

            flux_m = ev["source_path"].str.contains("/fluxatl/", na=False)
            phot_m = ev["source_path"].str.contains("/photatl/", na=False)
            out_rows = []
            for r in rows:
                b = r["beta"]
                Nf,Kf,Lf = _subset_L(b, flux_m)
                Np,Kp,Lp = _subset_L(b, phot_m)
                out_rows.append({"beta": b,
                                 "N_flux": Nf, "K_flux": Kf, "L_bits_flux": Lf,
                                 "N_phot": Np, "K_phot": Kp, "L_bits_phot": Lp})

            with open(outdir/"mdl_bandsplit.csv","w",newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "beta","N_flux","K_flux","L_bits_flux","N_phot","K_phot","L_bits_phot"
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
        import pandas as pd
        tmpdir = Path(tempfile.mkdtemp())
        subprocess.check_call([
            "python","-m","scripts.MDL.threadlaw_photoncode",
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
            for b in beta_list:
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

        # --- restrict to the physical region (avoid shallow-slope artifact) ---
        df = df_all.loc[df_all["beta"] <= -0.8].copy()
        if df.empty:
            df = df_all.copy()  # fallback: plot whatever we have

        # --- compute ΔL relative to the min within the physical region ---
        Lmin = float(df["L_bits"].min()) if not df.empty else float("nan")
        df["dL_bits"] = df["L_bits"] - Lmin

        # optional null (flat, for visual reference) – compute Δ to match ΔL scale
        dn = None
        null_path = outdir/"mdl_null.csv"
        if null_path.exists():
            dn_raw = pd.read_csv(null_path).sort_values("beta")
            dn = dn_raw.loc[dn_raw["beta"].isin(df["beta"])].copy()
            if not dn.empty:
                dn["dL_bits_null"] = dn["L_bits_null"] - Lmin

        def decorate(ax, zoom=False):
            ax.axvline(beta_alpha, ls="--", color="tab:blue", lw=1.0, alpha=0.9)
            # shaded band ±0.010 around log10 α (edit width as desired)
            ax.axvspan(beta_alpha - 0.010, beta_alpha + 0.010,
                       color="tab:blue", alpha=0.10, lw=0)
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel(r"$\Delta L$ (bits)")
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_title("zoom near log$_{10}\\,\\alpha$" if zoom else "full span (physical region)")

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,3.8), gridspec_kw={"wspace":0.28})

        # --- robust inlier band on ΔL (within physical region) ---
        y = df["dL_bits"].to_numpy()
        if len(y):
            ylo, yhi = np.nanpercentile(y, 2), np.nanpercentile(y, 98)
            mask_in = (y >= ylo) & (y <= yhi)
        else:
            mask_in = np.ones_like(y, dtype=bool)
        mask_out = ~mask_in

        # LEFT: physical full span, ΔL (single stats box; no legend)
        series_label = (args.series_name or args.label)

        # main curve (inliers)
        ax1.plot(
            df["beta"][mask_in], df["dL_bits"][mask_in],
            marker="o", ms=2.4, lw=1.0, color="tab:blue", alpha=0.95
        )

        # visible outliers (no legend entry)
        if mask_out.any():
            n_out = int(mask_out.sum())
            ax1.scatter(
                df["beta"][mask_out], df["dL_bits"][mask_out],
                marker="x", s=18, lw=0.8, color="0.55", alpha=0.70
            )

        # density-preserving null (no legend entry)
        if dn is not None and not dn.empty:
            ax1.plot(dn["beta"], dn["dL_bits_null"], lw=1.0, color="tab:orange", alpha=0.95)

        decorate(ax1, zoom=False)
        if len(y):
            pad = 0.06 * (yhi - ylo)
            ax1.set_ylim(max(0, ylo - pad), yhi + pad)  # ΔL is non-negative

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

        # RIGHT: tight zoom around α, ΔL (no legend)
        z_lo, z_hi = beta_alpha - 0.024, beta_alpha + 0.024
        dfz = df[(df["beta"] >= z_lo) & (df["beta"] <= z_hi)]

        ax2.plot(dfz["beta"], dfz["dL_bits"], marker="o", ms=2.4, lw=1.0, color="tab:blue")
        if dn is not None and not dn.empty:
            dnz = dn[(dn["beta"] >= z_lo) & (dn["beta"] <= z_hi)]
            if not dnz.empty:
                ax2.plot(dnz["beta"], dnz["dL_bits_null"], lw=1.0, color="tab:orange", alpha=0.95)

        ax2.set_xlim(z_lo, z_hi)
        if not dfz.empty:
            ypad = (dfz["dL_bits"].max() - dfz["dL_bits"].min()) * 0.15 + 0.5
            ax2.set_ylim(0, max(0.5, dfz["dL_bits"].max() + ypad))
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

        fig.suptitle(f"{args.label}: ΔL(β)", y=0.98, fontsize=11.5)
        fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
        fig.savefig(outdir / "L_vs_beta_full+zoom.png", dpi=220, bbox_inches="tight")
        fig.savefig(outdir / "L_vs_beta_full+zoom.svg", bbox_inches="tight")

    except Exception as e:
        print(f"[WARN] plot skipped: {e}")

if __name__ == "__main__":
    main()