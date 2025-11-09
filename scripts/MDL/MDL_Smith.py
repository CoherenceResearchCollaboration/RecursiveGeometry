# scripts/analysis_pipeline/MDL_Smith.py
"""
Version 2.2_broken_tables

# per-dataset sweeps (you already produce these with mdla_sweep.py)
python -m scripts.analysis_pipeline.MDL_Smith \
  --sweep MDLA_results/lamps/mdla/na_I_ritz_vac/mdl_sweep.csv \
         MDLA_results/lamps/mdla/neon_neI_ritz_vac/mdl_sweep.csv \
         MDLA_results/lamps/mdla/hg_II_ritz_vac/mdl_sweep.csv \
         MDLA_results/molecules/mdla/C2_Swan_visible/mdl_sweep.csv \
         MDLA_results/solar/mdla/mdl_sweep.csv \
         MDLA_results/stars/mdla/mdl_sweep.csv \
  --labels na_I neon_I hg_II C2_swan solar vega \
  --outdir MDLA_results/mdl_smith

"""

import argparse, json, math, pathlib, os, pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_derive(path):
    df = pd.read_csv(path).reset_index(drop=True)
    # force numerics so refine rows aren’t dropped
    df["beta"]   = pd.to_numeric(df.get("beta"),   errors="coerce")
    df["L_bits"] = pd.to_numeric(df.get("L_bits"), errors="coerce")
    if "beta" not in df.columns or "L_bits" not in df.columns:
        raise SystemExit(f"[ERR] {path} missing beta/L_bits columns.")
    if "N" in df.columns and pd.notna(df["N"].iloc[0]):
        N = int(df["N"].iloc[0])
    else:
        raise SystemExit(f"[ERR] {path} missing N (bins). Please include N in mdl_sweep.csv.")

    df = df[np.isfinite(df["beta"]) & np.isfinite(df["L_bits"])].copy()
    # dedupe duplicate betas: keep minimal L_bits at that beta
    df = (df.sort_values(["beta","L_bits"])
            .groupby("beta", as_index=False, sort=True)["L_bits"].min()
            .sort_values("beta").reset_index(drop=True))

    # ΔL
    L = df["L_bits"].to_numpy()
    DeltaL = L - L.min()
    DeltaL[DeltaL < 0] = 0.0
    df["DeltaL"] = DeltaL

    # helpers
    beta_alpha = math.log10(1/137.035999084)

    def plateau_center(xb, yb, center_pref):
        Lmin = float(yb.min())
        tied = xb[np.isclose(yb, Lmin, atol=1e-12)]
        if tied.size == 1:
            return float(tied[0])
        # nearest to center_pref; if several, take midpoint of that cluster
        j = int(np.argmin(np.abs(tied - center_pref)))
        near = tied[np.isclose(np.abs(tied - center_pref),
                               np.abs(tied[j] - center_pref), atol=1e-15)]
        return float(0.5*(near.min() + near.max()))

    # ---------- (A) GLOBAL MDL β* (free minimum over full domain) ----------
    beta_star_global = plateau_center(df["beta"].to_numpy(), df["DeltaL"].to_numpy(), beta_alpha)

    # ---------- (B) α-ANCHORED MDL β* (local minimum near α) ----------
    # narrow physical window; spline-refine if SciPy available, else plateau center
    phys = df[(df["beta"] >= beta_alpha - 0.03) & (df["beta"] <= beta_alpha + 0.03)]
    if phys.empty:
        phys = df.copy()
    try:
        from scipy.interpolate import UnivariateSpline
        xb = phys["beta"].to_numpy(); yb = phys["DeltaL"].to_numpy()
        # robust: require strictly increasing x for spline
        m = np.concatenate(([True], np.diff(xb) > 0))
        xb, yb = xb[m], yb[m]
        s = UnivariateSpline(xb, yb, k=3, s=0)
        roots = s.derivative(1).roots()
        if len(roots) > 0:
            # pick the root inside window that minimizes ΔL
            cand = np.array([r for r in roots if xb.min()-1e-9 <= r <= xb.max()+1e-9])
            if cand.size:
                vals = s(cand)
                beta_star_alpha = float(cand[np.argmin(vals)])
            else:
                beta_star_alpha = plateau_center(xb, yb, beta_alpha)
        else:
            beta_star_alpha = plateau_center(xb, yb, beta_alpha)
    except Exception:
        beta_star_alpha = plateau_center(phys["beta"].to_numpy(), phys["DeltaL"].to_numpy(), beta_alpha)

    # --- (A′) Global-anchored Smith mapping (use N* at β*_global) ----------

    i_match_g = np.where(np.isclose(df["beta"].to_numpy(), beta_star_global, atol=1e-12))[0]
    i_star_g  = int(i_match_g[0]) if len(i_match_g) else int(np.argmin(np.abs(df["beta"].to_numpy()-beta_star_global)))
    if "N_eff" in df.columns and pd.notna(df["N_eff"].iloc[i_star_g]) and (df["N_eff"].iloc[i_star_g] > 0):
        N_star_global = float(df["N_eff"].iloc[i_star_g])
    else:
        N_star_global = float(N)

    df_global_map = df.copy()
    df_global_map["delta_ell"] = df_global_map["DeltaL"] / N_star_global
    Gg = np.sqrt(1.0 - np.power(2.0, -df_global_map["delta_ell"].to_numpy()))
    df_global_map["Gamma_abs"] = np.clip(Gg, 0.0, 1.0 - 1e-9)
    df_global_map["z_info"]    = (1.0 + df_global_map["Gamma_abs"]) / (1.0 - df_global_map["Gamma_abs"])

    # --- α-anchored mapping (build AFTER global so we can return both) ----------
    i_match = np.where(np.isclose(df["beta"].to_numpy(), beta_star_alpha, atol=1e-12))[0]
    i_star  = int(i_match[0]) if len(i_match) else int(np.argmin(np.abs(df["beta"].to_numpy() - beta_star_alpha)))
    if "N_eff" in df.columns and pd.notna(df["N_eff"].iloc[i_star]) and (df["N_eff"].iloc[i_star] > 0):
        N_star = float(df["N_eff"].iloc[i_star])
    else:
        N_star = float(N)

    df_alpha = df.copy()
    df_alpha["delta_ell"] = df_alpha["DeltaL"] / N_star
    Gamma_abs = np.sqrt(1.0 - np.power(2.0, -df_alpha["delta_ell"].to_numpy()))
    df_alpha["Gamma_abs"] = np.clip(Gamma_abs, 0.0, 1.0 - 1e-9)
    df_alpha["z_info"] = (1.0 + df_alpha["Gamma_abs"]) / (1.0 - df_alpha["Gamma_abs"])

    # curvature A at β*_alpha
    # Use a symmetric β-span around the center to avoid sampling bias
    beta_arr   = df_alpha["beta"].to_numpy()
    DeltaL_arr = df_alpha["DeltaL"].to_numpy()
    # local step estimate
    dl = (beta_arr[i_star]   - beta_arr[i_star-1]) if i_star > 0 else np.nan
    dr = (beta_arr[i_star+1] - beta_arr[i_star])   if i_star+1 < len(beta_arr) else np.nan
    # choose a small symmetric span based on local resolution
    h_local = np.nanmin([abs(dl), abs(dr), np.nanmedian(np.diff(beta_arr))])
    span = 5.0 * h_local if np.isfinite(h_local) else 0.005
    span = float(np.clip(span, 0.003, 0.020))  # guardrails
    m = (beta_arr >= beta_star_alpha - span) & (beta_arr <= beta_star_alpha + span)
    xb = beta_arr[m] - beta_star_alpha
    yb = DeltaL_arr[m]
    # require both sides and >= 3 points
    if (xb.size >= 3) and np.any(xb <= 0) and np.any(xb >= 0):
        # normalize abscissa to improve conditioning; rescale curvature by s^2
        s = np.max(np.abs(xb));  s = 1.0 if (not np.isfinite(s) or s == 0.0) else s
        x = xb / s
        try:
            coef = np.polyfit(x, yb, 2)
            A = 2.0 * coef[0] / (s*s)
        except Exception:
            A = float("nan")
    else:
        A = float("nan")
    # fallback: 3-point second derivative at the nearest triplet (nonuniform formula)
    if (not np.isfinite(A)) or (A <= 0):
        i0 = int(np.argmin(np.abs(beta_arr - beta_star_alpha)))
        if 0 < i0 < len(beta_arr) - 1:
            xl, x0, xr = beta_arr[i0-1], beta_arr[i0], beta_arr[i0+1]
            yl, y0, yr = DeltaL_arr[i0-1], DeltaL_arr[i0], DeltaL_arr[i0+1]
            hl, hr = (x0 - xl), (xr - x0)
            if (hl > 0) and (hr > 0):
                A_fd = 2.0*( yl/((xl-x0)*(xl-xr)) + y0/((x0-xl)*(x0-xr)) + yr/((xr-xl)*(xr-x0)) )
                if np.isfinite(A_fd) and (A_fd > 0):
                    A = A_fd

    # two-sided one-bit width at ΔL = N* (robust to incomplete crossings)
    xs = df_alpha["beta"].to_numpy(); ys = df_alpha["DeltaL"].to_numpy()
    def crossing_both(xs, ys, y0, beta_star):

        # find first crossing to the left and right; return (left, right) half-widths
        def _first_cross(start, stop, step):
            for i in range(start, stop, step):
                j = i + step
                if j < 0 or j >= len(xs): break
                x1, x2 = xs[i], xs[j]; y1, y2 = ys[i], ys[j]
                if (y1 - y0) * (y2 - y0) <= 0 and x1 != x2:
                    t = 0.0 if (y2 == y1) else (y0 - y1)/(y2 - y1)
                    return abs((x1 + t*(x2 - x1)) - beta_star)
            return float("nan")
        i0 = int(np.argmin(np.abs(xs - beta_star)))
        left  = _first_cross(i0, 0, -1)
        right = _first_cross(i0, len(xs)-1, +1)

        return left, right

    ymax = np.nanmax(ys) if np.size(ys) else float("nan")
    if not (np.isfinite(ymax) and ymax > 0):
        wL = wR = float("nan")
        halfpow_width = float("nan")
        onebit_y_target = float("nan")
        onebit_target_clipped = False
    else:
        onebit_y_target = min(N_star, ymax)
        onebit_target_clipped = (onebit_y_target < N_star)
        wL, wR = crossing_both(xs, ys, onebit_y_target, beta_star_alpha)
        finite = [w for w in (wL, wR) if np.isfinite(w)]
        halfpow_width = (min(finite) if finite else float("nan"))

    # build metrics dict now (metrics exists!)
    metrics = {
        "beta_star_global": beta_star_global,
        "beta_star_alpha":  beta_star_alpha,
        "delta_beta":       beta_star_global - beta_star_alpha,
        "curvature_A":      A,
        "onebit_halfpower_width":       halfpow_width,
        "onebit_halfpower_width_left":  wL,
        "onebit_halfpower_width_right": wR,
        "onebit_y_target":              onebit_y_target,
        "onebit_target_clipped":        onebit_target_clipped,
        "N": N,
        "N_eff_star": N_star,
        "N_eff_star_global": N_star_global
    }

    # return both mappings + metrics
    return df_alpha, df_global_map, metrics

# ---------------- Helper metrics (module scope) ----------------

def _nearest_index(x: np.ndarray, x0: float) -> int:
    """Index of x closest to x0 (ties resolve to the first min)."""
    return int(np.argmin(np.abs(x - x0)))

def _local_quadratic_curvature(beta: np.ndarray, dL: np.ndarray, center: float, half_window: int = 3) -> float:
    """
    Fit ΔL(β) ≈ a (β-β0)^2 + b (β-β0) + c on a small window around β0=center.
    Return the second derivative at the center: A = d²/dβ² ΔL |_{β0} = 2a.
    """
    if beta is None or dL is None or len(beta) < 5 or len(beta) != len(dL):
        return float("nan")
    i0 = _nearest_index(beta, center)
    i_lo = max(0, i0 - half_window)
    i_hi = min(len(beta), i0 + half_window + 1)
    xb = beta[i_lo:i_hi] - center
    yb = dL[i_lo:i_hi]
    if len(xb) < 5:
        return float("nan")
    try:
        # Normalize abscissa for conditioning; rescale curvature by s^2
        s = float(np.max(np.abs(xb))) if np.isfinite(np.max(np.abs(xb))) and np.max(np.abs(xb)) != 0 else 1.0
        coef = np.polyfit(xb / s, yb, 2)  # coef[0] = a
        A = 2.0 * coef[0] / (s * s)
        return float(A)
    except Exception:
        return float("nan")

def _onebit_halfwidth(beta: np.ndarray,
                      dL: np.ndarray,
                      center: float,
                      N_star: float,
                      clip_to_max: bool = True,
                      y_target: float | None = None) -> float:
    """
    Return the smallest positive |β - center| where ΔL(β) = y0.
    y0 defaults to N*; if clip_to_max is True, uses y0 = min(N*, max(ΔL)) to avoid NaN when
    the sweep never reaches N*. If y_target is provided, uses y0 = min(y_target, N*, max(ΔL)).
    """
    if beta is None or dL is None or len(beta) < 2 or len(beta) != len(dL) or not np.isfinite(N_star):
        return float("nan")

    # choose target y0 (ΔL level)
    y0 = float(N_star)
    if clip_to_max or (y_target is not None):
        ymax = np.nanmax(dL) if len(dL) else float("nan")
        if np.isfinite(ymax):
            if y_target is not None and np.isfinite(y_target):
                y0 = min(float(y_target), y0, ymax)
            else:
                y0 = min(y0, ymax)

    g = dL - y0
    i0 = _nearest_index(beta, center)

    def _cross_from(i_start: int, step: int) -> float:
        i = i_start
        prev_y = g[i]; prev_b = beta[i]
        while 0 <= i + step < len(beta):
            i_next = i + step
            cur_y = g[i_next]; cur_b = beta[i_next]
            if prev_y == 0.0:
                return abs(prev_b - center)
            if (prev_y * cur_y) < 0.0 and cur_b != prev_b:
                # linear interpolation between (prev_b, prev_y) and (cur_b, cur_y)
                t = (0.0 - prev_y) / (cur_y - prev_y)
                b_cross = prev_b + t * (cur_b - prev_b)
                return abs(b_cross - center)
            i, prev_y, prev_b = i_next, cur_y, cur_b
        return float("nan")

    left  = _cross_from(i0, -1)
    right = _cross_from(i0, +1)

    candidates = [w for w in (left, right) if np.isfinite(w)]
    return float(min(candidates)) if candidates else float("nan")

def plot_realaxis_smith(df, outpdf, title):
    plt.figure(figsize=(4.2,2.8), dpi=180)
    # draw real axis ticks for z in [0.5, 2, 5], mark z=1
    z = df["z_info"].values
    b = df["beta"].values
    plt.scatter(z, np.zeros_like(z), c=b, cmap="viridis", s=10, alpha=0.85, edgecolors="none")
    plt.axvline(1.0, color="tab:red", lw=1.0, ls="--", alpha=0.9)
    plt.xlabel(r"$z_{\rm info}$ (real, normalized)")
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpdf, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Build Information–Smith metrics/plots from mdl_sweep.csv")
    ap.add_argument("--sweep", nargs="+", required=True, help="One or more mdl_sweep.csv files (per dataset).")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for datasets (same length as --sweep).")
    ap.add_argument("--outdir", required=True, help="Output directory for per-dataset CSV/JSON/PDF and envelope.")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    per_rows = []
    envelope_rows = []
    for idx, sweep in enumerate(args.sweep):
        label = (args.labels[idx] if args.labels and idx < len(args.labels)
                 else Path(sweep).parent.name)
        base = outdir / label
        base.mkdir(parents=True, exist_ok=True)

        df_alpha, df_global, metrics = load_and_derive(sweep)

        # α-anchored per-dataset outputs
        df_alpha.to_csv(base / "smith_per_dataset_alpha.csv", index=False)
        plot_realaxis_smith(df_alpha, base / "smith_locus_alpha.pdf", f"{label}: α-anchored locus")

        # global-anchored per-dataset outputs
        df_global.to_csv(base / "smith_per_dataset_global.csv", index=False)
        plot_realaxis_smith(df_global, base / "smith_locus_global.pdf", f"{label}: global locus")

        # envelope rows (α-anchored centering)
        df_env_a = df_alpha.loc[np.isfinite(df_alpha["Gamma_abs"]), ["beta","Gamma_abs"]].copy()
        df_env_a["beta_centered"] = df_env_a["beta"] - metrics["beta_star_alpha"]
        df_env_a["which"] = "alpha"; df_env_a["dataset"] = label
        envelope_rows.append(df_env_a)

        # envelope rows (global-anchored centering)
        df_env_g = df_global.loc[np.isfinite(df_global["Gamma_abs"]), ["beta","Gamma_abs"]].copy()
        df_env_g["beta_centered"] = df_env_g["beta"] - metrics["beta_star_global"]
        df_env_g["which"] = "global"; df_env_g["dataset"] = label
        envelope_rows.append(df_env_g)

        # ---- recompute anchor-specific metrics (inside loop) ----
        STRICT_ONEBIT = bool(int(os.environ.get("SMITH_STRICT_ONEBIT", "0")))
        # Use the correct effective code lengths for each anchoring
        N_star_a = metrics.get("N_eff_star", metrics["N"])
        N_star_g = metrics.get("N_eff_star_global", metrics["N"])
        beta_grid     = df_alpha["beta"].to_numpy()
        DeltaL_grid   = df_alpha["DeltaL"].to_numpy()
        curv_A_alpha  = _local_quadratic_curvature(beta_grid, DeltaL_grid, metrics["beta_star_alpha"])
        onebit_alpha  = _onebit_halfwidth(beta_grid, DeltaL_grid,
                                          metrics["beta_star_alpha"], N_star_a,
                                          clip_to_max=not STRICT_ONEBIT)

        beta_grid_g   = df_global["beta"].to_numpy()
        DeltaL_grid_g = df_global["DeltaL"].to_numpy()
        curv_A_global = _local_quadratic_curvature(beta_grid_g, DeltaL_grid_g, metrics["beta_star_global"])
        onebit_global = _onebit_halfwidth(beta_grid_g, DeltaL_grid_g,
                                          metrics["beta_star_global"], N_star_g,
                                          clip_to_max=not STRICT_ONEBIT)

        per_rows.append({
            "dataset": label,
            "beta_star_alpha":  metrics["beta_star_alpha"],
            "beta_star_global": metrics["beta_star_global"],
            "delta_beta":       metrics["beta_star_global"] - metrics["beta_star_alpha"],
            "curvature_A_alpha":  curv_A_alpha,
            "curvature_A_global": curv_A_global,
            "onebit_alpha":       onebit_alpha,
            "onebit_global":      onebit_global,
            "N": metrics["N"]
        })

    # save table for LaTeX
    pd.DataFrame(per_rows).to_csv(outdir / "smith_metrics_table.csv", index=False)

    # OPTIONAL: write LaTeX tabulars (alpha, global, compare)
    tbl = pd.DataFrame(per_rows)

    def esc_tex(s: str) -> str:
        """Escape underscores in dataset labels for LaTeX."""
        return str(s).replace("_", r"\_")

    def fmt_num(x, prec=6):
        """Format numeric values for siunitx S columns, else return '---'."""
        return (f"{float(x):.{prec}f}") if (pd.notna(x) and np.isfinite(x)) else r"---"
    # NOTE: per-table S-column guards use
    #   tbl['onebit_alpha'].isna().any() and tbl['onebit_global'].isna().any()

    # ---------------- Alpha-anchored table (β* near log10 α) ----------------
    # Column spec: l | S(-1.6) | S(7.6) | S(1.6 or c) | S(4.0)
    alpha_third_col = (r"c" if tbl['onebit_alpha'].isna().any() else r"S[table-format=1.6]")
    lines_alpha = [
        rf"\begin{{tabular}}{{@{{}}l S[table-format=-1.6] S[table-format=7.6] {alpha_third_col} S[table-format=4.0]@{{}}}}",
        r"\toprule",
        r"\textbf{Dataset} & {\boldmath$\beta^{\ast}_{\alpha}$} & {$A$} & {$\lvert\beta-\beta^{\ast}\rvert_{\Delta L=N}$} & {$N$} \\",
        r"\midrule"
    ]
    for _, r in tbl.iterrows():
        dataset = esc_tex(r["dataset"])
        bstar_a = fmt_num(r["beta_star_alpha"])
        A = fmt_num(r["curvature_A_alpha"])
        onebit = fmt_num(r["onebit_alpha"])
        Nval = str(int(r["N"])) if pd.notna(r["N"]) else r"---"
        # If third col is c and value is numeric, keep its text; siunitx not required
        lines_alpha.append(f"{dataset} & {bstar_a} & {A} & {onebit} & {Nval} \\\\")
    lines_alpha += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "smith_metrics_table_alpha.tex").write_text("\n".join(lines_alpha))

    # ---------------- Global (free) MDL table ----------------
    # This matches your requested format exactly:
    # l | S(-1.6) | S(7.6) | S(1.6 or c) | S(4.0)
    global_third_col = (r"c" if tbl['onebit_global'].isna().any() else r"S[table-format=1.6]")
    lines_global = [
        rf"\begin{{tabular}}{{@{{}}l S[table-format=-1.6] S[table-format=7.6] {global_third_col} S[table-format=4.0]@{{}}}}",
        r"\toprule",
        r"\textbf{Dataset} & {\boldmath$\beta^{\ast}_{\mathrm{glob}}$} & {$A$} & {$\lvert\beta-\beta^{\ast}\rvert_{\Delta L=N}$} & {$N$} \\",
        r"\midrule"
    ]
    for _, r in tbl.iterrows():
        dataset = esc_tex(r["dataset"])
        bstar_g = fmt_num(r["beta_star_global"])
        A = fmt_num(r["curvature_A_global"])
        onebit = fmt_num(r["onebit_global"])
        Nval = str(int(r["N"])) if pd.notna(r["N"]) else r"---"
        lines_global.append(f"{dataset} & {bstar_g} & {A} & {onebit} & {Nval} \\\\")
    lines_global += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "smith_metrics_table_global.tex").write_text("\n".join(lines_global))

    # ---------------- Compare table (shows Δβ) ----------------
    # l | S(-1.6) | S(-1.6) | S(-1.6) | S(4.0)
    lines_cmp = [
        r"\begin{tabular}{@{}l S[table-format=-1.6] S[table-format=-1.6] S[table-format=-1.6] S[table-format=4.0]@{}}",
        r"\toprule",
        r"\textbf{Dataset} & {\boldmath$\beta^{\ast}_{\mathrm{glob}}$} & {\boldmath$\beta^{\ast}_{\alpha}$} & {$\Delta\beta$} & {$N$} \\",
        r"\midrule"
    ]
    for _, r in tbl.iterrows():
        dataset = esc_tex(r["dataset"])
        bstar_g = fmt_num(r["beta_star_global"])
        bstar_a = fmt_num(r["beta_star_alpha"])
        d_beta  = fmt_num(r["delta_beta"])
        Nval = str(int(r["N"])) if pd.notna(r["N"]) else r"---"
        lines_cmp.append(f"{dataset} & {bstar_g} & {bstar_a} & {d_beta} & {Nval} \\\\")
    lines_cmp += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "smith_metrics_table_compare.tex").write_text("\n".join(lines_cmp))

    # envelope (median + IQR across datasets on a common β grid)
    if len(envelope_rows) >= 1:
        env = pd.concat(envelope_rows, ignore_index=True)

        def build_envelope(env_sub, out_csv, out_pdf, title):
            # Fixed, symmetric span around the center; identical for both anchorings.
            SPAN = float(os.environ.get("SMITH_SPAN", "0.04"))  # tweakable without code changes
            grid = np.linspace(-SPAN, +SPAN, 161)
            grids = []
            for label, df_g in env_sub.groupby("dataset"):
                x = df_g["beta_centered"].to_numpy(); y = df_g["Gamma_abs"].to_numpy()
                order = np.argsort(x); x, y = x[order], y[order]
                if len(x) >= 2:
                    mask = np.concatenate(([True], np.diff(x) > 0)); x, y = x[mask], y[mask]
                # Only include datasets that cover the full fixed span; avoids partial bias
                if len(x) >= 3 and (x.min() <= -SPAN) and (x.max() >= +SPAN):
                    yi = np.interp(grid, x, y)
                    grids.append(pd.DataFrame({"dataset": label,
                                               "beta_centered": grid,
                                               "Gamma_abs": yi}))
            if not grids: return
            G = pd.concat(grids, ignore_index=True)
            env_out = (
                G.groupby("beta_centered")["Gamma_abs"]
                 .agg(Gamma_med="median",
                      Gamma_q1=lambda s: s.quantile(0.25),
                      Gamma_q3=lambda s: s.quantile(0.75))
                 .reset_index()
            )
            env_out.to_csv(out_csv, index=False)
            plt.figure(figsize=(5.2, 3.2), dpi=180)
            plt.fill_between(env_out["beta_centered"], env_out["Gamma_q1"], env_out["Gamma_q3"],
                             color="0.85", edgecolor="none", label="IQR")
            plt.plot(env_out["beta_centered"], env_out["Gamma_med"], color="tab:blue", lw=1.6, label="median")
            plt.axvline(0.0, color="tab:red", lw=1.0, ls="--")
            plt.xlabel(r"$\beta-\beta^\ast$"); plt.ylabel(r"$|\Gamma_{\rm info}|$")
            plt.title(title); plt.legend(frameon=False); plt.tight_layout()
            plt.savefig(out_pdf, bbox_inches="tight"); plt.close()

        build_envelope(env[env["which"]=="alpha"],
                       outdir / "smith_envelope_alpha.csv",
                       outdir / "smith_envelope_alpha.pdf",
                       "Information–Impedance envelope (α-anchored)")
        build_envelope(env[env["which"]=="global"],
                       outdir / "smith_envelope_global.csv",
                       outdir / "smith_envelope_global.pdf",
                       "Information–Impedance envelope (global-anchored)")

    # quick envelope plot
    plt.figure(figsize=(5.6, 2.8), dpi=180)
    x = np.arange(len(tbl))
    y = tbl["delta_beta"].to_numpy()
    plt.axhline(0.0, color="0.7", lw=1)
    plt.stem(x, y, basefmt=" ")
    plt.xticks(x, tbl["dataset"].tolist(), rotation=0)
    plt.ylabel(r"$\Delta\beta\;(\beta^\ast_{\rm glob}-\beta^\ast_{\alpha})$")
    plt.title(r"Information–Impedance bias $\Delta\beta$")
    plt.tight_layout()
    plt.savefig(outdir / "smith_delta_beta_stem.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
