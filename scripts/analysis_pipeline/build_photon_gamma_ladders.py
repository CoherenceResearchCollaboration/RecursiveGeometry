#!/usr/bin/env python3
"""
build_photon_gamma_ladders.py — tag-aware
──────────────────────────────────────────
Extract photon emission ladders across gamma bins for each ion and attractor
site (n_i, n_k), writing to data/meta/ion_photon_ladders_<tag>/.

Updated August 21, 2025 to ensure that only NIST lines data is used (no circular arguments)

You can pass an overlay file OR just a tag and we resolve paths via path_config.

For the richest dataset, i.e. for mass estimation, keep all sites:

python -m scripts.analysis_pipeline.build_photon_gamma_ladders \
  --overlay   data/meta/photon_overlay_mu-1.csv \
  --gamma_bin 0.02 \
  --min_hits  1

Examples:
  # From explicit overlay
  python -m scripts.analysis_pipeline.build_photon_gamma_ladders \
    --overlay data/meta/photon_overlay_mu-1.csv --gamma_bin 0.02 --min_hits 1

  # From tag only (auto resolves overlay to CSV/Parquet)
  python -m scripts.analysis_pipeline.build_photon_gamma_ladders \
    --tag He_II --gamma_bin 0.02 --min_hits 1
"""
import re
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from scipy.constants import h, e  # Planck, elementary charge

# ── Project utilities ─────────────────────────────────────────────
from scripts.utils.io_helpers import load_overlay
from scripts.utils.provenance import write_csv_with_provenance
from scripts.utils.constants import apply_canonical_columns
from scripts.utils.path_config import META_DIR, get_paths

DEFAULT_GAMMA_BIN = 0.02
DEFAULT_MIN_HITS = 1

# Accepts _a3, _a3_2, _a3-02, _a3.200, /a3/02, etc., but requires a non-alnum
# boundary before 'a' so we don't match 'Na_I'
_ARE = re.compile(
    r"(?<![A-Za-z0-9])a(?P<int>\d+)(?:[._/\- ]?(?P<frac>\d+))?\b"
)

def _gamma_from_text(s: str | float | int | None) -> float | None:
    if s is None:
        return None
    s = str(s)
    m = _ARE.search(s)
    if not m:
        return None
    A = int(m.group("int"))
    frac = m.group("frac")
    if frac is None:
        return float(A)
    # scale by the number of digits present
    return A + (int(frac) / (10 ** len(frac)))

def _recover_gamma_from_column(series: pd.Series) -> pd.Series:
    # Work on strings; ignore lists/objects cleanly
    s = series.astype(str)
    g = s.map(_gamma_from_text)
    # Ensure float dtype; non-matches remain NaN
    return pd.to_numeric(g, errors="coerce")

def canon_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first present column name among candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ===== Audits (copy/paste) ==============================================
def audit_overlay_pre(df, gamma_bin):
    """Sanity checks right after gamma recovery + grid snap, before filtering/writing."""
    try:
        # 1) Did we accidentally match the 'a' in Na_I?
        if {"ion", "gamma_bin"}.issubset(df.columns):
            n_na = int(((df["ion"] == "Na_I") & df["gamma_bin"].notna()).sum())
            print(f"[audit] Na_I rows with gamma (should be legit): {n_na}")

        # 2) Off-grid check (tolerant)
        q = df["gamma_bin"].astype(float) / float(gamma_bin)
        ok = np.isclose(q, np.round(q), rtol=0, atol=1e-12)
        off_cnt = int((~ok).sum())
        print(f"[audit] off-grid gamma count: {off_cnt}")

        # 3) Distinct gamma bins per ion (top 10)
        if {"ion","gamma_bin"}.issubset(df.columns):
            gb = df.groupby("ion")["gamma_bin"].nunique().sort_values(ascending=False)
            print("[audit] top ions by distinct gamma bins:")
            print(gb.head(10).to_string())

        # 4) Duplicate row sanity (pre-filter)
        dupe_cols = [c for c in ["ion","n_i","n_k","gamma_bin","delta_e_ev"] if c in df.columns]
        if len(dupe_cols) >= 3:
            n_dupes = int(df.duplicated(subset=dupe_cols).sum())
            print(f"[audit] duplicates (pre-filter) by {dupe_cols}: {n_dupes}")
    except Exception as e:
        print(f"[audit] pre-checks error: {e}")

def audit_overlay_post(df_written_index_or_none, ladder_df_or_none):
    try:
        # Quick visibility on what we actually wrote
        if df_written_index_or_none:
            print(f"[post] wrote ladders for ions: {', '.join(map(str, df_written_index_or_none))}")
        if ladder_df_or_none is not None and not ladder_df_or_none.empty:
            print(f"[post] kept overlay rows: {len(ladder_df_or_none)}")
            if "frequency_source" in ladder_df_or_none.columns:
                print(ladder_df_or_none["frequency_source"].value_counts(dropna=False).to_string())
    except Exception as e:
        print(f"[post] audit error: {e}")

def infer_tag_from_filename(p: Path) -> str:
    """Attempt to infer overlay tag from filename like photon_overlay_mu-1.csv"""
    stem = p.stem  # filename without extension
    for prefix in ["photon_overlay_", "overlay_"]:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem  # fallback to the bare stem

# =======================================================================

def main(overlay, gamma_bin, min_hits, tag, allow_delta_e_fallback=False):
    overlay_path = Path(overlay) if overlay else None

    # Infer or validate tag
    if tag is None and overlay_path is not None:
        tag = infer_tag_from_filename(overlay_path)
    elif tag is None and overlay_path is None:
        raise SystemExit("[error] Provide either --tag or --overlay")

    # Resolve overlay path from tag if not provided
    if overlay_path is None:
        p = get_paths(tag)
        # prefer Parquet if present, else CSV
        overlay_path = p["overlay_parquet"]
        if not overlay_path.exists():
            overlay_path = p["overlay_csv"]
    if not overlay_path.exists():
        raise SystemExit(f"[error] Overlay not found: {overlay_path}")

    # Construct the output path
    out_dir = META_DIR / f"ion_photon_ladders_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_overlay(overlay_path)
    df = apply_canonical_columns(df)

    # ── PATCH START: recover gamma_bin & canonicalize n_i/n_k duplicates ─────────
    # resolve duplicate numeric columns (e.g., n_i / n_i.1)
    ni = canon_col(df, "n_i", "n_i.1")
    nk = canon_col(df, "n_k", "n_k.1")
    if ni and ni != "n_i":
        df = df.rename(columns={ni: "n_i"})
    if nk and nk != "n_k":
        df = df.rename(columns={nk: "n_k"})

    # Ensure n_i / n_k are stored as nullable integers
    for col in ("n_i", "n_k"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Ensure obs_hits is a clean integer count
    if "obs_hits" in df.columns:
        df["obs_hits"] = pd.to_numeric(df["obs_hits"], errors="coerce").fillna(0).astype(int)

    # Ensure the new per-γ counts are clean integers; backfill from obs_hits if needed
    for c in ("obs_hits_gamma", "n_hits_gamma"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Back-compat: if overlay lacks obs_hits_gamma but has obs_hits, alias it
    if "obs_hits_gamma" not in df.columns and "obs_hits" in df.columns:
        df["obs_hits_gamma"] = pd.to_numeric(df["obs_hits"], errors="coerce").astype("Int64")

    # basic overlay diagnostics
    n_rows = len(df)
    n_ions = df["ion"].nunique() if "ion" in df.columns else None
    print(f"[overlay] rows={n_rows} ions={n_ions}")

    # If gamma_bin is missing, try to recover from attractor_tag/power
    if "gamma_bin" not in df.columns or df["gamma_bin"].isna().all():
        print("[overlay] gamma_bin missing; trying to recover from attractor_tag or power…")
        if "attractor_tag" in df.columns:
            recovered = _recover_gamma_from_column(df["attractor_tag"])
            before = df["gamma_bin"].isna().sum() if "gamma_bin" in df.columns else None
            df["gamma_bin"] = recovered
            after  = df["gamma_bin"].isna().sum()
            print(f"[overlay] filled gamma_bin from attractor_tag; missing gamma_bin before={before} after={after}")
        if "power" in df.columns and df["gamma_bin"].isna().sum() > 0:
            # snap power to grid
            df.loc[df["gamma_bin"].isna(), "gamma_bin"] = (
                np.round(df.loc[df["gamma_bin"].isna(), "power"] / float(gamma_bin)) * float(gamma_bin)
            ).round(4)
            before = df["gamma_bin"].isna().sum()
            df["gamma_bin"] = df["gamma_bin"].astype(float)
            after = df["gamma_bin"].isna().sum()
            print(f"[overlay] snapped power to gamma_bin grid ({gamma_bin}); missing gamma_bin before={before} after={after}")

    has_power = "power" in df.columns
    has_de = "delta_e_ev" in df.columns
    print(f"[overlay] has power? {has_power} ; has delta_e_ev? {has_de}")
    # ── PATCH END ────────────────────────────────────────────────────────────────

    # Determine/derive gamma_bin
    if "gamma_bin" not in df.columns:
        if "power" in df.columns:
            df = df.dropna(subset=["power", "delta_e_ev"])
            df["gamma_bin"] = (np.round(df["power"] / gamma_bin) * gamma_bin).round(4)
        else:
            raise SystemExit("[error] Overlay missing 'gamma_bin' and 'power' columns.")
    df = df.dropna(subset=["gamma_bin", "delta_e_ev"])
    
    # --- NEW: prefer measured NIST wavelength for ν to avoid ΔE→ν circularity ---
    # We derive a preferred lambda (nm) in priority order:
    #  1) lambda_nist_match_nm_best (numeric, from process_photons overlay)
    #  2) first value in lambda_nist_match_nm (semicolon list)
    #  3) (optional fallback, only if --allow_delta_e_fallback) lambda_photon_nm from ΔE
    lam_pref = pd.to_numeric(df.get("lambda_nist_match_nm_best"), errors="coerce")
    if lam_pref.isna().all() and "lambda_nist_match_nm" in df.columns:
        # take the first match in the semicolon-separated list
        first = df["lambda_nist_match_nm"].astype(str).str.split(";").str[0]
        lam_pref = pd.to_numeric(first, errors="coerce")
    # Optional fallback: use ΔE-derived photon wavelength only if user allows it
    lam_from_de = None
    if allow_delta_e_fallback and "delta_e_ev" in df.columns:
        lam_from_de = (1239.841984 / pd.to_numeric(df["delta_e_ev"], errors="coerce"))  # nm
    # Combine choices
    if lam_from_de is None:
        lam_final = lam_pref
        # Use a pandas Series with string dtype to avoid dtype‑promotion errors
        lam_src = pd.Series(np.where(lam_pref.notna(), "nist", None), index=df.index, dtype="string")
    else:
        lam_final = lam_pref.copy()
        need = lam_final.isna()
        lam_final[need] = lam_from_de[need]
        lam_src = pd.Series("nist", index=df.index, dtype="string")
        lam_src[lam_pref.isna() & pd.notna(lam_from_de)] = "delta_e"

        df["lambda_photon_nm"] = lam_final.astype(float)
        df["frequency_hz"]     = 299_792_458.0 / (df["lambda_photon_nm"] * 1e-9)
        df["frequency_source"] = lam_src  # stays “string” dtype, no dtype errors

    # Drop any rows without a wavelength in strict mode
    before = len(df)
    df = df[lam_final.notna()].copy()
    after = len(df)
    if before != after:
        print(f"[filter] dropped {before-after} rows lacking NIST λ (strict={not allow_delta_e_fallback})")
    # Set canonical wavelength and frequency
    df["lambda_photon_nm"] = lam_final.astype(float)
    c_m_s = 299_792_458.0
    df["frequency_hz"] = c_m_s / (df["lambda_photon_nm"].astype(float) * 1e-9)
    df["frequency_source"] = lam_src

    # after df = apply_canonical_columns(df)
    print(f"[overlay] rows={len(df)} ions={df['ion'].nunique() if 'ion' in df.columns else 'n/a'}")
    missing_gamma = df["gamma_bin"].isna().sum() if "gamma_bin" in df.columns else None
    print(f"[overlay] has gamma_bin? {'gamma_bin' in df.columns} ; missing_gamma_bin={missing_gamma}")
  
    audit_overlay_pre(df, gamma_bin)

    # Group by ion, site, γ
    groups = df.groupby(["ion", "n_i", "n_k", "gamma_bin"], dropna=False)
    ladders = defaultdict(list)

    def _pick_const(s, name):
        if s is None or len(s) == 0:
            return np.nan
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return np.nan
        if s.nunique() > 1:
            print(f"[warn] {name} has {s.nunique()} distinct values within a group; using the first.")
        return int(s.iloc[0])

    for (ion, n_i, n_k, gamma), g in groups:
        # choose a single representative value per (ion, n_i, n_k, γ)
        obs_gamma = _pick_const(g["obs_hits_gamma"] if "obs_hits_gamma" in g.columns else None,
                                "obs_hits_gamma")
        n_gamma   = _pick_const(g["n_hits_gamma"]   if "n_hits_gamma"   in g.columns else None,
                                "n_hits_gamma")

        # Site-local evidence: how many photons matched this specific (n_i, n_k, γ) site
        n_photons_matched = int(len(g))

        # Filtering rule:
        # - Prefer per-γ context (obs_hits_gamma) when present (do not sum per-row values).
        # - Otherwise fall back to the site-local count.
        if pd.notna(obs_gamma):
            count_for_filter = int(obs_gamma)
        else:
            count_for_filter = n_photons_matched

        if count_for_filter < min_hits:
            continue

        ladders[ion].append({
            "ion": ion,
            "n_i": n_i,
            "n_k": n_k,
            "gamma_bin": gamma,
            "frequency_hz": g["frequency_hz"].mean(),
            "delta_e_ev": g["delta_e_ev"].mean(),
            # ✅ per-γ context (from gamma-affinity stage)
            "obs_hits_gamma": obs_gamma if pd.notna(obs_gamma) else None,
            "n_hits_gamma":   n_gamma   if pd.notna(n_gamma)   else None,
            # 🔎 site-local evidence for this ladder site
            "n_photons_matched": n_photons_matched,
            # ↩︎ explicit alias to avoid confusion with site-local counts
            "obs_hits_gamma_context": int(obs_gamma) if pd.notna(obs_gamma) else None,
            # new provenance
            "frequency_source": g["frequency_source"].mode().iat[0] if "frequency_source" in g.columns else None,
            "lambda_photon_nm": g["lambda_photon_nm"].mean() if "lambda_photon_nm" in g.columns else None,
        })

    # Write per-ion ladders
    for ion, rows in ladders.items():
        df_out = pd.DataFrame(rows)
        df_out = apply_canonical_columns(df_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ion}_photon_ladder.csv"
        write_csv_with_provenance(df_out, out_path)
    # ---------------------------------------------------

    # --- POST AUDIT (new) ---
    wrote_ions = sorted(ladders.keys())
    # df_keep: rows from the overlay that belong to ions we actually wrote
    df_keep = df[df["ion"].isin(wrote_ions)].copy() if "ion" in df.columns else None
    audit_overlay_post(df_written_index_or_none=wrote_ions, ladder_df_or_none=df_keep)
    # -------------------------

    print(f"[✓] Wrote {len(ladders)} ion ladder files → {out_dir}")

# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", default=None,
                        help="Photon overlay CSV/Parquet (if omitted, resolved from --tag)")
    parser.add_argument("--gamma_bin", type=float, default=DEFAULT_GAMMA_BIN,
                        help="Gamma bin resolution (default: 0.02)")
    parser.add_argument("--min_hits", type=int, default=DEFAULT_MIN_HITS,
                        help="Minimum photon hits per (n_i, n_k, γ) to include (default: 1)")
    parser.add_argument("--tag", default=None,
                        help="Output tag (e.g., beta, mu-1, H_I). If omitted, inferred from overlay filename.")
    parser.add_argument("--allow_delta_e_fallback", action="store_true",
                        help="If set, allow ΔE-derived wavelength as a fallback when no NIST match exists. "
                             "By default, rows without NIST λ are dropped to avoid circularity.")
    args = parser.parse_args()
    main(args.overlay, args.gamma_bin, args.min_hits, args.tag, args.allow_delta_e_fallback)
