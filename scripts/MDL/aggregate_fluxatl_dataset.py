#!/usr/bin/env python3
"""
Merge solar atlas slices -> one photon *line list* CSV (absorption lines only).

Run:
PYTHONPATH=. python -m scripts.MDL.aggregate_fluxatl_dataset \
  --indir data/solar/fluxatl \
  --out-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --unit nm --smooth-win 9 --win-mad 401 --k-sigma 6.0 --min-sep-px 7 \
  --merge-kms 2.0 --one-per-kappa

"""
import argparse, glob, os, math
import pandas as pd
import numpy as np

# package-safe imports from the solar adapter
try:
    from .adapter_to_photoncodes_solar_fluxatl import load_fluxatl
    from .line_eventizer import eventize, load_ascii_two_or_three_cols, load_photatl_wn
except Exception:
    from scripts.MDL.adapter_to_photoncodes_solar_fluxatl import load_fluxatl
    from scripts.MDL.line_eventizer import eventize, load_ascii_two_or_three_cols, load_photatl_wn

---

### D) `scripts/MDL/aggregate_vega_elodie_dataset.py`
Same idea: change imports that reference the old package.

**Find** (top-level):  
`from scripts.MDL.line_eventizer import eventize`  
**Replace**:  
`from scripts.MDL.line_eventizer import eventize`

**Find** (later):  
`from scripts.MDL.adapter_to_photoncodes_vega_elodie import (...)`  
**Replace**:  
`from scripts.MDL.adapter_to_photoncodes_vega_elodie import (...)`  :contentReference[oaicite:3]{index=3}

---

### E) (Optional but nice) Update the usage text inside a few headers
These don’t affect execution; they’re just printed examples users see.

- `adapter_to_photoncodes_neon.py`: change `python -m scripts.MDL...` → `python -m scripts.MDL...`  :contentReference[oaicite:4]{index=4}  
- `adapter_to_photoncodes_solar_fluxatl.py`: same  :contentReference[oaicite:5]{index=5}  
- `adapter_to_photoncodes_vega_elodie.py`: same  :contentReference[oaicite:6]{index=6}  
- `line_eventizer.py`: same in the “Typical usage” block  :contentReference[oaicite:7]{index=7}  
- `threadlaw_photoncode.py`: same in the usage examples  :contentReference[oaicite:8]{index=8}

(Again: purely cosmetic; the real fixes are A–D.)

---

# 2) Your updated, copy-paste prompts (new module paths)

Run from the repo root with your venv active:

### Adapters / aggregators (build the events CSVs)

**NIST lamps (Ne/Na/Hg → photons):**
```bash
PYTHONPATH=. python -m scripts.MDL.adapter_to_photoncodes_neon \
  --in data/raw/lines/Hg_II_lines_raw.csv \
  --out-csv results/lamps/raw_csv/hg_II_ritz_vac.csv \
  --prefer ritz --medium vacuum --min-intens 0

PYTHONPATH=. python -m scripts.MDL.adapter_to_photoncodes_neon \
  --in data/raw/lines/Na_I_lines_raw.csv \
  --out-csv results/lamps/raw_csv/na_I_ritz_vac.csv \
  --prefer ritz --medium vacuum --min-intens 0

PYTHONPATH=. python -m scripts.MDL.adapter_to_photoncodes_neon \
  --in data/raw/lines/Ne_I_lines_raw.txt \
  --out-csv results/lamps/raw_csv/neon_neI_ritz_vac.csv \
  --prefer ritz --medium vacuum --min-intens 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=False,
                    help="Directory of files (legacy; use --indir-glob for multi-atlas).")
    ap.add_argument("--indir-glob", required=False, default=None,
                    help="One or more globs separated by commas (supports **), e.g.: "
                         "'data/solar/fluxatl/**/*.txt,data/solar/photatl/**'")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--unit", default="auto", choices=["nm","angstrom","auto"])
    # you can expose these if you want to tune from CLI
    ap.add_argument("--smooth-win", type=int, default=9)
    ap.add_argument("--win-mad", type=int, default=401)
    ap.add_argument("--k-sigma", type=float, default=6.0)
    ap.add_argument("--min-sep-px", type=int, default=7)
    ap.add_argument("--merge-nm", type=float, default=None)
    ap.add_argument("--merge-kms", type=float, default=2.0,
                    help="If set, merge by |Δλ|/λ * c <= Δv (km/s); overrides --merge-nm when not None.")
    ap.add_argument("--one-per-kappa", action="store_true", default=False)
    ap.add_argument("--k-sigma-seed", type=float, default=None)
    ap.add_argument("--k-sigma-final", type=float, default=None)
    ap.add_argument("--sigmas-pix", type=str, default="2,3,5,8,12")

    args = ap.parse_args()

    # Resolve files: either a single directory or one/many globs (comma-separated).
    import re
    if args.indir_glob:
        patterns = [p.strip() for p in args.indir_glob.split(",") if p.strip()]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted(set(files))  # de-dupe
        # keep only regular files (skip directories matched by **)
        files = [f for f in files if os.path.isfile(f)]
        # STRICT: only accept fluxatl 'lm####.txt' and photatl 'wn####' (opt '.txt')
        good = []
        skipped = 0
        for f in files:
            base = os.path.basename(f).lower()
            norm = f.replace("\\", "/")
            if "/fluxatl/" in norm and base.startswith("lm") and base.endswith(".txt") and base[2:6].isdigit():
                good.append(f)
            elif "/photatl/" in norm and re.fullmatch(r"wn\d{4}(\.txt)?", base):
                good.append(f)
            else:
                skipped += 1
        if skipped:
            print(f"[aggregate] skipped {skipped} non-slice files")
        files = good
    else:
        if not args.indir:
            raise SystemExit("Provide --indir or --indir-glob")
        files = sorted(glob.glob(os.path.join(args.indir, "*")))
        files = [f for f in files if os.path.isfile(f)]
        # When using --indir (single atlas), still enforce expected naming
        base_dir = os.path.abspath(args.indir).replace("\\","/")
        if base_dir.endswith("/fluxatl") or "/fluxatl/" in base_dir:
            files = [f for f in files if os.path.basename(f).lower().startswith("lm")
                                  and os.path.basename(f).lower().endswith(".txt")
                                  and os.path.basename(f)[2:6].isdigit()]
        elif base_dir.endswith("/photatl") or "/photatl/" in base_dir:
            files = [f for f in files if re.fullmatch(r"wn\d{4}(\.txt)?", os.path.basename(f).lower())]
    if not files:
        where = args.indir_glob or args.indir
        raise SystemExit(f"No input files found for pattern: {where}")

    # 1) find lines file-by-file, compute weights (depth), collect rows
    rows = []
    def _auto_load(path: str, unit: str):
        name = os.path.basename(path).lower()
        # photatl pages are named 'wn####' (sometimes no extension, sometimes '.txt')
        if name.startswith("wn") and name[2:6].isdigit():
            return load_photatl_wn(path)

        # generic ASCII (2–3 columns; first = wavelength or wavenumber with 'auto' units)
        lam_nm, flux = load_ascii_two_or_three_cols(path, unit=("auto" if unit=="auto" else unit))
        if lam_nm.size == 0:
            # fallback to fluxatl-specific loader (λ[nm], flux)
            lam_nm, flux = load_fluxatl(path, unit=unit)
        return lam_nm, flux

    for path in files:
        lam_nm, flux = _auto_load(path, unit=args.unit)
        if lam_nm.size == 0:
            continue
        # Defensive clamp for photatl: keep ~1100–5400 nm only
        is_ir = ("/photatl/" in path.replace("\\","/")) or os.path.basename(path).lower().startswith("wn")
        if is_ir:
            m = np.isfinite(lam_nm) & (lam_nm >= 1000.0) & (lam_nm <= 6000.0)
            dropped = (~m).sum()
            if dropped:
                print(f"[aggregate] photatl clamp: dropped {dropped} out-of-range rows in {os.path.basename(path)}")
            lam_nm = lam_nm[m]; flux = flux[m]
            if lam_nm.size == 0:
                continue
        # Ensure ascending wavelength for downstream searchsorted/FWHM logic
        order = np.argsort(lam_nm)
        lam_nm = lam_nm[order]
        flux   = flux[order]
        # eventize this slice; photatl needs gentler thresholds (shallower features)
        # reuse is_ir here
        k_sigma  = (args.k_sigma  * (0.6 if is_ir else 1.0))
        k_seed   = ((args.k_sigma_seed  if args.k_sigma_seed  is not None else args.k_sigma) * (0.6 if is_ir else 1.0))
        k_final  = ((args.k_sigma_final if args.k_sigma_final is not None else args.k_sigma) * (0.6 if is_ir else 1.0))
        min_sep  = max(3, int(round(args.min_sep_px * (0.6 if is_ir else 1.0))))
        smooth   = max(5, int(round(args.smooth_win * (1.2 if is_ir else 1.0))))
        sigmas   = np.array([float(s) for s in args.sigmas_pix.split(",") if s.strip()], dtype=float)
        df_evt = eventize(
            lam_nm, flux,
            win_smooth=smooth, win_mad=args.win_mad,
            k_sigma=k_sigma,
            k_sigma_seed=k_seed,
            k_sigma_final=k_final,
            min_sep_px=min_sep,
            sigmas_pix=sigmas
        )
        if not df_evt.empty:
            df_evt = df_evt.copy()
            df_evt["source_path"] = path
            rows.append(df_evt)

    if not rows:
        raise SystemExit("No lines found with current thresholds.")

    # concatenate BEFORE any de-duplication
    df = pd.concat(rows, ignore_index=True)

    # --- HARD SAFETY CLAMPS by atlas on the concatenated events ---
    #   fluxatl (296–1300 nm nominal)
    #   photatl (~1100–5400 nm nominal)
    # Anything outside these ranges is certainly a parse/loader artifact.
    dropped_events = 0
    if "source_path" in df.columns:
        is_flux = df["source_path"].str.contains("/fluxatl/")
        is_phot = df["source_path"].str.contains("/photatl/")

        # drop fluxatl events outside [250, 1500] nm (buffered)
        oob_flux = df[is_flux & ((df["wavelength_nm"] < 250.0) | (df["wavelength_nm"] > 1500.0))].index
        if len(oob_flux):
            print(f"[aggregate] OOB events dropped (fluxatl): {len(oob_flux)}")
            df = df.drop(oob_flux)
            dropped_events += len(oob_flux)

        # drop photatl events outside [1000, 6000] nm (buffered)
        oob_phot = df[is_phot & ((df["wavelength_nm"] < 1000.0) | (df["wavelength_nm"] > 6000.0))].index
        if len(oob_phot):
            print(f"[aggregate] OOB events dropped (photatl): {len(oob_phot)}")
            df = df.drop(oob_phot)
            dropped_events += len(oob_phot)

    # defensive: remove any NaN/inf wavelengths
    bad = ~np.isfinite(df["wavelength_nm"].to_numpy())
    if bad.any():
        print(f"[aggregate] NaN/inf wavelength events dropped: {int(bad.sum())}")
        df = df.loc[~bad]

    if dropped_events:
        df = df.sort_values("wavelength_nm").reset_index(drop=True)

    # wavelength de-dup (Δv preferred; else Δλ)  -- single pass only
    df = df.sort_values("wavelength_nm").reset_index(drop=True)
    lam = df["wavelength_nm"].to_numpy()
    groups, cur = [], [0]
    if getattr(args, "merge_kms", None) is not None:
        c_kms = 299792.458
        for i in range(1, len(df)):
            dv = abs(lam[i] - lam[cur[-1]])/lam[i] * c_kms
            if dv <= args.merge_kms:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)
    else:
        merge_nm = getattr(args, "merge_nm", 0.004) or 0.004
        for i in range(1, len(df)):
            if lam[i] - lam[cur[-1]] <= merge_nm:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)

    winners = []
    for g in groups:
        sub = df.iloc[g]
        # choose strongest by weight; fall back to deepest (max depth)
        if "weight" in sub and sub["weight"].notna().any():
            j = sub["weight"].idxmax()
        elif "depth" in sub and sub["depth"].notna().any():
            j = sub["depth"].idxmax()
        else:
            j = sub.index[0]
        winners.append(df.loc[j])

    df = pd.DataFrame(winners).sort_values("wavelength_nm").reset_index(drop=True)

    # ---- Telemetry: show bandwidth & predicted κ span before thinning ----
    lam_min = float(df["wavelength_nm"].min())
    lam_max = float(df["wavelength_nm"].max())
    beta = math.log10(1/137.035999084)
    delta_kappa = math.log10(lam_max/lam_min) / abs(beta)
    print(f"[aggregate] λ-range: {lam_min:.3f}–{lam_max:.3f} nm  |  Δκ≈ {delta_kappa:.3f} (β=log10 α)")

    # Per-atlas wavelength spans (helps verify no out-of-band photatl rows)
    try:
        _phot = df[df["source_path"].str.contains("/photatl/")]
        _flux  = df[df["source_path"].str.contains("/fluxatl/")]
        if not _phot.empty:
            print(f"[aggregate] photatl λ-range: {_phot['wavelength_nm'].min():.1f}–{_phot['wavelength_nm'].max():.1f} nm")
        if not _flux.empty:
            print(f"[aggregate] fluxatl  λ-range: {_flux['wavelength_nm'].min():.1f}–{_flux['wavelength_nm'].max():.1f} nm")
    except Exception:
        pass

    # --- Per-atlas summary (helps debug contributions)
    tags = []
    for p in files:
        if "/photatl/" in p.replace("\\","/") or os.path.basename(p).lower().startswith("wn"):
            tags.append("photatl")
        elif "/fluxatl/" in p.replace("\\","/"):
            tags.append("fluxatl")
        else:
            tags.append("other")
    import pandas as _pd
    _df = _pd.DataFrame({"atlas": tags})
    atlas_counts = _df["atlas"].value_counts().to_dict()
    print(f"[aggregate] slices by atlas: {atlas_counts}")
    try:
        print("[aggregate] lines by atlas:",
              df.groupby(df["source_path"].str.contains("/photatl/"))["wavelength_nm"].count()
              .rename({True:"photatl", False:"non-photatl"}).to_dict())
    except Exception:
        pass

    # 3) optional: keep strongest one per κ-bin
    if args.one_per_kappa:
        beta = math.log10(1/137.035999084)
        # convert to log10 frequency
        y = np.log10(df["frequency_hz"].to_numpy())
        # use the shortest wavelength (highest frequency) as global anchor
        y0_anchor = np.log10(299792458.0 / (df["wavelength_nm"].min() * 1e-9))
        # compute κ relative to that anchor
        kappa = (y - y0_anchor) / beta
        # group photons in 0.002-κ bins, keep strongest (by weight)
        bin_idx = np.floor(kappa / 0.002).astype(int)
        keep_idx = (
            pd.DataFrame({
                "bin": bin_idx,
                "w": df["weight"].fillna(0.0) if "weight" in df else pd.Series(1.0, index=df.index)
            })
            .groupby("bin")["w"].idxmax()
            .to_numpy()
        )
        df = df.loc[keep_idx].sort_values("wavelength_nm").reset_index(drop=True)


    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df):,} solar lines from {len(files)} slices -> {args.out_csv}")
    print("[aggregate] Done.")


if __name__ == "__main__":
    main()
