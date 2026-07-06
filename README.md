# Recursive Geometry of Atomic Spectra  
### *Research Series and Software Reproducibility Pack*

# Status notice

This repository is preserved as a historical reproducibility pack for the Recursive Geometry / early Thread Frame work.

It remains useful for provenance, code review, and reconstruction of earlier analyses. However, the current program no longer treats this repository as the live claim ceiling.

Thread Frame is now being rebuilt as a levels-first spectroscopic audit instrument with stronger provenance, null comparisons, covariance handling, observed-channel cycle closure, and admission gates.

Current research home:

https://www.lucernaveritas.ai/

The historic record follows.

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17167687.svg)](https://doi.org/10.5281/zenodo.17167687) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17335815.svg)](https://doi.org/10.5281/zenodo.17335815) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17402933.svg)](https://doi.org/10.5281/zenodo.17402933)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**Series Overview**  
This repository unifies all open-source research software and data pipelines for the *Coherence Research Collaboration* series:  

| Paper / Module | DOI | Description |
|----------------|-----|--------------|
| *Recursive Geometry of Atomic Spectra* | [10.5281/zenodo.17167687](https://doi.org/10.5281/zenodo.17167687) | Foundational γ-ladder and thread-frame geometry |
| *Information-Theoretic Confirmation of the α-Affine Thread Frame* | [10.5281/zenodo.17335815](https://doi.org/10.5281/zenodo.17335815) | MDL validation of the universal slope β≈log₁₀α |
| *A Local Electromagnetic Wave Equation from Spectral Geometry* | [10.5281/zenodo.17402933](https://doi.org/10.5281/zenodo.17402933) | Maxwellian verification / Radiocode implementation — see [README_ELECTROMAG.md](README_ELECTROMAG.md) |

---

**Latest preprint:** [10.5281/zenodo.17167687](https://doi.org/10.5281/zenodo.17167687)  
**Other works:** [10.5281/zenodo.17335815](https://doi.org/10.5281/zenodo.17335815)  
**This version (v2):** [10.5281/zenodo.17188444](https://doi.org/10.5281/zenodo.17188444)  
**Narrative:** https://www.thecoherencecode.com  
**Kelly Heaton:** https://www.linkedin.com/in/kelly-heaton-studio/

> **Note:** To reproduce the γ-ladder, you’ll need basic Python skills. Consumer hardware is fine (this work was done on a 4-year-old M1 Mac using Terminal).  
> Source data are public at NIST:  
> https://physics.nist.gov/PhysRefData/ASD/levels_form.html  
> https://physics.nist.gov/PhysRefData/ASD/lines_form.html

---

## Purpose

This repository provides a minimal, end-to-end set of **scripts, configs, and data pointers** to reproduce and audit the γ-ladder pipeline, photon overlay, and intercept/mass checks from our preprint, *Recursive Geometry of Atomic Spectra* (Heaton & The Coherence Research Collaboration, 2025). Several visualization scripts are provided to see photon count vs. slope, gamma resonance by tower (no photons), and a heatmap of gamma resonance (no photons).
Code is released **as-is** for scientific verification; this is **no-support mode**.

---

## Contents

- What this repo reproduces
- Inventory of basic scripts
- Repository layout
- Quick start
- Data, provenance, determinism
- How to run each stage
- Deuterium (D I): scope & status
- Known limitations
- Citing this work
- License & support
- Auxiliary scripts (research extensions)

---

## What this repo reproduces

**Non-circular, two-phase pipeline.**  
(I) Build recursion depth **γ** from *levels only*.  
(II) Overlay photons post-hoc and analyze threads in the **Thread Frame** (γ, log₁₀ν).  
Tilt is anchored (**β ≈ log₁₀α**); physics lives in **intercepts χ** (mass/Z²/site transport) and **microslopes**.

---

## Inventory of basic scripts (γ pipeline)

> See the end of this file for **auxiliary** research scripts used in the preprint (not required to reproduce the γ-ladder).

- `scripts/preprocess/nist_levels_parser_v13.py` — levels → tidy levels (QA, provenance)
- `scripts/preprocess/nist_lines_parser_v1.py` — lines → tidy lines (Level_ID, selection rules)
- `scripts/analysis_pipeline/build_resonance_inventory.py` — batch runner: levels γ-sweep, inventory
- `scripts/utils/run_resonance_sweep.py` — γ loop; writes per-γ hitpairs & per-ion summary
- `scripts/utils/resonance_permutation_test.py` — permutation nulls (uniform/spacing)
- `scripts/analysis_pipeline/build_attractor_affinity.py` — aggregate γ-affinity (levels-only)
- `scripts/analysis_pipeline/process_photons.py` — overlay photons with γ-resonant levels
- `scripts/analysis_pipeline/build_photon_gamma_ladders.py` — organize photons into “towers”
- `scripts/analysis_pipeline/rgp_physics_v1.py` — χ–β fits (linear core; curvature AIC-gated, WIP)
- `scripts/analysis_pipeline/rgp_mass_estimator.py` — mass intercept tests (hydrogenic collapse, isotopes; WIP)
- `scripts/views/plot_gamma_affinity_heatmap_matrix.py` — gamma-resonance heatmap (levels only)
- `scripts/views/plot_gamma_ladder_views.py` — gamma-resonance by quantum "tower" (levels only)
- `scripts/views/viz_all-ion_photon-decay_no-tower.py` — photon count vs. slope (levels + photons)
- `scripts/utils/constants.py` — α, α² targets, canonical columns
- `scripts/utils/load_sigma.py` — **read** σ from `sigma.json` / `SIGMA`
- `scripts/utils/set_sigma.py` — **write** σ to `sigma.json` (CLI)
- `scripts/utils/path_config.py` — path registry (tag → folders)
- `scripts/utils/provenance.py` — provenance writers (hashes, thresholds, metadata)
- `scripts/utils/io_helpers.py` — CSV/Parquet I/O helpers
- `sigma.json` — default `{"sigma": 0.0072973525693}` (CODATA α)

---

## Repository layout

```text
├─ data/
│  ├─ raw/
│  │  ├─ levels/          # NIST *_levels_raw.csv (examples provided)
│  │  └─ lines/           # NIST *_lines_raw.csv (examples provided)
│  ├─ tidy/
│  │  ├─ levels/          # tidy levels + adjacency + .meta.json
│  │  └─ lines/           # tidy lines + .meta.json
│  ├─ meta/               # YAMLs for sweeps (incl. rgp_sample.yaml, D_I.yaml)
│  └─ results/
│     ├─ resonance_inventory_<TAG>/     # per-ion γ sweep outputs
│     └─ reports/levels/                # parser QA reports
├─ scripts/
│  ├─ preprocess/
│  │  ├─ nist_levels_parser_v13.py      # levels → tidy levels
│  │  └─ nist_lines_parser_v1.py        # lines  → tidy lines
│  ├─ analysis_pipeline/
│  │  ├─ build_resonance_inventory.py   # batch γ-sweeps/inventory (levels)
│  │  ├─ build_attractor_affinity.py    # see where an ion's levels resonate with γ (levels)
│  │  ├─ process_photons.py             # re-associate photons with ion γ (levels+lines)
│  │  ├─ build_photon_gamma_ladders.py  # organize the data by quantum "tower"
│  │  ├─ rgp_mass_estimator.py          # research with mass intercept calculations (WIP)
│  │  └─ rgp_physics_v1.py              # χ–β plane fits (optional curvature WIP)
│  ├─ utils/
│  │  ├─ constants.py                   # α, α² targets, column map
│  │  ├─ set_sigma.py                   # write σ (sigma.json/ENV)
│  │  ├─ load_sigma.py                  # read σ (sigma.json/ENV)
│  │  ├─ path_config.py                 # path registry (tag → folders)
│  │  ├─ provenance.py                  # provenance writers (hashes, thresholds, metadata)
│  │  ├─ io_helpers.py                  # CSV/Parquet I/O helpers
│  │  ├─ run_resonance_sweep.py         # γ loop, write hitpairs
│  │  └─ resonance_permutation_test.py  # permutation nulls
│  └─ views/
│     ├─ plot_gamma_affinity_heatmap_matrix.py        # heatmap of gamma-resonant levels
│     ├─ plot_gamma_ladder_views.py                   # plot gamma-resonant levels by quantum tower
│     └─ viz_all-ion_photon-decay_no-tower.py         # plot photons by count (no pattern) vs. frequency (pattern)
├─ sigma.json                           # default: { "sigma": 0.0072973525693 }
└─ README.md                            # this file
```

## Scripts in this repo implement the critical steps:

- Tidy parsers for NIST levels and lines with provenance sidecars and QA: nist_levels_parser_v13.py, nist_lines_parser_v1.py.

Note: to make a quick check easier, we have provided raw and tidy levels and lines files for a small set of ions and one isotope. You can use the tidy files to build the gamma ladder, or you can process the raw files yourself (and/or download other raw files from NIST, which is exactly what we did for our paper).

- Levels‑only γ‑sweep with permutation‑null, adaptive tolerances, and FDR; inventory writer: build_resonance_inventory.py, run_resonance_sweep.py. NOTE: To build the gamma ladder, you must have you directory configured properly (as shown), but then you interact with only one script: build_resonance_inventory.py. This script calls all of the other scripts automatically. If you do not use the sigma.json file that we provided, then you need to generate your own.

- Deterministic randomness and σ access (sigma.json or SIGMA env): load_sigma.py. load_sigma 
Note: Constants include CODATA α and the α² Rydberg target helper (alpha2_target) that underlies the γ‑ladder target spacings.

Select visualization scripts are provided so you can see your results.

The design principles we adhere to in this repo intentionally mirror the preprint:

- Levels‑only discovery → photons post‑hoc, preventing circularity. (See Fig. 1 and §4.1.)
- σ‑locking and FDR control when sweeping γ; seeding is deterministic and tied to (ion, σ, γ). run_resonance_sweep 
- Strict frequency provenance: overlay uses NIST wavelength conventions (vacuum / air) and works off observed lines only. (Preprint §2.1; footnote on frequency provenance.)

Deuterium (D I): scope, status, and how to run: Deuterium (D I) was used solely for the isotope mass‑intercept test and was intentionally excluded from the core β‑slope/γ‑ladder summaries. In the present catalog coverage, D I is sparse: only one matched tower survived reliability gates in our run, so reference statistics were not computable; we labeled H↦D data‑limited and refrained from claims pending richer D I ladders (Table 5).

To keep the main results faithful to the preprint and to avoid confusion:

- D I is siloed to a dedicated tag (e.g., D_I_micro) and not included in the main β sweep.
- For the levels‑only γ step, we set mu: 1.0 (mass enters later in χ). This matches the study design where μ̂≡1 during γ discovery; reduced mass appears in intercept transport (§3.1, footnote; Eq. 4). KBHeaton_Recursive Geometry of … 
- We provide a D I micro‑sweep YAML and example commands below so others can explore; current outcomes should be treated as inconclusive. Inventory & per‑γ summaries are produced by the sweep machinery.

# Quick start

## 0) Ensure sigma.json exists at the repo root (see below).
## 1) Tidy NIST levels & lines (examples for O_III):
python -m scripts.preprocess.nist_levels_parser_v13 --ion O_III
python -m scripts.preprocess.nist_lines_parser_v1 \
  --raw_lines data/raw/lines/O_III_lines_raw.csv \
  --tidy_levels data/tidy/levels/O_III_levels.csv \
  --out_dir data/tidy/lines --wavelength_medium vacuum --energy_tol_meV 0.50

## 2) Run a γ‑sweep from a YAML
python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/bio_vacuum_mu-1.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 5000 --q_thresh 0.01 --dedup --enrich_hitpairs

- The parsers write provenance‑header CSVs and sidecar JSON with file hashes, thresholds, and QA plots.
- The sweep writes per‑γ hitpair files and a per‑ion summary (*_resonance_summary.txt + .json sidecar). build_resonance_inventory 
Dependencies (minimal): python>=3.10, numpy, pandas, matplotlib, scipy, statsmodels, pyyaml. (Parsers/sweeps import these directly.)

# Data, provenance, and determinism

## Data source (NIST). 
We use only the “observed” dataset with standard NIST wavelength conventions (vacuum < 200 nm; air 200–2000 nm; vacuum > 2000 nm). Parsers preserve provenance, and overlay computations use NIST‑matched wavelengths (strict mode by default). (Preprint §2.1 and strict provenance footnote.) 
## Deterministic RNG.
Permutation nulls seed deterministically as a function of (ion, σ, γ), enabling reproducible p/q values across runs and machines. σ resolves from sigma.json (or SIGMA env) via load_sigma.py. resonance_permutation_test load_sigma 
## Outputs with sidecars.
Parsers emit .meta.json and QA; sweeps emit *_resonance_summary.txt + .json with links to tidy inputs and raw lineage. 

# How to run each stage

## 0) Set σ (fine‑structure constant) once

At repo root, ensure:

{ "sigma": 0.0072973525693 }
If sigma.json is missing, load_sigma.py will raise a friendly error; you may also override in terminal with export SIGMA=....

## 1) Parse NIST levels → tidy levels

python -m scripts.preprocess.nist_levels_parser_v13 --ion <ION>
or omit --ion to process all RAW_LEVELS_DIR/*_levels_raw.csv

Converts wavenumber to eV, sorts by energy, assigns stable Level_ID, flags dense/duplicate/limit rows, infers n with provenance, builds adjacency edges (energy & series), emits QA report and provenance sidecars. Outputs under data/tidy/levels/. nist_levels_parser_v13 

## 2) Parse NIST lines → tidy lines

python -m scripts.preprocess.nist_lines_parser_v1 \
  --raw_lines data/raw/lines/<ION>_lines_raw.csv \
  --tidy_levels data/tidy/levels/<ION>_levels.csv \
  --out_dir data/tidy/lines \
  --wavelength_medium vacuum \
  --energy_tol_meV 0.50

Normalizes wavelengths, converts to frequency/energy, matches upper/lower Level_ID using energy tolerances, stamps selection‑rule tags (E1, parity, ΔJ), and writes a provenance‑header CSV + sidecar JSON. Outputs under data/tidy/lines/. nist_lines_parser_v1 

## 3) Levels‑only γ‑sweep & resonance inventory

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/<YOUR>.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 5000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs

Your YAML lists ions with (Z, μ, γ grid). The sweep estimates target spacings at γ using α²E₀Z²μ and an α‑scale (σ) factor, applies adaptive tolerance ladders, and computes permutation‑null p‑values with BH–FDR q across γ. Hitpair CSVs + per‑ion summary and an inventory table are written under data/results/resonance_inventory_<TAG>/

## 4) (Optional) γ‑affinity & photon overlay → photon ladders
This step is post‑hoc: photons are not used to define γ (non‑circularity). After the sweep, regroup γ‑resonant level pairs by (nᵢ, nₖ) and then overlay observed photons onto the γ ladder using NIST wavelengths (Thread Frame analysis follows). See Methods §4 in the preprint for the operational details of overlay and per‑tower ladder construction.

Build the affinity map (one row per `(ion, gamma_bin)`):

```bash
python -m scripts.analysis_pipeline.build_attractor_affinity --tag <TAG>
Example:
python -m scripts.analysis_pipeline.build_attractor_affinity --tag D_I_micro
```

Outputs: per‑ion photon γ‑ladders indexed by (ion, tower (nᵢ,nₖ), γ) that drive thread fits, intercepts, and local microslopes (§4–5).

Build the gamma resonant files with photons re-associated (process_photons.py):

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

Single ion (or isotope):

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

## 5) (Optional, but recommended) Organize the data by quantum "tower" to see patterns:

build_photon_gamma_ladders.py

Examples:
## From explicit photon overlay (create the gamma ladder, affinity, and photon overlay first)

  python -m scripts.analysis_pipeline.build_photon_gamma_ladders \
    --overlay data/meta/photon_overlay_mu-1.csv --gamma_bin 0.02 --min_hits 1

## 6) "RGP Physics" v1 — χ–β plane fits (w/ optional curvature). WIP
Script: scripts/analysis_pipeline/rgp_physics_v1.py
This module operationalizes the Thread‑Frame (Def. D2) with β=log10α (Eq. 3) and returns the intercepts χ and tilt diagnostics used throughout the results; the quadratic term is included “locally, if AIC demands” (Eq. 2). 
Inputs: a directory of *_photon_ladder.csv files (from build_photon_gamma_ladders.py); optional gamma_attractor_affinity_*.csv for p/q enrichment.

## Outputs:

- rgp_v1_tower_fits.csv — per‑tower WLS fits: beta (slope), chi (intercept), theta_deg, rmse_log10_hz, coverage/weights, and—if enabled and justified—c_quad, mean_curv, torsion_index (see below).
- rgp_v1_pairwise_maps.csv — Δβ, Δχ, Δθ, scale 10Δβ, mapped‑overlap RMSE/R², reliability flags, optional p/q.
- rgp_v1_ion_summary.csv — weighted means and totals per ion (including share_curvature_needed).
- rgp_v1_tower_coverage.csv — number of γ‑bins and total photon weight per (ion, ni ,nk).

## CLI examples:

Linear-only (paper-aligned):
python -m scripts.analysis_pipeline.rgp_physics_v1 \
  --photon-dir data/meta/ion_photon_ladders_mu-1 \
  --out-dir    data/results/rgp_v1

With quadratic curvature (experimental) and p/q enrichment:
python -m scripts.analysis_pipeline.rgp_physics_v1 \
  --photon-dir data/meta/ion_photon_ladders_mu-1 \
  --out-dir    data/results/rgp_v1 \
  --curvature \
  --affinity data/meta/gamma_attractor_affinity_bio_vacuum_mu-1.csv

Note on "curvature":
(c_quad, mean_curv, torsion_index) are only non-zero if you pass --curvature, the tower has ≥ 3 γ-bins, and the quadratic beats linear by ΔAIC ≤ −2; otherwise the geometry is computed with c=0, so “curvature” will be 0/blank. If --curvature is set and a tower has ≥ 3 distinct γ‑bins, the script fits both linear and quadratic WLS and applies an AIC gate: use c only if ΔAIC≤−2. When the gate fails, geom_c is set to 0 and geometry‑derived fields like mean_curv become 0 (so a blank/zero “curvature” column is expected in many cases). Curvature and the derived torsion_index are exploratory and not required to reproduce the paper’s χ–β results.

Reliability.
Flags (fit_sparse_bins, fit_low_weight, fit_high_rmse, fit_curvature_needed) and a reliability_score help filter marginal towers and maps. Consider focusing on reliability_score ≥ 0.7 for summary plots. 

## 7) (Optional) Intercepts & isotope mass checks

With slope locked near β=log₁₀α, intercepts χ transport reduced mass and Z²; matched towers can be used to estimate isotope shifts (Eq. 5a) and check hydrogenic collapse (Eq. 5b). The preprint reports hydrogenic collapse at millidex precision and labels H↦D as data‑limited in this run (single tower).

Reminder for D I: keep μ=1.0 in the levels‑only pipeline (γ discovery); treat μ in the intercept stage (χ transport). This prevents circularity and matches the study design (§3.1 note).

Known limitations (read this):

Catalog sparsity at high γ. Sparse NIST coverage can truncate tails and under‑estimate terminal depths by a photon or two; this principally affects tail diagnostics and any ladder‑terminus analysis. (Preprint, “Limitations of the work”.) 

D I is currently data‑limited. Only one matched tower survived reliability gates in our run, so reference statistics for H↦D were not computed. Treat current D I outputs as exploratory. (Preprint Table 5.) 

Site factors. Intercepts include a tower/site factor F site  F_\text{site} Fsite (quantum‑defect/relativistic/correlation bundle) not yet explicitly modeled here; precision mass extraction across non‑hydrogenic ions requires tower‑resolved electronic factors (future work). (Preprint §5.1.)

## Notes on the mass estimator test with D I (WIP):

In our present D I runs, the matched‑tower count is low and permutation tests are under‑powered, consistent with the preprint’s “data‑limited” status. We therefore state openly that D I results are inconclusive and that expanded coverage is future work. Outputs go under data/results/resonance_inventory_D_I_micro/ (same file patterns as the main sweep).

Suggested minimal D I YAML (data/meta/D_I.yaml):

paths:
  tag: D_I_micro

defaults:
  mu: 1.0                 # levels-only; mass enters at intercepts
  sigma_exp_meV: 0.05
  full_ladder: &full_ladder
    [0.00, 0.02, 0.04, ... , 5.00]   # γ grid (Δγ = 0.02)

D_I:
  flag: strict
  Z: 1
  mu: 1.0
  gamma_bins: *full_ladder

Run the D I micro‑sweep (levels‑only, spacing‑null, strict):

python -m scripts.analysis_pipeline.build_resonance_inventory \
  --cfg data/meta/D_I.yaml \
  --null_mode spacing --spacing_jitter_meV 0.0 \
  --n_iter 10000 --q_thresh 0.01 --dedup \
  --enrich_hitpairs

## 8) Visualization scripts
Two of the scripts are for gamma-resonant levels (no photons):
- `scripts/views/plot_gamma_affinity_heatmap_matrix.py` — gamma-resonance heatmap (levels only)
- `scripts/views/plot_gamma_ladder_views.py` — gamma-resonance by quantum "tower" (levels only)
One script visualizes levels and photons:
- `scripts/views/viz_all-ion_photon-decay_no-tower.py` — photon count vs. slope (levels + photons)

# Visualization: γ-Attractor Affinity Heatmap

We also provide a script to **visualize γ-resonance affinity as a heatmap**,  
where each cell shows the resonance strength (obs_hits) for an (ion, γ) pair.

Run:

```bash
python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix \
  --sort_mode cluster
```

# Visualization: Cross-Ion Photon Decay (No Towers)

We provide a helper script to **visualize the universal slope β ≈ log₁₀α across ions**,  
without tower grouping. This makes the γ–ν relationship visible at a glance.

Run:

```bash
python -m scripts.views.cross_ion.viz_all-ion_photon-decay_no-tower \
  --input-dir data/results/photon_matched_resonant_pairs_mu-1/ \
  --output-dir data/results/plots/viz_all-ion_photon-decay_no-tower/ \
  --export-csv
  ```

# Visualization: γ-Ladder Tower Views

We provide a script to **visualize photon ladders by tower (nᵢ, nₖ)** in two orientations:

- **Rotated view** (x = nₖ, y = γ): “looking into” the γ-sheet.  
- **Original view** (x = γ, y = nₖ): conventional ladder portrait.

Run:

```bash
python -m scripts.views.ion_identity.plot_gamma_ladder_views \
  --ladders H_I:data/meta/ion_photon_ladders_mu-1/H_I_photon_ladder.csv \
  --ni 4 \
  --out  data/results/plots/gamma_levels_by-tower/H_I_4_gamma_vs_nk_grid.png \
  --marker x --cmap viridis
```

---

## Next-Stage Research Modules

The present repository also hosts the follow-on studies that extend the α-Affine Thread Frame into macroscopic field tests and compression geometry:

| Study | DOI / Tag | Scope |
|-------|------------|-------|
| **Information-Theoretic Confirmation of the α-Affine Thread Frame** | [10.5281/zenodo.17335814](https://doi.org/10.5281/zenodo.17335814) | MDL validation of the universal slope β ≈ log₁₀α |
| **A Local Electromagnetic Wave Equation from Spectral Geometry** | [10.5281/zenodo.17402933](https://doi.org/10.5281/zenodo.17402933) | Radiocode implementation and Maxwellian verification using public GRAPE V1 HF data |

### Electromagnetism (Radiocodes)

Reproduction scripts live under  
`scripts/analysis_pipeline/radio_delay_analysis.py` and `scripts/analysis_pipeline/pretty_dtau_plot.py`.  
They process the open GRAPE V1 Doppler dataset ([Collins et al., 2024](https://doi.org/10.5281/zenodo.13637199)) to reproduce the ionospheric verification shown in Fig. 3 of the electromagnetic paper.

A concise README for that module is provided at  
`README_ELECTROMAG.md`.

---

# To cite this work:

Heaton, K. B. & The Coherence Research Collaboration (2025).
Recursive Geometry of Atomic Spectra (preprint). 
Please cite the version you used and include the repository URL/commit for code.

License & support:
MIT License
## Copyright (c) 2025 Kelly B. Heaton and the Coherence Research Collaboration

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.

This is citizen science without funding. We cannot offer issue support or PR triage. If you publish results that use or critique this work, please cite the preprint and this repository.

# Appendix: FAQ (short)

Q: Why keep μ=1.0 during the γ‑sweep? A: To avoid circularity—γ discovery is levels‑only. Reduced mass appears later in intercept transport; see Eq. 4 and §3.1 notes in the preprint.

Q: How are p/q values computed? A: We use permutation nulls (uniform or spacing‑bootstrap), adaptive tolerance ladders, and BH–FDR across γ. RNG is deterministically seeded by (ion, σ, γ) so results are reproducible.
resonance_permutation_test
run_resonance_sweep
build_resonance_inventory

Q: What exactly do the parsers write out? A: Levels parser: tidy levels CSV with provenance header, adjacency parquet, .meta.json, QA histogram/markdown. Lines parser: tidy lines CSV with provenance header, .meta.json and selection‑rule tags.
nist_levels_parser_v13
nist_lines_parser_v1

## Collaboration, support, and other helpful feedback is welcome: kelly@circuiticon.com 

Follow the light of the lantern. 🌕🪔

## * Auxiliary scripts (research extensions; not required for basic γ-ladder reproduction)

scripts/analysis_pipeline/draw_ion_portraits_photons.py – ion portraits on (ni, nk) lattice (Figs. 2–3)
scripts/analysis_pipeline/microslope_extractor.py       – local δ(γ), θ(γ) fields (torsion corridors; Fig. 10)
scripts/analysis_pipeline/rgp_limit_analysis.py          – recursion limits γ*, νmin (Planck floor; Sec. 5.3)
scripts/analysis_pipeline/CTI_cross_thread_intersections.py – cross-thread intersections (CTI two-gate; Sec. 5.4, Figs. 12–13)
scripts/analysis_pipeline/threadlaw_photoncode.py        – κ-photoncodes (photons-only identity; Sec. 5.4)
scripts/analysis_pipeline/ion_photoncode_library.py      – build photoncode library for ions (Sec. 5.4)
scripts/analysis_pipeline/motif_maker.py                 – detect photoncode motifs with null/FDR (Sec. 5.4, Figs. 15–17)
scripts/analysis_pipeline/overlay_constellation.py       – visualize motif resonance constellations (Sec. 5.4, Fig. 17)
