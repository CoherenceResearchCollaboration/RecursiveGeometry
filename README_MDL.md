# 🧮 Recursive Geometry – MDL Extension
[📄 Preprint PDF (v1.0)](preprint/KBHeaton_Information_Theoretic_Confirmation_MDLSlope_2025.pdf)  
DOI: [10.5281/zenodo.17335815](https://doi.org/10.5281/zenodo.17335815)

### *Information-Theoretic Confirmation of the α-Affine Thread Frame*

This directory expands the **Recursive Geometry** repository with the complete
analysis and data pipeline that supports the paper:

> **Heaton et al. (2025)** — *Information-Theoretic Confirmation of the α-Affine Thread Frame*  
> *DOI: 10.5281/zenodo.17335815*

---

## 📂 Repository Layout

```

RecursiveGeometry/
├── data/                     ← public-domain source inputs
│   ├── raw/lines/            (NIST atomic line lists)
│   ├── solar/fluxatl/        (NSO FluxAtlas tiles + README_FLUXATL.txt)
│   ├── solar/photatl/        (NSO PhotAtlas tiles + README_PHOTATL.txt)
│   ├── stars/vega_elodie/    (ELODIE Vega spectra + README_VEGA.txt)
│   └── molecules/            (ExoMol C₂ Swan line list + README_C2.txt)
│
├── results/                  ← generated outputs (CSV + plots)
│
├── configs/mdl_runs/         ← YAML recipes for each dataset
│
└── scripts/
├── gamma_pipeline/       ← original γ-ladder recursive geometry code
├── MDL/                  ← new Minimum Description Length (β-slope) pipeline
│   ├── adapter_to_photoncodes_neon.py
│   ├── adapter_to_photoncodes_solar_fluxatl.py
│   ├── adapter_to_photoncodes_vega_elodie.py
│   ├── aggregate_fluxatl_dataset.py
│   ├── aggregate_vega_elodie_dataset.py
│   ├── line_eventizer.py
│   ├── threadlaw_photoncode.py
│   ├── mdla_sweep.py
│   └── run_mdla_from_yaml.py        ← convenience runner
└── views/ion_identity/              ← raw-spectrum plots

````

---

## 🧭 Quick-Start Guide

### 1. Create Environment and Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install numpy pandas matplotlib
````

### 2. Build “events” CSVs from raw data

**Example (Hg II lamp):**

```bash
PYTHONPATH=. python -m scripts.MDL.adapter_to_photoncodes_neon \
  --in data/raw/lines/Hg_II_lines_raw.csv \
  --out-csv results/lamps/raw_csv/hg_II_ritz_vac.csv \
  --prefer ritz --medium vacuum --min-intens 0
```

**Solar FluxAtlas merge:**

```bash
PYTHONPATH=. python -m scripts.MDL.aggregate_fluxatl_dataset \
  --indir data/solar/fluxatl \
  --out-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --unit nm --smooth-win 9 --win-mad 401 --k-sigma 6.0 --min-sep-px 7 \
  --merge-kms 2.0 --one-per-kappa
```

**Vega merge:**

```bash
PYTHONPATH=. python -m scripts.MDL.aggregate_vega_elodie_dataset \
  --indir data/stars/vega_elodie \
  --out-csv results/stars/raw_csv/vega_all_photons.csv \
  --unit nm --smooth-win 9 --win-mad 201 \
  --k-sigma-seed 2.5 --k-sigma-final 3.0 \
  --min-sep-px 9 --sigmas-pix 2,3,5,8,12,20,35 \
  --merge-kms 2.0 --one-per-kappa --fallback-simple
```

The molecular dataset `data/molecules/C2_Swan_visible.csv`
is already a processed line list; no adapter needed.

---

### 3. Run the MDL β-Slope Sweep

**Direct command:**

```bash
PYTHONPATH=. python -m scripts.MDL.mdla_sweep \
  --events-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --outdir results/solar/mdla \
  --label sun_fluxatl_all \
  --series-name "Sun (FluxAtlas, photons)" \
  --grid 0.002 --kmax 1.70 \
  --beta-list 'alpha,log10(1/128),log10(1/e),log10(1/phi),-2.40,-2.30,-2.10,-2.00' \
  --nulls 1 --null-beta alpha --null-windows 50
```

**Or via YAML runner:**

```bash
PYTHONPATH=. python scripts/MDL/run_mdla_from_yaml.py configs/mdl_runs/solar.yml
```

Repeat for Vega, the three lamp datasets, and C₂ Swan.

---

### 4. Optional – Rebuild Raw-Spectrum Plots

```bash
PYTHONPATH=. python -m scripts.views.ion_identity.plot_all_raw_spectra
PYTHONPATH=. python -m scripts.views.ion_identity.plot_solar_full_spectrum
```

---

## 📊 Outputs

Each run generates:

* `mdl_sweep.csv` – β-grid values and code lengths
* `mdl_best.json` – metadata for the minimum description length
* `L_vs_beta_full+zoom.png` – plots for publication

Composite results for all datasets appear in `results/.../mdla/`.

---

## 🪶 Citation

When citing this work, please reference both:

1. **Heaton et al. (2025)** – *Recursive Geometry of Atomic Spectra*
   DOI 10.5281/zenodo.17167687
2. **Heaton et al. (2025)** – *Information-Theoretic Confirmation of the α-Affine Thread Frame*
   DOI *10.5281/zenodo.17335815*

---

## ⚙️ License and Data Rights

All raw datasets included here are from public-domain or open-license sources
(NSO / NOAO, NIST, ELODIE Archive, and ExoMol).
Derived outputs are released under the MIT License.

---
