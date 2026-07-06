# A Local Electromagnetic Wave Equation from Spectral Geometry
### *Maxwellian Verification and Radiocode Implementation*

# Status notice

This repository is preserved as a historical reproducibility pack for the Recursive Geometry / early Thread Frame work.

It remains useful for provenance, code review, and reconstruction of earlier analyses. However, the current program no longer treats this repository as the live claim ceiling.

Thread Frame is now being rebuilt as a levels-first spectroscopic audit instrument with stronger provenance, null comparisons, covariance handling, observed-channel cycle closure, and admission gates.

Current research home:

https://www.lucernaveritas.ai/

The historic record follows.

---

[📄 Preprint PDF (v1.0)](preprint/A_Local_Electromagnetic_Wave_Equation_from_Spectral_Geometry.pdf)  
DOI 10.5281/zenodo.17402933

---
## 📡 Data Source

All GRAPE V1 Doppler data used by these scripts are publicly available from:

- Collins K., Gibbons J., Kazdan D., & Frissell N. (2024).  
  *Grape V1 Data: Frequency Estimation and Amplitudes of North American Time Standard Stations.*  
  Zenodo. [https://doi.org/10.5281/zenodo.13637199](https://doi.org/10.5281/zenodo.13637199)

Please download the desired years (2019–2020) and extract them under  
`data/Grape_V1/2019/` and `data/Grape_V1/2020/`.  
For methodology and network details, see Collins et al. (2023), *Earth Syst. Sci. Data* 15 (1403–1418).


---

## 📂 Repository Layout for "A Local Electromagnetic Wave Equation from Spectral Geometry"

  scripts/electromagnetism/
  ├── radio_delay_analysis.py
  └── pretty_dtau_plot.py

  scripts/MDL/
  ├── MDL_Smith.py
  └── mdla_sweep.py


The `electromagnetism` folder contains all scripts for reproducing the  
ionospheric Maxwellian regression analysis described in the paper.  
The `MDL` folder contains the β-slope and impedance post-processing  
pipeline shared with *Information-Theoretic Confirmation of the α-Affine Thread Frame*.

---

## ▶️ Quick-start Example

To reproduce Figure 3 (“Independent Maxwellian Verification”):

```bash
python -m scripts.analysis_pipeline.radio_delay_analysis \
    --data-root data/Grape_V1/2019/ data/Grape_V1/2020/ \
    --out-dir data/results/radio_delay/Grape_V1/ \
    --pool-by-grid --region-prefix EN

python -m scripts.analysis_pipeline.pretty_dtau_plot \
    --rel-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_rel_windows.csv \
    --fit-csv data/results/radio_delay/Grape_V1/pooled/EN/pooled_EN_DTau_fit_windows.csv \
    --out figures/pooled_EN_DTau_vs_invnu2_pretty.png

---

## 🧭 Notes on Data Reuse

The **MDL_Smith.py** analysis uses the same source datasets and results already published with the MDL paper (DOI 10.5281/zenodo.17335815).  
Those canonical CSVs are already included in this repository under  
`results/MDL_Smith/`.  

Therefore:
- No new MDL data need to be downloaded.
- Only the large GRAPE V1 Doppler dataset is hosted externally (see above).  

This ensures full scientific reproducibility without re-hosting third-party data.

