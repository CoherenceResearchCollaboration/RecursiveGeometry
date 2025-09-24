#!/usr/bin/env python3
"""
plot_gamma_affinity_heatmap_matrix.py
───────────────────────────────────────
Visualizes γ-attractor affinity as a log-scaled heatmap.
Each cell represents the resonance strength (obs_hits) for a specific (ion, γ) pair.

• Hue = log₁₀(obs_hits + 1)
• Rows = ions
• Columns = attractor exponents γ

Output:
    data/results/plots/attractor_affinity_matrix.png # either beta or mu-1 (change comments)

Usage:
    python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix

python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix --sort_mode peak
python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix --sort_mode centroid
python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix --sort_mode cluster

python -m scripts.views.cross_ion.plot_gamma_affinity_heatmap_matrix \
  --sort_mode cluster \
  --filter_site 2,1

--sort_mode peak --filter_site 2,2


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy.cluster.hierarchy import linkage, leaves_list

# ───────────────────────────────────────────────
# Load data
# csv_path = "data/meta/gamma_attractor_affinity_bio_vacuum_beta.csv"
csv_path = "data/meta/gamma_attractor_affinity_bio_vacuum_mu-1.csv"

df = pd.read_csv(csv_path)

df["ion"] = df["ion"].astype(str)
df["gamma"] = df["gamma_bin"].round(2)

# Pivot table: rows = ion, columns = gamma, values = obs_hits
# ─────────────── ARGUMENTS ───────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sort_mode", choices=["alphabet", "peak", "centroid", "cluster"], default="alphabet",
                    help="Ion sorting mode: alphabet | peak | centroid | cluster (default: alphabet)")
parser.add_argument("--filter_site", type=str, default=None,
                    help="Restrict data to (n_i,n_k) site, e.g. '2,2'")

args = parser.parse_args()

# Optional site filter
if args.filter_site:
    ni_str, nk_str = args.filter_site.split(",")
    ni, nk = int(ni_str), int(nk_str)
    df = df[(df["n_i"] == ni) & (df["n_k"] == nk)]
    print(f"✓ Filtering by site: (n_i={ni}, n_k={nk}) — rows now: {len(df)}")

# ─────────────── DATA PREP ───────────────
df_heatmap = df.pivot_table(index="ion", columns="gamma", values="obs_hits", fill_value=0)
df_log = np.log10(df_heatmap + 1)

# ─────────────── SORTING METHODS ───────────────
if args.sort_mode == "peak":
    df_peaks = df_heatmap.idxmax(axis=1).to_frame(name="peak_gamma")
    df_log["peak_gamma"] = df_peaks["peak_gamma"]
    df_log = df_log.sort_values("peak_gamma").drop(columns="peak_gamma")

elif args.sort_mode == "centroid":
    weights = df_heatmap.columns.values
    centroids = (df_heatmap * weights).sum(axis=1) / df_heatmap.sum(axis=1)
    df_log["gamma_centroid"] = centroids
    df_log = df_log.sort_values("gamma_centroid").drop(columns="gamma_centroid")

elif args.sort_mode == "cluster":
    from scipy.cluster.hierarchy import linkage, leaves_list
    Z = linkage(df_log.values, method="ward")
    order = leaves_list(Z)
    df_log = df_log.iloc[order]
# else 'alphabet' — do nothing

# Output path setup
outdir = Path("data/results/plots")
outdir.mkdir(parents=True, exist_ok=True)

# Plot heatmap
fig, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(
    df_log,
    cmap="plasma",
    cbar_kws={"label": "log₁₀(obs_hits + 1)"},
    linewidths=0.05,
    linecolor="gray",
    ax=ax
)

# Compose title dynamically
title = "γ-Attractor Affinity Matrix"

if args.filter_site:
    title += f" @ site ({args.filter_site})"

title += f"  |  sort: {args.sort_mode}  |  scale: log₁₀(obs_hits + 1)"
plt.title(title, fontsize=14)

plt.xlabel("γ (attractor exponent)")
plt.ylabel("Ion")
plt.tight_layout()

# Save figure
# Compose filename
filename_parts = ["attractor_affinity"]
if args.filter_site:
    filename_parts.append(f"site-{args.filter_site.replace(',','_')}")
filename_parts.append(f"sort-{args.sort_mode}")
filename_parts.append("mu-1_matrix.png")

out_name = "_".join(filename_parts)
fig.savefig(outdir / out_name, dpi=300)

print(f"✓ BOING! Saved heatmap → {out_name}")

