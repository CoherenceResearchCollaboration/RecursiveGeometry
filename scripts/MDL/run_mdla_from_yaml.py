#!/usr/bin/env python3
"""
Simple YAML runner for MDL analyses.

Usage: PYTHONPATH=. python scripts/MDL/run_mdla_from_yaml.py <config.yml>

For example:
PYTHONPATH=. python scripts/MDL/run_mdla_from_yaml.py configs/mdl_runs/solar.yml
"""

import sys, subprocess, yaml, shlex

if len(sys.argv) < 2:
    print("Usage: python tools/run_mdla_from_yaml.py <config.yml>")
    sys.exit(1)

# Load YAML file
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Build command list
cmd = [
    "python", "-m", "scripts.MDL.mdla_sweep",
    "--events-csv", cfg["events_csv"],
    "--outdir", cfg["outdir"],
    "--label", cfg["label"],
    "--series-name", cfg["series_name"],
    "--grid", str(cfg["grid"]),
    "--kmax", str(cfg["kmax"]),
    "--beta-list", str(cfg["beta_list"]),
    "--nulls", str(cfg["nulls"]),
    "--null-beta", str(cfg["null_beta"]),
    "--null-windows", str(cfg["null_windows"]),
]

print("\nRunning:", " ".join(shlex.quote(x) for x in cmd), "\n")
subprocess.run(cmd, check=True)
