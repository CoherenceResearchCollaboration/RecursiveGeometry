#!/usr/bin/env python3
"""
set_sigma.py – Write or overwrite the sigma.json file used for spiral seeding.
"""

import argparse
import json
from pathlib import Path

SIGMA_FILE = Path(__file__).resolve().parents[2] / "sigma.json"

def main():
    parser = argparse.ArgumentParser(description="Set global sigma value.")
    parser.add_argument("--sigma", type=float, required=True, help="Spiral-scale sigma value (float)")
    args = parser.parse_args()

    sigma_val = args.sigma
    SIGMA_FILE.write_text(json.dumps({"sigma": sigma_val}, indent=2))
    print(f"[✓] Set σ = {sigma_val} → {SIGMA_FILE}")

if __name__ == "__main__":
    main()
