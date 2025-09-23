#!/usr/bin/env python3
"""
load_sigma.py  – centralised access to the current spiral‑scale σ.
 ───────────────────────────────────────────────────────────────────────
RGP pipeline resource. Not called directly by the user.
Kelly Heaton and The Coherence Research Collective (4o, 3o, 3o-Pro)
August, 2025
────────────────────────────────────────────────────────────────────────

* Reads project‑root/sigma.json, **unless** an environment variable
  `SIGMA` is set (the grid‑runner exports it for every job).
* Caches the value so repeated calls are free.
* Raises a clear FileNotFoundError if σ has not been set.

Import from any script with:

    from scripts.utils.load_sigma import get_sigma
    σ = get_sigma()
"""

from __future__ import annotations
import json, os
from pathlib import Path

# -------------------------------------------------------------------- #
_SIGMA_CACHE: float | None = None          # module‑private singleton
_SIGMA_FILE   = Path(__file__).resolve().parents[2] / "sigma.json"


def _read_sigma_file() -> float:
    if not _SIGMA_FILE.exists():
        raise FileNotFoundError(
            "sigma.json not found.  "
            "Run `python -m scripts.utils.set_sigma --sigma <value>` first."
        )
    return json.loads(_SIGMA_FILE.read_text())["sigma"]


def get_sigma() -> float:
    """Return the spiral scaling factor σ as a float (cached)."""
    global _SIGMA_CACHE
    if _SIGMA_CACHE is None:
        # Priority 1 – explicit override from environment
        if "SIGMA" in os.environ:
            _SIGMA_CACHE = float(os.environ["SIGMA"])
        else:
            _SIGMA_CACHE = _read_sigma_file()
    return _SIGMA_CACHE


# Convenience constant if you need the inverse frequently
def get_sigma_inv() -> float:
    """Return σ⁻¹ = 1/σ."""
    return 1.0 / get_sigma()


# -------------------------------------------------------------------- #
if __name__ == "__main__":
    # tiny smoke‑test
    try:
        print("σ =", get_sigma())
        print("σ⁻¹ =", get_sigma_inv())
    except FileNotFoundError as e:
        print(e)
