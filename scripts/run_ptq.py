"""CLI script to run a Post-Training Quantization (PTQ) simulation."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ptq_sim.cli import run_ptq_main

if __name__ == "__main__":
    run_ptq_main()
