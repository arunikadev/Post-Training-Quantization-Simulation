"""Inspect keys, tensor shapes, dtypes, and FP32 size of any .pt/.pth checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ptq_sim.models.loader import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint or state_dict.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt/.pth file.")
    parser.add_argument("--state-dict-key", type=str, default=None, help="Optional nested key.")
    parser.add_argument("--max-preview", type=int, default=30, help="Max tensor entries to print.")
    args = parser.parse_args()

    info = inspect_checkpoint(
        checkpoint_path=args.checkpoint,
        state_dict_key=args.state_dict_key,
        max_preview=args.max_preview,
    )
    print("\n=== Checkpoint Inspection ===")
    print(f"Path               : {info['checkpoint_path']}")
    print(f"Detected Format    : {info['detected_format']}")
    print(f"Disk Size          : {info['file_size_mb']:.4f} MB")
    print(f"Tensor Entries     : {info['num_state_dict_entries']}")
    print(f"Total Elements     : {info['total_tensor_elements']:,}")
    print(f"FP32 Equivalent    : {info['fp32_equivalent_mb']:.4f} MB")
    print(f"DType Counts       : {info['dtype_counts']}")
    if info["top_level_keys"]:
        print(f"Top-Level Keys (<=30): {info['top_level_keys'][:10]}")
    print("\n--- Sample Tensors ---")
    for entry in info["preview_tensors"]:
        print(f"  {entry['key']:<45} shape={str(entry['shape']):<20} dtype={entry['dtype']}")


if __name__ == "__main__":
    main()
