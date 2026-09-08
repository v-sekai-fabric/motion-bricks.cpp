"""Pick the training GPU by name before CUDA initialises.

Torch enumerates the desk's GPUs fastest-first, which inverts nvidia-smi's order,
so an index is not a stable choice; a name is. Setting CUDA_DEVICE_ORDER to the
bus order makes CUDA_VISIBLE_DEVICES agree with nvidia-smi, and the index is then
read from nvidia-smi itself.
"""
from __future__ import annotations

import os
import subprocess
import sys


def select_gpu_by_name(name_fragment: str = "4090") -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("MJLAB_MOTIONBRICKS_GPU_SELECTED"):
        return int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout
    rows = [line.split(",", 1) for line in out.strip().splitlines() if "," in line]
    matches = [int(i.strip()) for i, n in rows if name_fragment in n]
    if not matches:
        sys.exit(f"FAIL: no GPU named like '{name_fragment}' among {[n.strip() for _, n in rows]}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(matches[0])
    os.environ["MJLAB_MOTIONBRICKS_GPU_SELECTED"] = "1"
    return matches[0]
