"""Print the stack's versions and the CUDA devices; refuse without the 4090."""
from __future__ import annotations

import importlib.metadata as md
import sys

from mjlab_motionbricks.gpu import select_gpu_by_name


def main() -> None:
    select_gpu_by_name("4090")
    import torch

    for dist in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch", "tensordict", "onnxscript"):
        try:
            print(f"{dist:<12} {md.version(dist)}")
        except md.PackageNotFoundError:
            print(f"{dist:<12} missing")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("cuda", torch.cuda.is_available(), names)
    if not any("4090" in n for n in names):
        sys.exit("FAIL: the 4090 is not visible to torch")


if __name__ == "__main__":
    main()
