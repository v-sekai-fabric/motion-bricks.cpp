"""mjlab's train entry with this package's tasks registered and the GPU picked by name.

    pixi run train Mjlab-Velocity-Flat-MotionBricks-G1 --env.scene.num-envs 1024 --agent.max-iterations 300
"""
from __future__ import annotations

from mjlab_motionbricks.gpu import select_gpu_by_name


def main() -> None:
    select_gpu_by_name("4090")
    import mjlab_motionbricks  # noqa: F401
    from mjlab.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
