"""mjlab's play entry with this package's tasks registered and the GPU picked by name."""
from __future__ import annotations

from mjlab_motionbricks.gpu import select_gpu_by_name


def main() -> None:
    select_gpu_by_name("4090")
    import mjlab_motionbricks  # noqa: F401
    from mjlab.scripts.play import main as play_main

    play_main()


if __name__ == "__main__":
    main()
