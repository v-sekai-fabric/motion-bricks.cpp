"""The G1 sim-to-real task family (RFD 2238). Importing the package registers the tasks."""

from mjlab_motionbricks import tasks as _tasks  # noqa: F401  (registration on import)

__all__ = ["TASK_FLAT"]
TASK_FLAT = "Mjlab-Velocity-Flat-MotionBricks-G1"
