"""Task registration: the flat velocity task at 50 Hz with the ROM terms."""
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_motionbricks.tasks.env_cfg import motionbricks_g1_flat_env_cfg
from mjlab_motionbricks.tasks.rl_cfg import motionbricks_g1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MotionBricks-G1",
    env_cfg=motionbricks_g1_flat_env_cfg(),
    play_env_cfg=motionbricks_g1_flat_env_cfg(play=True),
    rl_cfg=motionbricks_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
