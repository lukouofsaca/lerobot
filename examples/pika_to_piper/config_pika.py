import os
from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


def _default_piper_description_dir() -> str:
    env_path = os.getenv("PIPER_DESCRIPTION_DIR", "")
    candidates = [
        env_path,
        "/home/zhbs/pika_ros/install/piper_description/share/piper_description",
        os.path.expanduser("~/pika_ros/install/piper_description/share/piper_description"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


@TeleoperatorConfig.register_subclass("pika")
@dataclass
class PikaTeleoperatorConfig(TeleoperatorConfig):
    sense_port: str = "/dev/ttyUSB0"
    tracker_device: str = "T20"
    piper_description_dir: str = field(default_factory=_default_piper_description_dir)
    gripper_xyzrpy: list[float] = field(default_factory=lambda: [0.19, 0.0, 0.2, 0.0, 0.0, 0.0])
    home_joint_state: list[float] = field(default_factory=lambda: [0.0] * 6)
    filter_min_cutoff: float = 1.0
    filter_beta_pos: float = 1.0
    filter_beta_rot: float = 0.5
    ik_weight_position: float = 1.0
    ik_weight_orientation: float = 0.15
    ik_weight_regularization: float = 0.005
    ik_weight_smoothing: float = 0.05
    ik_jump_threshold_deg: float = 30.0
    lift: bool = False
    enable_safety_guard: bool = True
    use_command_state_enable: bool = False
    pose_timeout_sec: float = 0.25
    max_joint_step_deg: float = 8.0
    max_consecutive_ik_failures: int = 8
    force_disable_on_pose_stale: bool = True
