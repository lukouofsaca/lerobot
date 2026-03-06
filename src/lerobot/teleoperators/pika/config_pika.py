from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pika")
@dataclass
class PikaTeleoperatorConfig(TeleoperatorConfig):
    sense_port: str = "/dev/ttyUSB0"
    tracker_device: str = "T20"
    piper_description_dir: str = ""
    gripper_xyzrpy: list[float] = field(default_factory=lambda: [0.19, 0.0, 0.2, 0.0, 0.0, 0.0])
    home_joint_state: list[float] = field(default_factory=lambda: [0.0] * 6)
    filter_min_cutoff: float = 1.0
    filter_beta_pos: float = 1.0
    filter_beta_rot: float = 0.5
    lift: bool = False
