#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from lerobot.processor.hil_processor import InterventionActionProcessorStep
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)


def load_env_cfg() -> dict:
    cfg_path = Path("src/lerobot/configs/env_config_pika_piper.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_obs(joints: np.ndarray, gripper: float) -> dict[str, float]:
    obs = {f"joint_{index + 1}.pos": float(joints[index]) for index in range(6)}
    obs["gripper.pos"] = float(gripper)
    return obs


class GymManipulatorPikaIKRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ik_cfg = cfg["env"]["processor"]["inverse_kinematics"]
        self.motor_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        self.kinematics = RobotKinematics(
            urdf_path=ik_cfg["urdf_path"],
            target_frame_name=ik_cfg["target_frame_name"],
            joint_names=self.motor_names[:-1],
        )
        self.steps = [
            InterventionActionProcessorStep(
                use_gripper=True,
                terminate_on_success=True,
                force_teleop_action=True,
            ),
            MapTensorToDeltaActionDictStep(use_gripper=True),
            MapDeltaActionToRobotActionStep(noise_threshold=0.005),
            EEReferenceAndDelta(
                kinematics=self.kinematics,
                end_effector_step_sizes=ik_cfg["end_effector_step_sizes"],
                motor_names=self.motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(end_effector_bounds=ik_cfg["end_effector_bounds"]),
            GripperVelocityToJoint(clip_max=0.08, speed_factor=0.0, discrete_gripper=False, absolute_input=True),
            InverseKinematicsRLStep(
                kinematics=self.kinematics,
                motor_names=self.motor_names,
                initial_guess_current_joints=False,
            ),
        ]

    def run(self, teleop_action: dict[str, float], observation: dict[str, float]) -> dict[str, float]:
        transition = create_transition(
            observation=observation,
            action=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            info={},
            complementary_data={"teleop_action": dict(teleop_action)},
        )
        for step in self.steps:
            transition = step(transition)
        action = transition[TransitionKey.ACTION]
        assert isinstance(action, dict)
        return {key: float(value) for key, value in action.items() if key.endswith(".pos")}


def run_gym_manipulator_pika_ik(
    runner: GymManipulatorPikaIKRunner,
    teleop_action: dict[str, float],
    observation: dict[str, float],
) -> dict[str, float]:
    return runner.run(teleop_action, observation)


def run_record_identity_chain(teleop_action: dict[str, float]) -> dict[str, float]:
    return dict(teleop_action)


def can_send_to_piper(action: dict[str, float]) -> bool:
    expected = [f"joint_{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    return all(key in action for key in expected)


def jitter_drift_probe(runner: GymManipulatorPikaIKRunner, observation: dict[str, float], steps: int = 10) -> float:
    joints = np.array([observation[f"joint_{index}.pos"] for index in range(1, 7)], dtype=np.float64)
    for _ in range(steps):
        teleop_action = {
            "delta_x": 0.0032,
            "delta_y": -0.0032,
            "delta_z": 0.0032,
            "gripper": 0.5,
        }
        out = run_gym_manipulator_pika_ik(runner, teleop_action, observation)
        joints = np.array([out[f"joint_{index}.pos"] for index in range(1, 7)], dtype=np.float64)
        observation = make_obs(joints, out["gripper.pos"])
    return float(np.linalg.norm(joints))


def main() -> None:
    cfg = load_env_cfg()
    runner = GymManipulatorPikaIKRunner(cfg)
    observation = make_obs(np.zeros(6, dtype=np.float64), 0.04)

    print("=== Defaults from env_config_pika_piper.json ===")
    print("teleop.output_mode:", cfg["env"]["teleop"].get("output_mode"))
    print("processor.control_mode:", cfg["env"]["processor"].get("control_mode"))
    print("reset.fixed_reset_joint_positions:", cfg["env"]["processor"]["reset"].get("fixed_reset_joint_positions"))
    print("ik.target_frame_name:", cfg["env"]["processor"]["inverse_kinematics"].get("target_frame_name"))
    print()

    teleop_joint = {
        "joint_1.pos": 0.0,
        "joint_2.pos": 0.0,
        "joint_3.pos": 0.0,
        "joint_4.pos": 0.0,
        "joint_5.pos": 0.0,
        "joint_6.pos": 0.0,
        "gripper.pos": 0.04,
    }

    teleop_ee_delta = {
        "delta_x": 0.01,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "gripper": 0.5,
    }

    print("=== Case A: same JOINT input ===")
    record_out_joint = run_record_identity_chain(teleop_joint)
    gym_out_from_joint = run_gym_manipulator_pika_ik(runner, teleop_joint, observation)
    print("record_identity can_send_to_piper:", can_send_to_piper(record_out_joint))
    print("gym_manipulator_ik can_send_to_piper:", can_send_to_piper(gym_out_from_joint))
    print("record_identity action:", record_out_joint)
    print("gym_manipulator_ik action:", gym_out_from_joint)
    print()

    print("=== Case B: same EE_DELTA input ===")
    record_out_delta = run_record_identity_chain(teleop_ee_delta)
    gym_out_from_delta = run_gym_manipulator_pika_ik(runner, teleop_ee_delta, observation)
    print("record_identity can_send_to_piper:", can_send_to_piper(record_out_delta))
    print("gym_manipulator_ik can_send_to_piper:", can_send_to_piper(gym_out_from_delta))
    print("record_identity action:", record_out_delta)
    print("gym_manipulator_ik action:", gym_out_from_delta)
    print()

    print("=== Drift probe (10 steps tiny enabled deltas) ===")
    drift_norm = jitter_drift_probe(runner, observation, steps=10)
    print("joint L2 norm after 10 tiny-delta steps:", drift_norm)


if __name__ == "__main__":
    main()
