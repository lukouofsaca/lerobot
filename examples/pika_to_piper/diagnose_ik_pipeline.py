#!/usr/bin/env python3

"""Pika->Piper IK 管线分阶段诊断脚本（不修改 IK 本体）。

用途：
1) 先验证配置与 URDF 一致性（joint / frame）。
2) 再验证依赖导入与 RobotKinematics 初始化。
3) 最后逐步执行 RL IK action pipeline，定位具体失败步骤。

示例：
  conda run -n lerobot python examples/pika_to_piper/diagnose_ik_pipeline.py \
    --env-config src/lerobot/configs/env_config_pika_piper.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import create_transition
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.processor.core import TransitionKey
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose IK pipeline stage-by-stage")
    parser.add_argument(
        "--env-config",
        type=str,
        default="src/lerobot/configs/env_config_pika_piper.json",
        help="环境配置 JSON 路径",
    )
    parser.add_argument("--urdf", type=str, default=None, help="覆盖配置中的 URDF 路径")
    parser.add_argument("--target-frame", type=str, default=None, help="覆盖配置中的 target_frame_name")
    parser.add_argument(
        "--policy-action",
        type=float,
        nargs=4,
        default=[0.2, 0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ", "GRIPPER"),
        help="模拟策略输出 [delta_x, delta_y, delta_z, gripper]",
    )
    parser.add_argument(
        "--initial-joints-deg",
        type=float,
        nargs="*",
        default=None,
        help="初始观测关节角（度），长度需与电机数一致；默认全 0",
    )
    parser.add_argument("--print-traceback", action="store_true", help="失败时打印完整堆栈")
    return parser.parse_args()


def parse_urdf_names(urdf_path: str) -> tuple[list[str], list[str]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = [elem.attrib["name"] for elem in root.findall("link") if "name" in elem.attrib]
    joints = [
        elem.attrib["name"]
        for elem in root.findall("joint")
        if "name" in elem.attrib and elem.attrib.get("type") not in {"fixed"}
    ]
    return links, joints


def stage(name: str) -> None:
    print("\n" + "=" * 80)
    print(f"[STAGE] {name}")
    print("=" * 80)


def fail(name: str, exc: Exception, show_tb: bool) -> None:
    print(f"[FAIL] {name}: {exc}")
    if show_tb:
        traceback.print_exc()


def main() -> int:
    args = parse_args()

    env_config_path = Path(args.env_config)
    if not env_config_path.is_absolute():
        env_config_path = REPO_ROOT / env_config_path

    with env_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_cfg = cfg["env"]
    ik_cfg = env_cfg["processor"]["inverse_kinematics"]
    urdf_path = args.urdf or ik_cfg["urdf_path"]
    target_frame_name = args.target_frame or ik_cfg["target_frame_name"]
    end_effector_step_sizes = ik_cfg["end_effector_step_sizes"]
    end_effector_bounds = ik_cfg["end_effector_bounds"]

    motor_names = list(env_cfg["robot"]["motors"]["motors"].keys())
    ik_joint_names = [name for name in motor_names if name != "gripper"]

    stage("配置与 URDF 一致性")
    print(f"env_config: {env_config_path}")
    print(f"urdf_path: {urdf_path}")
    print(f"target_frame_name: {target_frame_name}")
    print(f"motor_names: {motor_names}")
    print(f"ik_joint_names: {ik_joint_names}")

    try:
        links, urdf_joints = parse_urdf_names(urdf_path)
    except Exception as exc:
        fail("解析 URDF", exc, args.print_traceback)
        return 2

    print(f"URDF links({len(links)}): {links}")
    print(f"URDF joints({len(urdf_joints)}): {urdf_joints}")
    print(f"target frame in URDF links: {target_frame_name in links}")
    unresolved = [j for j in ik_joint_names if (j not in urdf_joints and j.replace('_', '') not in urdf_joints)]
    print(f"unresolved ik joints (by exact/alias): {unresolved}")

    stage("RobotKinematics 初始化")
    try:
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=ik_joint_names,
        )
        print("[PASS] RobotKinematics created")
        print(f"resolved joint_names in solver: {kinematics.joint_names}")
    except Exception as exc:
        fail("RobotKinematics 初始化", exc, args.print_traceback)
        print("结论：问题发生在依赖导入或 IK 求解器初始化阶段，尚未进入 action pipeline 逻辑。")
        return 3

    stage("逐步执行 IK action pipeline")
    if args.initial_joints_deg is None or len(args.initial_joints_deg) == 0:
        q0 = np.zeros(len(motor_names), dtype=float)
    else:
        if len(args.initial_joints_deg) != len(motor_names):
            print(
                f"[FAIL] initial_joints_deg 长度({len(args.initial_joints_deg)}) != 电机数({len(motor_names)})"
            )
            return 4
        q0 = np.array(args.initial_joints_deg, dtype=float)

    observation = {f"{name}.pos": float(q0[idx]) for idx, name in enumerate(motor_names)}
    action_tensor = torch.tensor(args.policy_action, dtype=torch.float32)

    transition = create_transition(
        observation=observation,
        action=action_tensor,
        complementary_data={},
    )

    steps = [
        ("MapTensorToDeltaActionDictStep", MapTensorToDeltaActionDictStep(use_gripper=True)),
        ("MapDeltaActionToRobotActionStep", MapDeltaActionToRobotActionStep()),
        (
            "EEReferenceAndDelta",
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes=end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
        ),
        ("EEBoundsAndSafety", EEBoundsAndSafety(end_effector_bounds=end_effector_bounds)),
        ("GripperVelocityToJoint", GripperVelocityToJoint(clip_max=90.0, speed_factor=1.0, discrete_gripper=True)),
        (
            "InverseKinematicsRLStep",
            InverseKinematicsRLStep(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=False,
            ),
        ),
    ]

    for name, proc in steps:
        try:
            transition = proc(transition)
            action_snapshot = transition[TransitionKey.ACTION]
            if isinstance(action_snapshot, dict):
                print(f"[PASS] {name}: action keys -> {sorted(action_snapshot.keys())}")
            else:
                print(f"[PASS] {name}: action type -> {type(action_snapshot).__name__}")
        except Exception as exc:
            fail(name, exc, args.print_traceback)
            print(f"结论：失败发生在步骤 {name}，可聚焦该步骤输入输出约定。")
            return 5

    final_action = transition[TransitionKey.ACTION]
    stage("最终结果")
    print("[PASS] 整条 IK action pipeline 可执行")
    if isinstance(final_action, dict):
        summary_keys = [f"{name}.pos" for name in motor_names]
        summary = {k: final_action.get(k, None) for k in summary_keys}
        print(f"final joint action: {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
