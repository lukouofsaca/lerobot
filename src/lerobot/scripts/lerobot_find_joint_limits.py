#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to find joint limits and end-effector bounds via teleoperation.

Example:

```shell
lerobot-find-joint-limits \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760432981 \
  --robot.id=black \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem58760434471 \
  --teleop.id=blue \
  --urdf_path=<user>/SO-ARM100-main/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper \
  --teleop_time_s=30 \
  --warmup_time_s=5 \
  --control_loop_fps=30
```
"""

import time
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    pika,
    so_leader,
)
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path: str
    target_frame_name: str = "gripper"

    # Duration of the recording phase in seconds
    teleop_time_s: float = 30
    # Duration of the warmup phase in seconds
    warmup_time_s: float = 5
    # Control loop frequency
    control_loop_fps: int = 30


def resolve_urdf_for_placo(urdf_path: str) -> str:
    urdf_file = Path(urdf_path).resolve()
    text = urdf_file.read_text(encoding="utf-8")
    package_pattern = re.compile(r"package://([^/]+)/")
    package_names = sorted(set(package_pattern.findall(text)))
    if not package_names:
        return str(urdf_file)

    parts = urdf_file.parts
    if "share" not in parts:
        return str(urdf_file)

    share_idx = parts.index("share")
    if share_idx + 1 >= len(parts):
        return str(urdf_file)

    package_root = Path(*parts[: share_idx + 2]).as_posix().rstrip("/") + "/"
    rewritten = text
    for package_name in package_names:
        rewritten = rewritten.replace(f"package://{package_name}/", package_root)

    tmp = tempfile.NamedTemporaryFile(prefix="piper_joint_limits_", suffix=".urdf", delete=False)
    tmp.write(rewritten.encode("utf-8"))
    tmp.close()
    return tmp.name


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    print(f"Connecting to robot: {cfg.robot.type}...")
    teleop.connect()
    robot.connect()
    print("Devices connected.")

    # Initialize Kinematics
    try:
        urdf_for_placo = resolve_urdf_for_placo(cfg.urdf_path)
        print(f"Using URDF for placo: {urdf_for_placo}")
        observation_joint_keys = [name for name in robot.bus.motors.keys() if name != "gripper"]
        kinematics_joint_names = [name.replace("_", "") for name in observation_joint_keys]
        kinematics = RobotKinematics(
            urdf_for_placo,
            cfg.target_frame_name,
            joint_names=kinematics_joint_names,
        )
    except Exception as e:
        print(f"Error initializing kinematics: {e}")
        print("Ensure URDF path and target frame name are correct.")
        robot.disconnect()
        teleop.disconnect()
        return

    # Initialize variables
    max_pos = None
    min_pos = None
    max_ee = None
    min_ee = None

    start_t = time.perf_counter()
    warmup_done = False

    print("\n" + "=" * 40)
    print(f"  WARMUP PHASE ({cfg.warmup_time_s}s)")
    print("  Move the robot freely to ensure control works.")
    print("  Data is NOT being recorded yet.")
    print("=" * 40 + "\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Teleoperation Control Loop
            action = teleop.get_action()
            robot.send_action(action)

            # 2. Read Observations
            observation = robot.get_observation()
            raw_joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors], dtype=float)
            raw_kinematics_joint_positions = np.array(
                [observation[f"{key}.pos"] for key in observation_joint_keys], dtype=float
            )

            if cfg.robot.type == "piper_follower":
                # Piper SDK feedback is in 0.001 degree units.
                kinematics_joint_positions = raw_kinematics_joint_positions / 1000.0
                joint_positions = np.deg2rad(raw_joint_positions / 1000.0)
            else:
                kinematics_joint_positions = raw_kinematics_joint_positions
                joint_positions = raw_joint_positions

            # 3. Calculate Kinematics
            # Forward kinematics to get (x, y, z) translation
            ee_pos = kinematics.forward_kinematics(kinematics_joint_positions)[:3, 3]

            current_time = time.perf_counter()
            elapsed = current_time - start_t

            # 4. Handle Phases
            if elapsed < cfg.warmup_time_s:
                # Still in warmup
                pass

            else:
                # Phase Transition: Warmup -> Recording
                if not warmup_done:
                    print("\n" + "=" * 40)
                    print("  RECORDING STARTED")
                    print("  Move robot to ALL joint limits.")
                    print("  Press Ctrl+C to stop early and save results.")
                    print("=" * 40 + "\n")

                    # Initialize limits with current position at start of recording
                    max_pos = joint_positions.copy()
                    min_pos = joint_positions.copy()
                    max_ee = ee_pos.copy()
                    min_ee = ee_pos.copy()
                    warmup_done = True

                # Update Limits
                max_ee = np.maximum(max_ee, ee_pos)
                min_ee = np.minimum(min_ee, ee_pos)
                max_pos = np.maximum(max_pos, joint_positions)
                min_pos = np.minimum(min_pos, joint_positions)

                # Time check
                recording_time = elapsed - cfg.warmup_time_s
                remaining = cfg.teleop_time_s - recording_time

                # Simple throttle for print statements (every ~1 sec)
                if int(recording_time * 100) % 100 == 0:
                    print(f"Time remaining: {remaining:.1f}s", end="\r")

                if recording_time > cfg.teleop_time_s:
                    print("\nTime limit reached.")
                    break

            precise_sleep(max(1.0 / cfg.control_loop_fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping safely...")

    finally:
        # Safety: Disconnect devices
        print("\nDisconnecting devices...")
        robot.disconnect()
        teleop.disconnect()

    # Results Output
    if max_pos is not None:
        print("\n" + "=" * 40)
        print("FINAL RESULTS")
        print("=" * 40)

        # Rounding for readability
        r_max_ee = np.round(max_ee, 4).tolist()
        r_min_ee = np.round(min_ee, 4).tolist()
        r_max_pos = np.round(max_pos, 4).tolist()
        r_min_pos = np.round(min_pos, 4).tolist()

        print("\n# End Effector Bounds (x, y, z):")
        print(f"max_ee = {r_max_ee}")
        print(f"min_ee = {r_min_ee}")

        print("\n# Joint Position Limits (radians):")
        print(f"max_pos = {r_max_pos}")
        print(f"min_pos = {r_min_pos}")

    else:
        print("No data recorded (exited during warmup).")


def main():
    find_joint_and_ee_bounds()


if __name__ == "__main__":
    main()
