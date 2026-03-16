#!/usr/bin/env python3

"""LeRobot 框架下的 Pika -> 单个 Piper 遥操 Demo。

特点:
1. 使用 PikaTeleoperator + PiperFollower 抽象层（不是裸 IK 调用）
2. 每帧关节步长限制始终生效，避免原始 IK 跳变直接下发
3. 默认启用 safety guard 处理定位失效、连续 IK 失败和使能逻辑
4. 支持只打印动作（--no-send）或发送到真实机器人

运行示例:
    cd ~/sda/zyx/lerobot
    python examples/pika_to_piper/single_piper_teleop_demo.py \
      --sense-port /dev/ttyUSB81 \
      --tracker-device T20 \
      --piper-description-dir /home/zhbs/pika_ros/install/piper_description/share/piper_description \
      --can-name can0 \
      --fps 30

关闭保护示例:
    python examples/pika_to_piper/single_piper_teleop_demo.py ... --disable-safety-guard
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.teleoperators.pika.config_pika import PikaTeleoperatorConfig
from lerobot.teleoperators.pika.pika_teleoperator import PikaTeleoperator
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeRobot Pika -> single Piper teleop demo")

    parser.add_argument("--sense-port", type=str, default="/dev/ttyUSB81")
    parser.add_argument("--tracker-device", type=str, default="T20")
    parser.add_argument("--piper-description-dir", type=str, default=PikaTeleoperatorConfig().piper_description_dir)
    parser.add_argument("--can-name", type=str, default="can0")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 表示无限运行，直到 Ctrl+C")
    parser.add_argument("--no-send", action="store_true", help="只计算与打印动作，不发送到 Piper")
    parser.add_argument("--no-viz", action="store_true", help="关闭 Meshcat 可视化")

    parser.add_argument("--disable-safety-guard", action="store_true", help="关闭 safety guard")
    parser.add_argument("--use-command-state-enable", action="store_true", help="用 command_state 作为 enable 信号")
    parser.add_argument("--pose-timeout-sec", type=float, default=0.25)
    parser.add_argument("--max-joint-step-deg", type=float, default=8.0, help="每帧关节最大步长，始终生效")
    parser.add_argument("--max-consecutive-ik-failures", type=int, default=8)
    parser.add_argument("--no-force-disable-on-pose-stale", action="store_true")
    parser.add_argument("--ik-weight-position", type=float, default=1.0, help="IK 位置误差权重")
    parser.add_argument("--ik-weight-orientation", type=float, default=0.2, help="IK 姿态误差权重，增大可提升末端两节响应")
    parser.add_argument("--ik-weight-regularization", type=float, default=0.005, help="IK 零位正则权重，减小可提升折叠位形可达性")
    parser.add_argument("--ik-weight-smoothing", type=float, default=0.05, help="IK 平滑权重，过大时末端关节会发钝")
    parser.add_argument("--ik-jump-threshold-deg", type=float, default=30.0, help="IK 解跳变判定阈值")
    parser.add_argument("--filter-beta-rot", type=float, default=0.5, help="姿态滤波灵敏度，减小更稳，增大更跟手")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.piper_description_dir:
        raise ValueError("未找到 piper_description_dir，请通过 --piper-description-dir 指定，或设置 PIPER_DESCRIPTION_DIR")

    teleop_cfg = PikaTeleoperatorConfig(
        sense_port=args.sense_port,
        tracker_device=args.tracker_device,
        piper_description_dir=args.piper_description_dir,
        enable_safety_guard=(not args.disable_safety_guard),
        use_command_state_enable=args.use_command_state_enable,
        pose_timeout_sec=args.pose_timeout_sec,
        max_joint_step_deg=args.max_joint_step_deg,
        max_consecutive_ik_failures=args.max_consecutive_ik_failures,
        force_disable_on_pose_stale=(not args.no_force_disable_on_pose_stale),
        ik_weight_position=args.ik_weight_position,
        ik_weight_orientation=args.ik_weight_orientation,
        ik_weight_regularization=args.ik_weight_regularization,
        ik_weight_smoothing=args.ik_weight_smoothing,
        ik_jump_threshold_deg=args.ik_jump_threshold_deg,
        filter_beta_rot=args.filter_beta_rot,
    )
    teleop = PikaTeleoperator(teleop_cfg)

    robot = None
    if not args.no_send:
        robot_cfg = PiperFollowerConfig()
        robot_cfg.motors.can_name = args.can_name
        robot = PiperFollower(robot_cfg)

    print("Connecting teleoperator...")
    teleop.connect(calibrate=True)
    print("Teleoperator connected")

    if robot is not None:
        print("Connecting robot...")
        robot.connect()
        print("Robot connected")

    viz = None
    if not args.no_viz:
        if teleop.ik_solver is None:
            raise RuntimeError("IK solver 未初始化，无法启动 Meshcat")
        viz = MeshcatVisualizer(
            teleop.ik_solver.reduced_robot.model,
            teleop.ik_solver.reduced_robot.collision_model,
            teleop.ik_solver.reduced_robot.visual_model,
        )
        viz.initViewer(open=True)
        viz.loadViewerModel("pinocchio")
        viz.display(pin.neutral(teleop.ik_solver.reduced_robot.model))
        print(f"Meshcat URL: {viz.viewer.url()}")

    dt = 1.0 / max(args.fps, 1e-6)
    t_start = time.time()
    loop_idx = 0

    try:
        while True:
            t0 = time.perf_counter()

            action = teleop.get_action()

            if viz is not None:
                q = np.array([action[f"joint_{i}.pos"] for i in range(1, 7)], dtype=float)
                expected_nq = teleop.ik_solver.reduced_robot.model.nq
                if q.shape[0] > expected_nq:
                    q = q[:expected_nq]
                elif q.shape[0] < expected_nq:
                    q = np.pad(q, (0, expected_nq - q.shape[0]), mode="constant")
                viz.display(q)

            if robot is not None:
                robot.send_action(action)

            if loop_idx % int(max(args.fps, 1.0)) == 0:
                print(
                    f"loop={loop_idx} "
                    f"j2={action['joint_2.pos']:.3f} j3={action['joint_3.pos']:.3f} "
                    f"gripper={action['gripper.pos']:.3f}"
                )

            loop_idx += 1
            if args.duration > 0.0 and (time.time() - t_start) >= args.duration:
                break

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        teleop.disconnect()
        if robot is not None:
            robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
