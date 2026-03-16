#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 框架下直接控制 Piper 机械臂 Demo（不需要 Pika 遥操）。

功能:
1. 使用 PiperFollower 连接 & 使能机械臂
2. 移到初始位姿
3. 依次执行几个预设关节目标位，验证机械臂是否响应
4. 读取关节状态并打印
5. 断开连接

运行:
    cd ~/sda/zyx/lerobot
    python examples/pika_to_piper/demo_lerobot_piper_ctrl.py --can-name can0
"""

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower


def parse_args():
    p = argparse.ArgumentParser(description="LeRobot PiperFollower 控制 demo")
    p.add_argument("--can-name", type=str, default="can0", help="CAN 端口名")
    p.add_argument("--move-time", type=float, default=2.0, help="每个目标位的运行时间 (秒)")
    return p.parse_args()


# ---------- 预设关节目标序列 (单位: 弧度, 最后一个是 gripper 0~0.08 m) ----------
WAYPOINTS = [
    # 初始位姿
    [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0],
    # joint_1 左转 30°, 其余不变
    [math.radians(30), 0.0, 0.0, 0.0, 0.52, 0.0, 0.0],
    # joint_2 抬起 20°
    [math.radians(30), math.radians(20), 0.0, 0.0, 0.52, 0.0, 0.0],
    # joint_5 翻转, 夹爪张开
    [math.radians(30), math.radians(20), 0.0, 0.0, math.radians(60), 0.0, 0.05],
    # 夹爪关闭
    [math.radians(30), math.radians(20), 0.0, 0.0, math.radians(60), 0.0, 0.0],
    # 回到初始位姿
    [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0],
]

MOTOR_ORDER = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def joints_to_action(joints: list[float]) -> dict[str, float]:
    """将 7 元素列表转为 PiperFollower.send_action 需要的字典格式。"""
    return {f"{name}.pos": val for name, val in zip(MOTOR_ORDER, joints)}


def main():
    args = parse_args()

    # ---- 1. 构建配置 & 创建 robot ----
    cfg = PiperFollowerConfig()
    cfg.motors.can_name = args.can_name
    robot = PiperFollower(cfg)

    # ---- 2. 连接 (使能 + calibrate 移到初始位) ----
    print(f"[INFO] 连接 Piper (CAN={args.can_name}) ...")
    robot.connect()
    print("[INFO] 连接成功，等待 1s 让机械臂稳定 ...")
    time.sleep(1.0)

    try:
        # ---- 3. 读取当前关节状态 ----
        obs = robot.get_observation()
        print("[INFO] 当前关节状态:")
        for k, v in obs.items():
            if isinstance(v, float):
                print(f"  {k} = {v:.4f}")

        # ---- 4. 依次发送 waypoint ----
        for i, wp in enumerate(WAYPOINTS):
            action = joints_to_action(wp)
            print(f"\n[MOVE {i}] 目标: {[f'{v:.3f}' for v in wp]}")
            robot.send_action(action)

            # 在 move_time 内持续发送同一目标 (保证控制循环)
            t0 = time.time()
            while time.time() - t0 < args.move_time:
                robot.send_action(action)
                time.sleep(0.02)  # 50 Hz

            # 打印到达后的关节状态
            obs = robot.get_observation()
            actual = [obs.get(f"{m}.pos", 0.0) for m in MOTOR_ORDER]
            print(f"  实际: {[f'{v:.3f}' for v in actual]}")

        print("\n[INFO] 所有 waypoint 完成")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        print("[INFO] 断开连接 ...")
        robot.disconnect()
        print("[INFO] 完成")


if __name__ == "__main__":
    main()
