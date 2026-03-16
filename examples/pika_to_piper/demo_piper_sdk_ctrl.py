#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用 piper_sdk (C_PiperInterface_V2) 控制 Piper 机械臂 Demo。

不依赖 LeRobot 框架，展示 SDK 底层完整的:
  使能 -> 运动控制 -> 关节控制 -> 夹爪控制 -> 读状态 -> 失能

运行:
    python examples/pika_to_piper/demo_piper_sdk_ctrl.py --can-name can0
"""

import argparse
import math
import time

from piper_sdk import C_PiperInterface_V2

# rad -> SDK 单位 (0.001°) 的转换系数
JOINT_FACTOR = 57295.779513  # 1000 * 180 / π


def parse_args():
    p = argparse.ArgumentParser(description="piper_sdk 直接控制 demo")
    p.add_argument("--can-name", type=str, default="can0", help="CAN 端口名")
    p.add_argument("--move-time", type=float, default=2.0, help="每个目标位保持时间 (秒)")
    return p.parse_args()


def enable_arm(piper: C_PiperInterface_V2, timeout: float = 5.0) -> bool:
    """使能全部 6 个关节，带超时。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        info = piper.GetArmLowSpdInfoMsgs()
        states = [
            info.motor_1.foc_status.driver_enable_status,
            info.motor_2.foc_status.driver_enable_status,
            info.motor_3.foc_status.driver_enable_status,
            info.motor_4.foc_status.driver_enable_status,
            info.motor_5.foc_status.driver_enable_status,
            info.motor_6.foc_status.driver_enable_status,
        ]
        if all(states):
            print("[INFO] 6 个关节均已使能")
            return True
        time.sleep(0.5)
    print("[ERROR] 使能超时")
    return False


def disable_arm(piper: C_PiperInterface_V2, timeout: float = 5.0):
    """失能机械臂。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.DisableArm(7)
        piper.GripperCtrl(0, 1000, 0x02, 0)
        info = piper.GetArmLowSpdInfoMsgs()
        states = [
            info.motor_1.foc_status.driver_enable_status,
            info.motor_2.foc_status.driver_enable_status,
            info.motor_3.foc_status.driver_enable_status,
            info.motor_4.foc_status.driver_enable_status,
            info.motor_5.foc_status.driver_enable_status,
            info.motor_6.foc_status.driver_enable_status,
        ]
        if not any(states):
            print("[INFO] 已失能")
            return
        time.sleep(0.5)
    print("[WARN] 失能超时，请手动检查")


def send_joint_cmd(piper: C_PiperInterface_V2, joints_rad: list[float], gripper_m: float):
    """
    发送关节位置指令。

    joints_rad: 6 个关节弧度值
    gripper_m:  夹爪开合量 (米, 0~0.08)
    """
    j = [round(r * JOINT_FACTOR) for r in joints_rad]
    g = round(abs(gripper_m) * 1000 * 1000)  # 米 -> SDK 单位

    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])
    piper.GripperCtrl(g, 1000, 0x01, 0)


def read_joint_state(piper: C_PiperInterface_V2) -> dict:
    """读取当前关节角度 (SDK 0.001° 单位) 和夹爪角度。"""
    jm = piper.GetArmJointMsgs().joint_state
    gm = piper.GetArmGripperMsgs().gripper_state
    return {
        "joint_1": jm.joint_1,
        "joint_2": jm.joint_2,
        "joint_3": jm.joint_3,
        "joint_4": jm.joint_4,
        "joint_5": jm.joint_5,
        "joint_6": jm.joint_6,
        "gripper_angle": gm.grippers_angle,
        "gripper_effort": gm.grippers_effort,
    }


def print_state(state: dict):
    """以 弧度 / 0.001° 双单位打印关节状态。"""
    for k in [f"joint_{i}" for i in range(1, 7)]:
        raw = state[k]
        rad = raw / JOINT_FACTOR
        print(f"  {k}: {raw:>8} (0.001°)  = {rad:>8.4f} rad")
    print(f"  gripper_angle : {state['gripper_angle']}")
    print(f"  gripper_effort: {state['gripper_effort']}")


# ---- 预设 waypoint: [j1..j6] (弧度), gripper (米) ----
WAYPOINTS = [
    ([0.0, 0.0, 0.0, 0.0, 0.52, 0.0], 0.0),
    ([math.radians(30), 0.0, 0.0, 0.0, 0.52, 0.0], 0.0),
    ([math.radians(30), math.radians(20), 0.0, 0.0, 0.52, 0.0], 0.0),
    ([math.radians(30), math.radians(20), 0.0, 0.0, math.radians(60), 0.0], 0.05),
    ([math.radians(30), math.radians(20), 0.0, 0.0, math.radians(60), 0.0], 0.0),
    ([0.0, 0.0, 0.0, 0.0, 0.52, 0.0], 0.0),
]


def main():
    args = parse_args()

    # 1. 初始化 SDK
    print(f"[INFO] 连接 CAN 端口: {args.can_name}")
    piper = C_PiperInterface_V2(args.can_name)
    piper.ConnectPort()

    # 2. 使能
    if not enable_arm(piper):
        return

    try:
        # 3. 读取初始状态
        print("\n[INFO] 当前关节状态:")
        print_state(read_joint_state(piper))

        # 4. 遍历 waypoint
        for i, (joints, gripper) in enumerate(WAYPOINTS):
            print(f"\n[MOVE {i}] joints={[f'{math.degrees(j):.1f}°' for j in joints]}, gripper={gripper:.3f} m")
            t0 = time.time()
            while time.time() - t0 < args.move_time:
                send_joint_cmd(piper, joints, gripper)
                time.sleep(0.02)

            print("  到达后状态:")
            print_state(read_joint_state(piper))

        print("\n[INFO] 所有 waypoint 完成")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        # 5. 回初始位再失能
        print("[INFO] 回初始位 ...")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            send_joint_cmd(piper, [0.0, 0.0, 0.0, 0.0, 0.52, 0.0], 0.0)
            time.sleep(0.02)

        print("[INFO] 失能 ...")
        disable_arm(piper)
        print("[INFO] 完成")


if __name__ == "__main__":
    main()
