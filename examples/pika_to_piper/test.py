#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pika Vive Tracker -> lerobot PiperIK -> Meshcat 可视化验证脚本

功能:
1. 借鉴 pika_sdk/examples/sense/vive_tracker_example.py 的连接与设备探测逻辑
2. 使用 lerobot.teleoperators.pika.piper_ik 的 FK/IK 与 OneEuroFilter
3. 复用 pika_ros/arm_FIK.py 的 Meshcat 显示风格:
   - 机械臂模型
   - ee_target 坐标轴
   - pika_tracker 坐标轴(显示 Vive 原始位姿)

运行前提:
- 已安装: pinocchio, meshcat, casadi
- 可导入 pika SDK (from pika.sense import Sense)
- 提供 piper_description_dir (包含 urdf/ 与 meshes/)
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path

import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from lerobot.teleoperators.pika.piper_ik import (
    OneEuroFilter,
    PiperFK,
    PiperIK,
    create_transformation_matrix,
    euler_from_quaternion,
    matrix_to_xyzrpy,
)

logger = logging.getLogger("ik_meshcat_vive_viz")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize lerobot Piper IK with Vive tracker in Meshcat")
    parser.add_argument("--sense-port", type=str, default="/dev/ttyUSB0", help="Pika Sense serial port")
    parser.add_argument("--tracker-device", type=str, default="T20", help="Vive tracker device name")
    parser.add_argument("--piper-description-dir", type=str, required=True, help="Path to piper_description package dir")
    parser.add_argument("--lift", action="store_true", help="Use piper_description-lift.urdf")
    parser.add_argument("--gripper-xyzrpy", type=float, nargs=6, default=[0.19, 0.0, 0.2, 0.0, 0.0, 0.0])
    parser.add_argument("--home-joint-state", type=float, nargs=6, default=[0.0] * 6)
    parser.add_argument("--filter-min-cutoff", type=float, default=1.0)
    parser.add_argument("--filter-beta-pos", type=float, default=1.0)
    parser.add_argument("--filter-beta-rot", type=float, default=0.5)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--init-wait-sec", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def make_axis_lines(axis_length: float = 0.1, axis_width: int = 10):
    frame_axis_positions = np.array(
        [[0, 0, 0], [1, 0, 0],
         [0, 0, 0], [0, 1, 0],
         [0, 0, 0], [0, 0, 1]],
        dtype=np.float32,
    ).T

    frame_axis_colors = np.array(
        [[1, 0, 0], [1, 0.6, 0],
         [0, 1, 0], [0.6, 1, 0],
         [0, 0, 1], [0, 0.6, 1]],
        dtype=np.float32,
    ).T

    return mg.LineSegments(
        mg.PointsGeometry(
            position=axis_length * frame_axis_positions,
            color=frame_axis_colors,
        ),
        mg.LineBasicMaterial(
            linewidth=axis_width,
            vertexColors=True,
        ),
    )


def create_filters(min_cutoff: float, beta_pos: float, beta_rot: float):
    t0 = time.time()
    return [
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_pos),
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_pos),
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_pos),
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_rot),
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_rot),
        OneEuroFilter(t0, 0.0, min_cutoff=min_cutoff, beta=beta_rot),
    ]


def filter_pose_matrix(target_matrix: np.ndarray, filters: list[OneEuroFilter], reset: bool):
    pose_xyzrpy_raw = matrix_to_xyzrpy(target_matrix)

    if reset or (filters and filters[0].t_prev > 0 and (time.time() - filters[0].t_prev) > 0.5):
        t = time.time()
        for i, val in enumerate(pose_xyzrpy_raw):
            filters[i].x_prev = val
            filters[i].dx_prev = 0.0
            filters[i].t_prev = t
        reset = False

    t = time.time()
    pose_xyzrpy = []
    for i, val in enumerate(pose_xyzrpy_raw):
        if i >= 3:
            prev_val = filters[i].x_prev
            diff = val - prev_val
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            val = prev_val + diff
        pose_xyzrpy.append(filters[i](t, val))

    return create_transformation_matrix(*pose_xyzrpy), reset


def tracker_pose_to_matrix(pose):
    roll, pitch, yaw = euler_from_quaternion(pose.rotation)
    return create_transformation_matrix(
        pose.position[0],
        pose.position[1],
        pose.position[2],
        roll,
        pitch,
        yaw,
    )


def build_paths(piper_description_dir: str, lift: bool):
    pdir = Path(piper_description_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"piper_description_dir 不存在: {pdir}")

    urdf_name = "piper_description-lift.urdf" if lift else "piper_description.urdf"
    urdf_path = pdir / "urdf" / urdf_name
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    package_dirs = [str(pdir), str(pdir.parent)]
    return urdf_path, package_dirs


def init_meshcat(ik_solver: PiperIK):
    vis = MeshcatVisualizer(
        ik_solver.reduced_robot.model,
        ik_solver.reduced_robot.collision_model,
        ik_solver.reduced_robot.visual_model,
    )
    vis.initViewer(open=True)
    vis.loadViewerModel("pinocchio")
    vis.display(pin.neutral(ik_solver.reduced_robot.model))

    axis_obj = make_axis_lines(axis_length=0.1, axis_width=10)
    vis.viewer["ee_target"].set_object(axis_obj)
    vis.viewer["pika_tracker"].set_object(axis_obj)

    logger.info("Meshcat URL: %s", vis.viewer.url())
    return vis


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 借鉴 vive_tracker_example.py 的连接逻辑
    from pika.sense import Sense

    urdf_path, package_dirs = build_paths(args.piper_description_dir, args.lift)

    sense = Sense(port=args.sense_port)
    if not sense.connect():
        raise RuntimeError(f"连接 Sense 失败: {args.sense_port}")

    logger.info("连接 Sense 成功，初始化 Vive Tracker...")
    tracker = sense.get_vive_tracker()
    if tracker is None:
        raise RuntimeError("获取 ViveTracker 失败，请检查 pysurvive 依赖")

    time.sleep(args.init_wait_sec)

    devices = sense.get_tracker_devices()
    logger.info("初始检测到设备: %s", devices)

    retry_count = 0
    while args.tracker_device not in devices and retry_count < args.max_retries:
        retry_count += 1
        logger.info("未检测到 %s，重试 %d/%d", args.tracker_device, retry_count, args.max_retries)
        time.sleep(1.0)
        devices = sense.get_tracker_devices()
        logger.info("当前设备: %s", devices)

    if args.tracker_device not in devices:
        raise RuntimeError(f"未检测到目标设备 {args.tracker_device}")

    logger.info("检测到目标设备 %s，开始初始化 IK", args.tracker_device)

    fk_solver = PiperFK(
        urdf_path=str(urdf_path),
        package_dirs=package_dirs,
        gripper_xyzrpy=args.gripper_xyzrpy,
        lift=args.lift,
    )
    ik_solver = PiperIK(
        urdf_path=str(urdf_path),
        package_dirs=package_dirs,
        gripper_xyzrpy=args.gripper_xyzrpy,
        lift=args.lift,
    )

    # 标定：对齐 tracker 初始位姿 与机械臂 home 位姿
    pose0 = sense.get_pose(args.tracker_device)
    if pose0 is None:
        raise RuntimeError("标定失败：无法读取 tracker 初始位姿")

    initial_pika_matrix = tracker_pose_to_matrix(pose0)

    home_q = np.asarray(args.home_joint_state, dtype=float)
    expected_nq = fk_solver.reduced_robot.model.nq
    if home_q.shape[0] != expected_nq:
        raise ValueError(f"home_joint_state 维度错误: expected={expected_nq}, got={home_q.shape[0]}")

    home_arm_xyzrpy = fk_solver.get_pose(home_q)
    initial_arm_matrix = create_transformation_matrix(*home_arm_xyzrpy)

    filters = create_filters(args.filter_min_cutoff, args.filter_beta_pos, args.filter_beta_rot)
    filters_reset = True

    vis = init_meshcat(ik_solver)
    dt = 1.0 / max(args.hz, 1e-6)

    logger.info("开始主循环，Ctrl+C 退出")
    loop_idx = 0
    try:
        while True:
            t0 = time.perf_counter()

            pose = sense.get_pose(args.tracker_device)
            if pose is None:
                logger.warning("当前未取到 %s 位姿，跳过本帧", args.tracker_device)
                time.sleep(0.02)
                continue

            current_pika_matrix = tracker_pose_to_matrix(pose)
            vis.viewer["pika_tracker"].set_transform(current_pika_matrix)

            # 与 pika_teleoperator.py 对齐
            target_matrix = initial_arm_matrix @ np.dot(np.linalg.inv(initial_pika_matrix), current_pika_matrix)

            filtered_target_matrix, filters_reset = filter_pose_matrix(target_matrix, filters, filters_reset)
            vis.viewer["ee_target"].set_transform(filtered_target_matrix)

            gripper_mm = sense.get_gripper_distance()
            gripper_m = float(gripper_mm) / 1000.0
            sol_q, _, valid = ik_solver.ik_fun(filtered_target_matrix, gripper=gripper_m)

            if valid and sol_q is not None:
                vis.display(sol_q)

            if loop_idx % 30 == 0:
                logger.info(
                    "loop=%d valid=%s gripper=%.4fm target_xyz=[%.3f, %.3f, %.3f]",
                    loop_idx,
                    valid,
                    gripper_m,
                    filtered_target_matrix[0, 3],
                    filtered_target_matrix[1, 3],
                    filtered_target_matrix[2, 3],
                )
            loop_idx += 1

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("收到退出信号，正在关闭...")
    finally:
        sense.disconnect()


if __name__ == "__main__":
    main()