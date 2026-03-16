#!/usr/bin/env python
"""验证 HIL-SERL IK 管线是否可在 Piper URDF 上工作。

功能：
1) 解析 URDF，列出 joint/link 名称（便于确认 target frame/joint 名称）。
2) 使用 RobotKinematics(placo) 逐个尝试候选 target frame。
3) 对可用 frame 执行 FK -> IK 回环自检。

用法：
python examples/pika_to_piper/verify_piper_ik_pipeline.py \
  --urdf /home/zhbs/pika_ros/install/piper_description/share/piper_description/urdf/piper_description.urdf \
    --joint-names joint1 joint2 joint3 joint4 joint5 joint6
"""

from __future__ import annotations

import argparse
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics


DEFAULT_FRAME_CANDIDATES = [
    "link6",
    "gripper_base_link",
    "gripper",
    "gripper_link",
    "ee_link",
    "tool0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True, help="Piper URDF 路径")
    parser.add_argument(
        "--target-frame",
        type=str,
        default=None,
        help="指定单个 frame 验证；不指定时按候选列表自动尝试",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        help="参与 IK/FK 的关节名列表（不含 gripper）",
    )
    parser.add_argument(
        "--q-deg",
        nargs="*",
        type=float,
        default=None,
        help="用于 FK/IK 回环测试的初始关节角（度）",
    )
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument(
        "--package-root",
        type=str,
        default=None,
        help=(
            "URDF 包根目录（例如 /home/zhbs/pika_ros/install/piper_description/share/piper_description）。"
            "用于将 package://<pkg>/... 重写为本地绝对路径。"
        ),
    )
    return parser.parse_args()


def resolve_urdf_for_placo(urdf_path: str, package_root: str | None) -> tuple[str, str | None]:
    urdf_file = Path(urdf_path).resolve()
    text = urdf_file.read_text(encoding="utf-8")
    package_pattern = re.compile(r"package://([^/]+)/")
    package_names = sorted(set(package_pattern.findall(text)))
    if not package_names:
        return str(urdf_file), None

    resolved_root: Path | None = Path(package_root).resolve() if package_root else None
    if resolved_root is None:
        # 常见 ROS 安装布局：.../share/<pkg>/urdf/<file>.urdf
        parts = urdf_file.parts
        if "share" in parts:
            share_idx = parts.index("share")
            if share_idx + 1 < len(parts):
                resolved_root = Path(*parts[: share_idx + 2])

    if resolved_root is None:
        raise ValueError(
            "检测到 package:// URI，但无法自动推断包根目录。"
            "请显式传入 --package-root。"
        )

    rewritten = text
    for package_name in package_names:
        src = f"package://{package_name}/"
        dst = f"{resolved_root.as_posix().rstrip('/')}/"
        rewritten = rewritten.replace(src, dst)

    tmp = tempfile.NamedTemporaryFile(prefix="piper_verify_", suffix=".urdf", delete=False)
    tmp.write(rewritten.encode("utf-8"))
    tmp.close()
    return tmp.name, str(resolved_root)


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


def try_frame(urdf_path: str, frame_name: str, joint_names: list[str], q_deg: np.ndarray) -> tuple[bool, str]:
    try:
        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=frame_name,
            joint_names=joint_names,
        )
        fk = kin.forward_kinematics(q_deg)
        ik = kin.inverse_kinematics(
            current_joint_pos=q_deg,
            desired_ee_pose=fk,
            position_weight=1.0,
            orientation_weight=0.01,
        )
        err = float(np.max(np.abs(ik[: len(joint_names)] - q_deg[: len(joint_names)])))
        return True, f"FK/IK round-trip max joint error = {err:.6f} deg"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    args = parse_args()

    urdf_for_placo, used_package_root = resolve_urdf_for_placo(args.urdf, args.package_root)

    links, joints = parse_urdf_names(args.urdf)
    print("=" * 72)
    print("URDF 概览")
    print("=" * 72)
    print(f"links({len(links)}): {links}")
    print(f"non-fixed joints({len(joints)}): {joints}")

    q_deg = np.zeros(len(args.joint_names), dtype=float)
    if args.q_deg is not None and len(args.q_deg) > 0:
        if len(args.q_deg) != len(args.joint_names):
            raise ValueError(
                f"--q-deg 长度({len(args.q_deg)})必须等于关节数({len(args.joint_names)})"
            )
        q_deg = np.array(args.q_deg, dtype=float)

    frame_candidates = [args.target_frame] if args.target_frame else DEFAULT_FRAME_CANDIDATES

    print("=" * 72)
    print("验证 RobotKinematics(placo) on Piper URDF")
    print("=" * 72)
    print(f"joint_names={args.joint_names}")
    print(f"q_deg={q_deg.tolist()}")
    print(f"urdf_for_placo={urdf_for_placo}")
    if used_package_root is not None:
        print(f"resolved_package_root={used_package_root}")

    any_success = False
    for frame in frame_candidates:
        ok, detail = try_frame(urdf_for_placo, frame, args.joint_names, q_deg)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] target_frame={frame}: {detail}")
        any_success = any_success or ok

    print("=" * 72)
    if any_success:
        print("至少一个 target_frame 验证通过，可用于 env_config.processor.inverse_kinematics")
    else:
        print("所有 target_frame 验证失败，请检查 placo 安装、URDF 路径、joint 名称")
    print("=" * 72)


if __name__ == "__main__":
    main()
