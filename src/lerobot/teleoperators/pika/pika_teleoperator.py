import logging
import math
import os
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_pika import PikaTeleoperatorConfig
from .piper_ik import (
    OneEuroFilter,
    PiperFK,
    PiperIK,
    create_transformation_matrix,
    euler_from_quaternion,
    matrix_to_xyzrpy,
)
from .safety_guard import PikaSafetyGuard

logger = logging.getLogger(__name__)


class PikaTeleoperator(Teleoperator):
    config_class = PikaTeleoperatorConfig
    name = "pika"

    def __init__(self, config: PikaTeleoperatorConfig):
        super().__init__(config)
        self.config = config

        self.sense_device = None
        self.fk_solver: PiperFK | None = None
        self.ik_solver: PiperIK | None = None

        self.initial_pika_matrix: np.ndarray | None = None
        self.initial_arm_matrix: np.ndarray | None = None

        self.filters: list[OneEuroFilter] = []
        self.filters_reset = True
        self.safety_guard: PikaSafetyGuard | None = None

        self._is_connected = False
        self._is_calibrated = False

        self.last_valid_action = {**{f"joint_{i}.pos": 0.0 for i in range(1, 7)}, "gripper.pos": 0.0}
        self._max_joint_step_rad = np.deg2rad(float(self.config.max_joint_step_deg))

        if self.config.enable_safety_guard:
            self.safety_guard = PikaSafetyGuard.from_teleoperator_config(self.config)

        # ── HIL-SERL 事件状态 ──
        self._intervention_active: bool = False
        self._space_pressed: bool = False
        self._episode_success: bool = False
        self._episode_terminate: bool = False
        self._rerecord: bool = False
        self._keyboard_listener = None
        self._key_event_queue: Queue = Queue()

        # ── EE-delta 模式状态 ──
        self._prev_filtered_ee_pos: np.ndarray | None = None
        self._ee_alignment_rot: np.ndarray | None = None
        self._last_ee_delta_action: dict[str, float] = {
            "delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 0.0
        }

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.output_mode == "ee_delta":
            return {"delta_x": float, "delta_y": float, "delta_z": float, "gripper": float}
        return {**{f"joint_{i}.pos": float for i in range(1, 7)}, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import pika

        self.sense_device = pika.sense(self.config.sense_port)
        if not self.sense_device.connect():
            raise RuntimeError(f"Failed to connect pika sense on {self.config.sense_port}")

        tracker = self.sense_device.get_vive_tracker()
        if tracker is None:
            raise RuntimeError("Failed to initialize Vive tracker via pika sense")

        deadline = time.time() + 10.0
        tracker_ready = False
        while time.time() < deadline:
            pose = self.sense_device.get_pose(self.config.tracker_device)
            if pose is not None:
                tracker_ready = True
                break
            
            
        if not tracker_ready:
            raise RuntimeError(f"Tracker '{self.config.tracker_device}' not available after 10 seconds")

        piper_description_dir = Path(self.config.piper_description_dir)
        if not self.config.piper_description_dir:
            raise ValueError("config.piper_description_dir must be set")
        urdf_filename = "piper_description-lift.urdf" if self.config.lift else "piper_description.urdf"
        urdf_path = piper_description_dir / "urdf" / urdf_filename
        if not urdf_path.exists():
            raise FileNotFoundError(f"Piper URDF not found: {urdf_path}")

        package_dirs = [str(piper_description_dir), str(piper_description_dir.parent)]
        self.fk_solver = PiperFK(
            urdf_path=str(urdf_path),
            package_dirs=package_dirs,
            gripper_xyzrpy=self.config.gripper_xyzrpy,
            lift=self.config.lift,
        )
        self.ik_solver = PiperIK(
            urdf_path=str(urdf_path),
            package_dirs=package_dirs,
            gripper_xyzrpy=self.config.gripper_xyzrpy,
            lift=self.config.lift,
            weight_position=self.config.ik_weight_position,
            weight_orientation=self.config.ik_weight_orientation,
            weight_regularization=self.config.ik_weight_regularization,
            weight_smoothing=self.config.ik_weight_smoothing,
            jump_threshold_deg=self.config.ik_jump_threshold_deg,
        )

        t0 = time.time()
        self.filters = [
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_pos),
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_pos),
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_pos),
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_rot),
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_rot),
            OneEuroFilter(t0, 0.0, min_cutoff=self.config.filter_min_cutoff, beta=self.config.filter_beta_rot),
        ]

        self._is_connected = True
        if calibrate:
            self.calibrate()

        self._start_keyboard_listener()
        logger.info(f"{self} connected")

    def _get_tracker_matrix(self) -> np.ndarray | None:
        if self.sense_device is None:
            return None

        pose = self.sense_device.get_pose(self.config.tracker_device)
        if pose is None:
            return None

        roll, pitch, yaw = euler_from_quaternion(pose.rotation)
        return create_transformation_matrix(
            pose.position[0],
            pose.position[1],
            pose.position[2],
            roll,
            pitch,
            yaw,
        )

    @check_if_not_connected
    def calibrate(self) -> None:
        if self.fk_solver is None:
            raise RuntimeError("FK solver is not initialized")

        pika_matrix = self._get_tracker_matrix()
        if pika_matrix is None:
            raise RuntimeError(f"No tracker pose available for device '{self.config.tracker_device}'")

        expected_nq = self.fk_solver.reduced_robot.model.nq
        home_q = np.asarray(self.config.home_joint_state, dtype=float)
        if home_q.shape[0] != expected_nq:
            raise ValueError(
                f"home_joint_state length mismatch: expected {expected_nq}, got {home_q.shape[0]}"
            )

        arm_xyzrpy = self.fk_solver.get_pose(home_q)
        arm_matrix = create_transformation_matrix(*arm_xyzrpy)

        self.initial_pika_matrix = pika_matrix
        self.initial_arm_matrix = arm_matrix
        self._ee_alignment_rot = arm_matrix[:3, :3] @ np.linalg.inv(pika_matrix[:3, :3])
        self.filters_reset = True
        self._is_calibrated = True
        self._update_last_valid_action(home_q, self.last_valid_action["gripper.pos"])

        if self.ik_solver is not None:
            self.ik_solver.sync_state(home_q)

        self._prev_filtered_ee_pos = None

        if self.safety_guard is not None:
            self.safety_guard.reset()

    @check_if_not_connected
    def configure(self) -> None:
        pass

    def _filter_pose(self, target_matrix: np.ndarray) -> np.ndarray:
        pose_xyzrpy_raw = matrix_to_xyzrpy(target_matrix)

        if self.filters_reset or (
            self.filters and self.filters[0].t_prev > 0 and (time.time() - self.filters[0].t_prev) > 0.5
        ):
            t = time.time()
            for i, val in enumerate(pose_xyzrpy_raw):
                self.filters[i].x_prev = val
                self.filters[i].dx_prev = 0.0
                self.filters[i].t_prev = t
            self.filters_reset = False

        t = time.time()
        pose_xyzrpy: list[float] = []
        for i, val in enumerate(pose_xyzrpy_raw):
            if i >= 3:
                prev_val = self.filters[i].x_prev
                diff = val - prev_val
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                val = prev_val + diff
            pose_xyzrpy.append(self.filters[i](t, val))

        return create_transformation_matrix(*pose_xyzrpy)

    def _read_enable_signal(self) -> bool:
        if self.sense_device is None:
            return False

        if not self.config.use_command_state_enable:
            return True

        try:
            return bool(self.sense_device.get_command_state())
        except Exception:
            return True

    def _get_gripper_encoder_rad(self) -> float:
        if self.sense_device is None:
            return 0.0

        try:
            encoder_data = self.sense_device.get_encoder_data()
            return float(encoder_data.get("rad", 0.0))
        except Exception:
            return 0.0

    def _get_zero_ee_delta_action(self) -> dict[str, float]:
        gripper_mm = 0.0
        try:
            if self.sense_device is not None:
                raw = self.sense_device.get_gripper_distance()
                if raw is not None:
                    gripper_mm = float(raw)
        except Exception:
            pass
        gripper_norm = float(np.clip(gripper_mm / 90.0, 0.0, 1.0))
        self._last_ee_delta_action = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": gripper_norm,
        }
        return self._last_ee_delta_action.copy()

    def _update_last_gripper_action(self, gripper_rad: float) -> None:
        self.last_valid_action["gripper.pos"] = float(gripper_rad)

    def _update_last_valid_action(self, joint_positions: np.ndarray, gripper_rad: float) -> None:
        self.last_valid_action = {
            "joint_1.pos": float(joint_positions[0]),
            "joint_2.pos": float(joint_positions[1]),
            "joint_3.pos": float(joint_positions[2]),
            "joint_4.pos": float(joint_positions[3]),
            "joint_5.pos": float(joint_positions[4]),
            "joint_6.pos": float(joint_positions[5]),
            "gripper.pos": float(gripper_rad),
        }

    def _get_last_commanded_joint_positions(self) -> np.ndarray:
        return np.asarray(
            [self.last_valid_action[f"joint_{i}.pos"] for i in range(1, 7)],
            dtype=float,
        )

    def _shape_joint_command(
        self,
        candidate_q: np.ndarray,
        reference_q: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        candidate_q = np.asarray(candidate_q, dtype=float).reshape(-1)
        reference_q = np.asarray(reference_q, dtype=float).reshape(-1)

        if self._max_joint_step_rad <= 0.0:
            return candidate_q.copy(), False

        dq = candidate_q - reference_q
        dq_clipped = np.clip(dq, -self._max_joint_step_rad, self._max_joint_step_rad)
        shaped_q = reference_q + dq_clipped
        clipped = bool(np.any(np.abs(dq - dq_clipped) > 1e-9))
        return shaped_q, clipped

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        if not self._is_calibrated:
            self.calibrate()

        guard = self.safety_guard
        gripper_rad = self._get_gripper_encoder_rad()
        self._update_last_gripper_action(gripper_rad)

        current_pika_matrix = self._get_tracker_matrix()
        if current_pika_matrix is None:
            if guard is not None:
                guard.handle_pose_missing()
            if self.config.output_mode == "ee_delta":
                return self._get_zero_ee_delta_action()
            return self.last_valid_action.copy()

        if guard is not None:
            guard.note_pose_available()

            rising_edge = guard.update_enable_state(self._read_enable_signal())
            if rising_edge and self.fk_solver is not None:
                safe_q = guard.last_safe_q
                if safe_q is None:
                    safe_q = np.asarray(self.config.home_joint_state, dtype=float)
                self.initial_pika_matrix = current_pika_matrix
                self.initial_arm_matrix = create_transformation_matrix(*self.fk_solver.get_pose(safe_q))
                self.filters_reset = True
                self._prev_filtered_ee_pos = None
                if self.ik_solver is not None:
                    self.ik_solver.sync_state(safe_q)

            if not guard.enabled:
                if self.config.output_mode == "ee_delta":
                    return self._get_zero_ee_delta_action()
                return self.last_valid_action.copy()

        if self.config.output_mode == "ee_delta":
            mapped_pos = self._map_tracker_translation_to_ee_position(current_pika_matrix)
            return self._get_ee_delta_action_from_position(mapped_pos)

        target_matrix = self.initial_arm_matrix @ np.dot(np.linalg.inv(self.initial_pika_matrix), current_pika_matrix)
        filtered_target_matrix = self._filter_pose(target_matrix)

        return self._get_joint_action(filtered_target_matrix, gripper_rad)

    def _map_tracker_translation_to_ee_position(self, current_pika_matrix: np.ndarray) -> np.ndarray:
        if self.initial_pika_matrix is None or self.initial_arm_matrix is None:
            raise RuntimeError("Teleoperator must be calibrated before ee_delta mapping")

        align_rot = self._ee_alignment_rot
        if align_rot is None:
            align_rot = self.initial_arm_matrix[:3, :3] @ np.linalg.inv(self.initial_pika_matrix[:3, :3])
            self._ee_alignment_rot = align_rot

        tracker_delta = current_pika_matrix[:3, 3] - self.initial_pika_matrix[:3, 3]
        return self.initial_arm_matrix[:3, 3] + align_rot @ tracker_delta

    def _get_ee_delta_action_from_position(self, current_pos: np.ndarray) -> dict[str, float]:
        if self._prev_filtered_ee_pos is None:
            delta = np.zeros(3)
        else:
            delta = current_pos - self._prev_filtered_ee_pos
        self._prev_filtered_ee_pos = current_pos.copy()

        deadband_m = 0.0015
        delta[np.abs(delta) < deadband_m] = 0.0
        delta = np.clip(delta, -0.01, 0.01)

        gripper_mm = 0.0
        try:
            if self.sense_device is not None:
                raw = self.sense_device.get_gripper_distance()
                if raw is not None:
                    gripper_mm = float(raw)
        except Exception:
            pass
        gripper_norm = float(np.clip(gripper_mm / 90.0, 0.0, 1.0))

        self._last_ee_delta_action = {
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "gripper": gripper_norm,
        }
        return self._last_ee_delta_action.copy()

    def _get_joint_action(self, filtered_target_matrix: np.ndarray, gripper_rad: float) -> dict[str, float]:
        if self.ik_solver is None:
            raise RuntimeError("IK solver is not initialized")

        guard = self.safety_guard
        gripper_mm = self.sense_device.get_gripper_distance()
        gripper_m = float(gripper_mm) / 1000.0
        solver_reference_q = self._get_last_commanded_joint_positions()
        if guard is not None and guard.last_safe_q is not None:
            solver_reference_q = guard.last_safe_q

        sol_q, _, valid = self.ik_solver.ik_fun(
            filtered_target_matrix,
            gripper=gripper_m,
            motorstate=solver_reference_q,
        )

        shaped_q = None
        if valid and sol_q is not None:
            shaped_q, _ = self._shape_joint_command(sol_q, solver_reference_q)

        if guard is not None:
            safe_q, status = guard.apply_ik_result(shaped_q, valid)
            if status == "ik_invalid_repeated":
                guard.force_disable()
            if safe_q is not None:
                self._update_last_valid_action(safe_q, gripper_rad)
                self.ik_solver.sync_state(safe_q)
        elif shaped_q is not None:
            self._update_last_valid_action(shaped_q, gripper_rad)
            self.ik_solver.sync_state(shaped_q)

        return self.last_valid_action.copy()

    def _get_ee_delta_action(self, filtered_target_matrix: np.ndarray, gripper_rad: float) -> dict[str, float]:
        current_pos = filtered_target_matrix[:3, 3].copy()

        if self._prev_filtered_ee_pos is None:
            delta = np.zeros(3)
        else:
            delta = current_pos - self._prev_filtered_ee_pos
        self._prev_filtered_ee_pos = current_pos

        deadband_m = 0.0015
        delta[np.abs(delta) < deadband_m] = 0.0

        gripper_mm = 0.0
        try:
            if self.sense_device is not None:
                raw = self.sense_device.get_gripper_distance()
                if raw is not None:
                    gripper_mm = float(raw)
        except Exception:
            pass
        gripper_norm = float(np.clip(gripper_mm / 90.0, 0.0, 1.0))

        self._last_ee_delta_action = {
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
            "gripper": gripper_norm,
        }
        return self._last_ee_delta_action.copy()

    def _start_keyboard_listener(self) -> None:
        """启动 pynput 键盘监听线程（无 DISPLAY 或未安装时静默跳过）。"""
        try:
            if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
                logger.info("No DISPLAY set, keyboard listener disabled.")
                return
            from pynput import keyboard as _kbd

            def on_press(key):
                self._key_event_queue.put(("press", key))

            def on_release(key):
                self._key_event_queue.put(("release", key))

            listener = _kbd.Listener(on_press=on_press, on_release=on_release)
            listener.daemon = True
            listener.start()
            self._keyboard_listener = listener
            logger.info(
                "Keyboard listener started: "
                "Space=toggle intervention | s=success | q/esc=quit | r=rerecord"
            )
        except ImportError:
            logger.info("pynput not available; keyboard events disabled. pip install pynput")
        except Exception as exc:
            logger.warning(f"Could not start keyboard listener: {exc}")

    def _drain_key_events(self) -> None:
        """将队列中的键盘事件消费到内部状态。"""
        try:
            from pynput import keyboard as _kbd
        except ImportError:
            return

        while not self._key_event_queue.empty():
            try:
                event_type, key = self._key_event_queue.get_nowait()
            except Exception:
                break

            if event_type == "press":
                if key == _kbd.Key.space and not self._space_pressed:
                    self._space_pressed = True
                    self._intervention_active = not self._intervention_active
                    logger.debug(f"Intervention toggled → {self._intervention_active}")
                elif key == _kbd.Key.esc:
                    self._episode_success = False
                    self._episode_terminate = True
                    logger.info("[TeleopEvent] esc (failure + terminate)")
                elif hasattr(key, "char") and key.char is not None:
                    char = key.char.lower()
                    if char == "s":
                        self._episode_success = True
                        self._episode_terminate = True
                        logger.info("[TeleopEvent] success + terminate")
                    elif char == "q":
                        self._episode_success = False
                        self._episode_terminate = True
                        logger.info("[TeleopEvent] quit (failure + terminate)")
                    elif char == "r":
                        self._rerecord = True
                        self._episode_terminate = True
                        logger.info("[TeleopEvent] rerecord + terminate")
            elif event_type == "release" and key == _kbd.Key.space:
                self._space_pressed = False

    def get_teleop_events(self) -> dict[str, Any]:
        """HasTeleopEvents 协议实现。

        按键映射:
            Space   — 切换 intervention（按下激活，再按关闭）
            s       — 标记成功并终止 episode
            q / Esc — 标记失败并终止 episode
            r       — 标记重录并终止 episode
            Pika 双击夹爪 — 通过 command_state 作为附加 intervention 信号
        """
        self._drain_key_events()

        pika_intervention = False
        if self.sense_device is not None:
            try:
                pika_intervention = bool(self.sense_device.get_command_state())
            except Exception:
                pass

        events = {
            TeleopEvents.IS_INTERVENTION: self._intervention_active or pika_intervention,
            TeleopEvents.TERMINATE_EPISODE: self._episode_terminate,
            TeleopEvents.SUCCESS: self._episode_success,
            TeleopEvents.RERECORD_EPISODE: self._rerecord,
        }
        # 一次性事件读后重置
        self._episode_terminate = False
        self._episode_success = False
        self._rerecord = False
        return events

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        if self.sense_device is not None:
            self.sense_device.disconnect()

        self.sense_device = None
        self.fk_solver = None
        self.ik_solver = None
        self.initial_pika_matrix = None
        self.initial_arm_matrix = None
        self._is_calibrated = False
        self._is_connected = False
        self._intervention_active = False
        self._space_pressed = False
        self._prev_filtered_ee_pos = None
        if self.safety_guard is not None:
            self.safety_guard.reset()
