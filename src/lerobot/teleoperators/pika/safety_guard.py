from dataclasses import dataclass
import time

import numpy as np


@dataclass
class PikaSafetyGuardConfig:
    pose_timeout_sec: float = 0.25
    max_joint_step_deg: float = 8.0
    max_consecutive_ik_failures: int = 8
    force_disable_on_pose_stale: bool = True


class PikaSafetyGuard:
    def __init__(self, config: PikaSafetyGuardConfig):
        self.config = config
        self._max_joint_step_rad = np.deg2rad(config.max_joint_step_deg)

        self._enabled = False
        self._enabled_prev = False

        self._last_pose_ts: float | None = None
        self._consecutive_ik_failures = 0
        self._last_safe_q: np.ndarray | None = None

    @classmethod
    def from_teleoperator_config(cls, config) -> "PikaSafetyGuard":
        return cls(
            PikaSafetyGuardConfig(
                pose_timeout_sec=float(getattr(config, "pose_timeout_sec", 0.25)),
                max_joint_step_deg=float(getattr(config, "max_joint_step_deg", 8.0)),
                max_consecutive_ik_failures=int(getattr(config, "max_consecutive_ik_failures", 8)),
                force_disable_on_pose_stale=bool(getattr(config, "force_disable_on_pose_stale", True)),
            )
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_safe_q(self) -> np.ndarray | None:
        return None if self._last_safe_q is None else self._last_safe_q.copy()

    def update_enable_state(self, enabled_signal: bool) -> bool:
        self._enabled_prev = self._enabled
        self._enabled = bool(enabled_signal)
        return self._enabled and (not self._enabled_prev)

    def note_pose_available(self) -> None:
        self._last_pose_ts = time.monotonic()

    def note_pose_missing(self) -> None:
        if self._last_pose_ts is None:
            self._last_pose_ts = time.monotonic()

    def is_pose_fresh(self) -> bool:
        if self._last_pose_ts is None:
            return False
        return (time.monotonic() - self._last_pose_ts) <= self.config.pose_timeout_sec

    def force_disable(self) -> None:
        self._enabled = False

    def reset(self) -> None:
        self._enabled = False
        self._enabled_prev = False
        self._last_pose_ts = None
        self._consecutive_ik_failures = 0
        self._last_safe_q = None

    def handle_pose_missing(self) -> bool:
        self.note_pose_missing()
        if self.config.force_disable_on_pose_stale and (not self.is_pose_fresh()):
            self.force_disable()
            return True
        return False

    def apply_ik_result(self, sol_q: np.ndarray | None, valid: bool) -> tuple[np.ndarray | None, str]:
        if (not valid) or (sol_q is None):
            self._consecutive_ik_failures += 1
            if self._consecutive_ik_failures >= self.config.max_consecutive_ik_failures:
                return self.last_safe_q, "ik_invalid_repeated"
            return self.last_safe_q, "ik_invalid"

        candidate_q = np.asarray(sol_q, dtype=float).reshape(-1)

        if self._last_safe_q is None:
            self._last_safe_q = candidate_q.copy()
            self._consecutive_ik_failures = 0
            return self.last_safe_q, "ok_first"

        dq = candidate_q - self._last_safe_q
        dq_clipped = np.clip(dq, -self._max_joint_step_rad, self._max_joint_step_rad)
        safe_q = self._last_safe_q + dq_clipped

        clipped = bool(np.any(np.abs(dq - dq_clipped) > 1e-9))
        self._last_safe_q = safe_q.copy()
        self._consecutive_ik_failures = 0
        return self.last_safe_q, ("ok_clipped" if clipped else "ok")
