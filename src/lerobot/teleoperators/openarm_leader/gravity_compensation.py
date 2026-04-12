#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Gravity compensation support for the full-size OpenArm leader.

This module is intentionally decoupled from ``openarm_leader.py`` so the
teleoperator can keep using the existing torque-disabled manual mode when
gravity compensation is not requested.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from lerobot.motors.damiao import DamiaoMotorsBus

logger = logging.getLogger(__name__)

OPENARM_JOINT_MOTOR_NAMES = [f"joint_{idx}" for idx in range(1, 8)]


class GravityModel(Protocol):
    """Small interface for gravity-torque backends."""

    def compute(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        """Return gravity torques in Nm, ordered like ``joint_positions_rad``."""


def default_openarm_joint_names(side: str | None) -> list[str]:
    """Return the expected URDF joint names for an OpenArm arm."""

    if side is None:
        prefix = "openarm_"
    else:
        normalized_side = side.removesuffix("_arm")
        if normalized_side not in {"left", "right"}:
            raise ValueError(
                "`gravity_compensation_side` must be 'left', 'right', 'left_arm', 'right_arm', or None."
            )
        prefix = f"openarm_{normalized_side}_"

    return [f"{prefix}joint{idx}" for idx in range(1, 8)]


class PinocchioGravityModel:
    """URDF-based gravity model using Pinocchio.

    The model keeps the full URDF configuration and only writes the configured
    OpenArm joint values. This supports both single-arm URDFs and the bimanual
    OpenArm URDF used by the ROS2 reference implementation.
    """

    def __init__(self, urdf_path: str | Path, joint_names: list[str]) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "OpenArm leader gravity compensation requires Pinocchio. "
                "Install it with `pip install pin` before enabling `gravity_compensation`."
            ) from exc

        self._pin = pin
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"OpenArm gravity compensation URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.joint_names = joint_names

        missing = [name for name in joint_names if not self.model.existJointName(name)]
        if missing:
            raise ValueError(
                "OpenArm gravity compensation URDF is missing expected joints: "
                f"{missing}. Provide `gravity_compensation_joint_names` if this URDF uses different names."
            )

        self._q_indices = [self.model.idx_qs[self.model.getJointId(name)] for name in joint_names]
        self._v_indices = [self.model.idx_vs[self.model.getJointId(name)] for name in joint_names]
        self._q = pin.neutral(self.model)

    def compute(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        if joint_positions_rad.shape != (len(self.joint_names),):
            raise ValueError(
                f"Expected {len(self.joint_names)} joint positions, got shape {joint_positions_rad.shape}."
            )

        q = self._q.copy()
        for idx, joint_position in zip(self._q_indices, joint_positions_rad, strict=True):
            q[idx] = joint_position

        tau = self._pin.computeGeneralizedGravity(self.model, self.data, q)
        return np.array([tau[idx] for idx in self._v_indices], dtype=np.float64)


class OpenArmLeaderGravityCompensator:
    """Background gravity-compensation loop for an OpenArm leader arm."""

    def __init__(
        self,
        bus: DamiaoMotorsBus,
        gravity_model: GravityModel,
        *,
        motor_names: list[str] | None = None,
        factor: float = 0.8,
        max_torque: float = 8.0,
        frequency_hz: float = 200.0,
        kp: float = 0.0,
        kd: float = 0.0,
        joint_signs: list[float] | None = None,
        joint_offsets_deg: list[float] | None = None,
    ) -> None:
        self.bus = bus
        self.gravity_model = gravity_model
        self.motor_names = motor_names or OPENARM_JOINT_MOTOR_NAMES
        if len(self.motor_names) != 7:
            raise ValueError(f"OpenArm gravity compensation expects 7 arm motors, got {self.motor_names}.")
        if frequency_hz <= 0:
            raise ValueError("`frequency_hz` must be greater than 0.")
        if max_torque < 0:
            raise ValueError("`max_torque` must be greater than or equal to 0.")

        self.factor = factor
        self.max_torque = max_torque
        self.frequency_hz = frequency_hz
        self.period_s = 1.0 / frequency_hz
        self.kp = kp
        self.kd = kd
        self.joint_signs = np.array(joint_signs or [1.0] * 7, dtype=np.float64)
        self.joint_offsets_deg = np.array(joint_offsets_deg or [0.0] * 7, dtype=np.float64)
        if self.joint_signs.shape != (7,) or self.joint_offsets_deg.shape != (7,):
            raise ValueError("`joint_signs` and `joint_offsets_deg` must each contain 7 values.")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_states: dict[str, dict[str, float]] = {}
        self._last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def start(self) -> None:
        if self.is_running:
            return

        # Populate the state cache before get_action() switches to cache reads.
        self.step_once()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="openarm-leader-gravity-compensation",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    "OpenArm leader gravity compensation thread did not stop within %.1fs.", timeout_s
                )
                return
        self._thread = None

    def get_latest_states(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {motor: state.copy() for motor, state in self._latest_states.items()}

    def step_once(self) -> None:
        states = self.bus.sync_read_all_states()
        arm_positions_deg = np.array(
            [float(states[motor]["position"]) for motor in self.motor_names],
            dtype=np.float64,
        )
        model_positions_deg = arm_positions_deg * self.joint_signs + self.joint_offsets_deg
        model_positions_rad = np.radians(model_positions_deg)

        gravity_tau = self.gravity_model.compute(model_positions_rad)
        motor_tau = gravity_tau * self.joint_signs * self.factor
        motor_tau = np.clip(motor_tau, -self.max_torque, self.max_torque)

        commands = {
            motor: (self.kp, self.kd, 0.0, 0.0, float(torque))
            for motor, torque in zip(self.motor_names, motor_tau, strict=True)
        }
        self.bus._mit_control_batch(commands)

        with self._lock:
            self._latest_states = {motor: state.copy() for motor, state in states.items()}
            self._last_error = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            start_t = time.perf_counter()
            try:
                self.step_once()
            except Exception as exc:  # noqa: BLE001
                self._last_error = exc
                logger.exception("OpenArm leader gravity compensation step failed")

            elapsed_s = time.perf_counter() - start_t
            sleep_s = max(self.period_s - elapsed_s, 0.0)
            if self._stop_event.wait(sleep_s):
                break
