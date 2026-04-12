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

import numpy as np
import pytest

from lerobot.teleoperators.openarm_leader.gravity_compensation import (
    OpenArmLeaderGravityCompensator,
    default_openarm_joint_names,
)


class FakeGravityModel:
    def __init__(self) -> None:
        self.last_joint_positions_rad: np.ndarray | None = None

    def compute(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        self.last_joint_positions_rad = joint_positions_rad.copy()
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)


class FakeBus:
    def __init__(self) -> None:
        self.states = {
            f"joint_{idx}": {"position": float(idx * 10), "velocity": float(idx), "torque": 0.0}
            for idx in range(1, 8)
        }
        self.states["gripper"] = {"position": 12.0, "velocity": 0.0, "torque": 0.0}
        self.commands: list[dict[str, tuple[float, float, float, float, float]]] = []

    def sync_read_all_states(self) -> dict[str, dict[str, float]]:
        return {motor: state.copy() for motor, state in self.states.items()}

    def _mit_control_batch(self, commands: dict[str, tuple[float, float, float, float, float]]) -> None:
        self.commands.append(commands)


def test_default_openarm_joint_names() -> None:
    assert default_openarm_joint_names(None) == [f"openarm_joint{idx}" for idx in range(1, 8)]
    assert default_openarm_joint_names("left") == [f"openarm_left_joint{idx}" for idx in range(1, 8)]
    assert default_openarm_joint_names("right_arm") == [f"openarm_right_joint{idx}" for idx in range(1, 8)]

    with pytest.raises(ValueError, match="gravity_compensation_side"):
        default_openarm_joint_names("center")


def test_step_once_converts_positions_and_sends_gravity_torque() -> None:
    bus = FakeBus()
    gravity_model = FakeGravityModel()
    compensator = OpenArmLeaderGravityCompensator(
        bus,  # type: ignore[arg-type]
        gravity_model,
        factor=0.5,
        max_torque=2.5,
        kp=0.1,
        kd=0.2,
        joint_signs=[1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        joint_offsets_deg=[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    compensator.step_once()

    np.testing.assert_allclose(
        gravity_model.last_joint_positions_rad,
        np.radians([10.0, -15.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
    )

    assert len(bus.commands) == 1
    commands = bus.commands[0]
    assert set(commands) == {f"joint_{idx}" for idx in range(1, 8)}
    assert commands["joint_1"] == pytest.approx((0.1, 0.2, 0.0, 0.0, 0.5))
    assert commands["joint_2"] == pytest.approx((0.1, 0.2, 0.0, 0.0, -1.0))
    assert commands["joint_7"] == pytest.approx((0.1, 0.2, 0.0, 0.0, 2.5))

    latest_states = compensator.get_latest_states()
    assert latest_states["joint_1"]["position"] == 10.0
    assert latest_states["gripper"]["position"] == 12.0
