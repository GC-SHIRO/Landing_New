#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用仿真真值状态的动态甲板特权 PD 专家。"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class PrivilegedPDConfig:
    kp_xy: float = 0.85
    kd_xy: float = 0.35
    precision_kp_xy: float = 1.15
    precision_kd_xy: float = 0.50
    precision_height: float = 1.5
    approach_height: float = 2.5
    tracking_height: float = 1.2
    approach_gate_xy: float = 1.0
    descent_gate_xy: float = 0.25
    descent_gate_rel_speed: float = 0.25
    flare_height: float = 0.65
    touchdown_height: float = 0.12
    kp_z: float = 0.85
    max_climb_speed: float = 0.60
    max_descent_speed: float = 0.70
    flare_descent_speed: float = 0.22
    touchdown_descent_speed: float = 0.10
    max_xy_speed: float = 1.0
    max_action: float = 1.0
    body_z_down: bool = False

    def validate(self) -> None:
        if min(
            self.kp_xy,
            self.kd_xy,
            self.precision_kp_xy,
            self.precision_kd_xy,
            self.kp_z,
        ) < 0:
            raise ValueError("PD 增益必须非负")
        if self.approach_gate_xy <= self.descent_gate_xy:
            raise ValueError("approach_gate_xy 必须大于 descent_gate_xy")
        if self.approach_height <= self.tracking_height:
            raise ValueError("approach_height 必须大于 tracking_height")
        if self.tracking_height <= self.flare_height:
            raise ValueError("tracking_height 必须大于 flare_height")
        if self.flare_height <= self.touchdown_height:
            raise ValueError("flare_height 必须大于 touchdown_height")
        if min(self.max_xy_speed, self.max_action) <= 0:
            raise ValueError("动作限幅必须为正数")


@dataclass
class ExpertCommand:
    action: np.ndarray
    world_velocity: np.ndarray
    phase: str
    diagnostics: Dict[str, float]

    def to_metadata(self) -> Dict[str, object]:
        return {
            "phase": self.phase,
            "world_velocity": self.world_velocity.astype(float).tolist(),
            "diagnostics": dict(self.diagnostics),
        }


class PrivilegedPDExpert:
    """船速前馈、相对位置 PD 与分阶段下降组成的特权专家。"""

    def __init__(self, config: PrivilegedPDConfig = None):
        self.config = config or PrivilegedPDConfig()
        self.config.validate()

    def reset(self) -> None:
        return None

    def compute_action(
        self,
        drone_position: Sequence[float],
        drone_velocity: Sequence[float],
        drone_yaw: float,
        target_position: Sequence[float],
        target_velocity: Sequence[float],
    ) -> ExpertCommand:
        drone_position = self._vector3(drone_position, "drone_position")
        drone_velocity = self._vector3(drone_velocity, "drone_velocity")
        target_position = self._vector3(target_position, "target_position")
        target_velocity = self._vector3(target_velocity, "target_velocity")

        position_error = target_position - drone_position
        velocity_error = target_velocity - drone_velocity
        horizontal_error = float(np.linalg.norm(position_error[:2]))
        relative_xy_speed = float(np.linalg.norm(velocity_error[:2]))
        relative_height = float(drone_position[2] - target_position[2])

        precision_mode = relative_height <= self.config.precision_height
        kp_xy = self.config.precision_kp_xy if precision_mode else self.config.kp_xy
        kd_xy = self.config.precision_kd_xy if precision_mode else self.config.kd_xy
        world_xy = (
            target_velocity[:2]
            + kp_xy * position_error[:2]
            + kd_xy * velocity_error[:2]
        )
        world_xy = self._limit_norm(world_xy, self.config.max_xy_speed)

        phase, world_vz = self._vertical_command(
            horizontal_error=horizontal_error,
            relative_xy_speed=relative_xy_speed,
            relative_height=relative_height,
        )
        world_velocity = np.array(
            [world_xy[0], world_xy[1], world_vz], dtype=np.float32
        )
        action = self.world_enu_to_body_action(
            world_velocity=world_velocity,
            drone_yaw=float(drone_yaw),
            body_z_down=self.config.body_z_down,
        )
        action = np.clip(
            action, -self.config.max_action, self.config.max_action
        ).astype(np.float32)

        diagnostics = {
            "horizontal_error": horizontal_error,
            "relative_xy_speed": relative_xy_speed,
            "relative_height": relative_height,
            "position_error_x": float(position_error[0]),
            "position_error_y": float(position_error[1]),
            "position_error_z": float(position_error[2]),
            "target_speed": float(np.linalg.norm(target_velocity[:2])),
            "precision_mode": float(precision_mode),
        }
        return ExpertCommand(
            action=action,
            world_velocity=world_velocity,
            phase=phase,
            diagnostics=diagnostics,
        )

    def _vertical_command(
        self,
        horizontal_error: float,
        relative_xy_speed: float,
        relative_height: float,
    ) -> Tuple[str, float]:
        gate_open = (
            horizontal_error <= self.config.descent_gate_xy
            and relative_xy_speed <= self.config.descent_gate_rel_speed
        )

        if not gate_open:
            approach = horizontal_error > self.config.approach_gate_xy
            desired_height = self.config.approach_height if approach else self.config.tracking_height
            height_error = desired_height - relative_height
            world_vz = np.clip(
                self.config.kp_z * height_error,
                -self.config.max_descent_speed,
                self.config.max_climb_speed,
            )
            phase = "APPROACH" if approach else "MATCH"
            return phase, float(world_vz)

        if relative_height > self.config.flare_height:
            return "DESCEND", -float(self.config.max_descent_speed)
        if relative_height > self.config.touchdown_height:
            return "FLARE", -float(self.config.flare_descent_speed)
        return "TOUCHDOWN", -float(self.config.touchdown_descent_speed)

    @staticmethod
    def world_enu_to_body_action(
        world_velocity: Sequence[float],
        drone_yaw: float,
        body_z_down: bool = False,
    ) -> np.ndarray:
        velocity = PrivilegedPDExpert._vector3(world_velocity, "world_velocity")
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        body_x = cos_yaw * velocity[0] + sin_yaw * velocity[1]
        body_y = -sin_yaw * velocity[0] + cos_yaw * velocity[1]
        body_z = -velocity[2] if body_z_down else velocity[2]
        return np.array([body_x, body_y, body_z], dtype=np.float32)

    @staticmethod
    def build_privileged_observation(
        drone_position: Sequence[float],
        drone_yaw: float,
        target_position: Sequence[float],
    ) -> np.ndarray:
        drone_position = PrivilegedPDExpert._vector3(
            drone_position, "drone_position"
        )
        target_position = PrivilegedPDExpert._vector3(
            target_position, "target_position"
        )
        delta = target_position - drone_position
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        target_body_x = cos_yaw * delta[0] + sin_yaw * delta[1]
        target_body_y = -sin_yaw * delta[0] + cos_yaw * delta[1]
        height_above_deck = drone_position[2] - target_position[2]
        return np.array(
            [target_body_x, target_body_y, height_above_deck], dtype=np.float32
        )

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)

    @staticmethod
    def _limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm < 1e-9:
            return vector
        return vector * (max_norm / norm)

    @staticmethod
    def _vector3(value: Sequence[float], name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (3,):
            raise ValueError(f"{name} 必须为 3 维，当前形状为 {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} 包含非有限值: {vector}")
        return vector


def compute_transition_reward(
    previous_distance: float,
    current_distance: float,
    horizontal_error: float,
    relative_height: float,
    relative_xy_speed: float,
    action: Sequence[float],
    success: bool,
    done: bool,
) -> float:
    """为自动采集数据生成与动态目标一致的稠密奖励。"""
    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    progress = float(previous_distance) - float(current_distance)
    reward = 4.0 * progress
    reward -= 0.08 * float(horizontal_error)
    reward -= 0.03 * abs(float(relative_height))
    reward -= 0.04 * float(relative_xy_speed)
    reward -= 0.01 * float(np.dot(action_array, action_array))
    if success:
        reward += 100.0
    elif done:
        reward -= 25.0
    return float(reward)
