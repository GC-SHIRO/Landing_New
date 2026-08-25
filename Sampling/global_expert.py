#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用仿真全局真值控制动态甲板降落的简化专家。"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ExpertConfig:
    """专家参数，常用数值由采集脚本顶部统一传入。"""

    kp_xy: float = 0.85
    kd_xy: float = 0.35
    kp_z: float = 0.85
    align_xy_threshold: float = 1.0
    descend_xy_threshold: float = 0.30
    descend_vxy_threshold: float = 0.25
    stable_steps_required: int = 2
    tracking_height: float = 2.0
    flare_height: float = 0.60
    touchdown_height: float = 0.15
    near_ground_height: float = 0.60
    max_xy_speed: float = 1.0
    max_descent_speed: float = 0.50
    max_climb_speed: float = 0.40
    flare_descent_speed: float = 0.20
    touchdown_descent_speed: float = 0.08
    search_climb_speed: float = 0.25
    max_xy_delta: float = 0.15
    max_z_delta: float = 0.10
    max_action: float = 1.0

    def validate(self) -> None:
        """只检查会直接产生错误数据的参数。"""
        if self.stable_steps_required <= 0:
            raise ValueError("stable_steps_required 必须大于 0")
        if not self.tracking_height > self.flare_height > self.touchdown_height:
            raise ValueError("高度阈值必须满足 tracking > flare > touchdown")
        if self.near_ground_height < self.touchdown_height:
            raise ValueError("near_ground_height 不能低于 touchdown_height")
        if min(
            self.max_xy_speed,
            self.max_descent_speed,
            self.max_climb_speed,
            self.search_climb_speed,
            self.max_action,
        ) <= 0.0:
            raise ValueError("速度上限必须大于 0")
        if max(
            self.max_xy_speed,
            self.max_descent_speed,
            self.max_climb_speed,
            self.search_climb_speed,
        ) > self.max_action:
            raise ValueError("专家速度不能超过 max_action")


@dataclass
class ExpertCommand:
    """专家单步输出。"""

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


class GlobalLandingExpert:
    """用全局位置和速度真值生成三维机体速度动作。"""

    def __init__(self, config: Optional[ExpertConfig] = None):
        self.config = config or ExpertConfig()
        self.config.validate()
        self._stable_steps = 0
        self._previous_action: Optional[np.ndarray] = None
        self._was_searching = False

    def reset(self) -> None:
        self._stable_steps = 0
        self._previous_action = None
        self._was_searching = False

    def compute_action(
        self,
        drone_position: Sequence[float],
        drone_velocity: Sequence[float],
        drone_yaw: float,
        target_position: Sequence[float],
        target_velocity: Sequence[float],
        marker_visible: bool = True,
        last_action: Optional[Sequence[float]] = None,
    ) -> ExpertCommand:
        """计算正常降落或目标丢失后的 SEARCH 动作。"""
        drone_position = self._vector3(drone_position, "drone_position")
        drone_velocity = self._vector3(drone_velocity, "drone_velocity")
        target_position = self._vector3(target_position, "target_position")
        target_velocity = self._vector3(target_velocity, "target_velocity")

        position_error = target_position - drone_position
        velocity_error = target_velocity - drone_velocity
        horizontal_error = float(np.linalg.norm(position_error[:2]))
        relative_xy_speed = float(np.linalg.norm(velocity_error[:2]))
        relative_height = float(drone_position[2] - target_position[2])

        # 非近地失检时保持上一条指令的水平分量，并用正 z 搜寻 marker。
        if not marker_visible and relative_height > self.config.near_ground_height:
            action = self._search_action(last_action)
            world_velocity = self.body_action_to_world_enu(action, float(drone_yaw))
            self._previous_action = action.copy()
            self._stable_steps = 0
            self._was_searching = True
            return ExpertCommand(
                action=action,
                world_velocity=world_velocity,
                phase="SEARCH",
                diagnostics=self._diagnostics(
                    horizontal_error,
                    relative_xy_speed,
                    relative_height,
                    marker_visible,
                ),
            )

        correction_xy = (
            self.config.kp_xy * position_error[:2]
            + self.config.kd_xy * velocity_error[:2]
        )
        world_xy = self._limit_norm(
            target_velocity[:2] + correction_xy,
            self.config.max_xy_speed,
        )
        phase, world_vz = self._vertical_command(
            horizontal_error,
            relative_xy_speed,
            relative_height,
        )
        world_velocity = np.array(
            [world_xy[0], world_xy[1], world_vz], dtype=np.float32
        )
        raw_action = self.world_enu_to_body_action(world_velocity, float(drone_yaw))
        if self._was_searching:
            # 重新看到 marker 后立即恢复专家动作，不延续 SEARCH 的上升指令。
            action = np.clip(
                raw_action, -self.config.max_action, self.config.max_action
            ).astype(np.float32)
        else:
            action = self._smooth_and_clip(raw_action)
        self._previous_action = action.copy()
        self._was_searching = False

        return ExpertCommand(
            action=action,
            world_velocity=world_velocity,
            phase=phase,
            diagnostics=self._diagnostics(
                horizontal_error,
                relative_xy_speed,
                relative_height,
                marker_visible,
            ),
        )

    def _vertical_command(
        self,
        horizontal_error: float,
        relative_xy_speed: float,
        relative_height: float,
    ) -> Tuple[str, float]:
        stable = (
            horizontal_error <= self.config.descend_xy_threshold
            and relative_xy_speed <= self.config.descend_vxy_threshold
        )
        self._stable_steps = self._stable_steps + 1 if stable else 0
        if self._stable_steps < self.config.stable_steps_required:
            # 对准期间不继续盲目下降；低于跟踪高度时才缓慢爬升恢复高度。
            desired_height = max(self.config.tracking_height, relative_height)
            height_error = desired_height - relative_height
            vz = float(
                np.clip(
                    self.config.kp_z * height_error,
                    -self.config.max_descent_speed,
                    self.config.max_climb_speed,
                )
            )
            phase = (
                "ALIGN"
                if horizontal_error > self.config.align_xy_threshold
                else "TRACK"
            )
            return phase, vz

        if relative_height > self.config.flare_height:
            return "DESCEND", -float(self.config.max_descent_speed)
        if relative_height > self.config.touchdown_height:
            return "TOUCHDOWN", -float(self.config.flare_descent_speed)
        return "TOUCHDOWN", -float(self.config.touchdown_descent_speed)

    def _search_action(self, last_action: Optional[Sequence[float]]) -> np.ndarray:
        if last_action is None:
            previous = self._previous_action
        else:
            previous = self._vector3(last_action, "last_action").astype(np.float32)
        if previous is None:
            previous = np.zeros(3, dtype=np.float32)
        action = np.array(
            [previous[0], previous[1], self.config.search_climb_speed],
            dtype=np.float32,
        )
        return np.clip(
            action, -self.config.max_action, self.config.max_action
        ).astype(np.float32)

    def _smooth_and_clip(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(
            action, -self.config.max_action, self.config.max_action
        ).astype(np.float32)
        if self._previous_action is None:
            return action
        lower = self._previous_action - np.array(
            [self.config.max_xy_delta, self.config.max_xy_delta, self.config.max_z_delta],
            dtype=np.float32,
        )
        upper = self._previous_action + np.array(
            [self.config.max_xy_delta, self.config.max_xy_delta, self.config.max_z_delta],
            dtype=np.float32,
        )
        return np.clip(action, lower, upper).astype(np.float32)

    def _diagnostics(
        self,
        horizontal_error: float,
        relative_xy_speed: float,
        relative_height: float,
        marker_visible: bool,
    ) -> Dict[str, float]:
        return {
            "horizontal_error": float(horizontal_error),
            "relative_xy_speed": float(relative_xy_speed),
            "relative_height": float(relative_height),
            "marker_visible": float(bool(marker_visible)),
            "stable_steps": float(self._stable_steps),
        }

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)

    @staticmethod
    def world_enu_to_body_action(
        world_velocity: Sequence[float], drone_yaw: float
    ) -> np.ndarray:
        velocity = GlobalLandingExpert._vector3(world_velocity, "world_velocity")
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        body_x = cos_yaw * velocity[0] + sin_yaw * velocity[1]
        body_y = -sin_yaw * velocity[0] + cos_yaw * velocity[1]
        return np.array([body_x, body_y, velocity[2]], dtype=np.float32)

    @staticmethod
    def body_action_to_world_enu(
        action: Sequence[float], drone_yaw: float
    ) -> np.ndarray:
        body = GlobalLandingExpert._vector3(action, "action")
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        world_x = cos_yaw * body[0] - sin_yaw * body[1]
        world_y = sin_yaw * body[0] + cos_yaw * body[1]
        return np.array([world_x, world_y, body[2]], dtype=np.float32)

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
            raise ValueError(f"{name} 必须是 3 维，当前形状为 {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} 包含非有限值: {vector}")
        return vector
