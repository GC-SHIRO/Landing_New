#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用仿真真值状态的动态甲板特权 PD 专家。"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

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
    brake_gate_xy: float = 0.60
    descent_gate_xy: float = 0.30
    descent_gate_rel_speed: float = 0.25
    stable_steps_required: int = 2
    brake_xy_speed: float = 0.50
    stable_xy_speed: float = 0.20
    flare_height: float = 0.65
    touchdown_height: float = 0.12
    kp_z: float = 0.85
    max_climb_speed: float = 0.60
    max_descent_speed: float = 0.70
    flare_descent_speed: float = 0.22
    touchdown_descent_speed: float = 0.10
    max_xy_speed: float = 1.0
    max_action: float = 1.0
    # This PX4/MAVROS stack consumes the velocity command with z-up semantics.
    # Keep it aligned with the existing environment and successful manual data.
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
        if not self.approach_gate_xy > self.brake_gate_xy > self.descent_gate_xy:
            raise ValueError("要求 approach_gate_xy > brake_gate_xy > descent_gate_xy")
        if self.stable_steps_required <= 0:
            raise ValueError("stable_steps_required 必须为正数")
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
        self._stable_steps = 0

    def reset(self) -> None:
        self._stable_steps = 0

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
        correction_xy = (
            kp_xy * position_error[:2]
            + kd_xy * velocity_error[:2]
        )
        correction_speed_limit = self._xy_speed_limit(horizontal_error)
        # Keep the platform-velocity feed-forward intact.  Limiting the total
        # velocity here made the aircraft slower than a moving deck near the
        # target, so it could never settle enough to enter DESCEND.
        correction_xy = self._limit_norm(correction_xy, correction_speed_limit)
        world_xy = self._limit_norm(
            target_velocity[:2] + correction_xy, self.config.max_xy_speed
        )

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
            "correction_speed_limit": float(correction_speed_limit),
            "stable_steps": float(self._stable_steps),
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
        stable = (
            horizontal_error <= self.config.descent_gate_xy
            and relative_xy_speed <= self.config.descent_gate_rel_speed
        )
        self._stable_steps = self._stable_steps + 1 if stable else 0
        gate_open = self._stable_steps >= self.config.stable_steps_required

        if not gate_open:
            approach = horizontal_error > self.config.approach_gate_xy
            # Once horizontally close, hold (or regain) the tracking height
            # while braking.  Do not turn an unclosed horizontal velocity into
            # a landing descent merely because the vehicle is already above
            # the deck.
            desired_height = (
                self.config.approach_height
                if approach
                else max(self.config.tracking_height, relative_height)
            )
            height_error = desired_height - relative_height
            world_vz = np.clip(
                self.config.kp_z * height_error,
                -self.config.max_descent_speed,
                self.config.max_climb_speed,
            )
            if approach:
                phase = "APPROACH"
            elif horizontal_error > self.config.descent_gate_xy:
                phase = "BRAKE"
            else:
                phase = "STABLE_TRACK"
            return phase, float(world_vz)

        if relative_height > self.config.flare_height:
            return "DESCEND", -float(self.config.max_descent_speed)
        if relative_height > self.config.touchdown_height:
            return "FLARE", -float(self.config.flare_descent_speed)
        return "TOUCHDOWN", -float(self.config.touchdown_descent_speed)

    def _xy_speed_limit(self, horizontal_error: float) -> float:
        if horizontal_error <= self.config.descent_gate_xy:
            return self.config.stable_xy_speed
        if horizontal_error <= self.config.brake_gate_xy:
            return self.config.brake_xy_speed
        return self.config.max_xy_speed

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


# 本地复刻 Simulation/env_base.GazeboEnv.reward_setup。
# 仅放在 Sampling 内：不 import Simulation，也不做共享模块耦合。
SIM_REWARD_SCALE = 0.1
SIM_SUCCESS_REWARD = 300.0
SIM_FAIL_REWARD = -200.0
# env_base 终端成功框：-1.5 > x > -2.5 且 -1.5 > y > -2.5
SIM_SUCCESS_WORLD_X = (-2.5, -1.5)
SIM_SUCCESS_WORLD_Y = (-2.5, -1.5)


def _simulation_l3_shaping(observation: Sequence[float]) -> float:
    """稠密项：与 env_base.reward_setup 相同的视觉 L3 shaping。"""
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    if obs.shape[0] < 3:
        raise ValueError(f"observation 至少需要 3 维，当前形状为 {obs.shape}")
    delta_x = float(obs[0])
    delta_y = float(obs[1])
    height = float(obs[2])
    shape = -(
        (abs(delta_x) ** 3 + abs(delta_y) ** 3 + abs(height) ** 3) ** (1.0 / 3.0)
    )
    return float(SIM_REWARD_SCALE * shape)


def _in_simulation_success_box(world_x: float, world_y: float) -> bool:
    """是否落在 env_base 终端成功框：-2.5 < x < -1.5 且 -2.5 < y < -1.5。"""
    x_lo, x_hi = SIM_SUCCESS_WORLD_X
    y_lo, y_hi = SIM_SUCCESS_WORLD_Y
    return (x_lo < float(world_x) < x_hi) and (y_lo < float(world_y) < y_hi)


def compute_transition_reward(
    observation: Sequence[float],
    done: bool,
    success: bool,
    world_x: float,
    world_y: float,
    next_observation: Optional[Sequence[float]] = None,
    relative_xy_distance: Optional[float] = None,
    landing_xy_threshold: Optional[float] = None,
) -> float:
    """生成与 Simulation 动态落地判定对齐的逐步奖励。

    稠密项与 env_base.reward_setup 的视觉 L3 shaping 一致。
    终止项判定与 Simulation/env_base.step() 的权威落地判定一致：
        landing_success = deck_contact AND relative_xy_distance <= landing_xy_threshold
    （relative_xy_distance 是相对甲板的水平距离，非世界系固定坐标。）

    旧版成功判定使用静态世界框 (-2.5,-1.5)^2，动态甲板下成功落点常远离该框，
    导致 success=True 也被判为失败奖励 -200。此处改为相对甲板判定：
    - 传入 relative_xy_distance + landing_xy_threshold 时，与 env.step() 的
      position_ok 完全同式；
    - 不传时直接信任 success 标志（env 已按相对甲板 + deck_contact 判定）。

    调用方应传入：
    - observation：步进前视觉观测（与 env_base 一致）
    - world_x/world_y：步进后无人机世界坐标（仅为接口对称保留，不再参与判定）
    - relative_xy_distance：步进后无人机相对甲板的水平距离
    - landing_xy_threshold：env 的 landing_xy_threshold（默认 1.5）
    - next_observation：仅为与 env_base 接口对称保留，实际不使用
    """
    del next_observation  # env_base 也接收该参数，但未使用
    reward = _simulation_l3_shaping(observation)
    if not done:
        return float(reward)
    if bool(success):
        if relative_xy_distance is not None and landing_xy_threshold is not None:
            if float(relative_xy_distance) <= float(landing_xy_threshold):
                return float(SIM_SUCCESS_REWARD)
            return float(SIM_FAIL_REWARD)
        return float(SIM_SUCCESS_REWARD)
    return float(SIM_FAIL_REWARD)
