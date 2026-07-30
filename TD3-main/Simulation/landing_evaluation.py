#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机自主降落评估器 (Simulation 统一版)
========================================
SUCCESS 由在线环境的简化落地检测器决定:
  相对位置、相对高度和相对速度连续稳定指定时间

打分: 仅 SUCCESS 计分; 失败一律 0
  - 精度 70%: 相对 marker 的 3D 偏差
  - 动力学 30%: 相对船速度 + 加速度

无 ROS 依赖, 可离线重算与单测。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EpisodeData:
    """单个 episode 的原始数据, 由评估脚本填充后传入 evaluate()"""

    episode_id: int = 0

    drone_final_x: float = float("nan")
    drone_final_y: float = float("nan")
    drone_final_z: float = float("nan")
    target_x: float = float("nan")
    target_y: float = float("nan")
    target_z: float = float("nan")

    rel_vx: float = float("nan")
    rel_vy: float = float("nan")
    rel_vz: float = float("nan")
    rel_ax: float = float("nan")
    rel_ay: float = float("nan")
    rel_az: float = float("nan")
    impact_velocity_z: float = float("nan")

    steps: int = 0
    dt: float = 0.1
    init_dist_3d: float = float("nan")
    max_height: float = float("nan")
    max_dist_origin: float = float("nan")

    actions: List[List[float]] = field(default_factory=list)
    drone_positions: List[Tuple[float, float, float]] = field(default_factory=list)

    crashed: bool = False
    crash_msg: str = ""
    out_of_bounds: bool = False
    visual_out_of_bounds: bool = False
    max_steps_reached: bool = False
    lost_detection: bool = False
    landing_success: Optional[bool] = None
    relative_height: float = float("nan")
    visual_height: float = float("nan")
    visual_x: float = float("nan")
    visual_y: float = float("nan")
    terminal_reason: str = ""

    ckpt_dir: str = ""
    load_step: int = 0
    platform_type: str = "dynamic"


class LandingEvaluation:
    """降落评估器: 环境稳定降落判定 + 精度/动力学打分。"""

    W_ACCURACY = 0.70
    W_DYNAMICS = 0.30

    def __init__(
        self,
        platform_type: str = "dynamic",
        marker_offset_x: float = -0.20,
        marker_offset_y: float = 0.0,
        marker_offset_z: float = 1.3,
        err3d_threshold: float = 1.2,
        v_rel_max: float = 1.5,
        a_rel_max: float = 5.0,
        max_dist: float = 25.0,
        max_height: float = 12.0,
        **_legacy_kwargs,
    ):
        self.platform_type = platform_type
        self.marker_offset_x = float(marker_offset_x)
        self.marker_offset_y = float(marker_offset_y)
        self.marker_offset_z = float(marker_offset_z)
        self.err3d_threshold = float(err3d_threshold)
        self.v_rel_max = float(v_rel_max)
        self.a_rel_max = float(a_rel_max)
        self.max_dist = float(max_dist)
        self.max_height = float(max_height)

    def classify(self, data: EpisodeData) -> str:
        if data.crashed:
            return "CRASHED"
        if self._is_landing_success(data):
            return "SUCCESS"
        if data.visual_out_of_bounds or data.terminal_reason == "YOLO_OUT_OF_BOUNDS":
            return "YOLO_OUT_OF_BOUNDS"
        if data.out_of_bounds:
            return "OUT_OF_BOUNDS"
        if data.lost_detection:
            return "LOST_DETECTION"
        if data.max_steps_reached:
            return "MAX_STEPS"
        return "MISSED"

    def is_success(self, result: str) -> bool:
        return result == "SUCCESS"

    def _is_landing_success(self, data: EpisodeData) -> bool:
        return data.landing_success is True

    def score_accuracy(self, data: EpisodeData) -> float:
        err_3d = self._err_3d(data)
        if not np.isfinite(err_3d) or self.err3d_threshold <= 0:
            return 0.0
        return 100.0 * max(0.0, 1.0 - err_3d / self.err3d_threshold)

    def score_dynamics(self, data: EpisodeData) -> float:
        v = self._rel_speed(data)
        a = self._rel_accel(data)
        s_v = 0.0
        s_a = 0.0
        if np.isfinite(v) and self.v_rel_max > 0:
            s_v = 100.0 * max(0.0, 1.0 - abs(v) / self.v_rel_max)
        if np.isfinite(a) and self.a_rel_max > 0:
            s_a = 100.0 * max(0.0, 1.0 - abs(a) / self.a_rel_max)
        if np.isfinite(v) and np.isfinite(a):
            return 0.5 * s_v + 0.5 * s_a
        if np.isfinite(v):
            return s_v
        if np.isfinite(a):
            return s_a
        return 0.0

    def composite_score(self, s_acc: float, s_dyn: float) -> float:
        return self.W_ACCURACY * s_acc + self.W_DYNAMICS * s_dyn

    @staticmethod
    def rating(total: float) -> str:
        if total >= 90:
            return "A"
        if total >= 75:
            return "B"
        if total >= 60:
            return "C"
        if total >= 40:
            return "D"
        return "F"

    def evaluate(self, data: EpisodeData) -> Dict[str, Any]:
        result = self.classify(data)
        is_succ = self.is_success(result)
        if is_succ:
            s_acc = self.score_accuracy(data)
            s_dyn = self.score_dynamics(data)
            total = self.composite_score(s_acc, s_dyn)
            grade = self.rating(total)
        else:
            s_acc = s_dyn = total = 0.0
            grade = "F"

        herr = self._horiz_err(data)
        verr = self._vert_err(data)
        dz = self._signed_vert(data)
        err3d = self._err_3d(data)
        rel_speed = self._rel_speed(data)
        rel_accel = self._rel_accel(data)
        vxy = self._rel_vxy(data)
        vz_abs = self._rel_vz_abs(data)

        return {
            "Episode": data.episode_id,
            "Result": result,
            "CompositeScore": total,
            "Rating": grade,
            "Score_Accuracy": s_acc,
            "Score_Dynamics": s_dyn,
            "Score_Efficiency": 0.0,
            "Score_Smoothness": 0.0,
            "Score_Safety": 0.0,
            "FinalX": data.drone_final_x,
            "FinalY": data.drone_final_y,
            "FinalZ": data.drone_final_z,
            "TargetX": data.target_x,
            "TargetY": data.target_y,
            "TargetZ": data.target_z,
            "HorizErr": herr,
            "VertErr": verr,
            "SignedDz": dz,
            "Err3D": err3d,
            "RelSpeed": rel_speed,
            "RelAccel": rel_accel,
            "RelVxy": vxy,
            "RelVzAbs": vz_abs,
            "RelVx": data.rel_vx,
            "RelVy": data.rel_vy,
            "RelVz": data.rel_vz,
            "ImpactVelocityZ": data.impact_velocity_z,
            "Steps": data.steps,
            "TimeSec": data.steps * data.dt,
            "InitDist3D": data.init_dist_3d,
            "MaxHeight": data.max_height,
            "MaxDistOrigin": data.max_dist_origin,
            "LandingSuccess": data.landing_success,
            "RelativeHeight": data.relative_height,
            "VisualHeight": data.visual_height,
            "VisualX": data.visual_x,
            "VisualY": data.visual_y,
            "VisualOutOfBounds": data.visual_out_of_bounds,
            "TerminalReason": data.terminal_reason,
            "ErrorMsg": data.crash_msg,
            "CkptDir": data.ckpt_dir or "",
            "LoadStep": data.load_step,
            "Dt": data.dt,
            "MaxSteps": 0,
            "PlatformType": data.platform_type,
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }

    def summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        if n == 0:
            return {}

        def _count(name: str) -> int:
            return sum(1 for r in records if str(r.get("Result", "")) == name)

        n_reset = sum(1 for r in records if str(r.get("Result", "")).startswith("RESET"))
        n_crashed = _count("CRASHED")
        n_oob = _count("OUT_OF_BOUNDS")
        n_visual_oob = _count("YOLO_OUT_OF_BOUNDS")
        n_lost = _count("LOST_DETECTION")
        n_max = _count("MAX_STEPS")
        n_missed = _count("MISSED")
        n_success = _count("SUCCESS")
        n_aborted = n_oob + n_visual_oob + n_lost + n_max
        n_landed_like = n_success + n_missed

        def _mean_std(key: str):
            vals = [float(r[key]) for r in records if key in r and np.isfinite(float(r[key]))]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        m_total, s_total = _mean_std("CompositeScore")
        m_acc, _ = _mean_std("Score_Accuracy")
        m_dyn, _ = _mean_std("Score_Dynamics")
        m_herr, _ = _mean_std("HorizErr")
        m_verr, _ = _mean_std("VertErr")
        m_vrel, _ = _mean_std("RelSpeed")
        m_arel, _ = _mean_std("RelAccel")

        return {
            "N_total": n,
            "N_RESET_FAILED": n_reset,
            "N_CRASHED": n_crashed,
            "N_ABORTED": n_aborted,
            "N_OUT_OF_BOUNDS": n_oob,
            "N_YOLO_OUT_OF_BOUNDS": n_visual_oob,
            "N_LOST_DETECTION": n_lost,
            "N_MAX_STEPS": n_max,
            "N_MISSED": n_missed,
            "N_SUCCESS": n_success,
            "N_PERFECT": 0,
            "N_GOOD": 0,
            "N_ACCEPTABLE": n_success,
            "N_LANDED": n_landed_like,
            "SR": 100.0 * n_success / max(n, 1),
            "SR_usable": 100.0 * n_success / max(n_landed_like, 1),
            "SR_attempted": 100.0 * n_success / max(n - n_reset - n_crashed, 1),
            "Mean_CompositeScore": m_total,
            "Std_CompositeScore": s_total,
            "Mean_Score_Accuracy": m_acc,
            "Mean_Score_Dynamics": m_dyn,
            "Mean_HorizErr": m_herr,
            "Mean_VertErr": m_verr,
            "Mean_RelSpeed": m_vrel,
            "Mean_RelAccel": m_arel,
        }

    def _horiz_err(self, data: EpisodeData) -> float:
        dx = data.drone_final_x - data.target_x
        dy = data.drone_final_y - data.target_y
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return float("nan")
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def _signed_vert(self, data: EpisodeData) -> float:
        if not np.isfinite(data.drone_final_z) or not np.isfinite(data.target_z):
            return float("nan")
        return float(data.drone_final_z - data.target_z)

    def _vert_err(self, data: EpisodeData) -> float:
        dz = self._signed_vert(data)
        if not np.isfinite(dz):
            return float("nan")
        return float(abs(dz))

    def _err_3d(self, data: EpisodeData) -> float:
        herr = self._horiz_err(data)
        verr = self._vert_err(data)
        if not (np.isfinite(herr) and np.isfinite(verr)):
            return float("nan")
        return float(np.sqrt(herr ** 2 + verr ** 2))

    def _rel_vxy(self, data: EpisodeData) -> float:
        if np.isfinite(data.rel_vx) and np.isfinite(data.rel_vy):
            return float(np.sqrt(data.rel_vx ** 2 + data.rel_vy ** 2))
        return float("nan")

    def _rel_vz_abs(self, data: EpisodeData) -> float:
        if np.isfinite(data.rel_vz):
            return float(abs(data.rel_vz))
        if np.isfinite(data.impact_velocity_z):
            return float(abs(data.impact_velocity_z))
        return float("nan")

    def _rel_speed(self, data: EpisodeData) -> float:
        comps = [data.rel_vx, data.rel_vy, data.rel_vz]
        if all(np.isfinite(c) for c in comps):
            return float(np.sqrt(sum(c ** 2 for c in comps)))
        if np.isfinite(data.impact_velocity_z):
            return float(abs(data.impact_velocity_z))
        return float("nan")

    def _rel_accel(self, data: EpisodeData) -> float:
        comps = [data.rel_ax, data.rel_ay, data.rel_az]
        if all(np.isfinite(c) for c in comps):
            return float(np.sqrt(sum(c ** 2 for c in comps)))
        return float("nan")


def compute_dynamic_target(
    ship_x,
    ship_y,
    ship_z,
    ship_yaw=0.0,
    marker_offset_x=-0.20,
    marker_offset_y=0.0,
    marker_offset_z=1.3,
):
    cos_yaw = np.cos(ship_yaw)
    sin_yaw = np.sin(ship_yaw)
    world_offset_x = cos_yaw * marker_offset_x - sin_yaw * marker_offset_y
    world_offset_y = sin_yaw * marker_offset_x + cos_yaw * marker_offset_y
    return (
        ship_x + world_offset_x,
        ship_y + world_offset_y,
        ship_z + marker_offset_z,
    )


def estimate_velocity_from_positions(positions, dt):
    if len(positions) < 2 or dt <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, p0 = positions[-1], positions[-2]
    return ((p1[0]-p0[0])/dt, (p1[1]-p0[1])/dt, (p1[2]-p0[2])/dt)


def estimate_accel_from_positions(positions, dt):
    if len(positions) < 3 or dt <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p2 = np.asarray(positions[-1], dtype=np.float64)
    p1 = np.asarray(positions[-2], dtype=np.float64)
    p0 = np.asarray(positions[-3], dtype=np.float64)
    v1 = (p2 - p1) / dt
    v0 = (p1 - p0) / dt
    a = (v1 - v0) / dt
    return (float(a[0]), float(a[1]), float(a[2]))


def build_relative_dynamics(
    drone_vx, drone_vy, drone_vz,
    ship_vx, ship_vy, ship_vz=0.0,
    drone_ax=float("nan"), drone_ay=float("nan"), drone_az=float("nan"),
    ship_ax=0.0, ship_ay=0.0, ship_az=0.0,
):
    rel_vx = drone_vx - ship_vx if np.isfinite(drone_vx) and np.isfinite(ship_vx) else float("nan")
    rel_vy = drone_vy - ship_vy if np.isfinite(drone_vy) and np.isfinite(ship_vy) else float("nan")
    rel_vz = drone_vz - ship_vz if np.isfinite(drone_vz) and np.isfinite(ship_vz) else float("nan")
    if all(np.isfinite(c) for c in (drone_ax, drone_ay, drone_az)):
        rel_ax, rel_ay, rel_az = drone_ax - ship_ax, drone_ay - ship_ay, drone_az - ship_az
    else:
        rel_ax = rel_ay = rel_az = float("nan")
    return {
        "rel_vx": float(rel_vx) if np.isfinite(rel_vx) else float("nan"),
        "rel_vy": float(rel_vy) if np.isfinite(rel_vy) else float("nan"),
        "rel_vz": float(rel_vz) if np.isfinite(rel_vz) else float("nan"),
        "rel_ax": float(rel_ax) if np.isfinite(rel_ax) else float("nan"),
        "rel_ay": float(rel_ay) if np.isfinite(rel_ay) else float("nan"),
        "rel_az": float(rel_az) if np.isfinite(rel_az) else float("nan"),
    }


if __name__ == "__main__":
    evaluator = LandingEvaluation(platform_type="dynamic")

    ok = EpisodeData(
        episode_id=1,
        drone_final_x=10.1, drone_final_y=5.05, drone_final_z=1.43,
        target_x=10.0, target_y=5.0, target_z=1.4,
        rel_vx=0.05, rel_vy=0.0, rel_vz=0.05,
        rel_ax=0.1, rel_ay=0.0, rel_az=0.2,
        landing_success=True, visual_height=0.20,
        steps=100, dt=0.1, platform_type="dynamic",
    )
    r = evaluator.evaluate(ok)
    assert r["Result"] == "SUCCESS", r
    assert r["CompositeScore"] > 0, r
    print(f"OK SUCCESS score={r['CompositeScore']:.1f}")

    high = EpisodeData(
        episode_id=2,
        drone_final_x=10.1, drone_final_y=5.0, drone_final_z=1.8,
        target_x=10.0, target_y=5.0, target_z=1.4,
        rel_vx=0.0, rel_vy=0.0, rel_vz=0.0,
        landing_success=False, visual_height=0.8, steps=50, dt=0.1,
    )
    r2 = evaluator.evaluate(high)
    assert r2["Result"] == "MISSED", r2
    print("OK non-landing episode MISSED")

    visual_only = EpisodeData(
        episode_id=3,
        visual_height=0.20,
        steps=50, dt=0.1,
    )
    r3 = evaluator.evaluate(visual_only)
    assert r3["Result"] == "MISSED", r3
    print("OK visual observation alone is not SUCCESS")

    visual_oob = EpisodeData(
        episode_id=4,
        visual_height=1.0,
        visual_x=8.1,
        visual_out_of_bounds=True,
        terminal_reason="YOLO_OUT_OF_BOUNDS",
        steps=10,
        dt=0.1,
    )
    r4 = evaluator.evaluate(visual_oob)
    assert r4["Result"] == "YOLO_OUT_OF_BOUNDS", r4
    print("OK visual out-of-bounds failure")
    print("self-check passed")
