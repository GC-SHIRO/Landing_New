#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机自主降落评估器
====================
独立的评估类, 实现《无人机自主降落评估规范 v2.0》中的:
  - 结果分类 (RESET_FAILED / CRASHED / ABORTED / PERFECT / GOOD / ACCEPTABLE / MISSED / HARD_LANDING / OFF_PLATFORM)
  - 四维评分 (精度 35% / 效率 25% / 平滑性 20% / 安全性 20%)
  - 综合评分与评级 (A~F)

用法:
  evaluator = LandingEvaluation(platform_type="dynamic", marker_offset_z=1.3)
  record = evaluator.evaluate(episode_data)  # 单 episode 评估
  summary = evaluator.summarize(records)      # 批量汇总

依赖: 纯 Python + numpy, 无 ROS 依赖, 可离线重算。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 数据容器
# ============================================================

@dataclass
class EpisodeData:
    """单个 episode 的原始数据, 由评估脚本填充后传入 evaluate()"""

    episode_id: int = 0

    # ---- 终点状态 ----
    drone_final_x: float = float("nan")
    drone_final_y: float = float("nan")
    drone_final_z: float = float("nan")
    target_x: float = float("nan")
    target_y: float = float("nan")
    target_z: float = float("nan")

    # ---- 过程统计 ----
    steps: int = 0
    dt: float = 0.1
    init_dist_3d: float = float("nan")
    max_height: float = float("nan")
    max_dist_origin: float = float("nan")
    impact_velocity_z: float = float("nan")  # 触地垂速

    # ---- 轨迹序列 ----
    actions: List[List[float]] = field(default_factory=list)  # [[vx,vy,vz], ...]
    drone_positions: List[Tuple[float, float, float]] = field(default_factory=list)

    # ---- 标记 ----
    crashed: bool = False
    crash_msg: str = ""
    out_of_bounds: bool = False
    max_steps_reached: bool = False
    lost_detection: bool = False

    # ---- 元信息 ----
    ckpt_dir: str = ""
    load_step: int = 0
    platform_type: str = "dynamic"


# ============================================================
# LandingEvaluation
# ============================================================

class LandingEvaluation:
    """降落评估器: 分类 + 四维评分 + 综合评级"""

    # 评分权重
    W_ACCURACY = 0.35
    W_EFFICIENCY = 0.25
    W_SMOOTHNESS = 0.20
    W_SAFETY = 0.20

    def __init__(
        self,
        platform_type: str = "dynamic",
        # 判定阈值
        success_herr_thresh: float = 1.0,
        success_height_thresh: float = 0.6,
        perfect_herr_thresh: float = 0.3,
        perfect_height_thresh: float = 0.15,
        good_herr_thresh: float = 0.6,
        good_height_thresh: float = 0.3,
        impact_vz_limit: float = 1.5,
        max_dist: float = 25.0,
        max_height: float = 12.0,
        # 坐标参数
        marker_offset_z: float = 1.3,
        # 评分参数
        err3d_threshold: float = 2.0,
        tpm_ref: float = 1.0,    # 每米耗时参考值 s/m
        tpm_max: float = 5.0,    # 每米耗时容忍上限
        j_max: float = 0.5,      # 动作平滑度容忍上限
        z_max_allowed: float = 12.0,
        v_imp_max: float = 1.5   # 冲击速度容忍上限 m/s
    ):
        self.platform_type = platform_type

        # 判定阈值
        self.success_herr_thresh = success_herr_thresh
        self.success_height_thresh = success_height_thresh
        self.perfect_herr_thresh = perfect_herr_thresh
        self.perfect_height_thresh = perfect_height_thresh
        self.good_herr_thresh = good_herr_thresh
        self.good_height_thresh = good_height_thresh
        self.impact_vz_limit = impact_vz_limit
        self.max_dist = max_dist
        self.max_height = max_height

        # 坐标
        self.marker_offset_z = marker_offset_z

        # 评分参数
        self.err3d_threshold = err3d_threshold
        self.tpm_ref = tpm_ref
        self.tpm_max = tpm_max
        self.j_max = j_max
        self.z_max_allowed = z_max_allowed
        self.v_imp_max = v_imp_max

    # -------- 分类 --------

    def classify(self, data: EpisodeData) -> str:
        """根据 episode 数据返回结果分类字符串"""
        if data.crashed:
            return "CRASHED"

        if data.out_of_bounds:
            return "OUT_OF_BOUNDS"

        if data.max_steps_reached:
            return "MAX_STEPS"

        if data.lost_detection:
            return "LOST_DETECTION"

        # ---- LANDED: 计算触地误差 (使用相对目标的垂直误差) ----
        herr = self._horiz_err(data)
        verr = self._vert_err(data)
        impact_vz = abs(data.impact_velocity_z)

        # 判断是否触地 (垂直误差 < 2×成功阈值, 放宽避免漏判)
        landed = (
            np.isfinite(herr) and np.isfinite(verr)
            and verr < self.success_height_thresh * 2.0
        )
        if not landed:
            return "MAX_STEPS"

        # ---- 动态平台额外检查 ----
        if self.platform_type == "dynamic" and herr > 2.0:
            return "OFF_PLATFORM"

        # ---- 成功分级 (均使用相对目标误差) ----
        if herr < self.perfect_herr_thresh and verr < self.perfect_height_thresh and impact_vz < 0.5:
            return "PERFECT"
        if herr < self.good_herr_thresh and verr < self.good_height_thresh:
            return "GOOD"
        if herr < self.success_herr_thresh and verr < self.success_height_thresh:
            return "ACCEPTABLE"

        # ---- 失败分类 ----
        if impact_vz >= self.impact_vz_limit:
            return "HARD_LANDING"
        return "MISSED"

    def is_success(self, result: str) -> bool:
        return result in ("PERFECT", "GOOD", "ACCEPTABLE")

    # -------- 四维评分 --------

    def score_accuracy(self, data: EpisodeData) -> float:
        """终点精度评分 0~100"""
        err_3d = self._err_3d(data)
        if not np.isfinite(err_3d):
            return 0.0
        return 100.0 * max(0.0, 1.0 - err_3d / self.err3d_threshold)

    def score_efficiency(self, data: EpisodeData) -> float:
        """效率评分 0~100"""
        if data.init_dist_3d is None or data.init_dist_3d < 1e-8:
            return 0.0
        tpm = (data.steps * data.dt) / data.init_dist_3d
        return 100.0 * max(0.0, 1.0 - (tpm - self.tpm_ref) / self.tpm_max)

    def score_smoothness(self, data: EpisodeData) -> float:
        """平滑性评分 0~100"""
        j_act = self._action_smoothness(data.actions)
        if not np.isfinite(j_act):
            return 0.0
        return 100.0 * max(0.0, 1.0 - j_act / self.j_max)

    def score_safety(self, data: EpisodeData) -> float:
        """安全性评分 0~100"""
        if data.out_of_bounds:
            return 0.0

        height_ok = min(1.0, self.z_max_allowed / max(data.max_height, 1e-6))
        impact_ok = max(0.0, 1.0 - abs(data.impact_velocity_z) / self.v_imp_max)
        return 100.0 * (0.5 * height_ok + 0.5 * impact_ok)

    # -------- 综合评分 --------

    def composite_score(self, s_acc: float, s_eff: float, s_smooth: float, s_safe: float) -> float:
        return (
            self.W_ACCURACY * s_acc
            + self.W_EFFICIENCY * s_eff
            + self.W_SMOOTHNESS * s_smooth
            + self.W_SAFETY * s_safe
        )

    @staticmethod
    def rating(total: float) -> str:
        """A~F 评级"""
        if total >= 90:
            return "A"
        if total >= 75:
            return "B"
        if total >= 60:
            return "C"
        if total >= 40:
            return "D"
        return "F"

    # -------- 主入口 --------

    def evaluate(self, data: EpisodeData) -> Dict[str, Any]:
        """
        完整评估一个 episode, 返回可写入 JSONL 的 dict。
        包含分类、四维分、综合分、所有衍生指标。
        """
        result = self.classify(data)
        is_succ = self.is_success(result)

        # 仅成功着陆 (PERFECT/GOOD/ACCEPTABLE) 计算评分, 其余一律 0 分
        if is_succ:
            s_acc = self.score_accuracy(data)
            s_eff = self.score_efficiency(data)
            s_smooth = self.score_smoothness(data)
            s_safe = self.score_safety(data)
            total = self.composite_score(s_acc, s_eff, s_smooth, s_safe)
            rating = self.rating(total)
        else:
            s_acc = s_eff = s_smooth = s_safe = 0.0
            total = 0.0
            rating = "F"

        herr = self._horiz_err(data)
        verr = self._vert_err(data)
        err3d = self._err_3d(data)
        tpm = (data.steps * data.dt) / max(data.init_dist_3d, 1e-8)
        j_act = self._action_smoothness(data.actions)
        j_jerk = self._jerk(data.actions)
        dir_change = self._direction_change_rate(data.actions)
        path_ratio = self._path_ratio(data)

        record = {
            "Episode": data.episode_id,
            "Result": result,
            "CompositeScore": total,
            "Rating": rating,

            "Score_Accuracy": s_acc,
            "Score_Efficiency": s_eff,
            "Score_Smoothness": s_smooth,
            "Score_Safety": s_safe,

            "FinalX": data.drone_final_x,
            "FinalY": data.drone_final_y,
            "FinalZ": data.drone_final_z,
            "TargetX": data.target_x,
            "TargetY": data.target_y,
            "TargetZ": data.target_z,
            "HorizErr": herr,
            "VertErr": verr,
            "Err3D": err3d,
            "NormHorizErr": herr / max(data.init_dist_3d, 1e-8) if np.isfinite(herr) else float("nan"),

            "Steps": data.steps,
            "TimeSec": data.steps * data.dt,
            "InitDist3D": data.init_dist_3d,
            "TimePerMeter3D": tpm,
            "StepsPerMeter3D": data.steps / max(data.init_dist_3d, 1e-8),
            "PathRatio": path_ratio,

            "ActionSmoothness": j_act,
            "Jerk": j_jerk,
            "DirectionChangeRate": dir_change,

            "MaxHeight": data.max_height,
            "MaxDistOrigin": data.max_dist_origin,
            "ImpactVelocityZ": data.impact_velocity_z,

            "ErrorMsg": data.crash_msg,
            "CkptDir": data.ckpt_dir or "",
            "LoadStep": data.load_step,
            "Dt": data.dt,
            "MaxSteps": 0,  # filled by caller
            "PlatformType": data.platform_type,
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        return record

    # -------- 汇总 --------

    def summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """对一批 episode 记录做汇总统计"""
        n = len(records)
        if n == 0:
            return {}

        def _count(result_prefix: str) -> int:
            return sum(1 for r in records if str(r.get("Result", "")).startswith(result_prefix))

        n_reset_fail = _count("RESET_FAILED")
        n_crashed = _count("CRASHED")
        n_aborted = _count("OUT_OF_BOUNDS") + _count("MAX_STEPS") + _count("LOST_DETECTION")
        n_perfect = _count("PERFECT")
        n_good = _count("GOOD")
        n_acceptable = _count("ACCEPTABLE")
        n_success = n_perfect + n_good + n_acceptable
        n_landed = n_success + _count("MISSED") + _count("HARD_LANDING") + _count("OFF_PLATFORM")
        n_usable = n_success + n_landed  # LANDED episodes

        def _mean_std(key: str) -> Tuple[float, float]:
            vals = [float(r[key]) for r in records if key in r and np.isfinite(float(r[key]))]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        m_total, s_total = _mean_std("CompositeScore")
        m_acc, _ = _mean_std("Score_Accuracy")
        m_eff, _ = _mean_std("Score_Efficiency")
        m_smooth, _ = _mean_std("Score_Smoothness")
        m_safe, _ = _mean_std("Score_Safety")
        m_herr, _ = _mean_std("HorizErr")
        m_tpm, _ = _mean_std("TimePerMeter3D")
        m_impact, _ = _mean_std("ImpactVelocityZ")
        m_path, _ = _mean_std("PathRatio")

        return {
            "N_total": n,
            "N_RESET_FAILED": n_reset_fail,
            "N_CRASHED": n_crashed,
            "N_ABORTED": n_aborted,
            "N_LANDED": n_landed,
            "N_PERFECT": n_perfect,
            "N_GOOD": n_good,
            "N_ACCEPTABLE": n_acceptable,
            "SR": 100.0 * n_success / max(n, 1),
            "SR_usable": 100.0 * n_success / max(n_usable, 1),
            "SR_attempted": 100.0 * n_success / max(n - n_reset_fail - n_crashed, 1),
            "Mean_CompositeScore": m_total,
            "Std_CompositeScore": s_total,
            "Mean_Score_Accuracy": m_acc,
            "Mean_Score_Efficiency": m_eff,
            "Mean_Score_Smoothness": m_smooth,
            "Mean_Score_Safety": m_safe,
            "Mean_HorizErr": m_herr,
            "Mean_TimePerMeter3D": m_tpm,
            "Mean_ImpactVelocityZ": m_impact,
            "Mean_PathRatio": m_path,
        }

    # -------- 内部工具 --------

    def _horiz_err(self, data: EpisodeData) -> float:
        dx = data.drone_final_x - data.target_x
        dy = data.drone_final_y - data.target_y
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return float("nan")
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def _vert_err(self, data: EpisodeData) -> float:
        if not np.isfinite(data.drone_final_z) or not np.isfinite(data.target_z):
            return float("nan")
        return float(abs(data.drone_final_z - data.target_z))

    def _err_3d(self, data: EpisodeData) -> float:
        herr = self._horiz_err(data)
        verr = self._vert_err(data)
        if not (np.isfinite(herr) and np.isfinite(verr)):
            return float("nan")
        return float(np.sqrt(herr ** 2 + verr ** 2))

    @staticmethod
    def _action_smoothness(actions: List) -> float:
        if len(actions) < 2:
            return 0.0
        a = np.asarray(actions, dtype=np.float32)
        diffs = a[1:] - a[:-1]
        return float(np.mean(np.sum(diffs ** 2, axis=1)))

    @staticmethod
    def _jerk(actions: List) -> float:
        if len(actions) < 3:
            return 0.0
        a = np.asarray(actions, dtype=np.float32)
        d1 = a[1:] - a[:-1]
        d2 = d1[1:] - d1[:-1]
        return float(np.mean(np.sum(d2 ** 2, axis=1)))

    @staticmethod
    def _direction_change_rate(actions: List) -> float:
        if len(actions) < 2:
            return 0.0
        a = np.asarray(actions, dtype=np.float32)
        count = 0
        for i in range(1, len(a)):
            dot = np.dot(a[i - 1], a[i])
            norm = np.linalg.norm(a[i - 1]) * np.linalg.norm(a[i])
            if norm > 1e-9 and dot / norm < 0:  # angle > 90°
                count += 1
        return count / len(a)

    @staticmethod
    def _path_ratio(data: EpisodeData) -> float:
        if len(data.drone_positions) < 2 or data.init_dist_3d < 1e-8:
            return float("nan")
        total = 0.0
        for i in range(1, len(data.drone_positions)):
            dx = data.drone_positions[i][0] - data.drone_positions[i - 1][0]
            dy = data.drone_positions[i][1] - data.drone_positions[i - 1][1]
            dz = data.drone_positions[i][2] - data.drone_positions[i - 1][2]
            total += math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return total / data.init_dist_3d


# ============================================================
# 便捷函数: 从船舶位置计算动态平台目标点
# ============================================================

def compute_dynamic_target(ship_x: float, ship_y: float, ship_z: float,
                           marker_offset_z: float = 1.3) -> Tuple[float, float, float]:
    """根据船 base_link 位置计算甲板 marker 的世界坐标"""
    return (ship_x, ship_y, ship_z + marker_offset_z)


if __name__ == "__main__":
    # 简单自测
    evaluator = LandingEvaluation(platform_type="dynamic")

    data = EpisodeData(
        episode_id=1,
        drone_final_x=10.2, drone_final_y=5.1, drone_final_z=1.5,
        target_x=10.0, target_y=5.0, target_z=1.4,
        steps=150, dt=0.1, init_dist_3d=10.0,
        max_height=9.0, max_dist_origin=12.0, impact_velocity_z=0.3,
        actions=[[0.1, 0.0, -0.2]] * 150,
        platform_type="dynamic",
    )

    record = evaluator.evaluate(data)
    print(f"Result: {record['Result']}")
    print(f"Composite: {record['CompositeScore']:.1f}  Rating: {record['Rating']}")
    print(f"Acc={record['Score_Accuracy']:.1f} Eff={record['Score_Efficiency']:.1f} "
          f"Smooth={record['Score_Smoothness']:.1f} Safe={record['Score_Safety']:.1f}")
