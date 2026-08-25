#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式就地转换旧三维 JSONL 专家数据为十维 observation。"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


# ==================== 转换参数：直接修改这里 ====================
TIME_DELTA = 0.1
MAX_VISUAL_SPEED = 10.0
MAX_VISUAL_ACCELERATION = 100.0
STATE_DIM = 10
BACKUP_SUFFIX = ".pre_10d_backup"


class VisualMotionObservation:
    """与当前采集器一致的十维视觉状态构造器。"""

    def __init__(self, time_delta: float) -> None:
        if not math.isfinite(time_delta) or time_delta <= 0.0:
            raise ValueError("TIME_DELTA 必须为正的有限数值")
        self.time_delta = float(time_delta)
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.has_previous_velocity = False

    def initialize(self, position: Sequence[float], confidence: float) -> np.ndarray:
        self.position = _position(position, "第一帧 observation").astype(np.float32)
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.has_previous_velocity = False
        return self._compose(confidence)

    def update(
        self, position: Sequence[float], marker_visible: bool, confidence: float
    ) -> np.ndarray:
        if not marker_visible:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self.has_previous_velocity = False
            return self._compose(0.0)

        next_position = _position(position, "后续 observation").astype(np.float32)
        next_velocity = np.clip(
            (next_position - self.position) / self.time_delta,
            -MAX_VISUAL_SPEED,
            MAX_VISUAL_SPEED,
        ).astype(np.float32)
        if self.has_previous_velocity:
            next_acceleration = np.clip(
                (next_velocity - self.velocity) / self.time_delta,
                -MAX_VISUAL_ACCELERATION,
                MAX_VISUAL_ACCELERATION,
            ).astype(np.float32)
        else:
            next_acceleration = np.zeros(3, dtype=np.float32)

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        self.has_previous_velocity = True
        return self._compose(confidence)

    def _compose(self, confidence: float) -> np.ndarray:
        return np.concatenate(
            (self.position, self.velocity, self.acceleration, [_confidence(confidence)])
        ).astype(np.float32)


def convert_episode(episode: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将一条旧三维 episode 转为十维，并保持 transition 链严格连续。"""
    if not episode:
        raise ValueError("episode 不能为空")
    if any(not isinstance(step, dict) for step in episode):
        raise ValueError("episode 包含非对象 step")

    states: List[np.ndarray] = []
    builder = VisualMotionObservation(TIME_DELTA)
    first_visible = _marker_visible(episode[0], "marker_visible", True)
    states.append(builder.initialize(
        _position(episode[0].get("observation"), "第一帧 observation"),
        float(first_visible),
    ))

    for index, step in enumerate(episode):
        current_visible = _marker_visible(step, "marker_visible", first_visible)
        visible = _marker_visible(step, "next_marker_visible", current_visible)
        next_position = _position(
            step.get("next_observation"), f"第 {index} 步 next_observation"
        )
        states.append(builder.update(next_position, visible, float(visible)))

    converted: List[Dict[str, Any]] = []
    for index, step in enumerate(episode):
        converted_step = dict(step)
        converted_step["observation"] = states[index].astype(float).tolist()
        converted_step["next_observation"] = states[index + 1].astype(float).tolist()
        env_info = dict(converted_step.get("env_info") or {})
        env_info["observation_confidence"] = float(states[index][9])
        env_info["next_observation_confidence"] = float(states[index + 1][9])
        env_info["confidence_source"] = "legacy_marker_visible_proxy"
        converted_step["env_info"] = env_info
        converted.append(converted_step)
    return converted


def convert_file_inplace(path: Path) -> Path:
    """先完整写入临时文件，再备份并原子替换输入 JSONL。"""
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if backup_path.exists():
        raise FileExistsError(f"备份已存在，拒绝覆盖: {backup_path}")

    temp_path: Path | None = None
    converted_episodes = 0
    try:
        with path.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"第 {line_number} 行 JSON 解析失败: {error}") from error
                if not isinstance(episode, list):
                    raise ValueError(f"第 {line_number} 行不是 episode 列表")
                converted = convert_episode(episode)
                output.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_episodes += 1

        if converted_episodes == 0:
            raise ValueError("文件中没有可转换的 episode")
        shutil.copy2(path, backup_path)
        os.replace(temp_path, path)
        temp_path = None
        return backup_path
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _position(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} 必须是旧版 3 维数组，当前形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限数值")
    return vector


def _marker_visible(step: Dict[str, Any], key: str, default: bool) -> bool:
    env_info = step.get("env_info") or {}
    if not isinstance(env_info, dict):
        raise ValueError("env_info 必须是对象")
    return bool(env_info.get(key, default))


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(confidence, 0.0, 1.0)) if math.isfinite(confidence) else 0.0


def main() -> None:
    print("旧三维 JSONL 数据集转换为十维 observation")
    print("注意：旧数据没有真实 YOLO 置信度，将以 marker_visible 写入 1/0 代理值。")
    raw_path = input("请输入要就地转换的 JSONL 文件路径: ").strip()
    if not raw_path:
        print("未输入路径，已取消。")
        return
    path = Path(raw_path).expanduser().resolve()
    print(f"目标文件: {path}")
    print(f"将自动创建备份: {path.name}{BACKUP_SUFFIX}")
    if input("输入 YES 确认就地转换: ").strip() != "YES":
        print("未确认，已取消。")
        return
    try:
        backup_path = convert_file_inplace(path)
    except (OSError, ValueError) as error:
        print(f"转换失败，原文件未被替换: {error}")
        raise SystemExit(1) from error
    print(f"转换完成，原文件已更新；备份文件: {backup_path}")


if __name__ == "__main__":
    main()
