#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练前检查 global_expert.jsonl 是否满足 TD3_offline.py 的输入要求。"""

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# ==================== 验证参数：直接修改这里 ====================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "expert_data_dynamic", "global_expert.jsonl")
STATE_DIM = 10
ACTION_DIM = 3
MAX_ACTION = 1.0
SEQ_LEN = 8
BATCH_SIZE = 64
MIN_EPISODE_STEPS = 15
NEAR_GROUND_HEIGHT = 0.60

CORE_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "done",
)


def read_episodes(path: str) -> List[List[Dict[str, Any]]]:
    """读取每行一个完整 episode 的 JSONL。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    episodes: List[List[Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第 {line_number} 行 JSON 解析失败: {error}") from error
            if not isinstance(episode, list):
                raise ValueError(f"第 {line_number} 行不是 episode 列表")
            if any(not isinstance(step, dict) for step in episode):
                raise ValueError(f"第 {line_number} 行包含非字典 step")
            episodes.append(episode)
    if not episodes:
        raise ValueError("数据文件中没有 episode")
    return episodes


def validate_dataset(
    episodes: Sequence[Sequence[Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, Any]]:
    """返回错误列表和简单统计结果。"""
    errors: List[str] = []
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    search_steps = 0
    lost_steps = 0
    search_positive_z = 0
    near_ground_search = 0
    total_transitions = 0
    total_windows = 0

    for episode_index, episode in enumerate(episodes):
        prefix = f"episode {episode_index}"
        if len(episode) < MIN_EPISODE_STEPS:
            errors.append(
                f"{prefix}: 长度 {len(episode)} 小于 {MIN_EPISODE_STEPS}"
            )
        if not episode:
            continue
        if not bool(episode[-1].get("success", False)):
            errors.append(f"{prefix}: 最后一步不是成功落地")

        total_transitions += len(episode)
        total_windows += max(0, len(episode) - SEQ_LEN)

        for step_index, step in enumerate(episode):
            step_prefix = f"{prefix} step {step_index}"
            missing = [field for field in CORE_FIELDS if field not in step]
            if missing:
                errors.append(f"{step_prefix}: 缺少字段 {missing}")
                continue

            try:
                observation = _finite_vector(
                    step["observation"], STATE_DIM, "observation"
                )
                next_observation = _finite_vector(
                    step["next_observation"], STATE_DIM, "next_observation"
                )
                action = _finite_vector(step["action"], ACTION_DIM, "action")
                reward = float(step["reward"])
            except (TypeError, ValueError) as error:
                errors.append(f"{step_prefix}: {error}")
                continue

            if not math.isfinite(reward):
                errors.append(f"{step_prefix}: reward 非有限")
            if np.any(np.abs(action) > MAX_ACTION + 1e-6):
                errors.append(f"{step_prefix}: action 超出 [-{MAX_ACTION}, {MAX_ACTION}]")
            if not 0.0 <= observation[9] <= 1.0:
                errors.append(f"{step_prefix}: observation 置信度超出 [0, 1]")
            if not 0.0 <= next_observation[9] <= 1.0:
                errors.append(f"{step_prefix}: next_observation 置信度超出 [0, 1]")

            expected_done = step_index == len(episode) - 1
            if bool(step["done"]) != expected_done:
                errors.append(f"{step_prefix}: done 位置错误")

            expected_step_index = step.get("step_index", step_index)
            try:
                stored_step_index = int(expected_step_index)
            except (TypeError, ValueError):
                errors.append(f"{step_prefix}: step_index 不是整数")
            else:
                if stored_step_index != step_index:
                    errors.append(
                        f"{step_prefix}: step_index={expected_step_index} 不连续"
                    )

            if step_index + 1 < len(episode):
                try:
                    following = _finite_vector(
                        episode[step_index + 1]["observation"],
                        STATE_DIM,
                        "下一步 observation",
                    )
                    if not np.array_equal(next_observation, following):
                        errors.append(f"{step_prefix}: next_observation 链断裂")
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(f"{step_prefix}: {error}")

            observations.append(observation)
            actions.append(action)

            expert_info = step.get("expert") or {}
            env_info = step.get("env_info") or {}
            phase = str(expert_info.get("phase", ""))
            marker_visible = bool(env_info.get("marker_visible", True))
            next_marker_visible = bool(
                env_info.get("next_marker_visible", marker_visible)
            )
            relative_height = _optional_float(env_info.get("relative_height"))
            if not marker_visible:
                lost_steps += 1
                if step_index > 0:
                    try:
                        previous_observation = _finite_vector(
                            episode[step_index - 1]["observation"],
                            STATE_DIM,
                            "上一 observation",
                        )
                        if not np.array_equal(observation[:3], previous_observation[:3]):
                            errors.append(
                                f"{step_prefix}: 失检时没有保持上一有效位置"
                            )
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(f"{step_prefix}: {error}")
                if not np.array_equal(observation[3:9], np.zeros(6)):
                    errors.append(f"{step_prefix}: 失检时速度或加速度没有清零")
                if observation[9] != 0.0:
                    errors.append(f"{step_prefix}: 失检时置信度不是零")
            if not next_marker_visible:
                if not np.array_equal(next_observation[:3], observation[:3]):
                    errors.append(f"{step_prefix}: 下一帧失检但位置没有保持")
                if not np.array_equal(next_observation[3:9], np.zeros(6)):
                    errors.append(f"{step_prefix}: 下一帧失检但速度或加速度没有清零")
                if next_observation[9] != 0.0:
                    errors.append(f"{step_prefix}: 下一帧失检但置信度不是零")
            if phase == "SEARCH":
                search_steps += 1
                search_positive_z += int(action[2] > 0.0)
                if action[2] <= 0.0:
                    errors.append(f"{step_prefix}: SEARCH 的 z action 不是正值")
                if relative_height is not None and relative_height <= NEAR_GROUND_HEIGHT:
                    near_ground_search += 1
                    errors.append(f"{step_prefix}: 近地失检错误进入 SEARCH")
                if step_index == 0:
                    errors.append(f"{step_prefix}: 第一帧不应直接进入 SEARCH")
                else:
                    try:
                        previous_action = _finite_vector(
                            episode[step_index - 1]["action"],
                            ACTION_DIM,
                            "上一动作",
                        )
                        if not np.array_equal(action[:2], previous_action[:2]):
                            errors.append(
                                f"{step_prefix}: SEARCH 没有保持上一动作的水平分量"
                            )
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(f"{step_prefix}: {error}")

            if (
                not marker_visible
                and relative_height is not None
                and relative_height > NEAR_GROUND_HEIGHT
                and phase != "SEARCH"
            ):
                errors.append(f"{step_prefix}: 非近地失检但没有进入 SEARCH")

        first_observation = _finite_vector(
            episode[0]["observation"], STATE_DIM, f"{prefix} 第一帧 observation"
        )
        if not np.array_equal(first_observation[3:9], np.zeros(6)):
            errors.append(f"{prefix}: 第一帧速度或加速度没有清零")
        if len(episode) > 1:
            second_observation = _finite_vector(
                episode[1]["observation"], STATE_DIM, f"{prefix} 第二帧 observation"
            )
            if not np.array_equal(second_observation[6:9], np.zeros(3)):
                errors.append(f"{prefix}: 第二帧加速度没有清零")

    if total_windows < BATCH_SIZE:
        errors.append(
            f"可用 LSTM 窗口只有 {total_windows}，少于 batch_size={BATCH_SIZE}"
        )

    observation_array = (
        np.stack(observations, axis=0)
        if observations
        else np.empty((0, STATE_DIM), dtype=np.float64)
    )
    action_array = (
        np.stack(actions, axis=0)
        if actions
        else np.empty((0, ACTION_DIM), dtype=np.float64)
    )
    if len(observation_array):
        observation_std = np.std(observation_array, axis=0)
        if np.any(observation_std <= 1e-6):
            errors.append(f"observation 存在退化维度，std={observation_std.tolist()}")

    statistics = {
        "episodes": len(episodes),
        "transitions": total_transitions,
        "lstm_windows": total_windows,
        "lost_steps": lost_steps,
        "search_steps": search_steps,
        "search_positive_z_ratio": (
            float(search_positive_z) / float(search_steps) if search_steps else None
        ),
        "near_ground_search_steps": near_ground_search,
        "observation_mean": _statistic(observation_array, np.mean),
        "observation_std": _statistic(observation_array, np.std),
        "observation_min": _statistic(observation_array, np.min),
        "observation_max": _statistic(observation_array, np.max),
        "action_mean": _statistic(action_array, np.mean),
        "action_std": _statistic(action_array, np.std),
        "action_min": _statistic(action_array, np.min),
        "action_max": _statistic(action_array, np.max),
    }
    return errors, statistics


def _finite_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError(f"{name} 形状应为 ({dimension},)，当前为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限值")
    return vector


def _optional_float(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _statistic(array: np.ndarray, function: Any) -> Any:
    if not len(array):
        return None
    return function(array, axis=0).astype(float).tolist()


def main() -> None:
    print(f"检查数据: {DATA_PATH}")
    try:
        episodes = read_episodes(DATA_PATH)
        errors, statistics = validate_dataset(episodes)
    except (OSError, ValueError) as error:
        print(f"验证失败: {error}")
        raise SystemExit(1) from error

    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    if errors:
        print(f"验证失败，共发现 {len(errors)} 个问题:")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- 其余 {len(errors) - 50} 个问题未展开")
        raise SystemExit(1)
    print("验证通过，可以交给 TD3_offline.py 训练")


if __name__ == "__main__":
    main()
