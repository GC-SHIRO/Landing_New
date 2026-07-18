#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert old expert dataset (obs=3 dims) to new LSTM dataset format (obs=6 dims):
- observation becomes 6D: [x,y,z,vx,vy,vz], where (vx,vy,vz) := action
- next_observation becomes 6D similarly (velocity part := action)
- Output JSONL: each line is ONE episode (List[Dict])
- Episodes split by done==True; force last step done=True per episode
- Add episode_final_reward to EVERY step

直接在代码里设置路径：
- OLD_DATA_PATH: 旧数据集（JSON大list 或 JSONL每行step dict）
- NEW_DATA_PATH: 新数据集（JSONL，每行一个episode）
"""

import os
import json
from typing import Any, Dict, List
import numpy as np


# ============================================================
# 直接在这里改路径
# ============================================================
OLD_DATA_PATH = "/home/herbertlin/桌面/work/expert_data_old/expert_data_leftdown.json"
NEW_DATA_PATH = "/home/herbertlin/桌面/work/expert_data_lstm_leftdown.jsonl"
# ============================================================


def _as_f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _obs3(x: Any) -> np.ndarray:
    v = _as_f32(x)
    if v.shape[0] < 3:
        raise ValueError(f"observation dim < 3, got {v.shape[0]}")
    return v[:3]


def _act3(x: Any) -> np.ndarray:
    v = _as_f32(x)
    if v.shape[0] < 3:
        raise ValueError(f"action dim < 3, got {v.shape[0]}")
    return v[:3]


def load_flat_steps(in_path: str) -> List[Dict[str, Any]]:
    """
    Load steps as a flat list of dicts.

    Supported:
    1) JSON: a single list of step dicts
    2) JSONL: each line is a step dict
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    # 先整体读，尝试 json.loads（适配 JSON 大 list）
    with open(in_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    # Try JSON (single big list)
    try:
        obj = json.loads(txt)
        if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
            return obj
        raise ValueError("JSON loaded but not a list of step dicts.")
    except json.JSONDecodeError:
        # Fallback to JSONL (step-per-line)
        steps: List[Dict[str, Any]] = []
        with open(in_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error at line {ln}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {ln} is not a dict.")
                steps.append(obj)
        return steps


def split_by_done(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split flat steps into episodes by done==True.
    如果最后没有done=True，也会把最后残余作为一个episode。
    """
    episodes: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for st in steps:
        cur.append(st)
        if bool(st.get("done", False)):
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


def convert_episode_keep_obs3(ep: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep 3D observation:
    obs3 = first 3 dims of observation
    next_obs3 = first 3 dims of next_observation
    keep action as 3D
    add episode_final_reward to each step
    """
    ep_ret = float(sum(float(st.get("reward", 0.0)) for st in ep))

    new_ep: List[Dict[str, Any]] = []
    for st in ep:
        if "observation" not in st or "next_observation" not in st or "action" not in st:
            raise KeyError("Each step must have keys: observation, next_observation, action.")

        obs3 = _obs3(st["observation"])
        next_obs3 = _obs3(st["next_observation"])
        act = _act3(st["action"])

        new_step = {
            "observation": obs3.tolist(),            # 3D
            "action": act.tolist(),                  # 3D
            "reward": float(st.get("reward", 0.0)),
            "next_observation": next_obs3.tolist(),  # 3D
            "done": bool(st.get("done", False)),
            "episode_final_reward": ep_ret,
        }
        new_ep.append(new_step)

    if new_ep:
        new_ep[-1]["done"] = True

    return new_ep



def save_jsonl_episodes(episodes: List[List[Dict[str, Any]]], out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main():
    steps = load_flat_steps(OLD_DATA_PATH)
    episodes = split_by_done(steps)

    new_eps: List[List[Dict[str, Any]]] = []
    out_steps = 0
    for ep in episodes:
        if not ep:
            continue
        new_ep = convert_episode_keep_obs3(ep)

        new_eps.append(new_ep)
        out_steps += len(new_ep)

    save_jsonl_episodes(new_eps, NEW_DATA_PATH)

    print(f"[OK] OLD_DATA_PATH = {OLD_DATA_PATH}")
    print(f"[OK] NEW_DATA_PATH = {NEW_DATA_PATH}")
    print(f"[OK] in_steps={len(steps)} -> episodes={len(episodes)} -> out_episodes={len(new_eps)}")
    print(f"[OK] out_steps={out_steps}")
    print("[Hint] 输出是 JSONL：每一行一个 episode(List[Dict])，与你的 offline 训练读取逻辑匹配。")


if __name__ == "__main__":
    main()
