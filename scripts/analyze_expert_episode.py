#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途:
- 解析专家演示数据并生成统计结果, 支持 JSON / JSONL / 目录批处理输入。

用法:
- python analyze_expert_episode.py --input <json|jsonl|dir> --out_dir <dir>
- 输出 summary.csv / episodes.csv / steps.csv。

实现方式:
- 自动识别 episode-list、step-list 及 done 分段。
- 统计每回合长度、奖励、初末状态距离、动作幅值与效率指标。

依赖关系:
- 仅依赖 Python 标准库 + numpy。
- 与训练脚本解耦, 可独立用于数据质检。
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_obs_xyz(obs: Any) -> Optional[Tuple[float, float, float]]:
    """
    Extract (x,y,z) from observation.
    Expect obs to be list/tuple with at least 3 numbers.
    Return None if cannot parse.
    """
    if obs is None:
        return None
    if isinstance(obs, (list, tuple)) and len(obs) >= 3:
        try:
            x = float(obs[0])
            y = float(obs[1])
            z = float(obs[2])
            return x, y, z
        except Exception:
            return None
    return None


def l2(x: float, y: float, z: float) -> float:
    return float(math.sqrt(x * x + y * y + z * z))


def horiz(x: float, y: float) -> float:
    return float(math.sqrt(x * x + y * y))


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def is_transition_dict(x: Any) -> bool:
    """
    Heuristic: a transition step is a dict and usually contains at least observation/action/reward/done.
    We keep it permissive to avoid rejecting variants.
    """
    if not isinstance(x, dict):
        return False
    # "observation" and "action" are common in your dataset
    if "observation" in x or "action" in x or "reward" in x or "done" in x:
        return True
    return False


# -----------------------------
# Episode structures
# -----------------------------
@dataclass
class EpisodeStats:
    episode_idx: int
    steps: int
    done_flag_present: bool
    terminated_at_end: bool

    # Observation-derived
    init_x: Optional[float]
    init_y: Optional[float]
    init_z: Optional[float]
    init_dist3d: Optional[float]
    init_horiz: Optional[float]

    final_x: Optional[float]
    final_y: Optional[float]
    final_z: Optional[float]
    final_dist3d: Optional[float]
    final_horiz: Optional[float]

    # Reward
    reward_sum: Optional[float]
    reward_mean: Optional[float]
    reward_std: Optional[float]
    final_step_reward: Optional[float]
    episode_final_reward_field: Optional[float]  # from data if exists

    # Action stats
    action_mean_abs: Optional[float]   # mean(|a|) over all dims & steps
    action_rms: Optional[float]        # rms over all dims & steps
    action_max_abs: Optional[float]    # max(|a|)

    # Efficiency (requires dt & init_dist3d)
    time_s: Optional[float]
    time_per_meter3d: Optional[float]
    steps_per_meter3d: Optional[float]


def compute_action_stats(actions: List[List[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not actions:
        return None, None, None
    a = np.array(actions, dtype=np.float64)  # (T, A)
    abs_a = np.abs(a)
    mean_abs = float(abs_a.mean())
    rms = float(np.sqrt((a ** 2).mean()))
    max_abs = float(abs_a.max())
    return mean_abs, rms, max_abs


def compute_reward_stats(rewards: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not rewards:
        return None, None, None, None
    r = np.array(rewards, dtype=np.float64)
    return float(r.sum()), float(r.mean()), float(r.std(ddof=0)), float(r[-1])


def analyze_one_episode(transitions: List[Dict[str, Any]], episode_idx: int, dt: float) -> EpisodeStats:
    steps = len(transitions)
    done_flag_present = any("done" in t for t in transitions)
    terminated_at_end = bool(transitions[-1].get("done", False)) if steps > 0 else False

    # observations
    obs0 = parse_obs_xyz(transitions[0].get("observation")) if steps > 0 else None
    obsT = parse_obs_xyz(transitions[-1].get("observation")) if steps > 0 else None

    if obs0 is not None:
        init_x, init_y, init_z = obs0
        init_dist3d = l2(init_x, init_y, init_z)
        init_horiz = horiz(init_x, init_y)
    else:
        init_x = init_y = init_z = init_dist3d = init_horiz = None

    if obsT is not None:
        final_x, final_y, final_z = obsT
        final_dist3d = l2(final_x, final_y, final_z)
        final_horiz = horiz(final_x, final_y)
    else:
        final_x = final_y = final_z = final_dist3d = final_horiz = None

    # rewards
    rewards: List[float] = []
    episode_final_reward_field = None
    for t in transitions:
        rv = safe_float(t.get("reward"))
        if rv is not None:
            rewards.append(rv)
        # episode_final_reward usually same every step; take last non-null
        efr = safe_float(t.get("episode_final_reward"))
        if efr is not None:
            episode_final_reward_field = efr

    reward_sum, reward_mean, reward_std, final_step_reward = compute_reward_stats(rewards)

    # actions
    actions: List[List[float]] = []
    for t in transitions:
        a = t.get("action", None)
        if isinstance(a, (list, tuple)) and len(a) > 0:
            try:
                actions.append([float(x) for x in a])
            except Exception:
                pass
    action_mean_abs, action_rms, action_max_abs = compute_action_stats(actions)

    # efficiency
    time_s = float(steps * dt) if steps > 0 else None
    time_per_meter3d = None
    steps_per_meter3d = None
    if init_dist3d is not None and init_dist3d > 1e-9 and steps > 0:
        time_per_meter3d = float(time_s / init_dist3d) if time_s is not None else None
        steps_per_meter3d = float(steps / init_dist3d)

    return EpisodeStats(
        episode_idx=episode_idx,
        steps=steps,
        done_flag_present=done_flag_present,
        terminated_at_end=terminated_at_end,

        init_x=init_x, init_y=init_y, init_z=init_z,
        init_dist3d=init_dist3d,
        init_horiz=init_horiz,

        final_x=final_x, final_y=final_y, final_z=final_z,
        final_dist3d=final_dist3d,
        final_horiz=final_horiz,

        reward_sum=reward_sum,
        reward_mean=reward_mean,
        reward_std=reward_std,
        final_step_reward=final_step_reward,
        episode_final_reward_field=episode_final_reward_field,

        action_mean_abs=action_mean_abs,
        action_rms=action_rms,
        action_max_abs=action_max_abs,

        time_s=time_s,
        time_per_meter3d=time_per_meter3d,
        steps_per_meter3d=steps_per_meter3d,
    )


# -----------------------------
# Robust loading (fix Extra data)
# -----------------------------
def load_any_json_objects_from_text(text: str) -> List[Any]:
    """
    Parse possibly-multi JSON from text.

    Supports:
      1) single JSON (json.loads)
      2) JSONL: each line is JSON
      3) concatenated JSON values via JSONDecoder.raw_decode loop
    """
    text = text.strip()
    if not text:
        return []

    # 1) try normal single JSON
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        pass

    # 2) try JSON lines
    objs: List[Any] = []
    ok_jsonl = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            ok_jsonl = False
            break
    if ok_jsonl and objs:
        return objs

    # 3) fallback: concatenated JSON parsing
    dec = json.JSONDecoder()
    i = 0
    n = len(text)
    objs = []
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        objs.append(obj)
        i = end
    return objs


def normalize_obj_to_episodes(obj: Any) -> List[List[Dict[str, Any]]]:
    """
    Normalize arbitrary JSON object into List[Episode], where Episode is List[transition dict].
    Handles wrappers and common dataset layouts.
    """
    # unwrap common wrappers
    if isinstance(obj, dict):
        for k in ("episodes", "data", "dataset", "trajectories", "trajectory", "steps"):
            if k in obj:
                obj = obj[k]
                break

    # case A: one episode = list of transition dicts
    if isinstance(obj, list) and (len(obj) == 0 or all(is_transition_dict(x) for x in obj)):
        return [obj]  # single episode

    # case B: list of episodes
    if isinstance(obj, list) and len(obj) > 0 and all(isinstance(ep, list) for ep in obj):
        episodes: List[List[Dict[str, Any]]] = []
        for ep in obj:
            if isinstance(ep, list) and (len(ep) == 0 or all(is_transition_dict(x) for x in ep)):
                episodes.append(ep)
        if episodes:
            return episodes

    return []


def split_step_objects_into_episodes(step_objs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    For JSONL where each line is a transition dict, split into episodes by done==True.
    """
    episodes: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for s in step_objs:
        if not isinstance(s, dict):
            continue
        cur.append(s)
        if bool(s.get("done", False)):
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


def load_episodes_from_file(path: str) -> List[List[Dict[str, Any]]]:
    """
    Read a file and robustly parse episodes from .json or .jsonl.

    - If it is JSONL with episode-per-line: each line is a JSON list of transitions.
    - If it is JSONL with step-per-line: each line is a JSON dict transition; split by done.
    - If it is .json but contains multiple concatenated JSON values: parse all and normalize.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    top_objs = load_any_json_objects_from_text(text)
    if not top_objs:
        return []

    # If many objects and they all look like transition dicts -> step-per-line JSONL
    if len(top_objs) > 1 and all(is_transition_dict(o) for o in top_objs):
        return split_step_objects_into_episodes(top_objs)  # type: ignore[arg-type]

    # Otherwise normalize each object into episodes
    episodes: List[List[Dict[str, Any]]] = []
    for obj in top_objs:
        eps = normalize_obj_to_episodes(obj)
        if eps:
            episodes.extend(eps)

    # If still empty but single object is a list of dict transitions (possible even when len(top_objs)==1 handled above)
    if not episodes and len(top_objs) == 1:
        obj0 = top_objs[0]
        if isinstance(obj0, list) and all(is_transition_dict(x) for x in obj0):
            episodes = [obj0]

    if not episodes:
        raise ValueError(
            f"Could not parse episodes from file: {path}\n"
            f"Top-level objects parsed: {len(top_objs)}. "
            f"Expected episode list, list-of-episodes, JSONL episodes, or JSONL step dicts."
        )
    return episodes


def discover_episodes(input_path: str) -> List[List[Dict[str, Any]]]:
    if os.path.isfile(input_path):
        if input_path.endswith(".json") or input_path.endswith(".jsonl"):
            return load_episodes_from_file(input_path)
        raise ValueError("Input file must be .json or .jsonl")

    if os.path.isdir(input_path):
        episodes: List[List[Dict[str, Any]]] = []
        files = sorted(os.listdir(input_path))
        for fn in files:
            p = os.path.join(input_path, fn)
            if os.path.isdir(p):
                continue
            if fn.endswith(".json") or fn.endswith(".jsonl"):
                episodes.extend(load_episodes_from_file(p))
        if not episodes:
            raise ValueError(f"No .json/.jsonl episodes found in directory: {input_path}")
        return episodes

    raise ValueError(f"Not a file or directory: {input_path}")


# -----------------------------
# Aggregation + output
# -----------------------------
def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    v = np.array(vals, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=0))


def write_csv(path: str, rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser("Analyze expert demos (episode JSON arrays) -> summary + csv")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "data" / "expert_data" / "expert_data_lstm.json"),
                    help="Path to episode .json/.jsonl OR directory containing them")
    ap.add_argument("--out_dir", default="/home/herbertlin/桌面/work/Landing_new/experdateanl", help="Output directory")
    ap.add_argument("--dt", type=float, default=0.1, help="Seconds per step (for time stats)")
    ap.add_argument("--export_steps", action="store_true", help="Also export per-step steps.csv (can be large)")
    ap.add_argument("--success_horiz", type=float, default=1.0, help="Success threshold on horizontal error (m)")
    ap.add_argument("--success_z", type=float, default=0.6, help="Success threshold on abs(z) or z (see --z_mode)")
    ap.add_argument("--z_mode", choices=["raw", "abs"], default="raw",
                    help="raw: require final_z < success_z; abs: require abs(final_z) < success_z")
    args = ap.parse_args()

    episodes = discover_episodes(args.input)

    ep_stats: List[EpisodeStats] = []
    step_rows: List[List[Any]] = []

    for i, ep in enumerate(episodes):
        if not isinstance(ep, list):
            continue
        # ensure transitions are dicts (skip garbage)
        ep_clean = [t for t in ep if isinstance(t, dict)]
        st = analyze_one_episode(ep_clean, episode_idx=i, dt=args.dt)
        ep_stats.append(st)

        if args.export_steps:
            # per-step export: obs, action, reward
            for t_idx, t in enumerate(ep_clean):
                obs = parse_obs_xyz(t.get("observation"))
                act = t.get("action")
                rew = safe_float(t.get("reward"))
                done = bool(t.get("done", False))
                efr = safe_float(t.get("episode_final_reward"))
                if obs is None:
                    x = y = z = None
                    h = d3 = None
                else:
                    x, y, z = obs
                    h = horiz(x, y)
                    d3 = l2(x, y, z)
                # flatten action (up to 3 dims typical)
                a0 = a1 = a2 = None
                if isinstance(act, (list, tuple)):
                    if len(act) > 0:
                        a0 = safe_float(act[0])
                    if len(act) > 1:
                        a1 = safe_float(act[1])
                    if len(act) > 2:
                        a2 = safe_float(act[2])

                step_rows.append([i, t_idx, x, y, z, h, d3, a0, a1, a2, rew, done, efr])

    # Success definition based on final obs (if available)
    success_flags = []
    usable_flags = []
    for st in ep_stats:
        if st.final_horiz is None or st.final_z is None:
            usable_flags.append(False)
            success_flags.append(False)
            continue
        usable_flags.append(True)
        if args.z_mode == "abs":
            z_ok = abs(st.final_z) < args.success_z
        else:
            z_ok = st.final_z < args.success_z
        success_flags.append((st.final_horiz < args.success_horiz) and z_ok)

    episodes_all = len(ep_stats)
    episodes_usable = sum(usable_flags)
    success_count = sum(1 for u, s in zip(usable_flags, success_flags) if u and s)

    # Aggregate numeric metrics
    steps_list = [st.steps for st in ep_stats]
    time_list = [st.time_s for st in ep_stats if st.time_s is not None]
    initdist_list = [st.init_dist3d for st in ep_stats if st.init_dist3d is not None]
    finalh_list = [st.final_horiz for st in ep_stats if st.final_horiz is not None]
    finalz_list = [st.final_z for st in ep_stats if st.final_z is not None]

    reward_sum_list = [st.reward_sum for st in ep_stats if st.reward_sum is not None]
    reward_mean_list = [st.reward_mean for st in ep_stats if st.reward_mean is not None]
    final_step_reward_list = [st.final_step_reward for st in ep_stats if st.final_step_reward is not None]
    episode_final_reward_field_list = [st.episode_final_reward_field for st in ep_stats if st.episode_final_reward_field is not None]

    tpm_list = [st.time_per_meter3d for st in ep_stats if st.time_per_meter3d is not None]
    spm_list = [st.steps_per_meter3d for st in ep_stats if st.steps_per_meter3d is not None]

    action_mean_abs_list = [st.action_mean_abs for st in ep_stats if st.action_mean_abs is not None]
    action_rms_list = [st.action_rms for st in ep_stats if st.action_rms is not None]
    action_max_abs_list = [st.action_max_abs for st in ep_stats if st.action_max_abs is not None]

    steps_mean, steps_std = mean_std([float(x) for x in steps_list])
    time_mean, time_std = mean_std([float(x) for x in time_list])
    init_mean, init_std = mean_std([float(x) for x in initdist_list])
    fh_mean, fh_std = mean_std([float(x) for x in finalh_list])
    fz_mean, fz_std = mean_std([float(x) for x in finalz_list])

    rsum_mean, rsum_std = mean_std([float(x) for x in reward_sum_list])
    rmean_mean, rmean_std = mean_std([float(x) for x in reward_mean_list])
    fr_mean, fr_std = mean_std([float(x) for x in final_step_reward_list])
    efr_mean, efr_std = mean_std([float(x) for x in episode_final_reward_field_list])

    tpm_mean, tpm_std = mean_std([float(x) for x in tpm_list])
    spm_mean, spm_std = mean_std([float(x) for x in spm_list])

    ama_mean, ama_std = mean_std([float(x) for x in action_mean_abs_list])
    arms_mean, arms_std = mean_std([float(x) for x in action_rms_list])
    amax_mean, amax_std = mean_std([float(x) for x in action_max_abs_list])

    # summary.csv
    summary_rows = [
        ["Metric", "Value"],
        ["Episodes (all)", float(episodes_all)],
        ["Episodes usable (final obs parsed)", float(episodes_usable)],
        ["Success threshold", f"horiz<{args.success_horiz}, z({args.z_mode})<{args.success_z}"],
        ["Success count", float(success_count)],
        ["Success Rate over usable (%)", float(100.0 * success_count / episodes_usable) if episodes_usable > 0 else ""],
        ["Success Rate over all (%)", float(100.0 * success_count / episodes_all) if episodes_all > 0 else ""],

        ["Mean Steps", steps_mean if steps_mean is not None else ""],
        ["Std Steps", steps_std if steps_std is not None else ""],
        ["Mean Time (s)", time_mean if time_mean is not None else ""],
        ["Std Time (s)", time_std if time_std is not None else ""],

        ["Mean InitDist3D (m)", init_mean if init_mean is not None else ""],
        ["Std InitDist3D (m)", init_std if init_std is not None else ""],
        ["Mean Final HorizErr (m)", fh_mean if fh_mean is not None else ""],
        ["Std Final HorizErr (m)", fh_std if fh_std is not None else ""],
        ["Mean Final Z", fz_mean if fz_mean is not None else ""],
        ["Std Final Z", fz_std if fz_std is not None else ""],

        ["Mean TimePerMeter3D (s/m)", tpm_mean if tpm_mean is not None else ""],
        ["Std TimePerMeter3D (s/m)", tpm_std if tpm_std is not None else ""],
        ["Mean StepsPerMeter3D (steps/m)", spm_mean if spm_mean is not None else ""],
        ["Std StepsPerMeter3D (steps/m)", spm_std if spm_std is not None else ""],

        ["Mean RewardSum", rsum_mean if rsum_mean is not None else ""],
        ["Std RewardSum", rsum_std if rsum_std is not None else ""],
        ["Mean RewardMeanPerStep", rmean_mean if rmean_mean is not None else ""],
        ["Std RewardMeanPerStep", rmean_std if rmean_std is not None else ""],
        ["Mean FinalStepReward", fr_mean if fr_mean is not None else ""],
        ["Std FinalStepReward", fr_std if fr_std is not None else ""],

        ["Mean episode_final_reward field", efr_mean if efr_mean is not None else ""],
        ["Std episode_final_reward field", efr_std if efr_std is not None else ""],

        ["Mean |action| (all dims)", ama_mean if ama_mean is not None else ""],
        ["Std |action| (all dims)", ama_std if ama_std is not None else ""],
        ["Mean action RMS", arms_mean if arms_mean is not None else ""],
        ["Std action RMS", arms_std if arms_std is not None else ""],
        ["Mean max|action| per-episode", amax_mean if amax_mean is not None else ""],
        ["Std max|action| per-episode", amax_std if amax_std is not None else ""],
    ]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    # episodes.csv
    ep_header = list(asdict(ep_stats[0]).keys()) if ep_stats else ["episode_idx"]
    ep_rows = [ep_header]
    for st in ep_stats:
        d = asdict(st)
        ep_rows.append([d.get(k) for k in ep_header])
    write_csv(os.path.join(out_dir, "episodes.csv"), ep_rows)

    # steps.csv
    if args.export_steps:
        step_header = ["episode_idx", "t", "x", "y", "z", "horiz_err", "dist3d",
                       "a0", "a1", "a2", "reward", "done", "episode_final_reward"]
        write_csv(os.path.join(out_dir, "steps.csv"), [step_header] + step_rows)

    # print console summary
    print("=== Expert Demo Analysis ===")
    print(f"Input: {args.input}")
    print(f"Episodes (all): {episodes_all}")
    print(f"Episodes usable: {episodes_usable}")
    print(f"Success: {success_count}/{episodes_usable}  (threshold: horiz<{args.success_horiz}, z({args.z_mode})<{args.success_z})")
    if episodes_usable > 0:
        print(f"Success rate (usable): {100.0 * success_count / episodes_usable:.2f}%")
    if time_mean is not None:
        print(f"Mean time: {time_mean:.3f}s ± {time_std:.3f}s")
    if steps_mean is not None:
        print(f"Mean steps: {steps_mean:.3f} ± {steps_std:.3f}")
    if init_mean is not None:
        print(f"Mean init dist3d: {init_mean:.3f}m ± {init_std:.3f}m")
    if fh_mean is not None:
        print(f"Mean final horiz err: {fh_mean:.3f}m ± {fh_std:.3f}m")
    if fz_mean is not None:
        print(f"Mean final z: {fz_mean:.3f} ± {fz_std:.3f}")
    print(f"Outputs -> {out_dir}/summary.csv, {out_dir}/episodes.csv" + (", steps.csv" if args.export_steps else ""))


if __name__ == "__main__":
    main()
