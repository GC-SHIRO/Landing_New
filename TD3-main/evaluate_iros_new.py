#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用途:
- 在 Gazebo 仿真中评估 TD3/LSTM 策略, 输出按回合和按轨迹的论文口径指标。

用法:
- IDE 运行: 修改 IDE_RUN_NAME、ckpt_dir 等参数后直接运行。
- 命令行运行: python evaluate_iros_new.py --test_episodes N --ckpt_dir <dir> --load_step <k>

实现方式:
- 每回合执行策略并实时记录, 回合结束立即写入 jsonl/csv, 避免中断导致数据丢失。
- 支持自动/手动断点续跑, 并统计稳定后末态误差、时间效率、平滑性等指标。

依赖关系:
- 依赖 TD3_offline.py 中 TD3 封装加载策略权重。
- 依赖 landing_env.py 与 ROS/Gazebo 交互。
- 输出数据供 analyze_eval_results.py 二次汇总。
"""

import os
import time
import json
import argparse
from collections import deque
from types import SimpleNamespace

import numpy as np
import pandas as pd

from TD3_offline import TD3

from landing_env import GazeboEnv


# ================= 配置部分（母板结构，字段保留） =================
parser = argparse.ArgumentParser(description="IROS Eval (robust per-episode writing + IDE resume)")

parser.add_argument('--test_episodes', type=int, default=102, help='计划测试回合数（可中断后续跑）')
parser.add_argument('--save_data_path', type=str,
                    default='./Landing_new/evaluation_data/TD3_LSTM',
                    help='数据保存路径')

# ✅ 默认改成与你训练时一致的 ckpt_dir（非常关键）
parser.add_argument('--ckpt_dir', type=str,
                    default='/home/shiro/Landing_new/checkpoints/TD3/LSTM',
                    help='模型权重文件夹路径')
parser.add_argument('--load_step', type=int, default=60000, help='加载哪一步的模型权重')

parser.add_argument('--state_dim', default=3, type=int)
parser.add_argument('--action_dim', default=3, type=int)
parser.add_argument('--max_action', default=1.0, type=float)
parser.add_argument('--capacity', default=65536, type=int)

# 母板字段：保留，但绝对不要传入 TD3_offline.Args
parser.add_argument('--learning_rate', default=3e-4, type=float)

parser.add_argument('--policy_noise', default=0.0, type=float)  # 评估时建议 0
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau', default=0.005, type=float)

# ===== LSTM 相关（必须与训练一致）=====
parser.add_argument('--seq_len', type=int, default=8, help='必须与训练时一致')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--attn_hidden_dim', type=int, default=64)
parser.add_argument('--dropout_p', type=float, default=0.1)
parser.add_argument('--lr_actor', type=float, default=1e-4)
parser.add_argument('--lr_critic', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)

# ===== 评估控制 =====
parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--max_steps', type=int, default=600)

# ===== 物理稳定缓冲 =====
parser.add_argument('--settle_seconds', type=float, default=3.0,
                    help='触地后等待物理稳定的时间（秒）')

# ===== 判定阈值（论文可复现）=====
parser.add_argument('--success_herr_thresh', type=float, default=1.0, help='稳定后水平误差阈值(m)')
parser.add_argument('--success_height_thresh', type=float, default=0.6, help='稳定后高度阈值(m)')

# ====== 写盘/续跑参数（仍保留，但IDE可覆盖）======
parser.add_argument('--run_name', type=str, default='', help='本次评估名字，用于文件名区分')
parser.add_argument('--append', action='store_true', help='追加写入（便于断点续跑）；默认覆盖新文件')
parser.add_argument('--resume', action='store_true', help='从已有 episode_metrics.jsonl 自动续跑（跳过已写回合）')
parser.add_argument('--write_running_summary', action='store_true', help='每回合后写 running_summary.csv（累计统计）')
parser.add_argument('--max_reset_fail', type=int, default=3, help='reset 连续失败多少次后终止评估')


# ================== IDE 运行开关：A + B 结合（放这里） ==================
# A: IDE_RESUME 三态
#   None  => 自动判断（B）
#   True  => 强制续跑
#   False => 强制从头跑（会清空同 run_name 的旧文件）
IDE_RESUME = None

# 用于区分不同实验输出（强烈建议设一个）
IDE_RUN_NAME = "td3lstm_60000"

# B: 自动判断：文件存在且非空 => resume（仅当 IDE_RESUME=None 时生效）
AUTO_RESUME = True
# ======================================================================

# IDE 点运行：不读命令行参数（避免你必须在终端输入）
args = parser.parse_args(args=[])

# 覆盖 run_name（IDE）
args.run_name = str(IDE_RUN_NAME)

os.makedirs(args.save_data_path, exist_ok=True)


# ================= 文件命名 =================
def _tagged(name: str) -> str:
    rn = str(args.run_name).strip()
    if rn == "":
        return name
    base, ext = os.path.splitext(name)
    return f"{base}_{rn}{ext}"


EP_JSONL_PATH = os.path.join(args.save_data_path, _tagged("episode_metrics.jsonl"))
TRAJ_CSV_PATH = os.path.join(args.save_data_path, _tagged("detailed_trajectories.csv"))
RUN_SUM_PATH = os.path.join(args.save_data_path, _tagged("running_summary.csv"))


# ================= 断点续跑：A + B 合并逻辑（放这里：路径已生成） =================
def _auto_resume(jsonl_path: str) -> bool:
    return os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0


# 先决定 resume
if IDE_RESUME is None:
    if AUTO_RESUME:
        args.resume = _auto_resume(EP_JSONL_PATH)
    else:
        args.resume = False
else:
    args.resume = bool(IDE_RESUME)

# resume => 强制 append
if args.resume:
    args.append = True


def _load_done_episodes(jsonl_path: str):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ep = obj.get("Episode", None)
            if ep is None:
                continue
            try:
                done.add(int(ep))
            except Exception:
                pass
    return done


def _next_episode_to_run(done_set: set, total_planned: int):
    for ep in range(1, total_planned + 1):
        if ep not in done_set:
            return ep
    return None


# 全新跑：清空旧文件（只影响同 run_name 的文件）
if (not args.append) and (not args.resume):
    for p in [EP_JSONL_PATH, TRAJ_CSV_PATH, RUN_SUM_PATH]:
        if os.path.exists(p):
            os.remove(p)

DONE_EPISODES = set()
START_EPISODE = 1
if args.resume:
    DONE_EPISODES = _load_done_episodes(EP_JSONL_PATH)
    nxt = _next_episode_to_run(DONE_EPISODES, args.test_episodes)
    if nxt is None:
        print(f"[Resume] 已完成 1..{args.test_episodes} 所有回合，无需继续。")
        raise SystemExit(0)
    START_EPISODE = nxt
    print(f"[Resume] 已写入回合数={len(DONE_EPISODES)}，将从 Episode {START_EPISODE} 继续。")
else:
    print("[Run] 非 resume 模式：从 Episode 1 开始（会覆盖同名旧文件）。")
# ======================================================================


def calculate_smoothness(actions):
    if len(actions) < 2:
        return 0.0
    actions = np.asarray(actions, dtype=np.float32)
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def _filter_args_for_td3_offline(eval_args):
    allowed = {
        "state_dim", "action_dim", "max_action",
        "gamma", "tau", "policy_delay", "policy_noise", "noise_clip",
        "seq_len", "hidden_dim", "attn_hidden_dim", "dropout_p",
        "lr_actor", "lr_critic", "weight_decay",
        "capacity", "ckpt_dir",
    }
    d = vars(eval_args).copy()
    filtered = {k: d[k] for k in allowed if k in d}
    return SimpleNamespace(**filtered)


def _load_norm_stats(ckpt_dir, state_dim):
    mean_path = os.path.join(ckpt_dir, 'state_mean.npy')
    std_path = os.path.join(ckpt_dir, 'state_std.npy')

    if os.path.exists(mean_path) and os.path.exists(std_path):
        state_mean = np.load(mean_path).astype(np.float32)
        state_std = np.load(std_path).astype(np.float32)
        if state_mean.shape[0] != state_dim or state_std.shape[0] != state_dim:
            raise ValueError(
                f"归一化维度不匹配：mean/std={state_mean.shape}/{state_std.shape} 但 state_dim={state_dim}"
            )
        print("成功加载状态归一化参数！")
        return state_mean, state_std

    print("警告：未找到归一化参数文件，将使用 mean=0,std=1（不推荐）")
    return np.zeros(state_dim, dtype=np.float32), np.ones(state_dim, dtype=np.float32)


def _safe_append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _safe_append_traj_csv(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    header_needed = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    df.to_csv(path, mode="a", header=header_needed, index=False)
    # 强制 flush
    with open(path, "a", encoding="utf-8") as f:
        f.flush()
        os.fsync(f.fileno())


def _compute_running_summary(ep_records: list):
    if len(ep_records) == 0:
        return None

    def _isfinite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    def _mean_std(key):
        xs = [float(r.get(key)) for r in ep_records if _isfinite(r.get(key))]
        if len(xs) == 0:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs))

    n = len(ep_records)
    succ = sum(1 for r in ep_records if str(r.get("Result", "")).startswith("SUCCESS"))
    usable = sum(1 for r in ep_records if (str(r.get("Result", "")).startswith("SUCCESS")
                                          or str(r.get("Result", "")).startswith("FAILED")))
    sr_over_usable = 100.0 * succ / max(usable, 1)
    sr_over_all = 100.0 * succ / max(n, 1)

    m_time, s_time = _mean_std("TimeSec")
    m_steps, s_steps = _mean_std("Steps")
    m_herr, s_herr = _mean_std("HorizErr_stable")
    m_init3d, s_init3d = _mean_std("InitDist3D")
    m_tpm, s_tpm = _mean_std("TimePerMeter3D")

    return {
        "EpisodesFinished": n,
        "UsableEpisodes(S/F)": usable,
        "SuccessRate_over_usable(%)": sr_over_usable,
        "SuccessRate_over_all(%)": sr_over_all,
        "MeanTimeSec": m_time, "StdTimeSec": s_time,
        "MeanSteps": m_steps, "StdSteps": s_steps,
        "MeanHorizErrStable": m_herr, "StdHorizErrStable": s_herr,
        "MeanInitDist3D": m_init3d, "StdInitDist3D": s_init3d,
        "MeanTimePerMeter3D": m_tpm, "StdTimePerMeter3D": s_tpm,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def main():
    print("=== 初始化环境与智能体 ===")
    env = GazeboEnv("/home/shiro/PX4_Firmware/launch/sandisland.launch", "iris", '0')
    
    td3_args = _filter_args_for_td3_offline(args)
    agent = TD3(args.state_dim, args.action_dim, args.max_action, args.capacity, td3_args)

    state_mean, state_std = _load_norm_stats(args.ckpt_dir, args.state_dim)

    def normalize(raw_state):
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        return (raw_state - state_mean) / (state_std + 1e-6)

    # 加载模型
    try:
        agent.load(args.ckpt_dir, step=args.load_step)
        print(f"成功加载模型 Step: {args.load_step} | ckpt_dir={args.ckpt_dir}")
    except TypeError:
        agent.load(args.load_step)
        print(f"成功加载模型 Step: {args.load_step}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("\n=== 开始评估（每回合立刻落盘）===")
    print(f"episode jsonl -> {EP_JSONL_PATH}")
    print(f"traj csv      -> {TRAJ_CSV_PATH}")
    if args.write_running_summary:
        print(f"running summary -> {RUN_SUM_PATH}")
    print(f"判定: herr<{args.success_herr_thresh} & height<{args.success_height_thresh}")
    print("效率指标: 使用 InitDist3D 做归一化 (TimePerMeter3D / StepsPerMeter3D / ErrRatioStable3D)")

    EPS = 1e-8
    reset_fail_streak = 0

    # running summary：resume 时读取历史
    finished_records = []
    if args.write_running_summary and os.path.exists(EP_JSONL_PATH) and os.path.getsize(EP_JSONL_PATH) > 0:
        with open(EP_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    finished_records.append(json.loads(line))
                except Exception:
                    pass

    for episode_id in range(START_EPISODE, args.test_episodes + 1):
        if args.resume and episode_id in DONE_EPISODES:
            continue

        # ---------- reset：强保护 ----------
        try:
            obs = env.reset()
            reset_fail_streak = 0
        except Exception as e:
            reset_fail_streak += 1
            print(f"[ResetError] Episode {episode_id}: reset failed: {e} (streak={reset_fail_streak})")

            fail_record = {
                "Episode": episode_id,
                "Steps": 0,
                "TimeSec": 0.0,
                "InitX": None, "InitY": None, "InitZ": None, "InitDist3D": None,
                "FinalX_stable": None, "FinalY_stable": None, "FinalZ_stable": None,
                "HorizErr_stable": None, "FinalHeightAbs": None,
                "Smoothness": None,
                "TimePerMeter3D": None, "StepsPerMeter3D": None, "ErrRatioStable3D": None,
                "Result": "RESET_FAILED",
                "ErrorMsg": str(e),
                "CkptDir": args.ckpt_dir,
                "LoadStep": args.load_step,
            }
            _safe_append_jsonl(EP_JSONL_PATH, fail_record)
            finished_records.append(fail_record)

            if args.write_running_summary:
                summ = _compute_running_summary(finished_records)
                if summ is not None:
                    pd.DataFrame([summ]).to_csv(RUN_SUM_PATH, index=False)

            if reset_fail_streak >= int(args.max_reset_fail):
                print(f"[Abort] reset 连续失败 {reset_fail_streak} 次，终止评估。")
                break
            continue

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        norm_obs = normalize(obs)
        done = False

        # ---------- init dist 3D ----------
        init_x, init_y, init_z = float(obs[0]), float(obs[1]), float(obs[2])
        init_dist3d = float(np.sqrt(init_x ** 2 + init_y ** 2 + init_z ** 2))

        # ---------- LSTM 冷启动 ----------
        state_queue = deque(maxlen=args.seq_len)
        for _ in range(args.seq_len):
            state_queue.append(norm_obs)

        episode_actions = []
        step_count = 0
        traj_rows = []

        crashed = False
        crash_msg = ""

        # ---------- rollout：强保护 ----------
        try:
            while not done:
                seq = np.asarray(state_queue, dtype=np.float32)      # (T,D)
                seq_batch = seq[np.newaxis, :, :]                    # (1,T,D)

                out = agent.choose_action(seq_batch, noise=0.0)
                action = out[0] if isinstance(out, tuple) else out
                action = np.asarray(action, dtype=np.float32).reshape(-1)

                next_obs, done, _, _ = env.step(action)

                traj_rows.append({
                    "episode": episode_id,
                    "step": step_count,
                    "time_sec": step_count * args.dt,
                    # 注意：obs[0:3] 是你的观测（通常是误差/相对位置），字段名沿用母板 pos_x/y/z
                    "pos_x": float(obs[0]),
                    "pos_y": float(obs[1]),
                    "pos_z": float(obs[2]),
                    "action_x": float(action[0]),
                    "action_y": float(action[1]),
                    "action_z": float(action[2]),
                })
                episode_actions.append(action.tolist())

                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                norm_obs = normalize(obs)
                state_queue.append(norm_obs)

                step_count += 1
                if step_count > int(args.max_steps):
                    done = True

        except Exception as e:
            crashed = True
            crash_msg = str(e)
            print(f"[StepError] Episode {episode_id}: exception during rollout: {e}")

        # ---------- settle + stable real state：强保护 ----------
        final_x = final_y = final_z = float("nan")
        horiz_err = final_height_abs = float("nan")

        try:
            try:
                env.unpause()
            except Exception:
                pass
            time.sleep(float(args.settle_seconds))
            try:
                env.pause()
            except Exception:
                pass

            final_obs = np.asarray(env.get_real_state(), dtype=np.float32).reshape(-1)
            final_x, final_y, final_z = float(final_obs[0]), float(final_obs[1]), float(final_obs[2])
            horiz_err = float(np.sqrt(final_x ** 2 + final_y ** 2))
            final_height_abs = float(abs(final_z))

        except Exception as e:
            crashed = True
            crash_msg = (crash_msg + " | " if crash_msg else "") + f"get_real_state/settle failed: {e}"
            print(f"[FinalStateError] Episode {episode_id}: {e}")

        flight_time = float(step_count * args.dt)
        smoothness = calculate_smoothness(episode_actions)

        time_per_meter_3d = flight_time / (init_dist3d + EPS)
        steps_per_meter_3d = float(step_count) / (init_dist3d + EPS)
        err_ratio_stable_3d = (float(horiz_err) / (init_dist3d + EPS)) if np.isfinite(horiz_err) else float("nan")

        if crashed:
            result_str = "CRASHED"
        else:
            if (np.isfinite(horiz_err) and np.isfinite(final_height_abs) and
                    horiz_err < args.success_herr_thresh and final_height_abs < args.success_height_thresh):
                result_str = "SUCCESS"
            else:
                result_str = "FAILED"
                if np.isfinite(final_height_abs) and final_height_abs >= args.success_height_thresh:
                    result_str += " (未落地)"
                elif np.isfinite(horiz_err) and horiz_err >= args.success_herr_thresh:
                    result_str += " (偏离)"

        # ---------- 每回合立刻写盘 ----------
        try:
            _safe_append_traj_csv(TRAJ_CSV_PATH, traj_rows)
        except Exception as e:
            print(f"[WriteTrajError] Episode {episode_id}: {e}")

        ep_record = {
            "Episode": episode_id,
            "Steps": int(step_count),
            "TimeSec": float(flight_time),

            "InitX": init_x,
            "InitY": init_y,
            "InitZ": init_z,
            "InitDist3D": float(init_dist3d),

            "FinalX_stable": float(final_x) if np.isfinite(final_x) else None,
            "FinalY_stable": float(final_y) if np.isfinite(final_y) else None,
            "FinalZ_stable": float(final_z) if np.isfinite(final_z) else None,

            "HorizErr_stable": float(horiz_err) if np.isfinite(horiz_err) else None,
            "FinalHeightAbs": float(final_height_abs) if np.isfinite(final_height_abs) else None,

            "Smoothness": float(smoothness) if np.isfinite(smoothness) else None,

            "TimePerMeter3D": float(time_per_meter_3d) if np.isfinite(time_per_meter_3d) else None,
            "StepsPerMeter3D": float(steps_per_meter_3d) if np.isfinite(steps_per_meter_3d) else None,
            "ErrRatioStable3D": float(err_ratio_stable_3d) if np.isfinite(err_ratio_stable_3d) else None,

            "Result": result_str,
            "ErrorMsg": crash_msg,

            "CkptDir": args.ckpt_dir,
            "LoadStep": args.load_step,
            "Dt": args.dt,
            "MaxSteps": int(args.max_steps),
            "SettleSeconds": float(args.settle_seconds),
            "SuccessHerrThresh": float(args.success_herr_thresh),
            "SuccessHeightThresh": float(args.success_height_thresh),
        }

        _safe_append_jsonl(EP_JSONL_PATH, ep_record)
        finished_records.append(ep_record)

        if args.write_running_summary:
            summ = _compute_running_summary(finished_records)
            if summ is not None:
                pd.DataFrame([summ]).to_csv(RUN_SUM_PATH, index=False)

        print(
            f"Episode {episode_id}/{args.test_episodes} | {result_str} | "
            f"steps={step_count} time={flight_time:.1f}s | "
            f"stable_xy=({final_x:.3f},{final_y:.3f}) herr={horiz_err:.3f} | "
            f"init3d={init_dist3d:.2f} tpm3d={time_per_meter_3d:.2f}"
        )

    print("\n[Done] 评估结束（已完成回合均已落盘）")
    print(f"Episode JSONL: {EP_JSONL_PATH}")
    print(f"Traj CSV     : {TRAJ_CSV_PATH}")
    if args.write_running_summary:
        print(f"RunningSummary: {RUN_SUM_PATH}")


if __name__ == '__main__':
    main()
