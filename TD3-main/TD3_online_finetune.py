#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途:
- 在离线 TD3-BC(LSTM+Attention) 权重基础上, 继续进行仿真在线微调训练。

用法:
- python TD3_online_finetune.py --ckpt_dir <offline_ckpt_dir> --load_step 80000
- 可选: --online_ckpt_dir <save_dir> --offline_data_path <expert.json/jsonl> --env_module landing_env_listen

实现方式:
- 读取离线阶段保存的 state_mean/state_std 做状态归一化。
- 加载离线权重后与 GazeboEnv 交互采样, 将在线样本写入 SequenceReplayBuffer(is_expert=False)。
- 支持可选的离线数据预填充, 实现 offline+online 混合训练。
- 每步按配置执行多次梯度更新, 并定期保存 checkpoint 与 TensorBoard 日志。

依赖关系:
- 依赖 TD3_offline.py 提供 OfflineTD3BCLSTM、Args、数据加载与归一化工具。
- 依赖 landing_env.py 或 landing_env_listen.py 提供 GazeboEnv。
"""

import os
import argparse
import importlib
from collections import deque
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from TD3_offline import (
    Args,
    OfflineTD3BCLSTM,
    read_offline_episodes,
    normalize_episodes,
    fill_buffer_from_episodes,
)


def parse_args():
    p = argparse.ArgumentParser("TD3 online finetune after offline pretraining")

    p.add_argument("--ckpt_dir",
                   type=str,
                   default="./checkpoints/TD3/LSTM",
                   help="offline checkpoint dir containing actor/critic and state_mean/std")
    p.add_argument("--load_step", type=int,
                   default=60000,
                   help="offline checkpoint step to load")
    p.add_argument("--online_ckpt_dir", type=str, default="", help="save dir for online finetuned checkpoints")

    p.add_argument("--env_module", type=str, default="landing_env", choices=["landing_env", "landing_env_listen"], help="which env module to import GazeboEnv from")
    p.add_argument("--launchfile", type=str, default="/home/wantengyuan/PX4_Firmware/launch/sandisland.launch")
    p.add_argument("--vehicle_type", type=str, default="iris")
    p.add_argument("--vehicle_id", type=str, default="0")

    p.add_argument("--state_dim", type=int, default=3)
    p.add_argument("--action_dim", type=int, default=3)
    p.add_argument("--max_action", type=float, default=1.0)
    p.add_argument("--seq_len", type=int, default=8)

    p.add_argument("--online_episodes", type=int, default=300)
    p.add_argument("--max_steps_per_episode", type=int, default=400)
    p.add_argument("--random_steps", type=int, default=1000, help="pure random action steps at beginning")
    p.add_argument("--warmup_updates_after", type=int, default=200, help="minimum buffer size before updates")
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--exploration_noise", type=float, default=0.15)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--capacity", type=int, default=200000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--state_noise_std", type=float, default=0.0)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--attn_hidden_dim", type=int, default=64)
    p.add_argument("--dropout_p", type=float, default=0.1)

    p.add_argument("--use_expert_only_bc", action="store_true", default=True)
    p.add_argument("--bc_weight_init", type=float, default=0.3, help="online stage initial BC weight")
    p.add_argument("--bc_weight_final", type=float, default=0.0, help="online stage final BC weight")
    p.add_argument("--bc_anneal_steps", type=int, default=50000)
    p.add_argument("--use_td3bc_adaptive_lambda", action="store_true", default=True)
    p.add_argument("--td3bc_alpha", type=float, default=2.5)

    p.add_argument("--offline_data_path", type=str, default="", help="optional: preload expert data into replay buffer")
    p.add_argument("--max_offline_episodes", type=int, default=0, help="optional limit when preloading expert data, 0 means all")

    p.add_argument("--save_every_steps", type=int, default=10000)
    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--set_train_step_to_load_step", action="store_true", default=True)

    return p.parse_args()


def pad_left_to_len(seq_list: List[np.ndarray], target_len: int) -> np.ndarray:
    if len(seq_list) == 0:
        raise ValueError("seq_list is empty")
    if len(seq_list) >= target_len:
        return np.stack(seq_list[-target_len:], axis=0)
    first = seq_list[0]
    pads = [first.copy() for _ in range(target_len - len(seq_list))]
    return np.stack(pads + seq_list, axis=0)


def build_cfg(args) -> Args:
    return Args(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        max_action=args.max_action,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        training_steps=1,
        grad_clip=args.grad_clip,
        state_noise_std=args.state_noise_std,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        attn_hidden_dim=args.attn_hidden_dim,
        dropout_p=args.dropout_p,
        use_expert_only_bc=args.use_expert_only_bc,
        bc_weight_init=args.bc_weight_init,
        bc_weight_final=args.bc_weight_final,
        bc_anneal_steps=args.bc_anneal_steps,
        use_td3bc_adaptive_lambda=args.use_td3bc_adaptive_lambda,
        td3bc_alpha=args.td3bc_alpha,
        capacity=args.capacity,
        data_path=args.offline_data_path,
        ckpt_dir=args.online_ckpt_dir,
        save_every=args.save_every_steps,
        log_every=args.log_every_steps,
        seed=args.seed,
    )


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if args.online_ckpt_dir.strip() == "":
        args.online_ckpt_dir = os.path.join(args.ckpt_dir, "online_finetune")
    else:
        args.online_ckpt_dir = os.path.abspath(args.online_ckpt_dir)
    os.makedirs(args.online_ckpt_dir, exist_ok=True)

    mean_path = os.path.join(args.ckpt_dir, "state_mean.npy")
    std_path = os.path.join(args.ckpt_dir, "state_std.npy")
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        raise FileNotFoundError(f"Missing normalization stats in {args.ckpt_dir}")

    state_mean = np.load(mean_path).astype(np.float32)
    state_std = np.load(std_path).astype(np.float32)
    if state_mean.shape[0] != args.state_dim:
        raise RuntimeError(f"state_dim mismatch: mean has {state_mean.shape[0]}, args has {args.state_dim}")

    np.save(os.path.join(args.online_ckpt_dir, "state_mean.npy"), state_mean)
    np.save(os.path.join(args.online_ckpt_dir, "state_std.npy"), state_std)

    def normalize(obs):
        x = np.asarray(obs, dtype=np.float32)
        return (x - state_mean) / (state_std + 1e-6)

    cfg = build_cfg(args)
    agent = OfflineTD3BCLSTM(cfg)
    agent.load(args.ckpt_dir, step=args.load_step)
    if args.set_train_step_to_load_step:
        agent.train_step = int(args.load_step)

    print(f"[Model] loaded offline checkpoint step={args.load_step} from {args.ckpt_dir}")

    if args.offline_data_path.strip() != "":
        episodes = read_offline_episodes(args.offline_data_path)
        if args.max_offline_episodes > 0:
            episodes = episodes[: args.max_offline_episodes]
        episodes_norm = normalize_episodes(episodes, state_mean, state_std)
        added = fill_buffer_from_episodes(agent, episodes_norm)
        print(f"[Buffer] preload expert sequences: +{added}, size={len(agent.buffer)}")

    env_mod = importlib.import_module(args.env_module)
    GazeboEnv = getattr(env_mod, "GazeboEnv")
    env = GazeboEnv(args.launchfile, args.vehicle_type, args.vehicle_id)

    writer = SummaryWriter(log_dir=os.path.join(args.online_ckpt_dir, "runs"))

    global_env_steps = 0
    global_updates = 0

    try:
        for ep in range(1, args.online_episodes + 1):
            obs = env.reset()
            done = False
            success = False
            ep_return = 0.0
            ep_steps = 0

            state_hist = [normalize(obs)]
            action_hist = []
            reward_hist = []
            done_hist = []

            while not done and ep_steps < args.max_steps_per_episode:
                if global_env_steps < args.random_steps:
                    action = np.random.uniform(-args.max_action, args.max_action, size=(args.action_dim,)).astype(np.float32)
                else:
                    s_seq = pad_left_to_len(state_hist, args.seq_len)
                    action, _ = agent.choose_action(s_seq, hidden_state=None, noise=args.exploration_noise)
                    action = np.clip(action, -args.max_action, args.max_action).astype(np.float32)

                next_obs, done, success, _ = env.step(action)
                reward = float(env.reward_setup(obs, next_obs, done, success))

                next_state_norm = normalize(next_obs)

                action_hist.append(action)
                reward_hist.append(reward)
                done_hist.append(float(done))
                state_hist.append(next_state_norm)

                t = len(action_hist) - 1
                if t >= args.seq_len - 1:
                    s_seq = np.asarray(state_hist[t - args.seq_len + 1 : t + 1], dtype=np.float32)
                    a_seq = np.asarray(action_hist[t - args.seq_len + 1 : t + 1], dtype=np.float32)
                    s2_seq = np.asarray(state_hist[t - args.seq_len + 2 : t + 2], dtype=np.float32)
                    agent.buffer.add(
                        s_seq=s_seq,
                        a_seq=a_seq,
                        r_last=reward_hist[t],
                        s2_seq=s2_seq,
                        done_last=done_hist[t],
                        is_expert=False,
                    )

                if len(agent.buffer) >= max(args.batch_size, args.warmup_updates_after):
                    for _ in range(args.updates_per_step):
                        info = agent.train_one_step()
                        global_updates += 1

                        if global_updates % args.log_every_steps == 0:
                            writer.add_scalar("online/loss_critic", float(info["critic_loss"]), global_updates)
                            writer.add_scalar("online/loss_actor", float(info["actor_loss"]), global_updates)
                            writer.add_scalar("online/loss_bc", float(info["bc_loss"]), global_updates)
                            writer.add_scalar("online/loss_td3", float(info["td3_loss"]), global_updates)
                            writer.add_scalar("online/lambda", float(info["lambda"]), global_updates)
                            writer.add_scalar("online/bc_weight", float(info["bc_weight"]), global_updates)
                            writer.add_scalar("online/actor_updated", float(info["actor_updated"]), global_updates)

                ep_return += reward
                ep_steps += 1
                global_env_steps += 1
                obs = next_obs

                if global_env_steps % args.save_every_steps == 0 and global_env_steps > 0:
                    agent.save(args.online_ckpt_dir, step=agent.train_step)
                    print(f"[Save] env_steps={global_env_steps}, train_step={agent.train_step}, dir={args.online_ckpt_dir}")

            writer.add_scalar("online/episode_return", ep_return, ep)
            writer.add_scalar("online/episode_steps", ep_steps, ep)
            writer.add_scalar("online/buffer_size", len(agent.buffer), ep)
            writer.add_scalar("online/success", 1.0 if success else 0.0, ep)

            print(
                f"[Episode {ep:04d}] steps={ep_steps:4d} return={ep_return:9.3f} "
                f"success={int(success)} buffer={len(agent.buffer)} train_step={agent.train_step}"
            )

        agent.save(args.online_ckpt_dir, step=agent.train_step)
        print(f"[Done] final checkpoint saved at step={agent.train_step} -> {args.online_ckpt_dir}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
