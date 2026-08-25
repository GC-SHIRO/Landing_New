#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TD3-BC 离线训练入口。"""

import argparse
import os

import numpy as np

from model.td3_offline import (
    Args,
    OfflineTD3BCLSTM,
    compute_mean_std,
    fill_buffer_from_episodes,
    normalize_episodes,
    read_offline_episodes,
    set_seed,
)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="TD3-BC 离线训练")

    parser.add_argument("--data_path", type=str, default=Args.data_path)
    parser.add_argument("--ckpt_dir", type=str, default=Args.ckpt_dir)

    parser.add_argument("--training_steps", type=int, default=Args.training_steps)
    parser.add_argument("--batch_size", type=int, default=Args.batch_size)
    parser.add_argument("--capacity", type=int, default=Args.capacity)
    parser.add_argument("--seq_len", type=int, default=Args.seq_len)

    parser.add_argument("--state_dim", type=int, default=Args.state_dim)
    parser.add_argument("--action_dim", type=int, default=Args.action_dim)
    parser.add_argument("--max_action", type=float, default=Args.max_action)

    parser.add_argument("--gamma", type=float, default=Args.gamma)
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument("--policy_delay", type=int, default=Args.policy_delay)
    parser.add_argument("--policy_noise", type=float, default=Args.policy_noise)
    parser.add_argument("--noise_clip", type=float, default=Args.noise_clip)

    parser.add_argument("--lr_actor", type=float, default=Args.lr_actor)
    parser.add_argument("--lr_critic", type=float, default=Args.lr_critic)
    parser.add_argument("--weight_decay", type=float, default=Args.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=Args.grad_clip)
    parser.add_argument("--state_noise_std", type=float, default=Args.state_noise_std)

    parser.add_argument("--hidden_dim", type=int, default=Args.hidden_dim)
    parser.add_argument("--attn_hidden_dim", type=int, default=Args.attn_hidden_dim)
    parser.add_argument("--dropout_p", type=float, default=Args.dropout_p)

    parser.add_argument(
        "--use_expert_only_bc",
        action="store_true",
        default=Args.use_expert_only_bc,
    )
    parser.add_argument("--bc_weight_init", type=float, default=Args.bc_weight_init)
    parser.add_argument("--bc_weight_final", type=float, default=Args.bc_weight_final)
    parser.add_argument("--bc_anneal_steps", type=int, default=Args.bc_anneal_steps)
    parser.add_argument(
        "--use_td3bc_adaptive_lambda",
        action="store_true",
        default=Args.use_td3bc_adaptive_lambda,
    )
    parser.add_argument("--td3bc_alpha", type=float, default=Args.td3bc_alpha)

    parser.add_argument("--save_every", type=int, default=Args.save_every)
    parser.add_argument("--log_every", type=int, default=Args.log_every)
    parser.add_argument("--seed", type=int, default=Args.seed)

    return Args(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()

    # TensorBoard 只在实际训练时加载，参数帮助不依赖日志组件。
    from torch.utils.tensorboard import SummaryWriter

    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    log_dir = os.path.join(args.ckpt_dir, "runs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    try:
        episodes = read_offline_episodes(args.data_path)
        if not episodes:
            raise RuntimeError(f"没有从数据文件中读取到 episode: {args.data_path}")

        mean, std = compute_mean_std(episodes, args.state_dim)
        np.save(os.path.join(args.ckpt_dir, "state_mean.npy"), mean)
        np.save(os.path.join(args.ckpt_dir, "state_std.npy"), std)

        norm_episodes = normalize_episodes(episodes, mean, std)
        agent = OfflineTD3BCLSTM(args)
        added = fill_buffer_from_episodes(agent, norm_episodes)
        print(f"已读取 {len(norm_episodes)} 个 episode，构建 {added} 条序列")

        if len(agent.buffer) < max(10, args.batch_size):
            raise RuntimeError(
                f"序列回放池过小: {len(agent.buffer)}，"
                f"至少需要 batch_size={args.batch_size}"
            )

        print("===== 开始离线训练 =====")
        for index in range(args.training_steps):
            info = agent.train_one_step()
            step = int(info.get("step", agent.train_step))

            if (index + 1) % args.log_every == 0:
                writer.add_scalar("loss/critic", float(info["critic_loss"]), step)
                writer.add_scalar("loss/actor", float(info["actor_loss"]), step)
                writer.add_scalar("loss/bc", float(info["bc_loss"]), step)
                writer.add_scalar("loss/td3", float(info["td3_loss"]), step)
                writer.add_scalar("misc/lambda", float(info["lambda"]), step)
                writer.add_scalar("misc/bc_weight", float(info["bc_weight"]), step)
                writer.add_scalar(
                    "misc/actor_updated", float(info["actor_updated"]), step
                )
                print(
                    f"Step {index + 1}/{args.training_steps} | "
                    f"Critic {info['critic_loss']:.4f} | "
                    f"Actor {info['actor_loss']:.4f} | "
                    f"BC {info['bc_loss']:.4f} | "
                    f"TD3 {info['td3_loss']:.4f}"
                )
                writer.flush()

            if (index + 1) % args.save_every == 0:
                agent.save(args.ckpt_dir, step=agent.train_step)
                print(f"已保存 checkpoint: step={agent.train_step}")

        agent.save(args.ckpt_dir, step=agent.train_step)
        print(f"训练完成，最终 checkpoint step={agent.train_step}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
