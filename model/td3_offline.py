#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 离线训练模型封装

用途:
- 使用离线专家数据训练 TD3-BC(LSTM+Attention) 策略, 产出可直接评估/在线微调的权重。

用法:
- 训练入口: python -m scripts.train_offline
- 作为模型模块: from model.td3_offline import OfflineTD3BCLSTM

实现方式:
- 网络: LSTM Actor/Critic + 特征注意力。
- 学习: TD3 双 Q + 延迟策略更新 + TD3-BC 正则。
- 数据: 支持 JSON/JSONL 多格式解析, 自动按 done 切分 episode, 构建序列回放池。
- 归一化: 计算并保存 state_mean.npy / state_std.npy。

依赖关系:
- 被训练、在线微调和仿真评估入口通过 TD3/OfflineTD3BCLSTM 加载模型。
- 不直接依赖 ROS，仅依赖 NumPy 和 PyTorch。
"""

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Args / Hyperparams
# ============================================================
@dataclass
class Args:
    # environment dims
    state_dim: int = 10
    action_dim: int = 3
    max_action: float = 1.0

    # training
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    training_steps: int = 100000
    grad_clip: float = 1.0
    state_noise_std: float = 0.0

    # sequence
    seq_len: int = 8

    # network
    hidden_dim: int = 256
    attn_hidden_dim: int = 64
    dropout_p: float = 0.1

    # offline regularization (TD3-BC style)
    use_expert_only_bc: bool = True
    bc_weight_init: float = 1.0
    bc_weight_final: float = 0.2
    bc_anneal_steps: int = 150_000
    use_td3bc_adaptive_lambda: bool = True
    td3bc_alpha: float = 2.5

    # buffer
    capacity: int = 200_000

    # io
    data_path: str = str(PROJECT_ROOT / "data" / "expert_global" / "global_expert.jsonl")
    ckpt_dir: str = str(PROJECT_ROOT / "checkpoints" / "TD3" / "global_expert")
    save_every: int = 10000
    log_every: int = 61
    seed: int = 1


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_anneal(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    t = min(max(step / float(duration), 0.0), 1.0)
    return float(start + t * (end - start))


def _to_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _split_steps_by_done(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Some datasets store ALL steps in one big list, using done=True to delimit episodes.
    This splits it into episodes. If no done=True appears, it returns [steps].
    """
    episodes: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for st in steps:
        cur.append(st)
        d = st.get("done", False)
        if bool(d):
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


# ============================================================
# Robust dataset loading (JSON or JSONL)
# ============================================================
def _normalize_loaded_object_to_episodes(obj: Any) -> List[List[Dict[str, Any]]]:
    """
    Accepts:
      - Episode = list[dict]
      - Episodes = list[Episode]
      - Steps list = list[dict] with done delim (treated as many episodes)
    Returns: list[Episode]
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        # list[dict] -> either an episode, or a long steps list with done delim
        if isinstance(obj[0], dict):
            # if contains done flags, split by done (safe even if already single ep)
            return _split_steps_by_done(obj)
        # list[list[dict]] -> episodes
        if isinstance(obj[0], list):
            episodes: List[List[Dict[str, Any]]] = []
            for ep in obj:
                if not isinstance(ep, list) or (len(ep) > 0 and not isinstance(ep[0], dict)):
                    raise ValueError("JSON episodes format invalid: expected list[list[dict]].")
                # also split each ep if it's actually steps with done delim
                episodes.extend(_split_steps_by_done(ep))
            return episodes
    raise ValueError("Unsupported dataset format. Expect JSON/JSONL of episode(s) or steps-with-done-delim.")


def _read_jsonl(path: str) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {ln}: {e}") from e

            eps = _normalize_loaded_object_to_episodes(obj)
            episodes.extend(eps)
    return episodes


def _read_json(path: str) -> List[List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            # common "Extra data" for JSONL -> fallback
            return _read_jsonl(path)

    return _normalize_loaded_object_to_episodes(obj)


def read_offline_episodes(path: str) -> List[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return _read_json(path)


def compute_mean_std(episodes: List[List[Dict[str, Any]]], state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    all_obs = []
    for ep in episodes:
        for step in ep:
            obs = step.get("observation", None)
            if obs is None:
                raise KeyError("Missing key 'observation' in dataset.")
            all_obs.append(_to_np(obs))
    if len(all_obs) == 0:
        raise ValueError("No observations found in dataset.")

    all_obs = np.stack(all_obs, axis=0)  # (N, Ds)
    if all_obs.shape[1] != state_dim:
        raise ValueError(f"State dim mismatch: dataset has {all_obs.shape[1]}, expected {state_dim}")

    mean = np.mean(all_obs, axis=0)
    std = np.std(all_obs, axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_episodes(
    episodes: List[List[Dict[str, Any]]],
    mean: np.ndarray,
    std: np.ndarray,
) -> List[List[Dict[str, Any]]]:
    norm_eps: List[List[Dict[str, Any]]] = []
    for ep in episodes:
        new_ep = []
        for step in ep:
            s = (_to_np(step["observation"]) - mean) / std
            s2 = (_to_np(step["next_observation"]) - mean) / std
            a = _to_np(step["action"])
            r = float(step["reward"])
            d = 1.0 if bool(step["done"]) else 0.0
            new_ep.append(
                {
                    "observation": s,
                    "next_observation": s2,
                    "action": a,
                    "reward": r,
                    "done": d,
                }
            )
        norm_eps.append(new_ep)
    return norm_eps


# ============================================================
# Sequence Replay Buffer (Offline)
# Stores sequences: (s_seq, a_seq, r_last, s2_seq, done_last)
# plus a flag is_expert
# ============================================================
class SequenceReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seq_len: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)

        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, seq_len, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)
        self.is_expert = np.ones((capacity, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def add(
        self,
        s_seq: np.ndarray,
        a_seq: np.ndarray,
        r_last: float,
        s2_seq: np.ndarray,
        done_last: float,
        is_expert: bool = True,
    ):
        assert s_seq.shape == (self.seq_len, self.state_dim)
        assert a_seq.shape == (self.seq_len, self.action_dim)
        assert s2_seq.shape == (self.seq_len, self.state_dim)

        self.s[self.ptr] = s_seq
        self.a[self.ptr] = a_seq
        self.r[self.ptr, 0] = float(r_last)
        self.s2[self.ptr] = s2_seq
        self.d[self.ptr, 0] = float(done_last)
        self.is_expert[self.ptr, 0] = 1.0 if is_expert else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise RuntimeError("Buffer is empty.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s": self.s[idx],
            "a": self.a[idx],
            "r": self.r[idx],
            "s2": self.s2[idx],
            "d": self.d[idx],
            "is_expert": self.is_expert[idx],
        }

    def add_episode(
        self,
        states: np.ndarray,   # (N, Ds)
        actions: np.ndarray,  # (N, Da)
        rewards: np.ndarray,  # (N,)
        dones: np.ndarray,    # (N,) 0/1
        is_expert: bool = True,
    ) -> int:
        """
        Sliding-window sequences (no cross-episode contamination).
        For each t (T-1 <= t <= N-2):
          s_seq  = states[t-T+1 : t+1]
          a_seq  = actions[t-T+1 : t+1]
          r_last = rewards[t]
          s2_seq = states[t-T+2 : t+2]
          done_last = dones[t]
        Skip if any done occurred inside the window before t.
        """
        T = self.seq_len
        N = len(states)
        if N < T + 1:
            return 0

        added = 0
        for t in range(T - 1, N - 1):
            if np.any(dones[t - T + 1 : t]):
                continue

            s_seq = states[t - T + 1 : t + 1]
            a_seq = actions[t - T + 1 : t + 1]
            s2_seq = states[t - T + 2 : t + 2]

            self.add(
                s_seq=s_seq.astype(np.float32),
                a_seq=a_seq.astype(np.float32),
                r_last=float(rewards[t]),
                s2_seq=s2_seq.astype(np.float32),
                done_last=float(dones[t]),
                is_expert=is_expert,
            )
            added += 1
        return added


# ============================================================
# Feature Attention
# ============================================================
class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) or (B,D)
        if x.dim() == 2:
            w = self.net(x)
            return x * w
        if x.dim() == 3:
            B, T, D = x.shape
            w = self.net(x.reshape(B * T, D)).reshape(B, T, D)
            return x * w
        raise ValueError(f"Unexpected tensor shape: {x.shape}")


# ============================================================
# Actor / Critic with LSTM
# ============================================================
class ActorLSTM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        attn_hidden_dim: int = 64,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.max_action = float(max_action)
        self.attn = FeatureAttention(state_dim, attn_hidden_dim)
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, s_seq: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # s_seq: (B,T,Ds)
        s_seq = self.attn(s_seq)
        x = F.relu(self.ln(self.fc(s_seq)))
        x = self.dropout(x)
        x, next_hidden = self.lstm(x, hidden)
        last = x[:, -1, :]
        a = torch.tanh(self.out(last)) * self.max_action
        return a, next_hidden

class CriticLSTM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        attn_hidden_dim: int = 64,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.attn = FeatureAttention(state_dim, attn_hidden_dim)
        self.fc = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        s_seq: torch.Tensor,
        a_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # s_seq: (B,T,Ds), a_seq: (B,T,Da)
        s_seq = self.attn(s_seq)
        xu = torch.cat([s_seq, a_seq], dim=-1)
        x = F.relu(self.ln(self.fc(xu)))
        x = self.dropout(x)
        x, _ = self.lstm(x, hidden)
        last = x[:, -1, :]
        q = self.out(last)
        return q


# ============================================================
# Offline TD3-BC + LSTM + Attention + Mixed-history
# ============================================================
class OfflineTD3BCLSTM:
    def __init__(self, args: Args):
        self.args = args
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_action = float(args.max_action)
        self.seq_len = args.seq_len

        # Networks
        self.actor = ActorLSTM(
            self.state_dim, self.action_dim, self.max_action,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)
        self.actor_target = ActorLSTM(
            self.state_dim, self.action_dim, self.max_action,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)

        self.critic1 = CriticLSTM(
            self.state_dim, self.action_dim,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)
        self.critic1_target = CriticLSTM(
            self.state_dim, self.action_dim,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)

        self.critic2 = CriticLSTM(
            self.state_dim, self.action_dim,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)
        self.critic2_target = CriticLSTM(
            self.state_dim, self.action_dim,
            hidden_dim=args.hidden_dim, attn_hidden_dim=args.attn_hidden_dim, dropout_p=args.dropout_p
        ).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=args.lr_actor, weight_decay=args.weight_decay)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)

        # Buffer
        self.buffer = SequenceReplayBuffer(args.capacity, args.state_dim, args.action_dim, args.seq_len)

        self.train_step = 0

        # FIX: cache last actor metrics so logging doesn't show zeros on non-update steps
        self.last_actor_info = {
            "actor_loss": 0.0,
            "bc_loss": 0.0,
            "td3_loss": 0.0,
            "lambda": 0.0,
        }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    def soft_update(self, net: nn.Module, target: nn.Module, tau: float):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_((1.0 - tau) * tp.data + tau * p.data)

    def _apply_state_noise(self, s: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = float(self.args.state_noise_std)
        if std <= 0:
            return s, s2
        noise = torch.randn_like(s) * std
        return s + noise, s2 + noise

    @torch.no_grad()
    def choose_action(
        self,
        s_seq: np.ndarray,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        noise: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inference helper.
        s_seq: (T,Ds) or (1,T,Ds)
        hidden_state: LSTM hidden (h,c) or None
        noise: gaussian noise std added to action (then clipped)
        returns: (action_np (Da,), next_hidden)
        """
        self.actor.eval()
        if isinstance(s_seq, np.ndarray) is False:
            s_seq = np.asarray(s_seq, dtype=np.float32)

        if s_seq.ndim == 2:
            s_seq_t = torch.tensor(s_seq[None, ...], dtype=torch.float32, device=device)  # (1,T,Ds)
        elif s_seq.ndim == 3:
            s_seq_t = torch.tensor(s_seq, dtype=torch.float32, device=device)  # (B,T,Ds)
        else:
            raise ValueError(f"s_seq must be (T,Ds) or (B,T,Ds), got {s_seq.shape}")

        a_t, next_hidden = self.actor(s_seq_t, hidden_state)
        a = a_t.squeeze(0).detach().cpu().numpy()

        if noise and noise > 0:
            a = a + np.random.normal(0.0, float(noise), size=a.shape).astype(np.float32)

        a = np.clip(a, -self.max_action, self.max_action)
        self.actor.train()
        return a.astype(np.float32), next_hidden

    def train_one_step(self) -> Dict[str, float]:
        args = self.args
        batch = self.buffer.sample(args.batch_size)

        s = torch.tensor(batch["s"], dtype=torch.float32, device=device)      # (B,T,Ds)
        a = torch.tensor(batch["a"], dtype=torch.float32, device=device)      # (B,T,Da)
        r = torch.tensor(batch["r"], dtype=torch.float32, device=device)      # (B,1)
        s2 = torch.tensor(batch["s2"], dtype=torch.float32, device=device)    # (B,T,Ds)
        d = torch.tensor(batch["d"], dtype=torch.float32, device=device)      # (B,1)
        is_exp = torch.tensor(batch["is_expert"], dtype=torch.float32, device=device)  # (B,1)

        s, s2 = self._apply_state_noise(s, s2)

        # BC weight schedule should be visible EVERY step
        bc_w_now = linear_anneal(self.train_step, args.bc_weight_init, args.bc_weight_final, args.bc_anneal_steps)

        # --------------------------
        # Critic update
        # --------------------------
        self.critic1.train()
        self.critic2.train()

        with torch.no_grad():
            a2_last, _ = self.actor_target(s2)  # (B,Da)
            noise = (torch.randn_like(a2_last) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
            a2_last = (a2_last + noise).clamp(-self.max_action, self.max_action)

            # target action sequence: history from dataset actions + last from target policy
            a2_seq = torch.cat([a[:, 1:, :], a2_last.unsqueeze(1)], dim=1)  # (B,T,Da)

            q1_t = self.critic1_target(s2, a2_seq)
            q2_t = self.critic2_target(s2, a2_seq)
            q_t = torch.min(q1_t, q2_t)
            y = r + (1.0 - d) * args.gamma * q_t

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic1_opt.zero_grad(set_to_none=True)
        self.critic2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), args.grad_clip)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), args.grad_clip)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # default: report last actor stats (FIXED LOGGING)
        info: Dict[str, float] = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(self.last_actor_info["actor_loss"]),
            "bc_loss": float(self.last_actor_info["bc_loss"]),
            "td3_loss": float(self.last_actor_info["td3_loss"]),
            "lambda": float(self.last_actor_info["lambda"]),
            "bc_weight": float(bc_w_now),
            "actor_updated": 0.0,
            "step": float(self.train_step),
        }

        # --------------------------
        # Actor update (delayed)
        # --------------------------
        if self.train_step % args.policy_delay == 0:
            self.actor.train()

            # actor action for last step
            pi_last, _ = self.actor(s)  # (B,Da)

            # Mixed-history: first T-1 from dataset, last from actor
            a_mix = a.clone()
            a_mix[:, -1, :] = pi_last

            q_pi = self.critic1(s, a_mix)
            td3_loss = -q_pi.mean()

            # BC on LAST step
            a_last = a[:, -1, :]
            bc_loss_raw = F.mse_loss(pi_last, a_last, reduction="none").mean(dim=1, keepdim=True)  # (B,1)
            if args.use_expert_only_bc:
                bc_loss = (bc_loss_raw * is_exp).sum() / (is_exp.sum().clamp(min=1.0))
            else:
                bc_loss = bc_loss_raw.mean()

            # TD3-BC adaptive lambda
            if args.use_td3bc_adaptive_lambda:
                with torch.no_grad():
                    q_abs_mean = q_pi.abs().mean().clamp(min=1e-3)
                lam = float(args.td3bc_alpha / q_abs_mean.item())
            else:
                lam = 1.0

            actor_loss = td3_loss + (bc_w_now * lam) * bc_loss

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), args.grad_clip)
            self.actor_opt.step()

            # Soft update
            self.soft_update(self.actor, self.actor_target, args.tau)
            self.soft_update(self.critic1, self.critic1_target, args.tau)
            self.soft_update(self.critic2, self.critic2_target, args.tau)

            # update cache
            self.last_actor_info.update(
                {
                    "actor_loss": float(actor_loss.item()),
                    "bc_loss": float(bc_loss.item()),
                    "td3_loss": float(td3_loss.item()),
                    "lambda": float(lam),
                }
            )

            info.update(
                {
                    "actor_loss": float(actor_loss.item()),
                    "bc_loss": float(bc_loss.item()),
                    "td3_loss": float(td3_loss.item()),
                    "lambda": float(lam),
                    "actor_updated": 1.0,
                }
            )

        self.train_step += 1
        return info

    def save(self, ckpt_dir: str, step: int):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(ckpt_dir, f"actor_{step}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(ckpt_dir, f"critic1_{step}.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(ckpt_dir, f"critic2_{step}.pth"))

    def load(self, ckpt_dir: str, step: int):
        self.actor.load_state_dict(torch.load(os.path.join(ckpt_dir, f"actor_{step}.pth"), map_location=device))
        self.critic1.load_state_dict(torch.load(os.path.join(ckpt_dir, f"critic1_{step}.pth"), map_location=device))
        self.critic2.load_state_dict(torch.load(os.path.join(ckpt_dir, f"critic2_{step}.pth"), map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


# ============================================================
# 兼容旧脚本调用方式的 TD3 封装
# ============================================================
class TD3:
    """
    Compatibility layer for older code expecting:
      TD3(state_dim, action_dim, max_action, capacity, args)
      .choose_action(...)
      .load(step) or .load(ckpt_dir, step=...)
      .save(step) or .save(ckpt_dir, step)
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, capacity: int, args: Any):
        # args can be argparse.Namespace or Args; normalize to Args
        if isinstance(args, Args):
            cfg = args
            cfg.state_dim = state_dim
            cfg.action_dim = action_dim
            cfg.max_action = max_action
            cfg.capacity = capacity
        else:
            # build Args from namespace/dict
            d = vars(args) if hasattr(args, "__dict__") else dict(args)
            cfg = Args(**{**Args().__dict__, **d})
            cfg.state_dim = state_dim
            cfg.action_dim = action_dim
            cfg.max_action = max_action
            cfg.capacity = capacity

        self.cfg = cfg
        self.agent = OfflineTD3BCLSTM(cfg)
        self.buffer = self.agent.buffer  # optional exposure

    @property
    def train_step(self) -> int:
        return self.agent.train_step

    def train_one_step(self) -> Dict[str, float]:
        return self.agent.train_one_step()

    def choose_action(self, s_seq, hidden_state=None, noise=0.0):
        return self.agent.choose_action(s_seq, hidden_state=hidden_state, noise=noise)

    def save(self, step: int, ckpt_dir: Optional[str] = None):
        if ckpt_dir is None:
            ckpt_dir = self.cfg.ckpt_dir
        return self.agent.save(ckpt_dir, step)

    def load(self, *args, **kwargs):
        """
        Support both:
          load(step)
          load(ckpt_dir, step=step)
        """
        if len(args) == 1 and isinstance(args[0], int) and "step" not in kwargs:
            step = int(args[0])
            return self.agent.load(self.cfg.ckpt_dir, step)
        if len(args) >= 1 and isinstance(args[0], str):
            ckpt_dir = args[0]
            step = kwargs.get("step", None)
            if step is None and len(args) >= 2:
                step = args[1]
            if step is None:
                raise TypeError("load(ckpt_dir, step=...) requires step.")
            return self.agent.load(ckpt_dir, int(step))
        raise TypeError("Unsupported load signature. Use load(step) or load(ckpt_dir, step=...).")


# ============================================================
# Build buffer from normalized episodes
# ============================================================
def fill_buffer_from_episodes(agent: OfflineTD3BCLSTM, episodes: List[List[Dict[str, Any]]]) -> int:
    added_total = 0
    for ep in episodes:
        states = np.stack([_to_np(st["observation"]) for st in ep], axis=0)         # (N,Ds)
        actions = np.stack([_to_np(st["action"]) for st in ep], axis=0)             # (N,Da)
        rewards = np.asarray([float(st["reward"]) for st in ep], dtype=np.float32)  # (N,)
        dones = np.asarray([float(st["done"]) for st in ep], dtype=np.float32)      # (N,)

        if states.shape[1] != agent.state_dim:
            raise ValueError(f"Episode state_dim mismatch: {states.shape[1]} vs {agent.state_dim}")
        if actions.shape[1] != agent.action_dim:
            raise ValueError(f"Episode action_dim mismatch: {actions.shape[1]} vs {agent.action_dim}")

        added_total += agent.buffer.add_episode(states, actions, rewards, dones, is_expert=True)
    return added_total
