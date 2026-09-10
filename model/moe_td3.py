#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段约束 Causal Transformer MoE 的离线 TD3-BC 模型。

策略只读取十维视觉运动 observation；专家 phase 仅是离线 Router 监督标签，
绝不作为部署时的策略输入。
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_MAPPING_VERSION = "v1-align-track-descend-touchdown-search"
PHASE_TRANSITION_VERSION = "v2-sampling-stable-two-steps"

# 采集器已有标签到 MoE 内部阶段的唯一映射；索引是 checkpoint 契约的一部分。
PHASE_NAMES: Tuple[str, ...] = (
    "APPROACH", "MATCH", "DESCEND", "TOUCHDOWN", "SEARCH"
)
EXPERT_PHASE_TO_INDEX: Mapping[str, int] = {
    "ALIGN": 0,
    "TRACK": 1,
    "DESCEND": 2,
    "TOUCHDOWN": 3,
    "SEARCH": 4,
}

# 行表示上一阶段，列表示当前阶段；对应当前 Sampling 连续稳定两步的配置。
# 下降和近地阶段仍可能因误差变大而重新对准，近地阶段也可能随高度变化恢复下降。
_ALLOWED_TRANSITIONS = torch.tensor(
    [
        [True, True, False, False, True],   # APPROACH
        [True, True, True, True, True],      # MATCH
        [True, True, True, True, True],      # DESCEND
        [True, True, True, True, True],      # TOUCHDOWN
        [True, True, False, False, True],    # SEARCH
    ],
    dtype=torch.bool,
)


@dataclass
class Args:
    """MoE 训练参数；常用项集中于此，避免散落在训练循环。"""

    # 数据接口：必须与当前十维采集器一致。
    state_dim: int = 10
    action_dim: int = 3
    max_action: float = 1.0
    seq_len: int = 32
    validation_fraction: float = 0.2

    # Causal Transformer 与 Router。
    hidden_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 512
    dropout_p: float = 0.1
    n_phases: int = len(PHASE_NAMES)
    router_hidden_dim: int = 64
    router_temperature: float = 1.0

    # TD3。
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    training_steps: int = 100_000
    grad_clip: float = 1.0
    state_noise_std: float = 0.0

    # 阶段监督与 TD3-BC。
    bc_weight_init: float = 1.0
    bc_weight_final: float = 0.2
    bc_anneal_steps: int = 150_000
    use_td3bc_adaptive_lambda: bool = True
    td3bc_alpha: float = 2.5
    router_loss_weight: float = 1.0
    switch_loss_weight: float = 0.01

    # 存储。
    capacity: int = 200_000
    data_path: str = str(PROJECT_ROOT / "data" / "expert_global" / "global_expert.jsonl")
    ckpt_dir: str = str(PROJECT_ROOT / "checkpoints" / "MoE_TD3" / "global_expert")
    save_every: int = 10_000
    log_every: int = 61
    seed: int = 1

    def validate(self) -> None:
        if self.state_dim != 10:
            raise ValueError("MoE-TD3 当前只支持 state_dim=10 的视觉运动 observation")
        if self.action_dim != 3:
            raise ValueError("MoE-TD3 当前只支持 action_dim=3")
        if self.seq_len < 3:
            raise ValueError("seq_len 至少为 3，Router 平滑项需要最后三个 phase")
        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError("hidden_dim 必须能被 transformer_heads 整除")
        if self.n_phases != len(PHASE_NAMES):
            raise ValueError(f"n_phases 必须为 {len(PHASE_NAMES)}")
        if self.router_temperature <= 0.0 or self.max_action <= 0.0:
            raise ValueError("router_temperature 与 max_action 必须大于 0")
        if self.policy_delay <= 0:
            raise ValueError("policy_delay 必须大于 0")


def set_seed(seed: int = 0) -> None:
    """固定离线初始化和采样随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_anneal(step: int, start: float, end: float, duration: int) -> float:
    """返回限定在起止区间内的线性退火值。"""
    if duration <= 0:
        return float(end)
    ratio = min(max(step / float(duration), 0.0), 1.0)
    return float(start + ratio * (end - start))


def _to_np(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def phase_to_index(phase: Any) -> int:
    """将采集器 phase 映射为固定整数；未知标签必须在训练前失败。"""
    if isinstance(phase, (int, np.integer)):
        index = int(phase)
        if 0 <= index < len(PHASE_NAMES):
            return index
        raise ValueError(f"phase 索引超出范围: {phase}")
    name = str(phase).strip().upper()
    if name not in EXPERT_PHASE_TO_INDEX:
        raise ValueError(
            f"未知或缺失的专家 phase={phase!r}，只接受: {', '.join(EXPERT_PHASE_TO_INDEX)}"
        )
    return EXPERT_PHASE_TO_INDEX[name]


def phase_name(index: int) -> str:
    """返回 MoE 内部 phase 名称，用于日志和部署状态。"""
    if not 0 <= int(index) < len(PHASE_NAMES):
        raise ValueError(f"phase 索引超出范围: {index}")
    return PHASE_NAMES[int(index)]


def transition_is_allowed(previous_phase: int, current_phase: int) -> bool:
    """检查一个 phase 转移是否满足第一版有限状态约束。"""
    previous = phase_to_index(previous_phase)
    current = phase_to_index(current_phase)
    return bool(_ALLOWED_TRANSITIONS[previous, current].item())


def _extract_phase(step: Mapping[str, Any]) -> int:
    """兼容原始采集 metadata 与归一化后的内部 phase 字段。"""
    if "phase" in step:
        return phase_to_index(step["phase"])
    expert = step.get("expert")
    if not isinstance(expert, Mapping):
        raise ValueError("step 缺少 expert.phase，MoE Router 没有监督标签")
    return phase_to_index(expert.get("phase"))


def _split_steps_by_done(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """将按 done 拼接的 step 列表切为 episode，避免跨回合窗口。"""
    episodes: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for step in steps:
        current.append(step)
        if bool(step.get("done", False)):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def _normalize_loaded_object_to_episodes(obj: Any) -> List[List[Dict[str, Any]]]:
    """读取 JSON 或 JSONL 时统一为 ``list[episode]``。"""
    if not isinstance(obj, list):
        raise ValueError("数据格式错误：期望 episode 或 episode 列表")
    if not obj:
        return []
    if isinstance(obj[0], dict):
        return _split_steps_by_done(obj)
    if isinstance(obj[0], list):
        episodes: List[List[Dict[str, Any]]] = []
        for episode in obj:
            if not isinstance(episode, list) or any(not isinstance(step, dict) for step in episode):
                raise ValueError("数据格式错误：episode 必须是 step 字典列表")
            episodes.extend(_split_steps_by_done(episode))
        return episodes
    raise ValueError("数据格式错误：期望 episode 或 episode 列表")


def _read_jsonl(path: str) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                episodes.extend(_normalize_loaded_object_to_episodes(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 第 {line_number} 行解析失败: {error}") from error
    return episodes


def read_offline_episodes(path: str) -> List[List[Dict[str, Any]]]:
    """读取 JSON/JSONL 专家数据；phase 完整性在归一化时检查。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as input_file:
        try:
            return _normalize_loaded_object_to_episodes(json.load(input_file))
        except json.JSONDecodeError:
            return _read_jsonl(path)


def compute_mean_std(
    episodes: Sequence[Sequence[Mapping[str, Any]]], state_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """只根据 observation 计算归一化统计量，不读取特权 metadata。"""
    observations: List[np.ndarray] = []
    for episode in episodes:
        for step in episode:
            if "observation" not in step:
                raise KeyError("数据缺少 observation")
            observation = _to_np(step["observation"])
            if observation.shape != (state_dim,) or not np.isfinite(observation).all():
                raise ValueError(f"observation 不是有限的 ({state_dim},) 向量")
            observations.append(observation)
    if not observations:
        raise ValueError("数据中没有 observation")
    stacked = np.stack(observations, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32), (np.std(stacked, axis=0) + 1e-6).astype(np.float32)


def normalize_episodes(
    episodes: Sequence[Sequence[Mapping[str, Any]]], mean: np.ndarray, std: np.ndarray,
) -> List[List[Dict[str, Any]]]:
    """归一化视觉 observation，并显式保留 phase 监督标签。"""
    normalized: List[List[Dict[str, Any]]] = []
    for episode in episodes:
        normalized_episode: List[Dict[str, Any]] = []
        for step in episode:
            state = (_to_np(step["observation"]) - mean) / std
            next_state = (_to_np(step["next_observation"]) - mean) / std
            if not np.isfinite(state).all() or not np.isfinite(next_state).all():
                raise ValueError("归一化后出现非有限 observation")
            normalized_episode.append(
                {
                    "observation": state.astype(np.float32),
                    "next_observation": next_state.astype(np.float32),
                    "action": _to_np(step["action"]),
                    "reward": float(step["reward"]),
                    "done": 1.0 if bool(step["done"]) else 0.0,
                    "phase": _extract_phase(step),
                }
            )
        normalized.append(normalized_episode)
    return normalized


def _window_end_indices(dones: np.ndarray, seq_len: int) -> List[int]:
    """只取完整历史；终止帧可作窗口末尾，非终止尾帧仍需真实下一阶段。"""
    indices = []
    for index in range(seq_len - 1, len(dones)):
        if np.any(dones[index - seq_len + 1:index]):
            continue
        if index == len(dones) - 1 and not bool(dones[index]):
            continue
        indices.append(index)
    return indices


@dataclass
class PreparedOfflineData:
    """各 Stage 共用的 episode 划分、归一化结果及有效窗口阶段计数。"""

    train_episodes: List[List[Dict[str, Any]]]
    validation_episodes: List[List[Dict[str, Any]]]
    mean: np.ndarray
    std: np.ndarray
    train_indices: List[int]
    validation_indices: List[int]
    train_phase_counts: np.ndarray
    validation_phase_counts: np.ndarray
    seq_len: int

    def save(self, directory: str) -> None:
        """保存一份共用统计与划分记录，后续 Stage 不应重新拟合归一化。"""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "state_mean.npy", self.mean)
        np.save(output_dir / "state_std.npy", self.std)
        metadata = {
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "seq_len": self.seq_len,
            "phase_names": list(PHASE_NAMES),
            "train_phase_counts": self.train_phase_counts.tolist(),
            "validation_phase_counts": self.validation_phase_counts.tolist(),
        }
        (output_dir / "data_split.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def prepare_offline_data(
    episodes: Sequence[Sequence[Mapping[str, Any]]], args: Args,
) -> PreparedOfflineData:
    """按完整 episode 固定划分，仅从训练集拟合统计；阶段缺样只统计、不报错。"""
    if len(episodes) < 2:
        raise ValueError("按 episode 划分训练和验证集至少需要两个回合")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须位于 0 和 1 之间")
    if args.seq_len < 3:
        raise ValueError("seq_len 至少为 3")
    indices = np.random.default_rng(args.seed).permutation(len(episodes))
    validation_count = min(len(episodes) - 1, max(1, int(len(episodes) * args.validation_fraction)))
    validation_indices = sorted(indices[:validation_count].tolist())
    train_indices = sorted(indices[validation_count:].tolist())
    train_raw = [episodes[index] for index in train_indices]
    validation_raw = [episodes[index] for index in validation_indices]
    mean, std = compute_mean_std(train_raw, args.state_dim)
    train = normalize_episodes(train_raw, mean, std)
    validation = normalize_episodes(validation_raw, mean, std)

    def phase_counts(split: List[List[Dict[str, Any]]]) -> np.ndarray:
        counts = np.zeros(len(PHASE_NAMES), dtype=np.int64)
        for episode in split:
            dones = np.asarray([step["done"] for step in episode])
            for index in _window_end_indices(dones, args.seq_len):
                counts[episode[index]["phase"]] += 1
        return counts

    return PreparedOfflineData(
        train_episodes=train, validation_episodes=validation, mean=mean, std=std,
        train_indices=train_indices, validation_indices=validation_indices,
        train_phase_counts=phase_counts(train), validation_phase_counts=phase_counts(validation),
        seq_len=args.seq_len,
    )


class SequenceReplayBuffer:
    """保存不跨 episode 的 state/action/phase 序列窗口。"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, seq_len: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.pointer = 0
        self.size = 0
        self.s = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, seq_len, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)
        self.phase_sequence = np.zeros((capacity, seq_len), dtype=np.int64)
        self.phase_previous = np.zeros((capacity, 1), dtype=np.int64)
        self.phase = np.zeros((capacity, 1), dtype=np.int64)
        self.phase_next = np.zeros((capacity, 1), dtype=np.int64)

    def __len__(self) -> int:
        return self.size

    def add(self, state_sequence: np.ndarray, action_sequence: np.ndarray, reward: float,
            next_state_sequence: np.ndarray, done: float, phase_sequence: np.ndarray,
            phase_previous: int, phase: int, phase_next: int) -> None:
        """写入一个已完成 shape 和 phase 合法性检查的窗口。"""
        if state_sequence.shape != (self.seq_len, self.state_dim):
            raise ValueError("state_sequence shape 不匹配")
        if action_sequence.shape != (self.seq_len, self.action_dim):
            raise ValueError("action_sequence shape 不匹配")
        if next_state_sequence.shape != (self.seq_len, self.state_dim):
            raise ValueError("next_state_sequence shape 不匹配")
        if phase_sequence.shape != (self.seq_len,):
            raise ValueError("phase_sequence shape 不匹配")
        if not transition_is_allowed(phase_previous, phase):
            raise ValueError(f"非法阶段转移: {phase_name(phase_previous)} -> {phase_name(phase)}")
        if bool(done) and phase_next != -1:
            raise ValueError("终止样本的 phase_next 必须为 -1，表示不存在下一阶段")
        if not bool(done) and not transition_is_allowed(phase, phase_next):
            raise ValueError(f"非法阶段转移: {phase_name(phase)} -> {phase_name(phase_next)}")
        index = self.pointer
        self.s[index], self.a[index] = state_sequence, action_sequence
        self.r[index, 0], self.s2[index], self.d[index, 0] = float(reward), next_state_sequence, float(done)
        self.phase_sequence[index] = phase_sequence
        self.phase_previous[index, 0] = int(phase_previous)
        self.phase[index, 0] = int(phase)
        self.phase_next[index, 0] = int(phase_next)
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                    dones: np.ndarray, phases: np.ndarray, next_states: np.ndarray) -> int:
        """按真实 transition 建窗口，保留终止奖励、动作及 next_observation。"""
        total_steps = len(states)
        if total_steps < self.seq_len:
            return 0
        if not (len(actions) == len(rewards) == len(dones) == len(phases) == len(next_states) == total_steps):
            raise ValueError("episode 的 state/next_state/action/reward/done/phase 长度不一致")
        if np.any((phases < 0) | (phases >= len(PHASE_NAMES))):
            raise ValueError("episode 含非法 phase 索引")
        added = 0
        for time_index in _window_end_indices(dones, self.seq_len):
            self.add(
                states[time_index - self.seq_len + 1:time_index + 1],
                actions[time_index - self.seq_len + 1:time_index + 1],
                float(rewards[time_index]),
                next_states[time_index - self.seq_len + 1:time_index + 1],
                float(dones[time_index]),
                phases[time_index - self.seq_len + 1:time_index + 1],
                int(phases[time_index - 1]), int(phases[time_index]),
                -1 if bool(dones[time_index]) else int(phases[time_index + 1]),
            )
            added += 1
        return added

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise RuntimeError("回放池为空")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "s": self.s[indices], "a": self.a[indices], "r": self.r[indices],
            "s2": self.s2[indices], "d": self.d[indices],
            "phase_sequence": self.phase_sequence[indices],
            "phase_previous": self.phase_previous[indices],
            "phase": self.phase[indices], "phase_next": self.phase_next[indices],
        }


class CausalTransformerEncoder(nn.Module):
    """带固定位置编码和严格因果 mask 的时序编码器。"""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int, heads: int,
                 ffn_dim: int, dropout_p: float, max_seq_len: int):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("Transformer hidden_dim 必须能被 heads 整除")
        self.max_seq_len = int(max_seq_len)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=ffn_dim,
            dropout=dropout_p, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.register_buffer("position_encoding", self._position_encoding(self.max_seq_len, hidden_dim))
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1
        ), persistent=False)

    @staticmethod
    def _position_encoding(length: int, hidden_dim: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-np.log(10_000.0) / hidden_dim))
        encoding = torch.zeros(1, length, hidden_dim, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(positions * div_term)
        encoding[0, :, 1::2] = torch.cos(positions * div_term)
        return encoding

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """输入 `(B,T,D)`，输出每个仅依赖过去的 token `(B,T,H)`。"""
        if sequence.ndim != 3:
            raise ValueError(f"Transformer 输入必须是 (B,T,D)，实际为 {tuple(sequence.shape)}")
        _, time_steps, _ = sequence.shape
        if time_steps > self.max_seq_len:
            raise ValueError(f"序列长度 {time_steps} 超过配置上限 {self.max_seq_len}")
        encoded = self.input_projection(sequence)
        encoded = self.input_norm(encoded + self.position_encoding[:, :time_steps])
        encoded = self.dropout(encoded)
        encoded = self.encoder(encoded, mask=self.causal_mask[:time_steps, :time_steps])
        return self.output_norm(encoded)


class PhaseRouter(nn.Module):
    """根据当前 token 与上一阶段生成受转移表限制的路由概率。"""

    def __init__(self, hidden_dim: int, router_hidden_dim: int, temperature: float):
        super().__init__()
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(router_hidden_dim, len(PHASE_NAMES)),
        )
        self.register_buffer("allowed_transitions", _ALLOWED_TRANSITIONS.clone())

    def forward(self, hidden: torch.Tensor, previous_phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("Router hidden 必须是 (B,H)")
        previous_phase = previous_phase.long().reshape(-1)
        if hidden.shape[0] != previous_phase.shape[0]:
            raise ValueError("Router batch 与 previous_phase batch 不一致")
        if torch.any(previous_phase < 0) or torch.any(previous_phase >= len(PHASE_NAMES)):
            raise ValueError("previous_phase 包含越界索引")
        logits = self.network(hidden) / self.temperature
        allowed = self.allowed_transitions[previous_phase]
        masked_logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        return masked_logits, F.softmax(masked_logits, dim=-1)


class ExpertHeads(nn.Module):
    """五个独立动作头；共享表征但不共享阶段动作参数。"""

    def __init__(self, hidden_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = float(max_action)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, action_dim))
            for _ in PHASE_NAMES
        ])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """输出 `(B,5,3)`，每个专家动作都严格限制在动作范围内。"""
        return torch.tanh(torch.stack([head(hidden) for head in self.heads], dim=1)) * self.max_action


class MoEActor(nn.Module):
    """共享 Causal Transformer、阶段 Router 与五个 Actor 专家头。"""

    def __init__(self, args: Args):
        super().__init__()
        self.encoder = CausalTransformerEncoder(
            args.state_dim, args.hidden_dim, args.transformer_layers, args.transformer_heads,
            args.transformer_ffn_dim, args.dropout_p, args.seq_len,
        )
        self.router = PhaseRouter(args.hidden_dim, args.router_hidden_dim, args.router_temperature)
        self.experts = ExpertHeads(args.hidden_dim, args.action_dim, args.max_action)

    def forward(self, state_sequence: torch.Tensor, previous_phase: torch.Tensor,
                mode: str = "soft") -> Dict[str, torch.Tensor]:
        hidden_sequence = self.encoder(state_sequence)
        logits, weights = self.router(hidden_sequence[:, -1, :], previous_phase)
        all_actions = self.experts(hidden_sequence[:, -1, :])
        selected_phase = torch.argmax(weights, dim=-1)
        if mode == "soft":
            action = torch.sum(weights.unsqueeze(-1) * all_actions, dim=1)
        elif mode == "hard":
            action = all_actions.gather(
                1, selected_phase[:, None, None].expand(-1, 1, all_actions.shape[-1])
            ).squeeze(1)
        else:
            raise ValueError("mode 只能是 'soft' 或 'hard'")
        return {
            "action": action, "logits": logits, "weights": weights,
            "all_actions": all_actions, "hidden_sequence": hidden_sequence,
            "selected_phase": selected_phase,
        }


class CriticTransformer(nn.Module):
    """使用独立因果编码器评价 state/action 历史的单个 Q 值。"""

    def __init__(self, args: Args):
        super().__init__()
        self.encoder = CausalTransformerEncoder(
            args.state_dim + args.action_dim, args.hidden_dim, args.transformer_layers,
            args.transformer_heads, args.transformer_ffn_dim, args.dropout_p, args.seq_len,
        )
        self.value_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(inplace=True), nn.Linear(args.hidden_dim, 1)
        )

    def forward(self, state_sequence: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        if state_sequence.shape[:2] != action_sequence.shape[:2]:
            raise ValueError("Critic 的 state/action 序列时间维不一致")
        hidden = self.encoder(torch.cat([state_sequence, action_sequence], dim=-1))
        return self.value_head(hidden[:, -1, :])


class OfflineMoETD3BC:
    """包含 Router 监督、专家 BC 与 TD3 更新的离线训练器。"""

    def __init__(self, args: Args):
        args.validate()
        self.args = args
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.max_action, self.seq_len = float(args.max_action), args.seq_len
        self.actor = MoEActor(args).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic1 = CriticTransformer(args).to(device)
        self.critic1_target = copy.deepcopy(self.critic1).to(device)
        self.critic2 = CriticTransformer(args).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)
        # target 网络只生成稳定的 bootstrap 目标，不应受 dropout 随机性的影响。
        self.actor_target.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor, weight_decay=args.weight_decay)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)
        self.buffer = SequenceReplayBuffer(args.capacity, args.state_dim, args.action_dim, args.seq_len)
        self.train_step = 0
        self.last_actor_info: Dict[str, float] = {
            "actor_loss": 0.0, "bc_loss": 0.0, "td3_loss": 0.0,
            "router_loss": 0.0, "switch_loss": 0.0, "lambda": 0.0, "router_accuracy": 0.0,
        }

    @staticmethod
    def _soft_update(network: nn.Module, target: nn.Module, tau: float) -> None:
        for parameter, target_parameter in zip(network.parameters(), target.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(parameter.data, alpha=tau)

    def _apply_state_noise(self, state: torch.Tensor, next_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.state_noise_std <= 0.0:
            return state, next_state
        noise = torch.randn_like(state) * float(self.args.state_noise_std)
        return state + noise, next_state + noise

    @staticmethod
    def _replace_last_action(action_sequence: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        mixed = action_sequence.clone()
        mixed[:, -1, :] = last_action
        return mixed

    @staticmethod
    def _expert_action_for_phase(all_actions: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return all_actions.gather(
            1, phase[:, None, None].expand(-1, 1, all_actions.shape[-1])
        ).squeeze(1)

    @torch.no_grad()
    def choose_action(self, state_sequence: np.ndarray, previous_phase: int = 0,
                      noise: float = 0.0, mode: str = "hard") -> Tuple[np.ndarray, int]:
        """部署辅助函数；只需要视觉序列与上次 Router 输出的 phase。"""
        sequence = np.asarray(state_sequence, dtype=np.float32)
        if sequence.ndim == 2:
            sequence = sequence[None, ...]
        if sequence.ndim != 3 or sequence.shape[0] != 1 or sequence.shape[2] != self.state_dim:
            raise ValueError(f"state_sequence 必须是 (T,{self.state_dim}) 或 (1,T,{self.state_dim})")
        if sequence.shape[1] == 0:
            raise ValueError("state_sequence 不能为空")
        if sequence.shape[1] > self.seq_len:
            sequence = sequence[:, -self.seq_len:, :]
        elif sequence.shape[1] < self.seq_len:
            padding = np.repeat(sequence[:, :1, :], self.seq_len - sequence.shape[1], axis=1)
            sequence = np.concatenate([padding, sequence], axis=1)
        previous = torch.tensor([phase_to_index(previous_phase)], dtype=torch.long, device=device)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
        was_training = self.actor.training
        self.actor.eval()
        output = self.actor(sequence_tensor, previous, mode=mode)
        action = output["action"].squeeze(0).cpu().numpy()
        selected_phase = int(output["selected_phase"].item())
        if noise > 0.0:
            action += np.random.normal(0.0, float(noise), size=action.shape).astype(np.float32)
        self.actor.train(was_training)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32), selected_phase

    @torch.no_grad()
    def _compute_target_values(self, next_state: torch.Tensor, action: torch.Tensor,
                               reward: torch.Tensor, done: torch.Tensor,
                               phase: torch.Tensor) -> torch.Tensor:
        """终止样本只使用即时奖励，且完全不调用目标网络计算未来价值。"""
        target_value = reward.clone()
        continuing = done.squeeze(1) == 0
        if continuing.any():
            target_output = self.actor_target(next_state[continuing], phase[continuing], mode="soft")
            target_last_action = target_output["action"]
            target_noise = (torch.randn_like(target_last_action) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            target_last_action = (target_last_action + target_noise).clamp(-self.max_action, self.max_action)
            target_action_sequence = torch.cat(
                [action[continuing, 1:, :], target_last_action.unsqueeze(1)], dim=1
            )
            target_q = torch.minimum(
                self.critic1_target(next_state[continuing], target_action_sequence),
                self.critic2_target(next_state[continuing], target_action_sequence),
            )
            target_value[continuing] += self.args.gamma * target_q
        return target_value

    def train_one_step(self) -> Dict[str, float]:
        """执行一次 Critic 更新，按 policy_delay 执行 Actor/Router 更新。"""
        batch = self.buffer.sample(self.args.batch_size)
        state = torch.as_tensor(batch["s"], dtype=torch.float32, device=device)
        action = torch.as_tensor(batch["a"], dtype=torch.float32, device=device)
        reward = torch.as_tensor(batch["r"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(batch["s2"], dtype=torch.float32, device=device)
        done = torch.as_tensor(batch["d"], dtype=torch.float32, device=device)
        phase_sequence = torch.as_tensor(batch["phase_sequence"], dtype=torch.long, device=device)
        phase_previous = torch.as_tensor(batch["phase_previous"], dtype=torch.long, device=device).squeeze(1)
        phase = torch.as_tensor(batch["phase"], dtype=torch.long, device=device).squeeze(1)
        state, next_state = self._apply_state_noise(state, next_state)
        bc_weight = linear_anneal(self.train_step, self.args.bc_weight_init, self.args.bc_weight_final, self.args.bc_anneal_steps)

        target_value = self._compute_target_values(next_state, action, reward, done, phase)

        q1, q2 = self.critic1(state, action), self.critic2(state, action)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.args.grad_clip)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.args.grad_clip)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        info: Dict[str, float] = {
            "critic_loss": float(critic_loss.item()), "bc_weight": float(bc_weight),
            "actor_updated": 0.0, "step": float(self.train_step), **self.last_actor_info,
        }
        if self.train_step % self.args.policy_delay == 0:
            output = self.actor(state, phase_previous, mode="soft")
            q_policy = self.critic1(state, self._replace_last_action(action, output["action"]))
            if self.args.use_td3bc_adaptive_lambda:
                with torch.no_grad():
                    q_abs_mean = q_policy.abs().mean().clamp(min=1e-3)
                adaptive_lambda = float(self.args.td3bc_alpha / q_abs_mean.item())
            else:
                adaptive_lambda = 1.0
            td3_loss = -adaptive_lambda * q_policy.mean()
            bc_loss = F.mse_loss(self._expert_action_for_phase(output["all_actions"], phase), action[:, -1, :])
            router_loss = F.cross_entropy(output["logits"], phase)
            # 倒数第二 token 只用到它之前的因果历史，使用真实 p[t-2] 的转移 mask。
            _, previous_weights = self.actor.router(output["hidden_sequence"][:, -2, :], phase_sequence[:, -3])
            switch_loss = F.mse_loss(output["weights"], previous_weights)
            actor_loss = td3_loss + bc_weight * bc_loss + self.args.router_loss_weight * router_loss + self.args.switch_loss_weight * switch_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target, self.args.tau)
            self._soft_update(self.critic1, self.critic1_target, self.args.tau)
            self._soft_update(self.critic2, self.critic2_target, self.args.tau)
            router_accuracy = (output["selected_phase"] == phase).float().mean()
            self.last_actor_info = {
                "actor_loss": float(actor_loss.item()), "bc_loss": float(bc_loss.item()),
                "td3_loss": float(td3_loss.item()), "router_loss": float(router_loss.item()),
                "switch_loss": float(switch_loss.item()), "lambda": float(adaptive_lambda),
                "router_accuracy": float(router_accuracy.item()),
            }
            info.update(self.last_actor_info)
            info["actor_updated"] = 1.0
        self.train_step += 1
        return info

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "phase_mapping_version": PHASE_MAPPING_VERSION, "phase_names": list(PHASE_NAMES),
            "phase_transition_version": PHASE_TRANSITION_VERSION,
            "state_dim": self.args.state_dim, "action_dim": self.args.action_dim,
            "seq_len": self.args.seq_len, "hidden_dim": self.args.hidden_dim,
            "transformer_layers": self.args.transformer_layers,
            "transformer_heads": self.args.transformer_heads, "n_phases": self.args.n_phases,
            "args": asdict(self.args),
        }

    def save(self, checkpoint_dir: str, step: int) -> None:
        """保存 MoE 网络和结构契约，避免误加载 LSTM checkpoint。"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, f"actor_{step}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(checkpoint_dir, f"critic1_{step}.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(checkpoint_dir, f"critic2_{step}.pth"))
        with open(os.path.join(checkpoint_dir, f"metadata_{step}.json"), "w", encoding="utf-8") as output_file:
            json.dump(self._checkpoint_metadata(), output_file, ensure_ascii=False, indent=2)

    def load(self, checkpoint_dir: str, step: int) -> None:
        """加载同一 MoE 契约的 checkpoint，并拒绝不兼容的结构。"""
        metadata_path = os.path.join(checkpoint_dir, f"metadata_{step}.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError("MoE checkpoint 缺少 metadata，拒绝加载无法验证的权重")
        with open(metadata_path, "r", encoding="utf-8") as input_file:
            metadata = json.load(input_file)
        expected = self._checkpoint_metadata()
        # 转移表也是持久化 buffer；拒绝旧版本，避免加载权重时把修正后的表覆盖回去。
        for key in ("phase_mapping_version", "phase_transition_version", "phase_names", "state_dim", "action_dim", "seq_len",
                    "hidden_dim", "transformer_layers", "transformer_heads", "n_phases"):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"MoE checkpoint {key} 不匹配: {metadata.get(key)!r} != {expected[key]!r}")
        self.actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"actor_{step}.pth"), map_location=device))
        self.critic1.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"critic1_{step}.pth"), map_location=device))
        self.critic2.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"critic2_{step}.pth"), map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


def fill_buffer_from_episodes(agent: OfflineMoETD3BC,
                              episodes: Sequence[Sequence[Mapping[str, Any]]]) -> int:
    """从已归一化的 episode 构造含 phase 的 MoE replay buffer。"""
    added_total = 0
    for episode_index, episode in enumerate(episodes):
        if not episode:
            continue
        states = np.stack([_to_np(step["observation"]) for step in episode], axis=0)
        next_states = np.stack([_to_np(step["next_observation"]) for step in episode], axis=0)
        actions = np.stack([_to_np(step["action"]) for step in episode], axis=0)
        rewards = np.asarray([float(step["reward"]) for step in episode], dtype=np.float32)
        dones = np.asarray([float(step["done"]) for step in episode], dtype=np.float32)
        phases = np.asarray([_extract_phase(step) for step in episode], dtype=np.int64)
        if (states.shape[1:] != (agent.state_dim,) or next_states.shape != states.shape
                or actions.shape[1:] != (agent.action_dim,)):
            raise ValueError(f"episode {episode_index} 的 state_dim 或 action_dim 不匹配")
        if not np.isfinite(states).all() or not np.isfinite(next_states).all() or not np.isfinite(actions).all():
            raise ValueError(f"episode {episode_index} 存在非有限 state、next_state 或 action")
        added_total += agent.buffer.add_episode(states, actions, rewards, dones, phases, next_states)
    return added_total
