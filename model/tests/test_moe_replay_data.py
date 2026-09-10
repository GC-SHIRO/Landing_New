"""终止样本、TD3 目标及各阶段共用数据准备的离线回归测试。"""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.moe_td3 import (
    Args,
    OfflineMoETD3BC,
    SequenceReplayBuffer,
    compute_mean_std,
    device,
    fill_buffer_from_episodes,
    normalize_episodes,
    phase_to_index,
    prepare_offline_data,
)


def make_episode(length=40, offset=0.0, terminal_reward=300.0):
    """生成状态链连续的示范；终止状态与终止前状态不同，避免复制状态也能过测。"""
    return [
        {
            "observation": np.full(10, offset + index / 100, dtype=np.float32),
            "next_observation": np.full(10, offset + (index + 1) / 100, dtype=np.float32),
            "action": np.array([0.1, 0.0, -0.08 if index == length - 1 else -0.2], dtype=np.float32),
            "reward": terminal_reward if index == length - 1 else -0.1,
            "done": index == length - 1,
            "expert": {"phase": "TOUCHDOWN" if index == length - 1 else "TRACK"},
        }
        for index in range(length)
    ]


def replay_agent(seq_len=32):
    """装载测试只需要数据接口，不分配神经网络。"""
    return SimpleNamespace(state_dim=10, action_dim=3,
                           buffer=SequenceReplayBuffer(128, 10, 3, seq_len))


class TerminalReplayTests(unittest.TestCase):
    def test_terminal_reward_action_and_real_next_state_are_kept(self):
        """40 步生成 9 个窗口，最后一步的奖惩、动作、标签和真实下一状态均保留。"""
        for terminal_reward in (300.0, -200.0):
            with self.subTest(terminal_reward=terminal_reward):
                raw = make_episode(terminal_reward=terminal_reward)
                episodes = normalize_episodes([raw], np.zeros(10), np.ones(10))
                agent = replay_agent()
                self.assertEqual(fill_buffer_from_episodes(agent, episodes), 9)
                buffer = agent.buffer
                self.assertEqual(buffer.d[:9].sum(), 1.0)
                self.assertEqual(buffer.r[8, 0], terminal_reward)
                np.testing.assert_array_equal(buffer.a[8, -1], raw[-1]["action"])
                np.testing.assert_array_equal(buffer.s[8], np.stack([s["observation"] for s in raw[-32:]]))
                np.testing.assert_array_equal(buffer.s2[8], np.stack([s["next_observation"] for s in raw[-32:]]))
                self.assertEqual(buffer.phase_previous[8, 0], phase_to_index("TRACK"))
                self.assertEqual(buffer.phase[8, 0], phase_to_index("TOUCHDOWN"))
                self.assertEqual(buffer.phase_next[8, 0], -1)
                # 普通窗口依旧严格平移一帧，下一阶段仍是原始下一步的标签。
                np.testing.assert_array_equal(buffer.s2[0], np.stack([s["observation"] for s in raw[1:33]]))
                self.assertEqual(buffer.phase_next[7, 0], phase_to_index("TOUCHDOWN"))

    def test_exact_length_episode_and_separate_episodes(self):
        """恰好 32 步的回合可训练终止步，不同回合的状态窗口不能混在一起。"""
        raw = [make_episode(32, 0.0), make_episode(32, 100.0)]
        agent = replay_agent()
        self.assertEqual(fill_buffer_from_episodes(agent, raw), 2)
        for index in range(2):
            np.testing.assert_array_equal(agent.buffer.s[index], np.stack([s["observation"] for s in raw[index]]))
            self.assertEqual(agent.buffer.d[index, 0], 1.0)

    def test_short_history_and_unfinished_tail_are_not_fabricated(self):
        """短于窗口的回合不补帧；没有下一阶段的非终止尾帧不伪造 phase_next。"""
        agent = replay_agent()
        self.assertEqual(fill_buffer_from_episodes(agent, [make_episode(31)]), 0)
        unfinished = make_episode(40)
        unfinished[-1]["done"] = False
        self.assertEqual(fill_buffer_from_episodes(agent, [unfinished]), 8)
        self.assertEqual(agent.buffer.d[:8].sum(), 0.0)


class TerminalTargetTests(unittest.TestCase):
    def setUp(self):
        self.agent = OfflineMoETD3BC(Args(
            seq_len=3, hidden_dim=8, transformer_heads=2, transformer_layers=1,
            transformer_ffn_dim=16, router_hidden_dim=4, dropout_p=0.0,
            capacity=8, batch_size=2, policy_noise=0.0,
        ))

    def test_all_terminal_targets_skip_all_target_networks(self):
        """全终止 batch 直接使用奖惩，完全不读取目标动作、目标 Q 或下一阶段。"""
        reward = torch.tensor([[300.0], [-200.0]], device=device)
        with patch.object(self.agent.actor_target, "forward", side_effect=AssertionError("不应调用目标 Actor")), \
             patch.object(self.agent.critic1_target, "forward", side_effect=AssertionError("不应调用目标 Critic")), \
             patch.object(self.agent.critic2_target, "forward", side_effect=AssertionError("不应调用目标 Critic")):
            result = self.agent._compute_target_values(
                torch.full((2, 3, 10), float("nan"), device=device),
                torch.zeros(2, 3, 3, device=device), reward,
                torch.ones(2, 1, device=device), torch.full((2,), -1, device=device),
            )
        torch.testing.assert_close(result, reward)

    def test_mixed_targets_bootstrap_only_nonterminal_rows(self):
        """混合 batch 仅普通样本使用双 Q 最小值，终止行即使下一状态异常也不会污染目标。"""
        next_state = torch.zeros(3, 3, 10, device=device)
        next_state[1:] = float("nan")
        actions = torch.arange(27, device=device, dtype=torch.float32).reshape(3, 3, 3) / 100
        phase = torch.tensor([1, 3, 3], device=device)
        with patch.object(self.agent.actor_target, "forward", return_value={"action": torch.zeros(1, 3, device=device)}) as actor, \
             patch.object(self.agent.critic1_target, "forward", return_value=torch.tensor([[2.0]], device=device)) as q1, \
             patch.object(self.agent.critic2_target, "forward", return_value=torch.tensor([[3.0]], device=device)):
            result = self.agent._compute_target_values(
                next_state, actions, torch.tensor([[1.0], [300.0], [-200.0]], device=device),
                torch.tensor([[0.0], [1.0], [1.0]], device=device), phase,
            )
        torch.testing.assert_close(result, torch.tensor([[1.0 + self.agent.args.gamma * 2], [300.0], [-200.0]], device=device))
        torch.testing.assert_close(actor.call_args.args[0], next_state[:1])
        torch.testing.assert_close(actor.call_args.args[1], phase[:1])
        torch.testing.assert_close(q1.call_args.args[1][:, :-1], actions[:1, 1:])

    def test_terminal_only_batch_updates_actor_and_critics(self):
        """终止步可独立完成 Actor/双 Critic 更新，其真实动作参与 BC。"""
        raw = make_episode(3)
        self.assertEqual(fill_buffer_from_episodes(self.agent, [raw]), 1)
        state = torch.tensor(np.stack([s["observation"] for s in raw])[None], device=device)
        phase = torch.tensor([phase_to_index("TRACK")], device=device)
        with torch.no_grad():
            expert_action = self.agent.actor(state, phase)["all_actions"][:, phase_to_index("TOUCHDOWN")]
            expected_bc = torch.nn.functional.mse_loss(expert_action, torch.tensor(raw[-1]["action"][None], device=device)).item()
        before = [parameter.detach().clone() for parameter in self.agent.actor.parameters()]
        with patch.object(self.agent.actor_target, "forward", side_effect=AssertionError("终止样本不应调用目标 Actor")):
            info = self.agent.train_one_step()
        self.assertEqual(info["actor_updated"], 1.0)
        self.assertAlmostEqual(info["bc_loss"], expected_bc, places=5)
        self.assertTrue(all(np.isfinite(value) for value in info.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, self.agent.actor.parameters())))
        for network in (self.agent.actor, self.agent.critic1, self.agent.critic2):
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters()))


class SharedDataPreparationTests(unittest.TestCase):
    def test_split_is_by_episode_and_repeatable(self):
        """同一 seed 划分相同，训练/验证回合互不重叠，且不修改原始数据。"""
        episodes = [make_episode(40, float(index)) for index in range(5)]
        original_first = episodes[0][0]["observation"].copy()
        prepared = prepare_offline_data(episodes, Args())
        repeated = prepare_offline_data(episodes, Args())
        self.assertEqual(prepared.train_indices, repeated.train_indices)
        self.assertEqual(prepared.validation_indices, repeated.validation_indices)
        self.assertFalse(set(prepared.train_indices) & set(prepared.validation_indices))
        self.assertEqual(sorted(prepared.train_indices + prepared.validation_indices), list(range(5)))
        self.assertEqual([len(ep) for ep in prepared.train_episodes + prepared.validation_episodes], [40] * 5)
        np.testing.assert_array_equal(episodes[0][0]["observation"], original_first)
        # 缺少其他阶段不会报错，计数对应实际训练窗口并包含终止步。
        self.assertEqual(prepared.train_phase_counts.sum(), 4 * 9)
        self.assertEqual(prepared.train_phase_counts[phase_to_index("TOUCHDOWN")], 4)
        self.assertEqual(prepared.validation_phase_counts.sum(), 9)
        self.assertEqual(prepared.train_phase_counts[phase_to_index("SEARCH")], 0)
        agent = replay_agent()
        self.assertEqual(fill_buffer_from_episodes(agent, prepared.train_episodes), prepared.train_phase_counts.sum())

    def test_validation_data_cannot_change_training_statistics(self):
        """大幅修改验证集后，训练归一化统计不变，验证集仍使用原训练尺度。"""
        episodes = [make_episode(40, float(index)) for index in range(5)]
        prepared = prepare_offline_data(episodes, Args())
        changed = copy.deepcopy(episodes)
        validation_index = prepared.validation_indices[0]
        for step in changed[validation_index]:
            step["observation"] += 1000
            step["next_observation"] += 1000
        new = prepare_offline_data(changed, Args())
        mean, std = compute_mean_std([episodes[i] for i in prepared.train_indices], 10)
        np.testing.assert_array_equal(new.mean, mean)
        np.testing.assert_array_equal(new.std, std)
        np.testing.assert_array_equal(new.train_episodes[0][0]["observation"], prepared.train_episodes[0][0]["observation"])
        np.testing.assert_allclose(new.validation_episodes[0][0]["observation"],
                                   (changed[validation_index][0]["observation"] - mean) / std)

    def test_shared_statistics_and_split_can_be_saved(self):
        """各 Stage 可复用保存的统计量，并追溯训练与验证的 episode 索引。"""
        prepared = prepare_offline_data([make_episode(32, 0.0), make_episode(32, 2.0)], Args())
        with tempfile.TemporaryDirectory() as directory:
            prepared.save(directory)
            np.testing.assert_array_equal(np.load(Path(directory) / "state_mean.npy"), prepared.mean)
            np.testing.assert_array_equal(np.load(Path(directory) / "state_std.npy"), prepared.std)
            metadata = json.loads((Path(directory) / "data_split.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["train_indices"], prepared.train_indices)
            self.assertEqual(metadata["validation_indices"], prepared.validation_indices)
            self.assertEqual(sum(metadata["train_phase_counts"]), 1)


if __name__ == "__main__":
    unittest.main()
