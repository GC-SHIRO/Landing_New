"""验证当前 Sampling 专家的阶段变化不会被 MoE 的转移表误拒绝。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Sampling.global_expert import ExpertConfig, GlobalLandingExpert
from model.moe_td3 import (
    Args,
    OfflineMoETD3BC,
    PhaseRouter,
    SequenceReplayBuffer,
    phase_to_index,
    transition_is_allowed,
)


class SamplingPhaseTransitionTests(unittest.TestCase):
    @staticmethod
    def expert_phases(inputs):
        """输入水平误差、高度和可见性，直接取得现有专家输出的标签。"""
        expert = GlobalLandingExpert(ExpertConfig(stable_steps_required=2))
        return [
            expert.compute_action(
                drone_position=[horizontal_error, 0.0, height],
                drone_velocity=[0.0, 0.0, 0.0],
                drone_yaw=0.0,
                target_position=[0.0, 0.0, 0.0],
                target_velocity=[0.0, 0.0, 0.0],
                marker_visible=visible,
            ).phase
            for horizontal_error, height, visible in inputs
        ]

    def test_expert_recovery_transitions_enter_replay_and_router(self):
        """五条原先被误拒绝的专家路径必须能进入 32 帧 replay 并用于 Router 监督。"""
        cases = {
            ("TRACK", "TOUCHDOWN"): [(0.0, 0.3, True)] * 2,
            ("DESCEND", "ALIGN"): [(0.0, 2.0, True)] * 2 + [(2.0, 2.0, True)],
            ("TOUCHDOWN", "ALIGN"): [(0.0, 0.3, True)] * 2 + [(2.0, 0.3, True)],
            ("TOUCHDOWN", "TRACK"): [(0.0, 0.3, True)] * 2 + [(0.5, 0.3, True)],
            ("TOUCHDOWN", "DESCEND"): [(0.0, 0.3, True)] * 2 + [(0.0, 0.8, True)],
        }
        router = PhaseRouter(hidden_dim=8, router_hidden_dim=4, temperature=1.0)
        for transition, inputs in cases.items():
            with self.subTest(transition=transition):
                # 前置对准历史让待测跳转落到实际训练窗口末端，最后一帧作为终止步。
                names = self.expert_phases([(2.0, 2.0, True)] * 32 + inputs + inputs[-1:])
                self.assertIn(transition, list(zip(names, names[1:])))
                phases = np.asarray([phase_to_index(name) for name in names])
                count = len(phases)
                buffer = SequenceReplayBuffer(16, 10, 3, 32)
                states = np.zeros((count, 10), dtype=np.float32)
                actions = np.zeros((count, 3), dtype=np.float32)
                dones = np.zeros(count, dtype=np.float32)
                dones[-1] = 1.0
                added = buffer.add_episode(states, actions, np.zeros(count), dones, phases, states.copy())
                self.assertEqual(added, count - 32 + 1)
                pairs = list(zip(buffer.phase_previous[:added, 0], buffer.phase[:added, 0]))
                self.assertIn(tuple(phase_to_index(name) for name in transition), pairs)
                previous = torch.tensor(buffer.phase_previous[:added, 0])
                current = torch.tensor(buffer.phase[:added, 0])
                logits, weights = router(torch.zeros(added, 8), previous)
                self.assertTrue(torch.all(weights[torch.arange(added), current] > 0.0))
                loss = torch.nn.functional.cross_entropy(logits, current)
                self.assertTrue(torch.isfinite(loss))
                router.zero_grad()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in router.parameters()))

    def test_search_recovery_still_waits_for_stability(self):
        """连续稳定两步时，重新检测到 marker 后先跟踪，再恢复下降或近地控制。"""
        for height, final_phase in [(2.0, "DESCEND"), (0.3, "TOUCHDOWN")]:
            with self.subTest(height=height):
                names = self.expert_phases([
                    (0.0, 2.0, False), (0.0, height, True), (0.0, height, True),
                ])
                self.assertEqual(names, ["SEARCH", "TRACK", final_phase])
                for previous, current in zip(names, names[1:]):
                    self.assertTrue(transition_is_allowed(phase_to_index(previous), phase_to_index(current)))

    def test_router_keeps_forbidden_descent_masked(self):
        """允许纠偏路径后，对准和搜寻阶段仍不能直接选择下降或近地专家。"""
        router = PhaseRouter(hidden_dim=8, router_hidden_dim=4, temperature=1.0)
        previous = torch.tensor([phase_to_index("ALIGN"), phase_to_index("SEARCH")])
        _, weights = router(torch.zeros(2, 8), previous)
        torch.testing.assert_close(weights[:, 2:4], torch.zeros(2, 2))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))


class PhaseCheckpointTests(unittest.TestCase):
    def test_checkpoint_cannot_restore_old_transition_table(self):
        """新 checkpoint 可正常加载，缺少转移版本的旧文件须在加载网络前被拒绝。"""
        args = Args(hidden_dim=8, transformer_heads=2, transformer_layers=1,
                    transformer_ffn_dim=16, router_hidden_dim=4, capacity=1)
        agent = OfflineMoETD3BC(args)
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory, 0)
            agent.load(directory, 0)
            self.assertTrue(agent.actor.router.allowed_transitions[2, 0])
            metadata_path = Path(directory) / "metadata_0.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["phase_transition_version"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phase_transition_version"):
                agent.load(directory, 0)


if __name__ == "__main__":
    unittest.main()
