"""验证独立训练脚本的阶段衔接、参数冻结和可恢复的合成数据训练流程。"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.moe_td3 import Args, OfflineMoETD3BC, device, fill_buffer_from_episodes
from scripts.train import stage0_single_head, stage1_pretrain, stage2_joint
from scripts.train.common import evaluate, load_checkpoint, load_data, validation_buffer


def synthetic_episodes():
    """使用独立临时数据覆盖五阶段，所有窗口都来自同一回合的连续状态。"""
    phases = ["ALIGN"] * 32 + ["TRACK", "DESCEND", "TOUCHDOWN", "SEARCH", "TRACK", "DESCEND", "TOUCHDOWN", "TOUCHDOWN"]
    rng = np.random.default_rng(7)
    episodes = []
    for episode_index in range(5):
        states = rng.normal(0, 0.1, (41, 10)).astype(np.float32)
        states[:, 0] += episode_index * 0.1
        states[:, 9] = 0.9
        episode = []
        for index, phase in enumerate(phases):
            episode.append({
                "observation": states[index].tolist(), "next_observation": states[index + 1].tolist(),
                "action": [0.05, -0.05, 0.25 if phase == "SEARCH" else -0.1],
                "reward": 300.0 if index == 39 else -0.1, "done": index == 39,
                "expert": {"phase": phase},
            })
        episodes.append(episode)
    return episodes


def tiny_args(**changes):
    return replace(Args(hidden_dim=8, transformer_heads=2, transformer_layers=1,
                        transformer_ffn_dim=16, router_hidden_dim=4, dropout_p=0.0,
                        batch_size=2, capacity=64, training_steps=2, log_every=1, save_every=1), **changes)


class TrainingStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_single_head_is_copied_into_all_experts(self):
        """五头初始化后与单头输出相同，encoder 参数精确复制。"""
        args = tiny_args()
        source = OfflineMoETD3BC(args, single_head=True)
        target = OfflineMoETD3BC(args)
        target.initialize_from_single_head(source.actor.state_dict())
        source.actor.eval()
        target.actor.eval()
        state = torch.randn(2, 32, 10, device=device)
        previous = torch.tensor([0, 2], device=device)
        with torch.no_grad():
            expected = source.actor(state)["action"]
            actual = target.actor(state, previous)
        for index in range(5):
            torch.testing.assert_close(actual["all_actions"][:, index], expected)
        torch.testing.assert_close(actual["action"], expected)
        for name, parameter in source.actor.encoder.state_dict().items():
            torch.testing.assert_close(parameter, target.actor.encoder.state_dict()[name])

    def test_pretraining_freezes_encoder_and_both_critics(self):
        """Stage 1 的 BC/CE 能更新路由和专家，但冻结 encoder 与双 Critic 保持不变。"""
        agent = OfflineMoETD3BC(tiny_args())
        fill_buffer_from_episodes(agent, synthetic_episodes()[:1])
        counts = np.bincount(agent.buffer.phase[:len(agent.buffer), 0], minlength=5)
        agent.configure_pretraining(counts)
        before = {name: {key: value.clone() for key, value in getattr(agent, name).state_dict().items()}
                  for name in ("actor", "critic1", "critic2")}
        agent.pretrain_one_step()
        for name in ("critic1", "critic2"):
            for key, value in getattr(agent, name).state_dict().items():
                torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
        for key, value in agent.actor.encoder.state_dict().items():
            torch.testing.assert_close(value, before["actor"]["encoder." + key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before["actor"][key]) for key, value in agent.actor.state_dict().items()))
        self.assertFalse(agent.actor.encoder.training)

    def test_unfrozen_encoder_has_smaller_learning_rate_and_missing_class_is_finite(self):
        """可选择以较低学习率训练 encoder，缺样类别不会造成无穷大权重。"""
        agent = OfflineMoETD3BC(tiny_args(pretrain_freeze_encoder=False))
        agent.configure_pretraining(np.array([1, 2, 3, 0, 4]))
        self.assertTrue(all(p.requires_grad for p in agent.actor.encoder.parameters()))
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[1]["lr"], agent.args.lr_actor * 0.1)
        self.assertEqual(agent.router_class_weights[3].item(), 0)
        self.assertTrue(torch.isfinite(agent.router_class_weights).all())

    def test_all_stages_and_same_stage_resume(self):
        """实际运行三脚本，验证共享数据、Critic 继承、Stage 2 解冻/同步及续训一致性。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "synthetic.jsonl"
            data_path.write_text("\n".join(json.dumps(ep) for ep in synthetic_episodes()), encoding="utf-8")
            args = tiny_args(data_path=str(data_path), dropout_p=0.1, training_steps=3)
            checkpoint0 = stage0_single_head.run(args=args, output_dir=root / "stage0", eval_every=1)
            payload0 = load_checkpoint(checkpoint0)
            settings = dict(training_steps=3, batch_size=2, log_every=1, save_every=1)
            checkpoint1 = stage1_pretrain.run(input_checkpoint=checkpoint0, output_dir=root / "stage1",
                                               settings=settings, eval_every=1)
            payload1 = load_checkpoint(checkpoint1)
            self.assertEqual(payload0["data"], payload1["data"])
            for name in ("critic1", "critic2"):
                for key in payload0[name]:
                    torch.testing.assert_close(payload1[name][key], payload0[name][key], rtol=0, atol=0)
            stage1_pretrain.run(output_dir=root / "stage1", resume_checkpoint=root / "stage1" / "step_1.pt",
                                settings=settings, eval_every=1)
            resumed1 = load_checkpoint(checkpoint1)
            for key, value in payload1["actor"].items():
                torch.testing.assert_close(value, resumed1["actor"][key], rtol=0, atol=0)
            original_step = OfflineMoETD3BC.train_one_step

            def checked_step(agent):
                if agent.train_step == 0:
                    self.assertTrue(all(p.requires_grad for network in (agent.actor, agent.critic1, agent.critic2)
                                        for p in network.parameters()))
                    self.assertFalse(agent.actor_optimizer.state)
                    for name in ("actor", "critic1", "critic2"):
                        for key, value in getattr(agent, name).state_dict().items():
                            torch.testing.assert_close(value, getattr(agent, name + "_target").state_dict()[key], rtol=0, atol=0)
                return original_step(agent)

            with patch.object(OfflineMoETD3BC, "train_one_step", checked_step):
                checkpoint2 = stage2_joint.run(input_checkpoint=checkpoint1, output_dir=root / "stage2",
                                               settings=settings, eval_every=1)
            payload2 = load_checkpoint(checkpoint2)
            self.assertEqual(payload2["data"], payload0["data"])
            self.assertTrue(any(not torch.equal(value, payload1["critic1"][key]) for key, value in payload2["critic1"].items()))
            self.assertEqual(payload2["step"], 3)
            records = [json.loads(line) for line in (root / "stage2" / "validation.jsonl").read_text().splitlines()]
            self.assertEqual(len(records[-1]["confusion_matrix"]), 5)
            self.assertTrue(np.isfinite(records[-1]["rollout_hard"]["mse"]))
            self.assertEqual(records[-1]["samples"], 9)
            # 从同一次实验中间存档恢复，与未中断结果逐参数一致。
            stage2_joint.run(output_dir=root / "stage2", resume_checkpoint=root / "stage2" / "step_1.pt",
                             settings=settings, eval_every=1)
            resumed = load_checkpoint(checkpoint2)
            for name in ("actor", "critic1", "critic2", "actor_target", "critic1_target", "critic2_target"):
                for key, value in payload2[name].items():
                    torch.testing.assert_close(value, resumed[name][key], rtol=0, atol=0)
            # 保存的归一化和索引确实被复用，验证数据变动不能悄悄换掉实验输入。
            self.assertTrue((root / "shared" / "state_mean.npy").is_file())
            data_path.write_text(data_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数据文件已变化"):
                load_data(args, payload0["data"])

    def test_rollout_uses_own_previous_phase_and_resets_per_episode(self):
        """顺序评估的阶段来自模型自身，下一回合必须重置为 APPROACH。"""
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.json"
            data_path.write_text(json.dumps(synthetic_episodes()), encoding="utf-8")
            args = tiny_args(data_path=str(data_path), validation_fraction=0.4, batch_size=64)
            prepared, _ = load_data(args)
            buffer = validation_buffer(prepared.validation_episodes, args, prepared.validation_phase_counts.sum())
            agent = OfflineMoETD3BC(args)
            received = []

            def fake_forward(state, previous, mode="soft"):
                batch = state.shape[0]
                # 固定预测 SEARCH；真实标签在后段已转为其他阶段。
                if batch == 1:
                    received.append(int(previous.item()))
                return {"action": torch.zeros(batch, 3, device=device),
                        "all_actions": torch.zeros(batch, 5, 3, device=device),
                        "selected_phase": torch.full((batch,), 4, dtype=torch.long, device=device)}

            with patch.object(agent.actor, "forward", fake_forward):
                evaluate(agent, buffer, prepared.validation_episodes)
            self.assertEqual(received, ([0] + [4] * 39) * 2)


if __name__ == "__main__":
    unittest.main()
