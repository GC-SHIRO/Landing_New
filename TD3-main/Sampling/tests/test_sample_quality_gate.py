#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for expert-data quality gating near touchdown."""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Sampling import collect_dynamic_expert as collector  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
RAW_CANDIDATES = [
    Path(REPO_ROOT) / "expert_data_dynamic" / "step1_privileged_pd_all.jsonl",
    Path(REPO_ROOT) / "expert_data_dynamic" / "random_dynamic_privileged_pd_all.jsonl",
]


def _replay_episode(episode, near_ground_height=collector.DEFAULT_NEAR_GROUND_HEIGHT):
    training = []
    reject_counts = Counter()
    fatal_quality = False
    fatal_reason = ""
    repeat_count = 0
    for step in episode:
        info = dict(step.get("env_info") or {})
        phase = (step.get("expert") or {}).get("phase")
        relative_height = info.get("relative_height")
        accepted, repeat_count, reason, fatal = collector.sample_is_usable(
            step["observation"],
            step["next_observation"],
            step["action"],
            info,
            repeat_count,
            phase=phase,
            relative_height=relative_height,
            near_ground_height=near_ground_height,
        )
        if accepted:
            training.append(step)
        else:
            reject_counts[reason or "unknown"] += 1
            if fatal and not fatal_quality:
                fatal_quality = True
                fatal_reason = reason or "unknown"
    success = bool(episode[-1].get("success")) if episode else False
    should_save, save_reason = collector.episode_should_save(
        success=success,
        training_episode=training,
        total_steps=len(episode),
        fatal_quality=fatal_quality,
        min_success_steps=15,
        min_valid_ratio=collector.DEFAULT_MIN_VALID_RATIO,
    )
    return {
        "success": success,
        "should_save": should_save,
        "save_reason": save_reason,
        "valid_steps": len(training),
        "total_steps": len(episode),
        "reject_counts": dict(reject_counts),
        "fatal": fatal_quality,
        "fatal_reason": fatal_reason,
        "training": collector.finalize_training_episode(training),
    }


class SampleQualityGateTest(unittest.TestCase):
    def test_near_ground_stale_is_dropped_not_fatal(self):
        obs = np.array([0.1, 0.2, 0.5], dtype=np.float32)
        action = np.array([0.0, 0.0, -0.1], dtype=np.float32)
        info = {
            "detection_fresh": False,
            "tag_detected": False,
            "relative_height": 0.08,
            "phase": "TOUCHDOWN",
        }
        accepted, repeat_count, reason, fatal = collector.sample_is_usable(
            obs, obs, action, info, repeat_count=0, phase="TOUCHDOWN", relative_height=0.08
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "near_ground_stale")
        self.assertFalse(fatal)
        self.assertEqual(repeat_count, 0)

    def test_midair_stale_is_vision_invalid_not_fatal(self):
        obs = np.array([0.1, 0.2, 3.0], dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)
        info = {"detection_fresh": False, "tag_detected": False, "relative_height": 3.0}
        accepted, _, reason, fatal = collector.sample_is_usable(
            obs, obs, action, info, 0, phase="DESCEND", relative_height=3.0
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "vision_invalid")
        self.assertFalse(fatal)

    def test_near_ground_repeat_is_dropped_not_fatal(self):
        obs = np.array([0.05, -0.02, 0.48], dtype=np.float32)
        action = np.array([0.1, 0.0, -0.1], dtype=np.float32)
        info = {
            "detection_fresh": True,
            "tag_detected": True,
            "relative_height": 0.05,
            "phase": "TOUCHDOWN",
        }
        accepted, repeat_count, reason, fatal = collector.sample_is_usable(
            obs, obs, action, info, repeat_count=2, phase="TOUCHDOWN", relative_height=0.05
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "near_ground_repeat")
        self.assertFalse(fatal)
        self.assertEqual(repeat_count, 3)

    def test_action_limit_is_fatal(self):
        obs = np.array([0.1, 0.2, 1.0], dtype=np.float32)
        action = np.array([1.5, 0.0, 0.0], dtype=np.float32)
        info = {"detection_fresh": True, "tag_detected": True, "relative_height": 1.0}
        accepted, _, reason, fatal = collector.sample_is_usable(
            obs, obs, action, info, 0, phase="DESCEND", relative_height=1.0
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "action_limit")
        self.assertTrue(fatal)

    def test_depth_spike_above_old_limit_is_kept_if_within_new_limit(self):
        obs = np.array([0.1, 0.2, 10.0], dtype=np.float32)
        next_obs = np.array([0.2, 0.1, 22.5], dtype=np.float32)
        action = np.zeros(3, dtype=np.float32)
        info = {"detection_fresh": True, "tag_detected": True, "relative_height": 6.0}
        accepted, _, reason, fatal = collector.sample_is_usable(
            obs, next_obs, action, info, 0, phase="BRAKE", relative_height=6.0
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")
        self.assertFalse(fatal)

    def test_episode_should_save_requires_ratio_and_min_steps(self):
        training = [{"done": False} for _ in range(20)]
        ok, reason = collector.episode_should_save(
            success=True,
            training_episode=training,
            total_steps=21,
            fatal_quality=False,
            min_success_steps=15,
            min_valid_ratio=0.85,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

        ok, reason = collector.episode_should_save(
            success=True,
            training_episode=training,
            total_steps=40,
            fatal_quality=False,
            min_success_steps=15,
            min_valid_ratio=0.85,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_valid_ratio")

    def test_finalize_sets_terminal_done(self):
        episode = [{"done": False, "step_index": 0}, {"done": False, "step_index": 1}]
        finalized = collector.finalize_training_episode(episode)
        self.assertFalse(finalized[0]["done"])
        self.assertTrue(finalized[1]["done"])
        self.assertFalse(episode[1]["done"])

    def test_existing_raw_successful_episodes_become_saveable(self):
        raw_paths = [path for path in RAW_CANDIDATES if path.is_file() and path.stat().st_size > 0]
        if not raw_paths:
            self.skipTest("no raw expert jsonl available")

        saved = 0
        inspected = 0
        for path in raw_paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    if not episode or not episode[-1].get("success"):
                        continue
                    inspected += 1
                    result = _replay_episode(episode)
                    # 近地 YOLO 失检/冻结不应把整回合标成 fatal。
                    self.assertFalse(
                        result["fatal"],
                        msg=f"{path.name} unexpected fatal: {result['reject_counts']}",
                    )
                    if result["should_save"]:
                        self.assertGreaterEqual(result["valid_steps"], 15)
                        self.assertTrue(result["training"][-1]["done"])
                        saved += 1
                    else:
                        # 边界情况：长时间冻结仍可能因有效占比不足被拒。
                        self.assertIn(
                            result["save_reason"],
                            {"low_valid_ratio", "too_few_valid_steps"},
                            msg=(
                                f"{path.name} unexpected reject: "
                                f"{result['save_reason']} rejects={result['reject_counts']}"
                            ),
                        )
        self.assertGreater(inspected, 0)
        self.assertGreater(saved, 0)
        # 近地裁剪后，多数成功落地回合仍应保留。
        self.assertGreaterEqual(saved / float(inspected), 0.5)


if __name__ == "__main__":
    unittest.main()
