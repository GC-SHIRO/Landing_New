#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试 TD3 数据连续性和 SEARCH 数据约束。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Sampling.validate_expert_data import validate_dataset


def _valid_episode(length=72):
    episode = []
    for index in range(length):
        observation = np.array(
            [0.01 * index, -0.02 * index, 8.0 - 0.05 * index],
            dtype=np.float32,
        )
        next_observation = np.array(
            [0.01 * (index + 1), -0.02 * (index + 1), 8.0 - 0.05 * (index + 1)],
            dtype=np.float32,
        )
        episode.append(
            {
                "observation": observation.tolist(),
                "action": [0.1, -0.1, -0.3],
                "reward": -1.0,
                "next_observation": next_observation.tolist(),
                "done": index == length - 1,
                "success": index == length - 1,
                "step_index": index,
                "expert": {"phase": "DESCEND"},
                "env_info": {
                    "marker_visible": True,
                    "next_marker_visible": True,
                    "relative_height": 2.0,
                },
            }
        )
    return episode


class DatasetContinuityTests(unittest.TestCase):
    def test_valid_episode_passes(self):
        """满足 TD3 固定契约的数据不能被误报。"""
        errors, statistics = validate_dataset([_valid_episode()])
        self.assertEqual(errors, [])
        self.assertEqual(statistics["lstm_windows"], 64)

    def test_broken_next_observation_is_rejected(self):
        """相邻状态链断裂必须被检查出来。"""
        episode = _valid_episode()
        episode[8]["next_observation"] = [99.0, 99.0, 99.0]
        errors, _ = validate_dataset([episode])
        self.assertTrue(any("链断裂" in error for error in errors))

    def test_valid_search_step_is_kept(self):
        """失检 SEARCH 帧应保持 observation、水平动作并使用正 z。"""
        episode = _valid_episode()
        index = 10
        episode[index]["observation"] = list(episode[index - 1]["observation"])
        episode[index - 1]["next_observation"] = list(
            episode[index]["observation"]
        )
        episode[index]["next_observation"] = list(episode[index]["observation"])
        episode[index + 1]["observation"] = list(
            episode[index]["next_observation"]
        )
        episode[index]["action"] = [0.1, -0.1, 0.25]
        episode[index]["expert"] = {"phase": "SEARCH"}
        episode[index]["env_info"] = {
            "marker_visible": False,
            "next_marker_visible": False,
            "relative_height": 2.0,
        }
        errors, statistics = validate_dataset([episode])
        self.assertEqual(errors, [])
        self.assertEqual(statistics["search_steps"], 1)

    def test_near_ground_search_is_rejected(self):
        """近地失检时不能进入 SEARCH。"""
        episode = _valid_episode()
        index = 10
        episode[index]["observation"] = list(episode[index - 1]["observation"])
        episode[index - 1]["next_observation"] = list(
            episode[index]["observation"]
        )
        episode[index]["next_observation"] = list(episode[index]["observation"])
        episode[index + 1]["observation"] = list(
            episode[index]["next_observation"]
        )
        episode[index]["action"] = [0.1, -0.1, 0.25]
        episode[index]["expert"] = {"phase": "SEARCH"}
        episode[index]["env_info"] = {
            "marker_visible": False,
            "next_marker_visible": False,
            "relative_height": 0.30,
        }
        errors, _ = validate_dataset([episode])
        self.assertTrue(any("非近地" not in error and "SEARCH" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
