#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试简化全局专家和 SEARCH 行为。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Sampling.global_expert import ExpertConfig, GlobalLandingExpert


class GlobalExpertTests(unittest.TestCase):
    def setUp(self):
        self.config = ExpertConfig(stable_steps_required=1)
        self.expert = GlobalLandingExpert(self.config)

    def test_horizontal_action_points_to_target(self):
        """目标在机体前方时，x 动作必须为正。"""
        command = self.expert.compute_action(
            drone_position=[0.0, 0.0, 3.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[1.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
        )
        self.assertGreater(command.action[0], 0.0)

    def test_stable_high_state_descends(self):
        """高空且水平稳定时，专家必须输出负 z 下降。"""
        command = self.expert.compute_action(
            drone_position=[0.0, 0.0, 3.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[0.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
        )
        self.assertEqual(command.phase, "DESCEND")
        self.assertLess(command.action[2], 0.0)

    def test_midair_loss_keeps_xy_and_climbs(self):
        """非近地失检时保持上一水平动作并用正 z 上升。"""
        previous = np.array([0.31, -0.22, -0.40], dtype=np.float32)
        command = self.expert.compute_action(
            drone_position=[0.0, 0.0, 3.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[0.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
            marker_visible=False,
            last_action=previous,
        )
        self.assertEqual(command.phase, "SEARCH")
        np.testing.assert_array_equal(command.action[:2], previous[:2])
        self.assertAlmostEqual(command.action[2], self.config.search_climb_speed)

    def test_near_ground_loss_does_not_search(self):
        """近地失检不能触发上升搜寻。"""
        command = self.expert.compute_action(
            drone_position=[0.0, 0.0, 0.30],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[0.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
            marker_visible=False,
            last_action=[0.0, 0.0, -0.1],
        )
        self.assertEqual(command.phase, "TOUCHDOWN")
        self.assertLess(command.action[2], 0.0)

    def test_reacquisition_immediately_leaves_search(self):
        """重新看到 marker 后不能继续沿用 SEARCH 上升动作。"""
        self.expert.compute_action(
            drone_position=[0.0, 0.0, 3.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[0.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
            marker_visible=False,
            last_action=[0.1, 0.0, -0.2],
        )
        command = self.expert.compute_action(
            drone_position=[0.0, 0.0, 3.0],
            drone_velocity=[0.0, 0.0, 0.0],
            drone_yaw=0.0,
            target_position=[0.0, 0.0, 0.0],
            target_velocity=[0.0, 0.0, 0.0],
            marker_visible=True,
        )
        self.assertEqual(command.phase, "DESCEND")
        self.assertLess(command.action[2], 0.0)

    def test_action_always_stays_in_td3_range(self):
        """专家动作必须满足 TD3_offline.py 的 [-1, 1] 范围。"""
        command = self.expert.compute_action(
            drone_position=[-10.0, -10.0, 8.0],
            drone_velocity=[-2.0, -2.0, 0.0],
            drone_yaw=1.2,
            target_position=[10.0, 10.0, 0.0],
            target_velocity=[0.8, 0.8, 0.0],
        )
        self.assertTrue(np.all(np.abs(command.action) <= 1.0))


if __name__ == "__main__":
    unittest.main()
