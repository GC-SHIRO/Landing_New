#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线校验：Sampling 奖励与 Simulation/env_base 动态落地判定语义一致。

稠密项与 env_base.reward_setup 的视觉 L3 shaping 公式一致；
终止项与 env_base.step() 的权威落地判定一致：
    landing_success = deck_contact AND relative_xy_distance <= landing_xy_threshold
（相对甲板水平距离，而非旧静态世界框 (-2.5,-1.5)^2。）
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Sampling.privileged_pd_expert import (  # noqa: E402
    SIM_FAIL_REWARD,
    SIM_SUCCESS_REWARD,
    compute_transition_reward,
)


def _env_base_dense(observation):
    """内联复刻 env_base.reward_setup 的稠密项数学式，用于交叉核对。"""
    delta_x = observation[0]
    delta_y = observation[1]
    height2 = observation[2]
    shape2 = -((abs(delta_x) ** 3 + abs(delta_y) ** 3 + abs(height2) ** 3) ** (1 / 3))
    return 0.1 * shape2


class SimulationAlignedRewardTest(unittest.TestCase):
    def test_dense_matches_env_base_formula(self):
        obs = [0.4, -0.2, 2.5]
        got = compute_transition_reward(
            observation=obs,
            done=False,
            success=False,
            world_x=0.0,
            world_y=0.0,
            next_observation=[0.3, -0.1, 2.4],
        )
        expected = _env_base_dense(obs)
        self.assertAlmostEqual(got, expected, places=10)
        self.assertLess(got, 0.0)

    def test_terminal_success_within_relative_threshold(self):
        # 相对甲板距离 <= landing_xy_threshold（与 env.step position_ok 同式）
        obs = [0.1, 0.1, 0.5]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=True,
            world_x=12.0,
            world_y=5.0,
            relative_xy_distance=0.3,
            landing_xy_threshold=1.5,
        )
        self.assertEqual(got, SIM_SUCCESS_REWARD)

    def test_terminal_success_beyond_relative_threshold_is_fail_reward(self):
        # 成功标志与距离不符（保护性双保险）：以相对距离为准判失败
        obs = [0.05, 0.02, 0.3]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=True,
            world_x=12.0,
            world_y=5.0,
            relative_xy_distance=2.0,
            landing_xy_threshold=1.5,
        )
        self.assertEqual(got, SIM_FAIL_REWARD)

    def test_terminal_success_by_flag_when_no_distance(self):
        # 未传距离参数时信任 env 的 success 标志（env 已按相对甲板+deck_contact 判定）
        obs = [0.05, 0.02, 0.3]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=True,
            world_x=12.0,
            world_y=5.0,
        )
        self.assertEqual(got, SIM_SUCCESS_REWARD)

    def test_terminal_failure(self):
        obs = [1.0, 1.0, 3.0]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=False,
            world_x=-2.0,
            world_y=-2.0,
            relative_xy_distance=0.3,
            landing_xy_threshold=1.5,
        )
        self.assertEqual(got, SIM_FAIL_REWARD)

    def test_next_observation_ignored_like_env_base(self):
        obs = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        a = compute_transition_reward(obs, False, False, 0.0, 0.0, next_observation=[9, 9, 9])
        b = compute_transition_reward(obs, False, False, 0.0, 0.0, next_observation=[0, 0, 0])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
