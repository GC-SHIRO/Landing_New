#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线校验：Sampling 奖励与 Simulation/env_base.reward_setup 公式一致。"""

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


def _env_base_formula(observation, done, success, world_x, world_y):
    """内联复刻 env_base.reward_setup 数学式，用于交叉核对。"""
    delta_x = observation[0]
    delta_y = observation[1]
    height2 = observation[2]
    shape2 = -((abs(delta_x) ** 3 + abs(delta_y) ** 3 + abs(height2) ** 3) ** (1 / 3))
    reward = 0.1 * shape2
    if done:
        if -1.5 > world_x > -2.5 and -1.5 > world_y > -2.5 and success:
            return 300.0
        return -200.0
    return float(reward)


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
        expected = _env_base_formula(obs, False, False, 0.0, 0.0)
        self.assertAlmostEqual(got, expected, places=10)
        self.assertLess(got, 0.0)

    def test_terminal_success_inside_box(self):
        obs = [0.1, 0.1, 0.5]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=True,
            world_x=-2.0,
            world_y=-2.0,
        )
        self.assertEqual(got, SIM_SUCCESS_REWARD)
        self.assertEqual(
            got,
            _env_base_formula(obs, True, True, -2.0, -2.0),
        )

    def test_terminal_success_outside_box_is_fail_reward(self):
        # 动态甲板成功落点常远离旧静态 pad 框，应按失败终端奖励处理。
        obs = [0.05, 0.02, 0.3]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=True,
            world_x=12.0,
            world_y=5.0,
        )
        self.assertEqual(got, SIM_FAIL_REWARD)
        self.assertEqual(
            got,
            _env_base_formula(obs, True, True, 12.0, 5.0),
        )

    def test_terminal_failure(self):
        obs = [1.0, 1.0, 3.0]
        got = compute_transition_reward(
            observation=obs,
            done=True,
            success=False,
            world_x=-2.0,
            world_y=-2.0,
        )
        self.assertEqual(got, SIM_FAIL_REWARD)

    def test_box_boundaries_are_exclusive(self):
        obs = [0.0, 0.0, 1.0]
        # 边界上：env_base 使用严格不等式，边界点不算成功框内。
        for world_xy in ((-2.5, -2.0), (-1.5, -2.0), (-2.0, -2.5), (-2.0, -1.5)):
            got = compute_transition_reward(
                observation=obs,
                done=True,
                success=True,
                world_x=world_xy[0],
                world_y=world_xy[1],
            )
            self.assertEqual(got, SIM_FAIL_REWARD, msg=world_xy)

    def test_next_observation_ignored_like_env_base(self):
        obs = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        a = compute_transition_reward(obs, False, False, 0.0, 0.0, next_observation=[9, 9, 9])
        b = compute_transition_reward(obs, False, False, 0.0, 0.0, next_observation=[0, 0, 0])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
