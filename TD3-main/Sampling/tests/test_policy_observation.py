#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试策略 observation 的失检保持规则。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Sampling.collect_global_expert import (
    parse_args,
    policy_observation_after_step,
    runtime_settings,
)


class PolicyObservationTests(unittest.TestCase):
    def test_test_option_uses_small_independent_dataset(self):
        """--test 必须切换到少量样本和独立输出文件。"""
        self.assertFalse(parse_args([]).test)
        self.assertTrue(parse_args(["--test"]).test)
        normal = runtime_settings(False)
        smoke = runtime_settings(True)
        self.assertEqual(smoke["target_saved"], 2)
        self.assertEqual(smoke["max_attempts"], 5)
        self.assertTrue(smoke["clear_output"])
        self.assertNotEqual(smoke["training_output"], normal["training_output"])
        self.assertNotEqual(smoke["raw_output"], normal["raw_output"])

    def test_visible_marker_uses_new_observation(self):
        """marker 可见时必须使用环境返回的新视觉状态。"""
        current = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        raw_next = np.array([0.8, 1.7, 2.9], dtype=np.float32)
        result = policy_observation_after_step(current, raw_next, True)
        np.testing.assert_array_equal(result, raw_next)

    def test_lost_marker_holds_previous_observation(self):
        """marker 失检时不能写默认值或真值。"""
        current = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        raw_next = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        result = policy_observation_after_step(current, raw_next, False)
        np.testing.assert_array_equal(result, current)

    def test_reacquisition_immediately_uses_fresh_observation(self):
        """连续失检后重新检测到 marker 时立即恢复视觉状态。"""
        observation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        for _ in range(4):
            observation = policy_observation_after_step(
                observation, [0.0, 0.0, 10.0], False
            )
        fresh = np.array([-0.2, 0.1, 3.5], dtype=np.float32)
        observation = policy_observation_after_step(observation, fresh, True)
        np.testing.assert_array_equal(observation, fresh)


if __name__ == "__main__":
    unittest.main()
