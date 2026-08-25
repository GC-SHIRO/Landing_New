#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试十维策略 observation 的构造规则。"""

import unittest

import numpy as np

from Sampling.collect_global_expert import (
    VisualMotionObservation,
    parse_args,
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

    def test_initial_and_first_velocity_rules(self):
        """初始帧导数为零，第二帧只计算速度。"""
        builder = VisualMotionObservation(0.1)
        initial = builder.initialize([1.0, 2.0, 3.0], 0.8)
        second = builder.update([1.1, 1.8, 3.3], True, 0.9)
        self.assertEqual(initial.shape, (10,))
        np.testing.assert_array_equal(initial[3:9], np.zeros(6))
        np.testing.assert_allclose(second[:3], [1.1, 1.8, 3.3])
        np.testing.assert_allclose(second[3:6], [1.0, -2.0, 3.0], atol=1e-6)
        np.testing.assert_array_equal(second[6:9], np.zeros(3))
        self.assertAlmostEqual(float(second[9]), 0.9)

    def test_third_frame_calculates_acceleration(self):
        """第三个连续有效视觉帧开始计算加速度。"""
        builder = VisualMotionObservation(0.1)
        builder.initialize([0.0, 0.0, 0.0], 0.7)
        builder.update([0.1, 0.0, 0.0], True, 0.8)
        result = builder.update([0.3, 0.0, 0.0], True, 0.9)
        np.testing.assert_allclose(result[3:6], [2.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[6:9], [10.0, 0.0, 0.0], atol=1e-6)

    def test_lost_marker_holds_position_and_clears_motion(self):
        """失检不能写真值，且导数和置信度必须清零。"""
        builder = VisualMotionObservation(0.1)
        builder.initialize([1.0, 2.0, 3.0], 0.8)
        builder.update([1.1, 2.0, 3.0], True, 0.9)
        result = builder.update([0.0, 0.0, 10.0], False, 0.6)
        np.testing.assert_allclose(result[:3], [1.1, 2.0, 3.0])
        np.testing.assert_array_equal(result[3:9], np.zeros(6))
        self.assertEqual(float(result[9]), 0.0)

    def test_reacquisition_uses_new_position_without_false_acceleration(self):
        """重获首帧重新计算速度，但不跨失检段计算加速度。"""
        builder = VisualMotionObservation(0.1)
        builder.initialize([0.0, 0.0, 0.0], 0.8)
        builder.update([0.1, 0.0, 0.0], True, 0.9)
        builder.update([0.0, 0.0, 10.0], False, 0.0)
        result = builder.update([0.3, 0.0, 0.0], True, 0.95)
        np.testing.assert_allclose(result[:3], [0.3, 0.0, 0.0])
        np.testing.assert_allclose(result[3:6], [2.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_array_equal(result[6:9], np.zeros(3))
        self.assertAlmostEqual(float(result[9]), 0.95)


if __name__ == "__main__":
    unittest.main()
