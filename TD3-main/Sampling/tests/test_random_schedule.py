#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试：随机采集调度的 7 类均匀性与运动配置映射（不依赖 ROS/Gazebo）。"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Sampling.random_dynamic_expert import (
    MAX_SHIP_SPEED,
    MOTION_CLASSES,
    STEP_OF_CLASS,
    configure_random_motion,
    pick_motion_class,
)


class _DummyController:
    """桩控制器：只记录调用，不依赖 rospy/Gazebo。"""

    def __init__(self):
        self.calls = []

    def set_mode_constant(self, *args, **kwargs):
        self.calls.append(("constant", args, kwargs))

    def set_mode_varspeed(self, *args, **kwargs):
        self.calls.append(("varspeed", args, kwargs))

    def set_mode_sine(self, *args, **kwargs):
        self.calls.append(("sine", args, kwargs))

    def set_mode_circle(self, *args, **kwargs):
        self.calls.append(("circle", args, kwargs))

    def set_mode_combined(self, *args, **kwargs):
        self.calls.append(("combined", args, kwargs))


class RandomScheduleTests(unittest.TestCase):
    def test_seven_classes_defined_and_mapped(self):
        """必须恰好 7 类，且每类都有兼容的 Step 标签。"""
        self.assertEqual(len(MOTION_CLASSES), 7)
        self.assertIn("static", MOTION_CLASSES)
        for cls in MOTION_CLASSES:
            self.assertIn(cls, STEP_OF_CLASS)

    def test_pick_motion_class_balances_saved_counts(self):
        """‘最缺补谁’策略应让 7 类已保存数始终相差不超过 1。"""
        counts = {cls: 0 for cls in MOTION_CLASSES}
        rng = np.random.RandomState(0)
        for _ in range(1000):
            cls = pick_motion_class(counts, rng)
            counts[cls] += 1
        values = list(counts.values())
        self.assertLessEqual(max(values) - min(values), 1)

    def test_configure_modes_for_all_classes(self):
        """7 类各自应生成预期的船运动模式。"""
        expected_mode = {
            "static": "near_static",
            "line_constant": "constant",
            "line_varspeed": "varspeed",
            "sine_constant": "sine",
            "sine_varspeed": "combined_sine",
            "circle_constant": "circle",
            "circle_varspeed": "combined_circle",
        }
        for cls in MOTION_CLASSES:
            for seed in (1, 2, 3):
                ctrl = _DummyController()
                scenario = configure_random_motion(ctrl, cls, seed)
                self.assertEqual(scenario["motion_class"], cls)
                self.assertEqual(scenario["step"], STEP_OF_CLASS[cls])
                self.assertEqual(scenario["mode"], expected_mode[cls])
                self.assertTrue(ctrl.calls)

    def test_speeds_stay_within_range(self):
        """速度与速度区间必须落在 [0, MAX_SHIP_SPEED]。"""
        for cls in (
            "line_constant",
            "line_varspeed",
            "sine_constant",
            "sine_varspeed",
            "circle_constant",
            "circle_varspeed",
        ):
            for seed in range(20):
                ctrl = _DummyController()
                scenario = configure_random_motion(ctrl, cls, seed)
                if scenario["mode"] in ("constant", "sine", "circle"):
                    self.assertGreaterEqual(scenario["speed"], 0.0)
                    self.assertLessEqual(scenario["speed"], MAX_SHIP_SPEED)
                elif scenario["mode"] in (
                    "varspeed",
                    "combined_sine",
                    "combined_circle",
                ):
                    self.assertLessEqual(scenario["speed_min"], scenario["speed_max"])
                    self.assertGreaterEqual(scenario["speed_min"], 0.0)
                    self.assertLessEqual(scenario["speed_max"], MAX_SHIP_SPEED)

    def test_static_uses_zero_velocity(self):
        """静止类必须下发零速度。"""
        ctrl = _DummyController()
        scenario = configure_random_motion(ctrl, "static", 7)
        self.assertEqual(scenario["speed"], 0.0)
        self.assertEqual(ctrl.calls[0][0], "constant")
        self.assertEqual(ctrl.calls[0][1][0], 0.0)
        self.assertEqual(ctrl.calls[0][1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
