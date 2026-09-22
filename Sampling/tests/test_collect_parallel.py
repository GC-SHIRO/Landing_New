#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试并行采集的 worker 配置、早停判断和输出合并。"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from Sampling import collect_parallel as parallel


class WorkerSettingsTests(unittest.TestCase):
    def test_default_workers_are_fully_isolated(self):
        """默认全部 worker 的端口、种子、飞行器和输出均不能重叠。"""
        settings = [parallel.worker_settings(worker, False) for worker in range(parallel.NUM_WORKERS)]
        for key in ("ros_port", "gazebo_port", "display", "seed", "vehicle_id", "training_output", "raw_output"):
            self.assertEqual(len({item[key] for item in settings}), parallel.NUM_WORKERS)
        self.assertEqual(sum(item["target_saved"] for item in settings), parallel.TARGET_SAVED_EPISODES)

    def test_ports_seeds_and_outputs_are_distinct(self):
        """每个 worker 的端口、种子、vehicle_id 和输出文件互不相同。"""
        first = parallel.worker_settings(0, False, num_workers=2)
        second = parallel.worker_settings(1, False, num_workers=2)
        self.assertEqual(second["ros_port"], first["ros_port"] + 1)
        self.assertEqual(second["gazebo_port"], first["gazebo_port"] + 1)
        self.assertNotEqual(first["display"], second["display"])
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertNotEqual(first["training_output"], second["training_output"])
        self.assertNotEqual(first["training_output"], first["raw_output"])
        self.assertEqual(second["vehicle_id"], "1")
        self.assertIn("ID:=1", second["launch_args"])
        self.assertIn("gui:=false", second["launch_args"])
        self.assertIn(
            f"python_executable:={parallel.YOLO_PYTHON_EXECUTABLE}",
            second["yolo_launch_args"],
        )
        environment = parallel.worker_environment(second)
        self.assertTrue(environment["ROS_MASTER_URI"].endswith(str(second["ros_port"])))
        self.assertTrue(environment["GAZEBO_MASTER_URI"].endswith(str(second["gazebo_port"])))
        self.assertEqual(environment["DISPLAY"], second["display"])
        self.assertEqual(environment["LIBGL_ALWAYS_SOFTWARE"], "1")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")

    def test_xvfb_command_uses_worker_display(self):
        """每个 worker 的 Xvfb 命令必须绑定其独立显示器且禁止网络监听。"""
        settings = parallel.worker_settings(1, False, num_workers=2)
        command = parallel.xvfb_command(settings["display"])
        self.assertEqual(command[:2], ["Xvfb", settings["display"]])
        self.assertIn(parallel.XVFB_SCREEN, command)
        self.assertIn("-nolisten", command)

    def test_yolo_ready_wait_covers_model_startup(self):
        """视觉流就绪等待必须长于单局初始检测等待，避免后启动 worker 空耗尝试。"""
        self.assertGreater(
            parallel.YOLO_READY_TIMEOUT_SECONDS,
            parallel.base.INITIAL_DETECTION_WAIT_SECONDS,
        )
        self.assertGreater(
            parallel.PARALLEL_INITIAL_DETECTION_WAIT_SECONDS,
            parallel.base.INITIAL_DETECTION_WAIT_SECONDS,
        )

    def test_targets_split_evenly(self):
        """总目标平均分配，余数分给靠前的 worker，总和不变。"""
        self.assertEqual(parallel.split_evenly(300, 2), [150, 150])
        self.assertEqual(parallel.split_evenly(301, 2), [151, 150])
        with patch.object(parallel, "TARGET_SAVED_EPISODES", 7), patch.object(parallel, "MAX_ATTEMPTS", 10):
            targets = [parallel.worker_settings(k, False, num_workers=3)["target_saved"] for k in range(3)]
            attempts = [parallel.worker_settings(k, False, num_workers=3)["max_attempts"] for k in range(3)]
        self.assertEqual(targets, [3, 2, 2])
        self.assertEqual(attempts, [4, 3, 3])

    def test_test_mode_uses_separate_files(self):
        """冒烟测试数据独立，且各 worker 使用同一已验证的起点序列。"""
        normal = parallel.worker_settings(0, False)
        test = parallel.worker_settings(0, True)
        other_test = parallel.worker_settings(1, True, num_workers=2)
        self.assertNotEqual(normal["training_output"], test["training_output"])
        self.assertEqual(test["target_saved"], parallel.TEST_TARGET_PER_WORKER)
        self.assertEqual(test["seed"], other_test["seed"])


class EarlyAbortTests(unittest.TestCase):
    def test_long_search_streak_aborts(self):
        """连续 SEARCH 超过阈值后才触发，中间恢复检测会清零计数。"""
        monitor = parallel.EarlyAbort(max_search_streak=3, stall_window_steps=100, min_descent_progress=0.3)
        for _ in range(3):
            self.assertIsNone(monitor.update("SEARCH", 5.0))
        self.assertIsNone(monitor.update("TRACK", 5.0))
        for _ in range(3):
            self.assertIsNone(monitor.update("SEARCH", 5.0))
        self.assertEqual(monitor.update("SEARCH", 5.0), "EARLY_ABORT_SEARCH")

    def test_stall_without_descent_aborts(self):
        """窗口内高度没有下降足够距离时判定停滞；持续下降则不触发。"""
        monitor = parallel.EarlyAbort(max_search_streak=40, stall_window_steps=10, min_descent_progress=0.3)
        # 第 1 步记录基准高度，此后整整 10 步没有下降 0.3 米即触发。
        for _ in range(10):
            self.assertIsNone(monitor.update("ALIGN", 8.0))
        self.assertEqual(monitor.update("ALIGN", 7.9), "EARLY_ABORT_STALL")

        monitor.reset()
        height = 8.0
        for _ in range(50):
            height -= 0.05
            self.assertIsNone(monitor.update("DESCEND", height))

    def test_reset_clears_state(self):
        monitor = parallel.EarlyAbort(max_search_streak=2, stall_window_steps=5, min_descent_progress=0.3)
        monitor.update("SEARCH", 5.0)
        monitor.update("SEARCH", 5.0)
        monitor.reset()
        self.assertIsNone(monitor.update("SEARCH", 5.0))
        self.assertIsNone(monitor.update("SEARCH", 5.0))
        self.assertEqual(monitor.update("SEARCH", 5.0), "EARLY_ABORT_SEARCH")


class MergeTests(unittest.TestCase):
    def test_merge_appends_and_removes_sources(self):
        """合并保持每行一个回合，跳过缺失文件，合并后删除来源文件。"""
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "merged.jsonl")
            with open(destination, "w", encoding="utf-8") as output_file:
                output_file.write(json.dumps([{"episode": "existing"}]) + "\n")
            sources = []
            for index in range(2):
                path = os.path.join(directory, f"worker{index}.jsonl")
                with open(path, "w", encoding="utf-8") as output_file:
                    output_file.write(json.dumps([{"episode": index}]) + "\n")
                sources.append(path)
            sources.append(os.path.join(directory, "missing.jsonl"))
            merged = parallel.merge_worker_outputs(sources, destination)
            self.assertEqual(merged, 2)
            with open(destination, "r", encoding="utf-8") as input_file:
                episodes = [json.loads(line) for line in input_file]
            self.assertEqual([item[0]["episode"] for item in episodes], ["existing", 0, 1])
            self.assertFalse(any(os.path.exists(path) for path in sources[:2]))

    def test_merge_refuses_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "same.jsonl")
            open(path, "w", encoding="utf-8").close()
            with self.assertRaisesRegex(ValueError, "不能与合并目标相同"):
                parallel.merge_worker_outputs([path], path)


if __name__ == "__main__":
    unittest.main()
