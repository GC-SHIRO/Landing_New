#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试旧三维 JSONL 转换脚本。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.convert_legacy_3d_dataset import convert_episode, convert_file_inplace


def _legacy_episode():
    return [
        {
            "observation": [0.0, 0.0, 3.0],
            "next_observation": [0.1, 0.0, 3.0],
            "action": [0.0, 0.0, 0.0],
            "reward": 0.0,
            "done": False,
            "env_info": {"marker_visible": True, "next_marker_visible": True},
        },
        {
            "observation": [0.1, 0.0, 3.0],
            "next_observation": [99.0, 99.0, 99.0],
            "action": [0.0, 0.0, 0.0],
            "reward": 0.0,
            "done": True,
            "env_info": {"marker_visible": True, "next_marker_visible": False},
        },
    ]


class LegacyDatasetConversionTests(unittest.TestCase):
    def test_episode_conversion_preserves_chain_and_loss_rule(self):
        """转换后必须保持十维链，失检状态使用零代理置信度。"""
        converted = convert_episode(_legacy_episode())
        self.assertEqual(len(converted), 2)
        for index, step in enumerate(converted):
            self.assertEqual(len(step["observation"]), 10)
            self.assertEqual(len(step["next_observation"]), 10)
            if index + 1 < len(converted):
                np.testing.assert_array_equal(
                    step["next_observation"], converted[index + 1]["observation"]
                )
        np.testing.assert_allclose(converted[0]["next_observation"][3:6], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            converted[1]["next_observation"][:3], [0.1, 0.0, 3.0], atol=1e-6
        )
        np.testing.assert_array_equal(converted[1]["next_observation"][3:9], np.zeros(6))
        self.assertEqual(converted[1]["next_observation"][9], 0.0)

    def test_file_conversion_replaces_source_and_keeps_backup(self):
        """就地转换必须更新原路径且保留未修改备份。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.jsonl"
            original = json.dumps(_legacy_episode(), ensure_ascii=False) + "\n"
            path.write_text(original, encoding="utf-8")
            backup = convert_file_inplace(path)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_text(encoding="utf-8"), original)
            converted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(converted[0]["observation"]), 10)


if __name__ == "__main__":
    unittest.main()
