#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling.privileged_pd_expert import (
    PrivilegedPDConfig,
    PrivilegedPDExpert,
    compute_transition_reward,
)


class PrivilegedPDExpertTest(unittest.TestCase):
    def setUp(self):
        self.expert = PrivilegedPDExpert()

    def test_aligned_command_contains_ship_feedforward(self):
        command = self.expert.compute_action(
            drone_position=(10.0, 5.0, 4.4),
            drone_velocity=(0.4, 0.0, 0.0),
            drone_yaw=0.0,
            target_position=(10.0, 5.0, 1.4),
            target_velocity=(0.4, 0.0, 0.0),
        )
        self.assertEqual(command.phase, "DESCEND")
        self.assertAlmostEqual(float(command.world_velocity[0]), 0.4, places=5)
        self.assertLess(float(command.action[2]), 0.0)

    def test_world_to_body_rotation(self):
        action = self.expert.world_enu_to_body_action(
            world_velocity=(1.0, 0.0, -0.2),
            drone_yaw=math.pi / 2.0,
            body_z_down=True,
        )
        np.testing.assert_allclose(action, (0.0, -1.0, 0.2), atol=1e-6)

    def test_privileged_observation_uses_body_relative_marker(self):
        observation = self.expert.build_privileged_observation(
            drone_position=(12.0, 5.0, 4.4),
            drone_yaw=math.pi / 2.0,
            target_position=(10.0, 5.0, 1.4),
        )
        np.testing.assert_allclose(observation, (0.0, 2.0, 3.0), atol=1e-6)

    def test_horizontal_action_is_bounded(self):
        command = self.expert.compute_action(
            drone_position=(0.0, 0.0, 8.0),
            drone_velocity=(0.0, 0.0, 0.0),
            drone_yaw=0.0,
            target_position=(20.0, 20.0, 1.4),
            target_velocity=(0.4, 0.0, 0.0),
        )
        self.assertLessEqual(
            float(np.linalg.norm(command.world_velocity[:2])),
            self.expert.config.max_xy_speed + 1e-6,
        )
        self.assertTrue(np.all(np.abs(command.action) <= 1.0 + 1e-6))

    def test_flare_is_slower_than_descent(self):
        descend = self.expert.compute_action(
            (10.0, 5.0, 3.0),
            (0.4, 0.0, 0.0),
            0.0,
            (10.0, 5.0, 1.4),
            (0.4, 0.0, 0.0),
        )
        flare = self.expert.compute_action(
            (10.0, 5.0, 1.9),
            (0.4, 0.0, 0.0),
            0.0,
            (10.0, 5.0, 1.4),
            (0.4, 0.0, 0.0),
        )
        self.assertEqual(descend.phase, "DESCEND")
        self.assertEqual(flare.phase, "FLARE")
        self.assertGreater(abs(float(descend.world_velocity[2])), abs(float(flare.world_velocity[2])))

    def test_success_reward_has_terminal_bonus(self):
        regular = compute_transition_reward(
            3.0, 2.8, 0.2, 0.5, 0.1, (0.2, 0.0, 0.1), False, False
        )
        success = compute_transition_reward(
            3.0, 2.8, 0.2, 0.5, 0.1, (0.2, 0.0, 0.1), True, True
        )
        self.assertAlmostEqual(success - regular, 100.0, places=5)

    def test_step1_point_mass_grid_converges(self):
        config = PrivilegedPDConfig(max_xy_speed=1.0, max_action=1.0)
        expert = PrivilegedPDExpert(config)
        initial_offsets = (-4.0, -2.0, 2.0, 4.0)
        initial_heights = (7.5, 8.5, 9.0)
        terminal_errors = []
        total = 0

        for offset_x in initial_offsets:
            for offset_y in initial_offsets:
                for initial_z in initial_heights:
                    total += 1
                    terminal_error = self._simulate_step1(
                        expert, offset_x, offset_y, initial_z
                    )
                    if terminal_error is not None:
                        terminal_errors.append(terminal_error)

        self.assertEqual(len(terminal_errors), total)
        self.assertLess(max(terminal_errors), 0.01)

    @staticmethod
    def _simulate_step1(expert, offset_x, offset_y, initial_z):
        dt = 0.1
        ship_position = np.array([10.0, 5.0, 1.4], dtype=np.float64)
        ship_velocity = np.array([0.4, 0.0, 0.0], dtype=np.float64)
        drone_position = np.array(
            [10.0 + offset_x, 5.0 + offset_y, initial_z], dtype=np.float64
        )
        drone_velocity = np.zeros(3, dtype=np.float64)

        for _ in range(600):
            command = expert.compute_action(
                drone_position,
                drone_velocity,
                0.0,
                ship_position,
                ship_velocity,
            )
            drone_velocity = command.world_velocity.astype(np.float64)
            drone_position += drone_velocity * dt
            ship_position += ship_velocity * dt
            relative = drone_position - ship_position
            if relative[2] <= 0.0 and np.linalg.norm(relative[:2]) <= 1.5:
                return float(np.linalg.norm(relative[:2]))
        return None


if __name__ == "__main__":
    unittest.main()
