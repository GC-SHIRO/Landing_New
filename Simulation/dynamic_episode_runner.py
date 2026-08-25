"""ROS-free episode runtime boundary for dynamic landing evaluation."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


MARKER_OFFSET_X = -0.20
MARKER_OFFSET_Y = 0.0
MARKER_OFFSET_Z = 1.3


class InfrastructureError(RuntimeError):
    """Raised when an episode cannot reach its first policy execution."""


@dataclass
class EpisodeRunResult:
    record: Dict[str, Any]
    trajectory_rows: List[Dict[str, Any]]


def configure_controller(controller, scenario, motion_seed):
    """Configure a ship controller for one approved experiment scenario."""
    params = scenario.parameters
    if scenario.step == 0:
        controller.set_mode_constant(0.0, 0.0)
    elif scenario.step == 1:
        controller.set_mode_constant(params["vx"], params["vy"])
    elif scenario.step == 2:
        controller.set_mode_varspeed(
            (1.0, 0.0),
            (params["speed_min"], params["speed_max"]),
            motion_seed,
        )
    elif scenario.step == 3:
        controller.set_mode_circle(params["radius"], params["period"], 1.0, 0.0)
    elif scenario.step == 4:
        controller.set_mode_circle(3.0, 40.0, 1.0, 0.0)
        controller.set_mode_combined(
            "circle",
            (params["speed_min"], params["speed_max"]),
            motion_seed,
        )
    else:
        raise ValueError("unsupported dynamic scenario step: {}".format(scenario.step))
    return scenario.motion


def bind_dynamic_landing(
    env,
    controller,
    landing_xy_threshold,
    visual_x_limit,
    visual_y_limit,
    yolo_lost_timeout,
):
    """Bind the environment landing detector to live ship truth callbacks."""
    env.landing_target_fn = lambda: controller.get_landing_target(
        marker_offset_z=MARKER_OFFSET_Z,
        marker_offset_x=MARKER_OFFSET_X,
        marker_offset_y=MARKER_OFFSET_Y,
    )
    env.landing_velocity_fn = controller.get_landing_velocity
    env.landing_xy_threshold = float(landing_xy_threshold)
    env.visual_x_limit = float(visual_x_limit)
    env.visual_y_limit = float(visual_y_limit)
    env.yolo_lost_timeout = float(yolo_lost_timeout)


def create_live_runtime(*args, **kwargs):
    """Construct ROS-backed runtime objects without imposing ROS on imports."""
    from Simulation.env.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController

    env_kwargs = kwargs.pop("env_kwargs", {})
    controller_kwargs = kwargs.pop("controller_kwargs", {})
    if kwargs:
        raise TypeError("unexpected runtime arguments: {}".format(sorted(kwargs)))
    env = GazeboEnv(*args, **env_kwargs)
    try:
        controller = ShipMotionController(**controller_kwargs)
    except Exception:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise
    return env, controller


def _seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _point_tuple(point):
    if point is None:
        return None
    return (float(point.x), float(point.y), float(point.z))


def _finite_vector(values, length=3):
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < length:
        return np.full(length, np.nan, dtype=np.float64)
    return vector[:length]


def _estimate_velocity(positions, dt):
    if len(positions) < 2 or dt <= 0:
        return np.full(3, np.nan, dtype=np.float64)
    return (np.asarray(positions[-1]) - np.asarray(positions[-2])) / dt


def _estimate_acceleration(positions, dt):
    if len(positions) < 3 or dt <= 0:
        return np.full(3, np.nan, dtype=np.float64)
    p2, p1, p0 = (np.asarray(position) for position in positions[-3:])
    return ((p0 - p1) - (p1 - p2)) / (dt * dt)


def _target_from_ship(ship_state, evaluator):
    pos = _finite_vector(ship_state["pos"], 2)
    yaw = float(ship_state.get("yaw", 0.0))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    offset_x = cos_yaw * evaluator.marker_offset_x - sin_yaw * evaluator.marker_offset_y
    offset_y = sin_yaw * evaluator.marker_offset_x + cos_yaw * evaluator.marker_offset_y
    return np.array(
        [
            pos[0] + offset_x,
            pos[1] + offset_y,
            float(ship_state["pos_z"]) + evaluator.marker_offset_z,
        ],
        dtype=np.float64,
    )


def run_episode(
    *,
    env,
    controller,
    agent,
    evaluator,
    scenario,
    episode_id,
    episode_seed,
    motion_seed,
    state_mean,
    state_std,
    seq_len,
    max_steps,
    dt,
    ckpt_dir="",
    load_step=0,
    settle_delay=0.3,
):
    """Run one deterministic policy attempt and return record plus trajectory."""
    from Simulation.landing_evaluation import EpisodeData

    try:
        configure_controller(controller, scenario, motion_seed)
    except Exception as exc:
        raise InfrastructureError("controller configuration failed: {}".format(exc)) from exc
    try:
        _seed_runtime(episode_seed)
    except Exception as exc:
        raise InfrastructureError("episode seeding failed: {}".format(exc)) from exc
    try:
        env.quadrant_idx = int(episode_id) % 4
        controller.teleport_to_origin()
        env.unpause()
        if settle_delay > 0:
            time.sleep(settle_delay)
        env.pause()
        observation = _finite_vector(env.reset())
        ship_state = controller.get_state()
        initial_drone_position = _point_tuple(env.comm.current_position)
        if initial_drone_position is None:
            raise RuntimeError("initial drone truth is unavailable")
        initial_target = _target_from_ship(ship_state, evaluator)
    except Exception as exc:
        raise InfrastructureError("episode startup failed: {}".format(exc)) from exc

    mean = _finite_vector(state_mean)
    std = _finite_vector(state_std)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise InfrastructureError("invalid normalization arrays")
    normalized = ((observation - mean) / (std + 1e-6)).astype(np.float32)
    states = deque((normalized.copy() for _ in range(seq_len)), maxlen=seq_len)

    actions = []
    positions = []
    truth_positions = [initial_drone_position]
    drone_velocities = []
    target_velocities = []
    trajectory_rows = []
    last_info = {}
    terminal_reason = "RUNNING"
    landing_success = False
    out_of_bounds = False
    visual_out_of_bounds = False
    lost_detection = False
    yolo_lost_steps = 0
    stale_observation_steps = 0
    crashed = False
    crash_message = ""
    policy_started = False
    cumulative_reward = 0.0
    cumulative_reward_available = callable(getattr(env, "reward_setup", None))

    try:
        for step_index in range(max_steps):
            controller.step(step_index * dt)
            sequence = np.asarray(states, dtype=np.float32)[np.newaxis, :, :]
            output = agent.choose_action(sequence, noise=0.0)
            policy_started = True
            action = output[0] if isinstance(output, tuple) else output
            action = _finite_vector(action).astype(np.float32)
            next_observation, done, success, info = env.step(action)
            if cumulative_reward_available:
                try:
                    reward = float(env.reward_setup(observation, next_observation, done, success))
                    if np.isfinite(reward):
                        cumulative_reward += reward
                    else:
                        cumulative_reward_available = False
                except Exception:
                    cumulative_reward_available = False
            observation = next_observation
            info = dict(info or {})
            last_info = info

            current_position = _point_tuple(env.comm.current_position)
            if current_position is not None:
                positions.append(current_position)
                truth_positions.append(current_position)
            ship_state = controller.get_state()
            target = _target_from_ship(ship_state, evaluator)
            ship_velocity = _finite_vector(
                list(_finite_vector(ship_state["vel"], 2)) + [ship_state.get("vel_z", 0.0)]
            )
            target_velocities.append(tuple(ship_velocity))
            drone_velocity = _estimate_velocity(truth_positions, dt)
            drone_velocities.append(tuple(drone_velocity))
            relative_velocity = drone_velocity - ship_velocity

            detection_fresh = bool(info.get("detection_fresh", info.get("tag_detected", False)))
            step_lost = bool(info.get("lost_detection", False))
            yolo_lost_steps += int(step_lost)
            stale_observation_steps += int(not detection_fresh)
            lost_detection = lost_detection or step_lost
            out_of_bounds = out_of_bounds or bool(info.get("out_of_bounds", False))
            visual_out_of_bounds = visual_out_of_bounds or bool(info.get("visual_out_of_bounds", False))
            landing_success = bool(info.get("landing_success", success))
            terminal_reason = str(info.get("terminal_reason", "RUNNING"))
            raw_next = _finite_vector(next_observation)
            normalized_next = ((raw_next - mean) / (std + 1e-6)).astype(np.float32)
            states.append(normalized_next)
            actions.append(action.tolist())

            drone_position = list(current_position) if current_position is not None else [float("nan")] * 3
            horizontal_error = float(np.linalg.norm(np.asarray(drone_position[:2]) - target[:2]))
            trajectory_rows.append(
                {
                    "scenario": scenario.name,
                    "episode_seed": episode_seed,
                    "step": step_index,
                    "sim_time": step_index * dt,
                    "observation": raw_next.tolist(),
                    "normalized_observation": normalized_next.tolist(),
                    "action": action.tolist(),
                    "drone_position": drone_position,
                    "drone_velocity": drone_velocity.tolist(),
                    "target_position": target.tolist(),
                    "target_velocity": ship_velocity.tolist(),
                    "horizontal_error": horizontal_error,
                    "relative_height": float(info.get("relative_height", np.nan)),
                    "relative_velocity": relative_velocity.tolist(),
                    "detection_fresh": detection_fresh,
                    "lost_detection": step_lost,
                    "done": bool(done),
                    "success": bool(success),
                    "terminal_reason": terminal_reason,
                }
            )
            if done:
                break
        else:
            terminal_reason = "MAX_STEPS"
    except Exception as exc:
        if not policy_started:
            raise InfrastructureError("policy startup failed: {}".format(exc)) from exc
        crashed = True
        crash_message = str(exc)

    step_count = len(actions)
    final_position = positions[-1] if positions else (float("nan"),) * 3
    try:
        ship_state = controller.get_state()
        target = _target_from_ship(ship_state, evaluator)
        ship_velocity = _finite_vector(
            list(_finite_vector(ship_state["vel"], 2)) + [ship_state.get("vel_z", 0.0)]
        )
        drone_velocity = _estimate_velocity(truth_positions, dt)
        drone_acceleration = _estimate_velocity(drone_velocities, dt)
        target_acceleration = _estimate_velocity(target_velocities, dt)
        relative_velocity = drone_velocity - ship_velocity
        relative_acceleration = drone_acceleration - target_acceleration
    except Exception as exc:
        crashed = True
        crash_message = "{}{}truth: {}".format(crash_message, " | " if crash_message else "", exc)
        target = np.full(3, np.nan)
        drone_velocity = np.full(3, np.nan)
        drone_acceleration = np.full(3, np.nan)
        relative_velocity = np.full(3, np.nan)
        relative_acceleration = np.full(3, np.nan)

    initial_distance = (
        float(np.linalg.norm(np.asarray(initial_drone_position) - initial_target))
        if np.all(np.isfinite(initial_target))
        else float("nan")
    )
    max_height = max((position[2] for position in positions), default=float("nan"))
    max_dist_origin = max((float(np.linalg.norm(position)) for position in positions), default=float("nan"))
    episode_data = EpisodeData(
        episode_id=episode_id,
        drone_final_x=final_position[0],
        drone_final_y=final_position[1],
        drone_final_z=final_position[2],
        target_x=target[0],
        target_y=target[1],
        target_z=target[2],
        rel_vx=relative_velocity[0],
        rel_vy=relative_velocity[1],
        rel_vz=relative_velocity[2],
        rel_ax=relative_acceleration[0],
        rel_ay=relative_acceleration[1],
        rel_az=relative_acceleration[2],
        impact_velocity_z=drone_velocity[2],
        steps=step_count,
        dt=dt,
        init_dist_3d=initial_distance,
        max_height=max_height,
        max_dist_origin=max_dist_origin,
        actions=actions,
        drone_positions=positions,
        crashed=crashed,
        crash_msg=crash_message,
        out_of_bounds=out_of_bounds,
        visual_out_of_bounds=visual_out_of_bounds,
        max_steps_reached=step_count >= max_steps and not landing_success,
        lost_detection=lost_detection,
        landing_success=landing_success,
        relative_height=float(last_info.get("relative_height", np.nan)),
        visual_height=float(last_info.get("visual_height", np.nan)),
        visual_x=float(last_info.get("visual_x", np.nan)),
        visual_y=float(last_info.get("visual_y", np.nan)),
        terminal_reason=terminal_reason,
        ckpt_dir=str(ckpt_dir),
        load_step=load_step,
        platform_type="dynamic",
    )
    record = evaluator.evaluate(episode_data)
    record.setdefault("ErrorMsg", crash_message)
    record.update(
        {
            "scenario": scenario.name,
            "Step": scenario.step,
            "motion": scenario.motion,
            "motion_parameters": dict(scenario.parameters),
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "motion_seed": motion_seed,
            "MaxSteps": max_steps,
            "cumulative_reward": cumulative_reward if cumulative_reward_available else None,
            "cumulative_reward_available": cumulative_reward_available,
            "yolo_lost_steps": yolo_lost_steps,
            "stale_observation_steps": stale_observation_steps,
            "valid_model_attempt": True,
            "infrastructure_error": False,
        }
    )
    return EpisodeRunResult(record=record, trajectory_rows=trajectory_rows)
