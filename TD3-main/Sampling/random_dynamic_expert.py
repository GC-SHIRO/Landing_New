#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集覆盖 Step0-4 的分层随机动态甲板特权 PD 专家数据。"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling import collect_dynamic_expert as collector
from Sampling.privileged_pd_expert import PrivilegedPDConfig, PrivilegedPDExpert
from Simulation.env_base import GazeboEnv, TIME_DELTA
from Simulation.ship_motion import ShipMotionController


DEFAULT_LAUNCH = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
MAX_SHIP_SPEED = 1.0
MAX_EXPERT_SPEED = 1.0
NEAR_STATIC_SPEED = 0.05
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="随机采集 Step0-4 动态甲板特权 PD 专家数据"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=TIME_DELTA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--launch", type=str, default=DEFAULT_LAUNCH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--raw_output", type=str, default=None)
    parser.add_argument("--reset_settle", type=float, default=0.4)
    parser.add_argument("--roscore_wait", type=float, default=2.0)
    parser.add_argument("--gazebo_wait", type=float, default=20.0)
    parser.add_argument("--startup_warmup", type=float, default=3.0)
    parser.add_argument("--min_success_steps", type=int, default=15)
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=collector.DEFAULT_MIN_VALID_RATIO,
        help="成功回合中可训练 transition 的最低占比",
    )
    parser.add_argument(
        "--near_ground_height",
        type=float,
        default=collector.DEFAULT_NEAR_GROUND_HEIGHT,
        help="相对甲板高度低于该值时，视觉失检/重复帧只裁剪不整回合否决",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes 和 max_steps 必须大于 0")
    if abs(args.dt - TIME_DELTA) > 1e-9:
        parser.error(f"当前环境物理步长固定为 {TIME_DELTA}，请保持 --dt 一致")
    if args.min_success_steps <= 0:
        parser.error("min_success_steps 必须大于 0")
    if not 0.0 < args.min_valid_ratio <= 1.0:
        parser.error("min_valid_ratio 必须落在 (0, 1]")
    if args.near_ground_height < 0.0:
        parser.error("near_ground_height 必须非负")
    if args.output is None:
        args.output = os.path.join(
            REPO_ROOT, "expert_data_dynamic", "random_dynamic_privileged_pd.jsonl"
        )
    if args.raw_output is None:
        args.raw_output = os.path.join(
            REPO_ROOT,
            "expert_data_dynamic",
            "random_dynamic_privileged_pd_all.jsonl",
        )


def build_expert() -> PrivilegedPDExpert:
    return PrivilegedPDExpert(
        PrivilegedPDConfig(
            max_xy_speed=MAX_EXPERT_SPEED,
            max_action=MAX_EXPERT_SPEED,
            max_descent_speed=0.50,
            max_climb_speed=0.60,
            flare_descent_speed=0.22,
            touchdown_descent_speed=0.10,
            body_z_down=False,
        )
    )


def build_step_schedule(episodes: int, seed: int) -> List[int]:
    """均分 Step0-4，再确定性打乱以保证每类都被覆盖。"""
    base_count, remainder = divmod(episodes, 5)
    schedule = []
    for step in range(5):
        schedule.extend([step] * (base_count + int(step < remainder)))
    np.random.RandomState(seed).shuffle(schedule)
    return schedule


def _random_direction(rng: np.random.RandomState) -> Tuple[float, float, float]:
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    return heading, math.cos(heading), math.sin(heading)


def _random_speed(rng: np.random.RandomState) -> float:
    return float(rng.uniform(0.0, MAX_SHIP_SPEED))


def _random_speed_range(rng: np.random.RandomState) -> Tuple[float, float]:
    low = _random_speed(rng)
    high = float(rng.uniform(low, MAX_SHIP_SPEED))
    return low, high


def _curve_geometry(rng: np.random.RandomState) -> Tuple[str, float, float]:
    curve = "sine" if rng.rand() < 0.5 else "circle"
    if curve == "sine":
        return curve, float(rng.uniform(0.8, 2.0)), float(rng.uniform(6.0, 12.0))
    return curve, 0.0, float(rng.uniform(1.5, 3.0))


def configure_random_motion(
    controller: ShipMotionController, step: int, episode_seed: int
) -> Dict[str, Any]:
    rng = np.random.RandomState(episode_seed)
    heading, direction_x, direction_y = _random_direction(rng)
    scenario: Dict[str, Any] = {
        "step": step,
        "episode_seed": episode_seed,
        "heading_rad": heading,
    }

    if step == 0:
        speed = 0.0 if rng.rand() < 0.5 else float(rng.uniform(0.0, NEAR_STATIC_SPEED))
        controller.set_mode_constant(speed * direction_x, speed * direction_y)
        scenario.update(
            mode="near_static",
            vx=speed * direction_x,
            vy=speed * direction_y,
            speed=speed,
            near_static=True,
        )
        return scenario

    if step == 1:
        speed = _random_speed(rng)
        controller.set_mode_constant(speed * direction_x, speed * direction_y)
        scenario.update(
            mode="constant",
            vx=speed * direction_x,
            vy=speed * direction_y,
            speed=speed,
            near_static=speed <= NEAR_STATIC_SPEED,
        )
        return scenario

    if step == 2:
        speed_min, speed_max = _random_speed_range(rng)
        controller.set_mode_varspeed((direction_x, direction_y), (speed_min, speed_max), episode_seed)
        scenario.update(
            mode="varspeed",
            vx=direction_x,
            vy=direction_y,
            speed_min=speed_min,
            speed_max=speed_max,
            near_static=speed_max <= NEAR_STATIC_SPEED,
        )
        return scenario

    curve, amplitude, wavelength_or_radius = _curve_geometry(rng)
    if step == 3:
        speed = _random_speed(rng)
        if speed <= NEAR_STATIC_SPEED:
            controller.set_mode_constant(0.0, 0.0)
            scenario.update(
                mode="near_static",
                curve=curve,
                speed=speed,
                vx=0.0,
                vy=0.0,
                near_static=True,
            )
            return scenario
        _configure_curve(
            controller, curve, speed, amplitude, wavelength_or_radius, direction_x, direction_y
        )
        scenario.update(
            mode=curve,
            curve=curve,
            speed=speed,
            vx=direction_x,
            vy=direction_y,
            amplitude=amplitude if curve == "sine" else None,
            wavelength=wavelength_or_radius if curve == "sine" else None,
            radius=wavelength_or_radius if curve == "circle" else None,
            period=(2.0 * math.pi * wavelength_or_radius / speed)
            if curve == "circle"
            else None,
            near_static=False,
        )
        return scenario

    speed_min, speed_max = _random_speed_range(rng)
    if speed_max <= NEAR_STATIC_SPEED:
        controller.set_mode_constant(0.0, 0.0)
        scenario.update(
            mode="near_static",
            curve=curve,
            speed_min=speed_min,
            speed_max=speed_max,
            vx=0.0,
            vy=0.0,
            near_static=True,
        )
        return scenario
    _configure_curve(
        controller,
        curve,
        max(speed_max, NEAR_STATIC_SPEED),
        amplitude,
        wavelength_or_radius,
        direction_x,
        direction_y,
    )
    controller.set_mode_combined(curve, (speed_min, speed_max), episode_seed)
    scenario.update(
        mode=f"combined_{curve}",
        curve=curve,
        speed_min=speed_min,
        speed_max=speed_max,
        vx=direction_x,
        vy=direction_y,
        amplitude=amplitude if curve == "sine" else None,
        wavelength=wavelength_or_radius if curve == "sine" else None,
        radius=wavelength_or_radius if curve == "circle" else None,
        period=(2.0 * math.pi * wavelength_or_radius / speed_max)
        if curve == "circle"
        else None,
        near_static=False,
    )
    return scenario


def _configure_curve(
    controller: ShipMotionController,
    curve: str,
    speed: float,
    amplitude: float,
    wavelength_or_radius: float,
    direction_x: float,
    direction_y: float,
) -> None:
    if curve == "sine":
        controller.set_mode_sine(
            speed, amplitude, wavelength_or_radius, direction_x, direction_y
        )
        return
    radius = wavelength_or_radius
    period = 2.0 * math.pi * radius / speed
    controller.set_mode_circle(radius, period, direction_x, direction_y)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if os.path.abspath(args.output) == os.path.abspath(args.raw_output):
        parser.error("--output 与 --raw_output 必须是不同文件")
    collector.reset_output_file(args.output)
    collector.reset_output_file(args.raw_output)
    schedule = build_step_schedule(args.episodes, args.seed)
    expert = build_expert()

    print("=" * 72)
    print("随机 Step0-4 特权 PD 专家数据采集")
    print(f"回合数: {args.episodes}，速度范围: 0-{MAX_SHIP_SPEED:.1f} m/s")
    print(f"成功数据: {args.output}")
    print(f"全部数据: {args.raw_output}")
    print("=" * 72)

    env = None
    controller = None
    success_count = 0
    raw_count = 0
    try:
        env = GazeboEnv(
            args.launch, VEHICLE_TYPE, VEHICLE_ID,
            max_dist=60.0,
            max_height=12.0,
            landing_xy_threshold=0.25,
            visual_x_limit=20.0,
            visual_y_limit=20.0,
            yolo_lost_timeout=2.0,
            enable_yolo=True, system_warmup_seconds=args.startup_warmup,
            roscore_wait_seconds=args.roscore_wait, gazebo_wait_seconds=args.gazebo_wait,
            configure_rc_loss_exception=False, mavros_state_timeout=30.0,
        )
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(collector.SHIP_INIT_X, collector.SHIP_INIT_Y),
            init_z=collector.SHIP_INIT_Z,
            max_speed=MAX_SHIP_SPEED,
        )
        env.landing_target_fn = lambda: controller.get_landing_target(
            marker_offset_z=collector.MARKER_OFFSET_Z,
            marker_offset_x=collector.MARKER_OFFSET_X,
            marker_offset_y=collector.MARKER_OFFSET_Y,
        )
        env.landing_velocity_fn = controller.get_landing_velocity
        if not controller.wait_for_odom(timeout=15.0):
            raise RuntimeError("未收到 WAM-V 模型状态")

        for episode_id, step in enumerate(schedule, start=1):
            episode_seed = args.seed + episode_id - 1
            scenario = configure_random_motion(controller, step, episode_seed)
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(args.reset_settle)
            env.pause()
            expert.reset()
            try:
                observation = np.asarray(env.reset(), dtype=np.float32).reshape(-1)
            except Exception as error:
                print(f"Ep {episode_id:04d}: RESET_FAILED: {error}")
                continue

            episode = []
            training_episode = []
            reject_counts: Counter = Counter()
            fatal_quality = False
            fatal_reason = ""
            repeat_count = 0
            success = False
            terminal_reason = "RUNNING"
            phase_counts: Dict[str, int] = {}
            try:
                for step_index in range(args.max_steps):
                    controller.step(step_index * args.dt)
                    truth_before = collector.current_truth(env, controller)
                    command = expert.compute_action(*truth_before)
                    phase_counts[command.phase] = phase_counts.get(command.phase, 0) + 1
                    next_observation, env_done, env_success, info = env.step(command.action)
                    next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
                    truth_after = collector.current_truth(env, controller)
                    horizontal_error = float(np.linalg.norm((truth_after[3] - truth_after[0])[:2]))
                    relative_height = float(truth_after[0][2] - truth_after[3][2])
                    relative_xy_speed = float(np.linalg.norm((truth_after[4] - truth_after[1])[:2]))
                    reached_limit = step_index + 1 >= args.max_steps
                    done = bool(env_done or reached_limit)
                    success = bool(info.get("landing_success", env_success))
                    terminal_reason = str(info.get("terminal_reason", "RUNNING"))
                    if reached_limit and not env_done:
                        terminal_reason = "MAX_STEPS"
                    # 与 Simulation 对齐的本地奖励（不 import env_base）。
                    reward = collector.compute_transition_reward(
                        observation=observation,
                        done=done,
                        success=success,
                        world_x=float(truth_after[0][0]),
                        world_y=float(truth_after[0][1]),
                        next_observation=next_observation,
                    )
                    quality_info = dict(info)
                    quality_info["phase"] = command.phase
                    quality_info["relative_height"] = relative_height
                    accepted, repeat_count, reason, fatal = collector.sample_is_usable(
                        observation,
                        next_observation,
                        command.action,
                        quality_info,
                        repeat_count,
                        phase=command.phase,
                        relative_height=relative_height,
                        near_ground_height=args.near_ground_height,
                    )
                    step_data = {
                        "observation": observation.astype(float).tolist(),
                        "action": command.action.astype(float).tolist(),
                        "reward": reward,
                        "next_observation": next_observation.astype(float).tolist(),
                        "done": done,
                        "success": success,
                        "episode_id": episode_id,
                        "step_index": step_index,
                        "terminal_reason": terminal_reason,
                        "scenario": scenario,
                        "expert": command.to_metadata(),
                        "privileged_state": collector.truth_metadata(*truth_before),
                        "next_privileged_state": collector.truth_metadata(*truth_after),
                        "env_info": {
                            "deck_contact": bool(info.get("deck_contact", False)),
                            "relative_xy_distance": float(
                                info.get("relative_xy_distance", horizontal_error)
                            ),
                            "relative_height": float(
                                info.get("relative_height", relative_height)
                            ),
                            "relative_xy_speed": float(
                                info.get("rel_xy_speed", relative_xy_speed)
                            ),
                            "tag_detected": bool(info.get("tag_detected", False)),
                            "detection_fresh": bool(
                                info.get(
                                    "detection_fresh",
                                    info.get("tag_detected", False),
                                )
                            ),
                        },
                        "quality_accepted": bool(accepted),
                        "quality_reason": reason,
                        "quality_fatal": bool(fatal),
                    }
                    episode.append(step_data)
                    if accepted:
                        training_episode.append(step_data)
                    else:
                        reject_counts[reason or "unknown"] += 1
                        if fatal and not fatal_quality:
                            fatal_quality = True
                            fatal_reason = reason or "unknown"
                    observation = next_observation
                    if done:
                        break
            except Exception as error:
                terminal_reason = "EXCEPTION"
                if episode:
                    episode[-1].update(done=True, terminal_reason=terminal_reason, error=str(error))
                print(f"Ep {episode_id:04d}: EXCEPTION: {error}")

            if episode:
                collector.append_episode(args.raw_output, episode)
                raw_count += 1
            should_save, save_reason = collector.episode_should_save(
                success=success,
                training_episode=training_episode,
                total_steps=len(episode),
                fatal_quality=fatal_quality,
                min_success_steps=args.min_success_steps,
                min_valid_ratio=args.min_valid_ratio,
            )
            quality_summary = collector.summarize_quality(
                reject_counts,
                accepted_steps=len(training_episode),
                total_steps=len(episode),
                fatal_reason=fatal_reason,
            )
            if should_save:
                saved_episode = collector.finalize_training_episode(training_episode)
                collector.append_episode(args.output, saved_episode)
                success_count += 1
                saved = "SAVED"
            else:
                saved = "REJECTED"
                if save_reason != "ok":
                    quality_summary = f"{save_reason}|{quality_summary}"
            print(
                f"Ep {episode_id:04d}: Step{step} {scenario['mode']} {saved} "
                f"success={success} steps={len(episode):3d} valid={len(training_episode):3d} "
                f"reason={terminal_reason:24s} quality={quality_summary:28s} "
                f"phases={phase_counts} SR={success_count / episode_id * 100.0:5.1f}%"
            )
    except KeyboardInterrupt:
        print("\n用户中断采集")
    finally:
        if controller is not None:
            controller.shutdown()
        if env is not None:
            env.close()
        print(f"采集完成: 成功 {success_count}/{args.episodes}，原始回合 {raw_count}")


if __name__ == "__main__":
    main()
