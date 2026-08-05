#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集覆盖 Step0-4 的分层随机动态甲板特权 PD 专家数据。

默认按 7 类运动（静止 / 直线 / 正弦 / 圆圈 × 匀速 / 变速）均匀采集，
直到保存满目标条数（默认 300）为止；每局速度随机。
"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling import collect_dynamic_expert as collector
from Sampling.privileged_pd_expert import PrivilegedPDConfig, PrivilegedPDExpert

if TYPE_CHECKING:
    from Simulation.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController

# 与 collect_dynamic_expert 保持一致：离线可测，运行期校验物理步长一致。
TIME_DELTA = 0.1

DEFAULT_LAUNCH = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
MAX_SHIP_SPEED = 1.0
MAX_EXPERT_SPEED = 1.0
NEAR_STATIC_SPEED = 0.05
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 7 类运动：静止 + 直线 / 正弦 / 圆圈 × 匀速 / 变速，默认均匀各占 1/7。
MOTION_CLASSES: Tuple[str, ...] = (
    "static",
    "line_constant",
    "line_varspeed",
    "sine_constant",
    "sine_varspeed",
    "circle_constant",
    "circle_varspeed",
)
# 兼容旧字段：记录该类别对应的 Step 标签（0-4）。
STEP_OF_CLASS: Dict[str, int] = {
    "static": 0,
    "line_constant": 1,
    "line_varspeed": 2,
    "sine_constant": 3,
    "sine_varspeed": 4,
    "circle_constant": 3,
    "circle_varspeed": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="7 类均匀随机采集动态甲板特权 PD 专家数据（直到保存满目标条数）"
    )
    parser.add_argument(
        "--target_saved",
        type=int,
        default=300,
        help="目标保存条数：直到保存满该数量的有效回合才停止",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="（已废弃）等价于 --target_saved，仅作兼容",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=1500,
        help="尝试次数安全上限，防止某类长期采集不到时无限运行",
    )
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
    if args.episodes is not None:
        args.target_saved = args.episodes
    if args.target_saved <= 0 or args.max_steps <= 0:
        parser.error("target_saved 和 max_steps 必须大于 0")
    if args.max_attempts < args.target_saved:
        parser.error("max_attempts 不能小于 target_saved")
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


def pick_motion_class(
    saved_counts: Dict[str, int], rng: np.random.RandomState
) -> str:
    """从“已保存最少”的类别中随机选一个作为下一局运动类别。

    只统计保存成功的回合；失败、被拒或异常都不计入，
    因此最终保存集会在 7 类之间自动趋于均匀（各约 1/7），
    同时避免前期某类扎堆的观感。
    """
    min_count = min(saved_counts.values())
    candidates = [
        cls for cls in MOTION_CLASSES if saved_counts[cls] == min_count
    ]
    return str(candidates[rng.randint(len(candidates))])


def _random_direction(rng: np.random.RandomState) -> Tuple[float, float, float]:
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    return heading, math.cos(heading), math.sin(heading)


def _random_speed(rng: np.random.RandomState) -> float:
    return float(rng.uniform(0.0, MAX_SHIP_SPEED))


def _random_speed_range(rng: np.random.RandomState) -> Tuple[float, float]:
    low = _random_speed(rng)
    high = float(rng.uniform(low, MAX_SHIP_SPEED))
    return low, high


def _curve_geometry(rng: np.random.RandomState, curve: str) -> Tuple[float, float]:
    """按给定曲线类型生成几何参数：sine -> (振幅, 波长)，circle -> (0, 半径)。"""
    if curve == "sine":
        return float(rng.uniform(0.8, 2.0)), float(rng.uniform(6.0, 12.0))
    return 0.0, float(rng.uniform(1.5, 3.0))


def configure_random_motion(
    controller: "ShipMotionController", motion_class: str, episode_seed: int
) -> Dict[str, Any]:
    """按 7 类运动生成本局船的运动配置，速度/几何参数随机。"""
    rng = np.random.RandomState(episode_seed)
    heading, direction_x, direction_y = _random_direction(rng)
    scenario: Dict[str, Any] = {
        "step": STEP_OF_CLASS[motion_class],
        "episode_seed": episode_seed,
        "heading_rad": heading,
        "motion_class": motion_class,
    }

    if motion_class == "static":
        controller.set_mode_constant(0.0, 0.0)
        scenario.update(
            mode="near_static",
            vx=0.0,
            vy=0.0,
            speed=0.0,
            near_static=True,
        )
        return scenario

    if motion_class == "line_constant":
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

    if motion_class == "line_varspeed":
        speed_min, speed_max = _random_speed_range(rng)
        controller.set_mode_varspeed(
            (direction_x, direction_y), (speed_min, speed_max), episode_seed
        )
        scenario.update(
            mode="varspeed",
            vx=direction_x,
            vy=direction_y,
            speed_min=speed_min,
            speed_max=speed_max,
            near_static=speed_max <= NEAR_STATIC_SPEED,
        )
        return scenario

    curve = "sine" if motion_class.startswith("sine") else "circle"
    amplitude, wavelength_or_radius = _curve_geometry(rng, curve)
    constant = motion_class.endswith("constant")
    if constant:
        speed = _random_speed(rng)
        _configure_curve(
            controller,
            curve,
            speed,
            amplitude,
            wavelength_or_radius,
            direction_x,
            direction_y,
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
            near_static=speed <= NEAR_STATIC_SPEED,
        )
        return scenario

    speed_min, speed_max = _random_speed_range(rng)
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
        near_static=speed_max <= NEAR_STATIC_SPEED,
    )
    return scenario


def _configure_curve(
    controller: "ShipMotionController",
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
    # 运行期懒导入 ROS/Gazebo 依赖：保证模块可被离线测试导入，
    # 同时脚本直接运行时也能拿到真实实现（与 collect_dynamic_expert 一致）。
    from Simulation.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController

    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if os.path.abspath(args.output) == os.path.abspath(args.raw_output):
        parser.error("--output 与 --raw_output 必须是不同文件")
    collector.reset_output_file(args.output)
    collector.reset_output_file(args.raw_output)
    expert = build_expert()

    print("=" * 72)
    print("随机 Step0-4 特权 PD 专家数据采集（7 类均匀：静止/直线/正弦/圆圈 × 匀速/变速）")
    print(f"目标保存: {args.target_saved} 条，尝试上限: {args.max_attempts}，"
          f"速度范围: 0-{MAX_SHIP_SPEED:.1f} m/s")
    print(f"成功数据: {args.output}")
    print(f"全部数据: {args.raw_output}")
    print("=" * 72)

    env = None
    controller = None
    success_count = 0
    raw_count = 0
    attempt_id = 0
    saved_counts = {cls: 0 for cls in MOTION_CLASSES}
    class_rng = np.random.RandomState(args.seed)
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

        while success_count < args.target_saved and attempt_id < args.max_attempts:
            attempt_id += 1
            motion_class = pick_motion_class(saved_counts, class_rng)
            episode_seed = args.seed + attempt_id - 1
            scenario = configure_random_motion(controller, motion_class, episode_seed)
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(args.reset_settle)
            env.pause()
            expert.reset()
            try:
                observation = np.asarray(env.reset(), dtype=np.float32).reshape(-1)
            except Exception as error:
                print(f"Ep {attempt_id:04d}: RESET_FAILED: {error}")
                continue

            episode = []
            training_episode = []
            reject_counts: Counter = Counter()
            fatal_quality = False
            fatal_reason = ""
            repeat_count = 0
            success = False
            terminal_reason = "RUNNING"
            episode_total_reward = 0.0
            try:
                for step_index in range(args.max_steps):
                    controller.step(step_index * args.dt)
                    truth_before = collector.current_truth(env, controller)
                    command = expert.compute_action(*truth_before)
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
                    # 终止成功判定与 env.step() 的动态落地判定一致：
                    # landing_success = deck_contact AND rel_xy<=landing_xy_threshold。
                    reward = collector.compute_transition_reward(
                        observation=observation,
                        done=done,
                        success=success,
                        world_x=float(truth_after[0][0]),
                        world_y=float(truth_after[0][1]),
                        relative_xy_distance=horizontal_error,
                        landing_xy_threshold=env.landing_xy_threshold,
                        next_observation=next_observation,
                    )
                    episode_total_reward += float(reward)
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
                        "episode_id": attempt_id,
                        "step_index": step_index,
                        "terminal_reason": terminal_reason,
                        "scenario": scenario,
                        "expert": command.to_metadata(),
                        "privileged_state": collector.truth_metadata(*truth_before),
                        "next_privileged_state": collector.truth_metadata(*truth_after),
                        "env_info": {
                            "deck_contact": bool(info.get("deck_contact", False)),
                            "landing_success": bool(success),
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
                print(f"Ep {attempt_id:04d}: EXCEPTION: {error}")

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
                saved_counts[motion_class] += 1
                saved = "SAVED"
            else:
                saved = "REJECTED"
                if save_reason != "ok":
                    quality_summary = f"{save_reason}|{quality_summary}"
            print(
                f"Ep {attempt_id:04d}({success_count}/{args.target_saved}): "
                f"{motion_class} | reward={episode_total_reward:.1f} | "
                f"success={success} | {saved} | "
                f"steps={len(episode)} | valid={len(training_episode)} | "
                f"reason={terminal_reason}"
            )
    except KeyboardInterrupt:
        print("\n用户中断采集")
    finally:
        if controller is not None:
            controller.shutdown()
        if env is not None:
            env.close()
        print(
            f"采集完成: 已保存 {success_count}/{args.target_saved} 条，"
            f"尝试 {attempt_id} 次，原始回合 {raw_count}"
        )
        if attempt_id >= args.max_attempts and success_count < args.target_saved:
            print("警告: 触达尝试上限仍未满目标，已保留现有数据")
        for cls in MOTION_CLASSES:
            print(f"  {cls:18s}: 保存 {saved_counts[cls]:3d}")


if __name__ == "__main__":
    main()
