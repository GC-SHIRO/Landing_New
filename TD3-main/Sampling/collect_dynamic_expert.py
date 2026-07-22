#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用特权 PD 专家自动采集 Step1-4 动态降落数据。"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Simulation.env_base import GazeboEnv, TIME_DELTA
from Sampling.privileged_pd_expert import (
    PrivilegedPDConfig,
    PrivilegedPDExpert,
    compute_transition_reward,
)
from Simulation.ship_motion import ShipMotionController


DEFAULT_LAUNCH = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"
SHIP_INIT_X = 10.0
SHIP_INIT_Y = 5.0
SHIP_INIT_Z = 0.1
MARKER_OFFSET_Z = 1.3
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="动态甲板特权 PD 专家数据采集")
    parser.add_argument("--step", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=TIME_DELTA)
    parser.add_argument("--launch", type=str, default=DEFAULT_LAUNCH)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="仅保存成功回合的训练数据",
    )
    parser.add_argument(
        "--raw_output",
        type=str,
        default=None,
        help="保存全部回合，供失败分析",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--v_min", type=float, default=0.3)
    parser.add_argument("--v_max", type=float, default=0.8)
    parser.add_argument("--randomize_step1_speed", action="store_true")
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--amp", type=float, default=2.0)
    parser.add_argument("--wavelen", type=float, default=8.0)
    parser.add_argument("--curve", choices=("sine", "circle"), default="sine")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--period", type=float, default=40.0)

    parser.add_argument("--kp_xy", type=float, default=0.85)
    parser.add_argument("--kd_xy", type=float, default=0.35)
    parser.add_argument("--precision_kp_xy", type=float, default=1.15)
    parser.add_argument("--precision_kd_xy", type=float, default=0.50)
    parser.add_argument("--precision_height", type=float, default=1.5)
    parser.add_argument("--kp_z", type=float, default=0.85)
    parser.add_argument("--approach_height", type=float, default=2.5)
    parser.add_argument("--tracking_height", type=float, default=1.2)
    parser.add_argument("--approach_gate_xy", type=float, default=1.0)
    parser.add_argument("--descent_gate_xy", type=float, default=0.25)
    parser.add_argument("--descent_gate_rel_speed", type=float, default=0.25)
    parser.add_argument("--flare_height", type=float, default=0.65)
    parser.add_argument("--touchdown_height", type=float, default=0.12)
    parser.add_argument("--max_xy_speed", type=float, default=1.0)
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--max_descent_speed", type=float, default=0.70)
    parser.add_argument("--max_climb_speed", type=float, default=0.60)
    parser.add_argument("--flare_descent_speed", type=float, default=0.22)
    parser.add_argument("--touchdown_descent_speed", type=float, default=0.10)
    parser.add_argument("--landing_xy_thresh", type=float, default=0.25)
    parser.add_argument("--visual_x_limit", type=float, default=20.0)
    parser.add_argument("--visual_y_limit", type=float, default=20.0)
    parser.add_argument("--yolo_lost_timeout", type=float, default=2.0)
    parser.add_argument("--max_dist", type=float, default=60.0)
    parser.add_argument("--max_height", type=float, default=12.0)
    parser.add_argument("--reset_settle", type=float, default=0.4)
    parser.add_argument("--roscore_wait", type=float, default=2.0)
    parser.add_argument("--gazebo_wait", type=float, default=10.0)
    parser.add_argument("--startup_warmup", type=float, default=3.0)
    parser.add_argument("--min_success_steps", type=int, default=15)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes 和 max_steps 必须大于 0")
    if abs(args.dt - TIME_DELTA) > 1e-9:
        parser.error(f"当前环境物理步长固定为 {TIME_DELTA}，请保持 --dt 一致")
    if args.v_min <= 0 or args.v_max < args.v_min:
        parser.error("要求 0 < v_min <= v_max")
    if args.vx is None:
        args.vx = 0.5 if args.step == 1 else 0.7
    if args.output is None:
        args.output = os.path.join(
            REPO_ROOT,
            "expert_data_dynamic",
            f"step{args.step}_privileged_pd.jsonl",
        )
    if args.raw_output is None:
        args.raw_output = os.path.join(
            REPO_ROOT,
            "expert_data_dynamic",
            f"step{args.step}_privileged_pd_all.jsonl",
        )
    if math.hypot(args.vx, args.vy) >= args.max_xy_speed:
        parser.error("船速必须小于 max_xy_speed，专家需要保留闭合误差的速度余量")


def build_expert(args: argparse.Namespace) -> PrivilegedPDExpert:
    config = PrivilegedPDConfig(
        kp_xy=args.kp_xy,
        kd_xy=args.kd_xy,
        precision_kp_xy=args.precision_kp_xy,
        precision_kd_xy=args.precision_kd_xy,
        precision_height=args.precision_height,
        approach_height=args.approach_height,
        tracking_height=args.tracking_height,
        approach_gate_xy=args.approach_gate_xy,
        descent_gate_xy=args.descent_gate_xy,
        descent_gate_rel_speed=args.descent_gate_rel_speed,
        flare_height=args.flare_height,
        touchdown_height=args.touchdown_height,
        kp_z=args.kp_z,
        max_climb_speed=args.max_climb_speed,
        max_descent_speed=args.max_descent_speed,
        flare_descent_speed=args.flare_descent_speed,
        touchdown_descent_speed=args.touchdown_descent_speed,
        max_xy_speed=args.max_xy_speed,
        max_action=args.max_action,
        body_z_down=False,
    )
    return PrivilegedPDExpert(config)


def configure_motion(
    controller: ShipMotionController,
    args: argparse.Namespace,
    episode_seed: int,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    vx = float(args.vx)
    vy = float(args.vy)
    if args.step == 1 and args.randomize_step1_speed:
        speed = float(rng.uniform(args.v_min, args.v_max))
        direction = np.array([vx, vy], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-9)
        vx, vy = float(direction[0] * speed), float(direction[1] * speed)

    if args.step == 1:
        controller.set_mode_constant(vx, vy)
        mode = "constant"
    elif args.step == 2:
        controller.set_mode_varspeed(
            (vx, vy), (args.v_min, args.v_max), episode_seed
        )
        mode = "varspeed"
    elif args.step == 3 and args.curve == "sine":
        controller.set_mode_sine(args.speed, args.amp, args.wavelen, vx, vy)
        mode = "sine"
    elif args.step == 3:
        controller.set_mode_circle(args.radius, args.period, vx, vy)
        mode = "circle"
    else:
        if args.curve == "sine":
            controller.set_mode_sine(args.speed, args.amp, args.wavelen, vx, vy)
        else:
            controller.set_mode_circle(args.radius, args.period, vx, vy)
        controller.set_mode_combined(
            args.curve, (args.v_min, args.v_max), episode_seed
        )
        mode = "combined_" + args.curve

    return {
        "step": int(args.step),
        "mode": mode,
        "vx": vx,
        "vy": vy,
        "v_min": float(args.v_min),
        "v_max": float(args.v_max),
        "seed": int(episode_seed),
    }


def current_truth(
    env: GazeboEnv,
    controller: ShipMotionController,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    if env.comm.current_position is None:
        raise RuntimeError("无人机位置真值不可用")
    drone_position = np.array(
        [
            env.comm.current_position.x,
            env.comm.current_position.y,
            env.comm.current_position.z,
        ],
        dtype=np.float64,
    )
    if env.drone_linear_velocity is None:
        drone_velocity = np.zeros(3, dtype=np.float64)
    else:
        drone_velocity = np.array(
            [
                env.drone_linear_velocity.x,
                env.drone_linear_velocity.y,
                env.drone_linear_velocity.z,
            ],
            dtype=np.float64,
        )
    target_position = np.asarray(
        controller.get_landing_target(MARKER_OFFSET_Z), dtype=np.float64
    )
    target_velocity = np.asarray(
        controller.get_landing_velocity(), dtype=np.float64
    )
    return (
        drone_position,
        drone_velocity,
        float(env.comm.current_yaw),
        target_position,
        target_velocity,
    )


def truth_metadata(
    drone_position: np.ndarray,
    drone_velocity: np.ndarray,
    drone_yaw: float,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> Dict[str, Any]:
    return {
        "drone_position": drone_position.astype(float).tolist(),
        "drone_velocity": drone_velocity.astype(float).tolist(),
        "drone_yaw": float(drone_yaw),
        "target_position": target_position.astype(float).tolist(),
        "target_velocity": target_velocity.astype(float).tolist(),
    }


def append_episode(path: str, episode: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(episode, ensure_ascii=False) + "\n")
        output_file.flush()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    expert = build_expert(args)
    rng = np.random.RandomState(args.seed)

    print("=" * 72)
    print(f"Step{args.step} 特权 PD 专家自动采集")
    print(f"成功数据: {args.output}")
    print(f"全部数据: {args.raw_output}")
    print(f"船初始速度: ({args.vx:.3f}, {args.vy:.3f}) m/s")
    print("训练观测: YOLO [marker_x, marker_y, marker_z]")
    print("YOLO: 启用")
    print("动作坐标: BODY_NED, z 向上为正")
    print("=" * 72)

    env = None
    controller = None
    success_count = 0
    raw_count = 0

    try:
        print("[启动 1/3] 启动 Gazebo、PX4 与 MAVROS...", flush=True)
        env = GazeboEnv(
            args.launch,
            VEHICLE_TYPE,
            VEHICLE_ID,
            max_dist=args.max_dist,
            max_height=args.max_height,
            landing_xy_threshold=args.landing_xy_thresh,
            visual_x_limit=args.visual_x_limit,
            visual_y_limit=args.visual_y_limit,
            yolo_lost_timeout=args.yolo_lost_timeout,
            enable_yolo=True,
            system_warmup_seconds=args.startup_warmup,
            roscore_wait_seconds=args.roscore_wait,
            gazebo_wait_seconds=args.gazebo_wait,
            configure_rc_loss_exception=False,
            mavros_state_timeout=30.0,
        )
        print("[启动 2/3] GazeboEnv 初始化完成", flush=True)
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(SHIP_INIT_X, SHIP_INIT_Y),
            init_z=SHIP_INIT_Z,
        )
        env.landing_target_fn = lambda: controller.get_landing_target(
            MARKER_OFFSET_Z
        )
        env.landing_velocity_fn = controller.get_landing_velocity

        if not controller.wait_for_odom(timeout=15.0):
            raise RuntimeError("未收到 WAM-V 模型状态")
        print("[启动 3/3] WAM-V 状态就绪，开始采集", flush=True)

        for episode_id in range(1, args.episodes + 1):
            episode_seed = int(args.seed + episode_id - 1)
            scenario = configure_motion(controller, args, episode_seed, rng)
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(args.reset_settle)
            env.pause()
            expert.reset()

            try:
                observation = np.asarray(
                    env.reset(), dtype=np.float32
                ).reshape(-1)
            except Exception as error:
                print(f"Ep {episode_id:04d}: RESET_FAILED: {error}")
                continue

            episode = []
            success = False
            terminal_reason = "RUNNING"
            phase_counts: Dict[str, int] = {}

            try:
                for step_index in range(args.max_steps):
                    sim_time = step_index * args.dt
                    controller.step(sim_time)
                    truth_before = current_truth(env, controller)
                    command = expert.compute_action(*truth_before)
                    phase_counts[command.phase] = phase_counts.get(command.phase, 0) + 1

                    previous_distance = float(
                        np.linalg.norm(truth_before[3] - truth_before[0])
                    )
                    next_observation, env_done, env_success, info = env.step(
                        command.action
                    )
                    next_observation = np.asarray(
                        next_observation, dtype=np.float32
                    ).reshape(-1)
                    truth_after = current_truth(env, controller)
                    current_distance = float(
                        np.linalg.norm(truth_after[3] - truth_after[0])
                    )
                    horizontal_error = float(
                        np.linalg.norm((truth_after[3] - truth_after[0])[:2])
                    )
                    relative_height = float(truth_after[0][2] - truth_after[3][2])
                    relative_xy_speed = float(
                        np.linalg.norm((truth_after[4] - truth_after[1])[:2])
                    )

                    reached_limit = step_index + 1 >= args.max_steps
                    done = bool(env_done or reached_limit)
                    success = bool(info.get("landing_success", env_success))
                    terminal_reason = str(info.get("terminal_reason", "RUNNING"))
                    if reached_limit and not env_done:
                        terminal_reason = "MAX_STEPS"

                    reward = compute_transition_reward(
                        previous_distance=previous_distance,
                        current_distance=current_distance,
                        horizontal_error=horizontal_error,
                        relative_height=relative_height,
                        relative_xy_speed=relative_xy_speed,
                        action=command.action,
                        success=success,
                        done=done,
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
                        "privileged_state": truth_metadata(*truth_before),
                        "next_privileged_state": truth_metadata(*truth_after),
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
                        },
                    }
                    episode.append(step_data)
                    observation = next_observation
                    if done:
                        break
            except Exception as error:
                terminal_reason = "EXCEPTION"
                if episode:
                    episode[-1]["done"] = True
                    episode[-1]["terminal_reason"] = terminal_reason
                    episode[-1]["error"] = str(error)
                print(f"Ep {episode_id:04d}: EXCEPTION: {error}")

            if episode:
                append_episode(args.raw_output, episode)
                raw_count += 1
            if success and len(episode) >= args.min_success_steps:
                append_episode(args.output, episode)
                success_count += 1
                saved = "SAVED"
            else:
                saved = "REJECTED"

            print(
                f"Ep {episode_id:04d}: {saved} success={success} "
                f"steps={len(episode):3d} reason={terminal_reason:24s} "
                f"phases={phase_counts} SR={success_count / episode_id * 100.0:5.1f}%"
            )

        print("=" * 72)
        print(f"采集完成: 成功 {success_count}/{args.episodes}")
        print(f"训练回合写入: {args.output}")
        print(f"原始回合写入: {args.raw_output} ({raw_count} episodes)")
        print("=" * 72)
    except KeyboardInterrupt:
        print("\n用户中断采集")
    finally:
        if controller is not None:
            controller.shutdown()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
