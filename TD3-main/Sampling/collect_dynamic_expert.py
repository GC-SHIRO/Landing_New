#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用特权 PD 专家自动采集 Step1-4 动态降落数据。"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling.privileged_pd_expert import (
    PrivilegedPDConfig,
    PrivilegedPDExpert,
    compute_transition_reward,
)

if TYPE_CHECKING:
    from Simulation.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController

# Keep the quality-gate helpers importable without ROS/Gazebo.  The live
# collector still validates against the environment step size below.
TIME_DELTA = 0.1


DEFAULT_LAUNCH = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"
SHIP_INIT_X = 10.0
SHIP_INIT_Y = 5.0
SHIP_INIT_Z = 0.1
MARKER_OFFSET_X = -0.23
MARKER_OFFSET_Y = 0.0
MARKER_OFFSET_Z = 1.3
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MAX_SPEED = 1.0
# Early YOLO depth spikes often exceed 15 m; keep mid-air outliers droppable
# without rejecting an otherwise good landing trajectory.
MAX_VISUAL_DISTANCE = 30.0
MIN_VISUAL_DISTANCE = 0.25
MAX_CONSECUTIVE_REPEATS = 2
DEFAULT_NEAR_GROUND_HEIGHT = 0.65
DEFAULT_MIN_VALID_RATIO = 0.85
NEAR_GROUND_PHASES = frozenset({"FLARE", "TOUCHDOWN"})
FATAL_QUALITY_REASONS = frozenset({"shape", "non_finite", "action_limit"})


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
    parser.add_argument("--descent_gate_xy", type=float, default=0.30)
    parser.add_argument("--brake_gate_xy", type=float, default=0.60)
    parser.add_argument("--descent_gate_rel_speed", type=float, default=0.25)
    parser.add_argument("--stable_steps_required", type=int, default=2)
    parser.add_argument("--brake_xy_speed", type=float, default=0.50)
    parser.add_argument("--stable_xy_speed", type=float, default=0.20)
    parser.add_argument("--flare_height", type=float, default=0.65)
    parser.add_argument("--touchdown_height", type=float, default=0.12)
    parser.add_argument("--max_xy_speed", type=float, default=1.0)
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--max_descent_speed", type=float, default=0.50)
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
    parser.add_argument("--gazebo_wait", type=float, default=20.0)
    parser.add_argument("--startup_warmup", type=float, default=3.0)
    parser.add_argument("--min_success_steps", type=int, default=15)
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=DEFAULT_MIN_VALID_RATIO,
        help="成功回合中可训练 transition 的最低占比",
    )
    parser.add_argument(
        "--near_ground_height",
        type=float,
        default=DEFAULT_NEAR_GROUND_HEIGHT,
        help="相对甲板高度低于该值时，视觉失检/重复帧只裁剪不整回合否决",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes 和 max_steps 必须大于 0")
    if abs(args.dt - TIME_DELTA) > 1e-9:
        parser.error(f"当前环境物理步长固定为 {TIME_DELTA}，请保持 --dt 一致")
    if args.v_min <= 0 or args.v_max < args.v_min:
        parser.error("要求 0 < v_min <= v_max")
    if args.min_success_steps <= 0:
        parser.error("min_success_steps 必须大于 0")
    if not 0.0 < args.min_valid_ratio <= 1.0:
        parser.error("min_valid_ratio 必须落在 (0, 1]")
    if args.near_ground_height < 0.0:
        parser.error("near_ground_height 必须非负")
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
    speed_values = (args.v_max, args.speed, args.max_xy_speed, args.max_action,
                    args.max_descent_speed, args.max_climb_speed,
                    args.flare_descent_speed, args.touchdown_descent_speed)
    if any(value > MAX_SPEED for value in speed_values):
        parser.error("船和飞机的所有速度上限必须不超过 1.0 m/s")


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
        brake_gate_xy=args.brake_gate_xy,
        descent_gate_xy=args.descent_gate_xy,
        descent_gate_rel_speed=args.descent_gate_rel_speed,
        stable_steps_required=args.stable_steps_required,
        brake_xy_speed=args.brake_xy_speed,
        stable_xy_speed=args.stable_xy_speed,
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
    controller: "ShipMotionController",
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
    env: "GazeboEnv",
    controller: "ShipMotionController",
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
        controller.get_landing_target(
            marker_offset_z=MARKER_OFFSET_Z,
            marker_offset_x=MARKER_OFFSET_X,
            marker_offset_y=MARKER_OFFSET_Y,
        ),
        dtype=np.float64,
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


def reset_output_file(path: str) -> None:
    """Start a collection run from an empty file at the existing output path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def is_near_ground(
    info: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    relative_height: Optional[float] = None,
    near_ground_height: float = DEFAULT_NEAR_GROUND_HEIGHT,
) -> bool:
    """True when the aircraft is in the final flare/touchdown regime."""
    info = info or {}
    phase_name = str(phase or info.get("phase") or "").upper()
    if phase_name in NEAR_GROUND_PHASES:
        return True
    height = relative_height
    if height is None:
        height = info.get("relative_height", info.get("visual_height"))
    height_value = _as_float(height, default=float("inf"))
    if math.isfinite(height_value) and height_value <= float(near_ground_height):
        return True
    return bool(info.get("deck_contact", False) or info.get("landed", False))


def sample_is_usable(
    observation,
    next_observation,
    action,
    info,
    repeat_count,
    phase: Optional[str] = None,
    relative_height: Optional[float] = None,
    near_ground_height: float = DEFAULT_NEAR_GROUND_HEIGHT,
):
    """Return (accepted, updated_repeat_count, reason, fatal).

    Near-ground YOLO dropouts are expected during touchdown.  Those frames are
    dropped from the training trajectory, but they no longer poison an otherwise
    successful episode unless the failure is structural (shape/NaN/action).
    """
    info = info or {}
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    near_ground = is_near_ground(
        info=info,
        phase=phase,
        relative_height=relative_height,
        near_ground_height=near_ground_height,
    )

    if observation.shape != (3,) or next_observation.shape != (3,) or action.shape != (3,):
        return False, repeat_count, "shape", True
    if not (
        np.all(np.isfinite(observation))
        and np.all(np.isfinite(next_observation))
        and np.all(np.isfinite(action))
    ):
        return False, repeat_count, "non_finite", True
    if np.any(np.abs(action) > MAX_SPEED + 1e-6):
        return False, repeat_count, "action_limit", True

    # 触地成功终止帧（env 的 landing_success = deck_contact AND rel_xy<=threshold）：
    # 必须保留为训练终止转移。触地瞬间 YOLO 冻结/视觉退化是预期行为，若在此
    # 裁剪，成功回合的终末 step 会丢失 success 标志与 +300 终止奖励。
    # 结构性错误（shape/non_finite/action_limit）仍在上方 fatal 拒绝。
    if bool(info.get("landing_success", False)):
        return True, repeat_count, "", False

    detection_fresh = bool(info.get("detection_fresh", info.get("tag_detected", False)))
    visual_xy = np.abs(next_observation[:2])
    visual_z = float(next_observation[2])
    if not detection_fresh:
        reason = "near_ground_stale" if near_ground else "vision_invalid"
        return False, 0, reason, False
    if np.any(visual_xy > 20.0):
        reason = "near_ground_vision" if near_ground else "vision_invalid"
        return False, 0, reason, False
    if visual_z > MAX_VISUAL_DISTANCE:
        return False, 0, "vision_invalid", False
    if visual_z < MIN_VISUAL_DISTANCE and not near_ground:
        return False, 0, "vision_invalid", False

    unchanged = np.array_equal(observation, next_observation)
    repeat_count = repeat_count + 1 if unchanged else 0
    if repeat_count > MAX_CONSECUTIVE_REPEATS:
        reason = "near_ground_repeat" if near_ground else "repeated_vision"
        # Keep counting so a long freeze remains visible in diagnostics, but do
        # not treat the expected touchdown freeze as a fatal episode failure.
        return False, repeat_count, reason, False
    return True, repeat_count, "", False


def finalize_training_episode(training_episode: Sequence[Dict[str, Any]]) -> list:
    """Ensure a saved trajectory ends with done=True for offline loaders."""
    episode = [dict(step) for step in training_episode]
    if not episode:
        return episode
    episode[-1]["done"] = True
    return episode


def episode_should_save(
    success: bool,
    training_episode: Sequence[Dict[str, Any]],
    total_steps: int,
    fatal_quality: bool,
    min_success_steps: int,
    min_valid_ratio: float,
) -> Tuple[bool, str]:
    """Decide whether a successful raw episode becomes training data."""
    if not success:
        return False, "not_success"
    if fatal_quality:
        return False, "fatal_quality"
    valid_steps = len(training_episode)
    if valid_steps < int(min_success_steps):
        return False, "too_few_valid_steps"
    ratio_denom = max(int(total_steps), 1)
    valid_ratio = float(valid_steps) / float(ratio_denom)
    if valid_ratio < float(min_valid_ratio):
        return False, "low_valid_ratio"
    return True, "ok"


def summarize_quality(
    reject_counts: Counter,
    accepted_steps: int,
    total_steps: int,
    fatal_reason: str = "",
) -> str:
    if fatal_reason:
        return f"FATAL:{fatal_reason}"
    if total_steps <= 0:
        return "empty"
    if not reject_counts:
        return "OK"
    top_reason, top_count = reject_counts.most_common(1)[0]
    valid_ratio = float(accepted_steps) / float(max(total_steps, 1))
    return f"drop:{top_reason}x{top_count}@{valid_ratio:.0%}"


def main() -> None:
    # ROS/Gazebo imports stay inside main so offline quality-gate tests can
    # import this module without a full simulation stack.
    from Simulation.env_base import GazeboEnv, TIME_DELTA as ENV_TIME_DELTA
    from Simulation.ship_motion import ShipMotionController

    if abs(ENV_TIME_DELTA - TIME_DELTA) > 1e-9:
        raise RuntimeError(
            f"采集脚本 TIME_DELTA={TIME_DELTA} 与环境 {ENV_TIME_DELTA} 不一致"
        )

    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if os.path.abspath(args.output) == os.path.abspath(args.raw_output):
        parser.error("--output 与 --raw_output 必须是不同文件")
    # The requested collection replaces, rather than appends to, the current
    # dynamic expert corpus.  The filenames remain unchanged for training.
    reset_output_file(args.output)
    reset_output_file(args.raw_output)
    expert = build_expert(args)
    rng = np.random.RandomState(args.seed)

    print("=" * 72)
    print(f"Step{args.step} 特权 PD 专家自动采集")
    print(f"成功数据: {args.output}")
    print(f"全部数据: {args.raw_output}")
    print(f"船初始速度: ({args.vx:.3f}, {args.vy:.3f}) m/s")
    print("训练观测: YOLO [marker_x, marker_y, marker_z]")
    print("YOLO: 启用")
    print("动作坐标: BODY_NED, z 向上为正（与当前 PX4 接口实测一致）")
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
            max_speed=MAX_SPEED,
        )
        env.landing_target_fn = lambda: controller.get_landing_target(
            marker_offset_z=MARKER_OFFSET_Z,
            marker_offset_x=MARKER_OFFSET_X,
            marker_offset_y=MARKER_OFFSET_Y,
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
                    sim_time = step_index * args.dt
                    controller.step(sim_time)
                    truth_before = current_truth(env, controller)
                    command = expert.compute_action(*truth_before)
                    phase_counts[command.phase] = phase_counts.get(command.phase, 0) + 1

                    next_observation, env_done, env_success, info = env.step(
                        command.action
                    )
                    next_observation = np.asarray(
                        next_observation, dtype=np.float32
                    ).reshape(-1)
                    truth_after = current_truth(env, controller)
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

                    # 与 Simulation/env_base.step() 的动态落地判定对齐（本地复刻，不 import）：
                    # 稠密项用步进前视觉观测；终止项成功判定与 env 的
                    # landing_success = deck_contact AND rel_xy<=threshold 一致，
                    # 使用相对甲板水平误差（horizontal_error），而非旧静态世界框。
                    reward = compute_transition_reward(
                        observation=observation,
                        done=done,
                        success=success,
                        world_x=float(truth_after[0][0]),
                        world_y=float(truth_after[0][1]),
                        relative_xy_distance=horizontal_error,
                        landing_xy_threshold=env.landing_xy_threshold,
                        next_observation=next_observation,
                    )
                    quality_info = dict(info)
                    quality_info["phase"] = command.phase
                    quality_info["relative_height"] = relative_height
                    accepted, repeat_count, reason, fatal = sample_is_usable(
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
                        "privileged_state": truth_metadata(*truth_before),
                        "next_privileged_state": truth_metadata(*truth_after),
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
                    episode[-1]["done"] = True
                    episode[-1]["terminal_reason"] = terminal_reason
                    episode[-1]["error"] = str(error)
                print(f"Ep {episode_id:04d}: EXCEPTION: {error}")

            if episode:
                append_episode(args.raw_output, episode)
                raw_count += 1

            should_save, save_reason = episode_should_save(
                success=success,
                training_episode=training_episode,
                total_steps=len(episode),
                fatal_quality=fatal_quality,
                min_success_steps=args.min_success_steps,
                min_valid_ratio=args.min_valid_ratio,
            )
            quality_summary = summarize_quality(
                reject_counts,
                accepted_steps=len(training_episode),
                total_steps=len(episode),
                fatal_reason=fatal_reason,
            )
            if should_save:
                saved_episode = finalize_training_episode(training_episode)
                append_episode(args.output, saved_episode)
                success_count += 1
                saved = "SAVED"
            else:
                saved = "REJECTED"
                if save_reason != "ok":
                    quality_summary = f"{save_reason}|{quality_summary}"

            print(
                f"Ep {episode_id:04d}: {saved} success={success} "
                f"steps={len(episode):3d} valid={len(training_episode):3d} "
                f"reason={terminal_reason:24s} quality={quality_summary:28s} "
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
