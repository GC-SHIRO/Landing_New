#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动采集可由 TD3_offline.py 直接训练的动态降落专家数据。"""

import argparse
import json
import math
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Sampling.global_expert import ExpertConfig, GlobalLandingExpert

if TYPE_CHECKING:
    from Simulation.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController


# ==================== 采集参数：直接修改这里 ====================
TARGET_SAVED_EPISODES = 300
MAX_ATTEMPTS = 1500
MAX_STEPS = 600
TIME_DELTA = 0.1
RANDOM_SEED = 42
MIN_SUCCESS_STEPS = 15

# 视觉运动状态使用控制周期差分；限幅只抑制偶发检测跳变，不能替代 YOLO 标定。
MAX_VISUAL_SPEED = 10.0
MAX_VISUAL_ACCELERATION = 100.0
STATE_DIM = 10

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINING_OUTPUT = os.path.join(
    REPO_ROOT, "expert_data_dynamic", "global_expert.jsonl"
)
RAW_OUTPUT = os.path.join(
    REPO_ROOT, "expert_data_dynamic", "global_expert_raw.jsonl"
)

# False 表示向现有文件追加；开始新实验时可改为 True。
CLEAR_OUTPUT_ON_START = False

# --test 使用独立文件，只采集少量成功样本，不污染正式数据。
TEST_TARGET_SAVED_EPISODES = 2
TEST_MAX_ATTEMPTS = 5
TEST_MAX_STEPS = 600
TEST_TRAINING_OUTPUT = os.path.join(
    REPO_ROOT, "expert_data_dynamic", "global_expert_test.jsonl"
)
TEST_RAW_OUTPUT = os.path.join(
    REPO_ROOT, "expert_data_dynamic", "global_expert_test_raw.jsonl"
)


# ==================== 仿真参数 ====================
LAUNCH_FILE = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"
SHIP_INITIAL_X = 10.0
SHIP_INITIAL_Y = 5.0
SHIP_INITIAL_Z = 0.1
MARKER_OFFSET_X = -0.23
MARKER_OFFSET_Y = 0.0
MARKER_OFFSET_Z = 1.3
LANDING_XY_THRESHOLD = 0.30
MAX_WORLD_DISTANCE = 60.0
MAX_FLIGHT_HEIGHT = 12.0
RESET_SETTLE_SECONDS = 0.4
INITIAL_DETECTION_WAIT_SECONDS = 5.0
ROSCORE_WAIT_SECONDS = 2.0
GAZEBO_WAIT_SECONDS = 20.0
STARTUP_WARMUP_SECONDS = 3.0

# 允许 YOLO 短暂漏帧；超过该时间才进入 SEARCH。
YOLO_LOST_TIMEOUT = 0.5


# ==================== 专家参数 ====================
EXPERT_CONFIG = ExpertConfig(
    kp_xy=0.85,
    kd_xy=0.35,
    kp_z=0.85,
    align_xy_threshold=1.0,
    descend_xy_threshold=0.30,
    descend_vxy_threshold=0.25,
    stable_steps_required=2,
    tracking_height=2.0,
    flare_height=0.60,
    touchdown_height=0.15,
    near_ground_height=0.60,
    max_xy_speed=1.0,
    max_descent_speed=0.50,
    max_climb_speed=0.40,
    flare_descent_speed=0.20,
    touchdown_descent_speed=0.08,
    search_climb_speed=0.25,
    max_xy_delta=0.15,
    max_z_delta=0.10,
    max_action=1.0,
)


# ==================== 动态甲板参数 ====================
MAX_SHIP_SPEED = 0.80
MIN_MOVING_SPEED = 0.20
MOTION_CLASSES: Tuple[str, ...] = (
    "static",
    "line_constant",
    "line_variable",
    "sine_constant",
    "sine_variable",
    "circle_constant",
    "circle_variable",
)


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """只保留一个用于少量样本冒烟测试的命令行开关。"""
    parser = argparse.ArgumentParser(description="全局真值专家自动采集")
    parser.add_argument(
        "--test",
        action="store_true",
        help="采集 2 个成功回合到独立测试文件",
    )
    return parser.parse_args(arguments)


def runtime_settings(test_mode: bool) -> Dict[str, Any]:
    """根据是否为冒烟测试选择数量和输出文件。"""
    if test_mode:
        return {
            "target_saved": TEST_TARGET_SAVED_EPISODES,
            "max_attempts": TEST_MAX_ATTEMPTS,
            "max_steps": TEST_MAX_STEPS,
            "training_output": TEST_TRAINING_OUTPUT,
            "raw_output": TEST_RAW_OUTPUT,
            "clear_output": True,
        }
    return {
        "target_saved": TARGET_SAVED_EPISODES,
        "max_attempts": MAX_ATTEMPTS,
        "max_steps": MAX_STEPS,
        "training_output": TRAINING_OUTPUT,
        "raw_output": RAW_OUTPUT,
        "clear_output": CLEAR_OUTPUT_ON_START,
    }


def choose_motion_class(
    saved_counts: Dict[str, int], rng: np.random.RandomState
) -> str:
    """优先补齐已保存数量最少的运动类别。"""
    minimum = min(saved_counts.values())
    candidates = [name for name in MOTION_CLASSES if saved_counts[name] == minimum]
    return str(candidates[int(rng.randint(len(candidates)))])


def configure_motion(
    controller: "ShipMotionController", motion_class: str, episode_seed: int
) -> Dict[str, Any]:
    """为一局配置简单的随机甲板运动。"""
    rng = np.random.RandomState(episode_seed)
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    direction_x = math.cos(heading)
    direction_y = math.sin(heading)
    scenario: Dict[str, Any] = {
        "motion_class": motion_class,
        "episode_seed": int(episode_seed),
        "heading": heading,
    }

    if motion_class == "static":
        controller.set_mode_constant(0.0, 0.0)
        scenario.update(mode="constant", speed=0.0)
        return scenario

    if motion_class == "line_constant":
        speed = float(rng.uniform(MIN_MOVING_SPEED, MAX_SHIP_SPEED))
        controller.set_mode_constant(speed * direction_x, speed * direction_y)
        scenario.update(mode="constant", speed=speed)
        return scenario

    if motion_class == "line_variable":
        speed_min, speed_max = _random_speed_range(rng)
        controller.set_mode_varspeed(
            (direction_x, direction_y),
            (speed_min, speed_max),
            episode_seed,
        )
        scenario.update(
            mode="varspeed", speed_min=speed_min, speed_max=speed_max
        )
        return scenario

    curve = "sine" if motion_class.startswith("sine") else "circle"
    variable = motion_class.endswith("variable")
    speed = float(rng.uniform(0.30, MAX_SHIP_SPEED))

    if curve == "sine":
        amplitude = float(rng.uniform(0.8, 1.5))
        wavelength = float(rng.uniform(8.0, 12.0))
        controller.set_mode_sine(
            speed, amplitude, wavelength, direction_x, direction_y
        )
        scenario.update(
            mode="sine",
            speed=speed,
            amplitude=amplitude,
            wavelength=wavelength,
        )
    else:
        radius = float(rng.uniform(1.5, 2.5))
        period = 2.0 * math.pi * radius / speed
        controller.set_mode_circle(radius, period, direction_x, direction_y)
        scenario.update(
            mode="circle", speed=speed, radius=radius, period=period
        )

    if variable:
        speed_min, speed_max = _random_speed_range(rng)
        controller.set_mode_combined(curve, (speed_min, speed_max), episode_seed)
        scenario.update(
            mode=f"combined_{curve}",
            speed_min=speed_min,
            speed_max=speed_max,
        )
    return scenario


def _random_speed_range(rng: np.random.RandomState) -> Tuple[float, float]:
    speed_min = float(rng.uniform(MIN_MOVING_SPEED, 0.50))
    speed_max = float(rng.uniform(speed_min, MAX_SHIP_SPEED))
    return speed_min, speed_max


def current_truth(
    env: "GazeboEnv", controller: "ShipMotionController"
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """读取专家控制需要的无人机和甲板全局真值。"""
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


class VisualMotionObservation:
    """将连续 YOLO 位置转换为十维策略 observation。"""

    def __init__(self, time_delta: float) -> None:
        if not math.isfinite(time_delta) or time_delta <= 0.0:
            raise ValueError("视觉状态差分时间间隔必须为正的有限数值")
        self.time_delta = float(time_delta)
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.has_previous_velocity = False

    def initialize(
        self, position: Sequence[float], confidence: float
    ) -> np.ndarray:
        """每局第一帧只记录位置和置信度，导数统一置零。"""
        self.position = _vector3(position, "initial_position").astype(np.float32)
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.has_previous_velocity = False
        return self._compose(confidence)

    def update(
        self,
        raw_position: Sequence[float],
        marker_visible: bool,
        confidence: float,
    ) -> np.ndarray:
        """可见时差分；失检时保持位置并清零导数和置信度。"""
        if not marker_visible:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self.has_previous_velocity = False
            return self._compose(0.0)

        next_position = _vector3(raw_position, "raw_position").astype(np.float32)
        next_velocity = np.clip(
            (next_position - self.position) / self.time_delta,
            -MAX_VISUAL_SPEED,
            MAX_VISUAL_SPEED,
        ).astype(np.float32)
        if self.has_previous_velocity:
            next_acceleration = np.clip(
                (next_velocity - self.velocity) / self.time_delta,
                -MAX_VISUAL_ACCELERATION,
                MAX_VISUAL_ACCELERATION,
            ).astype(np.float32)
        else:
            # 第一条速度及失检重获后的首条速度没有前一有效速度可比较。
            next_acceleration = np.zeros(3, dtype=np.float32)

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        self.has_previous_velocity = True
        return self._compose(confidence)

    def _compose(self, confidence: float) -> np.ndarray:
        confidence = _confidence(confidence)
        return np.concatenate(
            (self.position, self.velocity, self.acceleration, [confidence])
        ).astype(np.float32)


def compute_reward(
    next_observation: Sequence[float], done: bool, success: bool
) -> float:
    """生成连续距离奖励，并保留真实终止奖励。"""
    if done:
        return 300.0 if success else -200.0
    observation = _state_vector(next_observation, "next_observation")
    distance = float(np.sum(np.abs(observation[:3]) ** 3) ** (1.0 / 3.0))
    return -0.1 * distance


def episode_is_trainable(episode: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    """只做进入训练文件前的必要结构检查。"""
    if len(episode) < MIN_SUCCESS_STEPS:
        return False, "回合长度不足"
    if not bool(episode[-1].get("success", False)):
        return False, "回合没有成功落地"
    for index, step in enumerate(episode):
        try:
            observation = _state_vector(step["observation"], "observation")
            next_observation = _state_vector(
                step["next_observation"], "next_observation"
            )
            action = _vector3(step["action"], "action")
            reward = float(step["reward"])
        except (KeyError, TypeError, ValueError) as error:
            return False, f"第 {index} 步字段错误: {error}"
        if not math.isfinite(reward):
            return False, f"第 {index} 步 reward 非有限"
        if np.any(np.abs(action) > 1.0 + 1e-6):
            return False, f"第 {index} 步 action 超出 [-1, 1]"
        expected_done = index == len(episode) - 1
        if bool(step["done"]) != expected_done:
            return False, f"第 {index} 步 done 位置错误"
        if index + 1 < len(episode):
            following = _state_vector(
                episode[index + 1]["observation"], "following_observation"
            )
            if not np.array_equal(next_observation, following):
                return False, f"第 {index} 步 next_observation 链断裂"
    return True, "ok"


def append_episode(path: str, episode: Sequence[Dict[str, Any]]) -> None:
    """每行追加一个完整 episode。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(list(episode), ensure_ascii=False) + "\n")
        output_file.flush()


def prepare_output_files(
    training_output: str, raw_output: str, clear_output: bool
) -> None:
    """按顶部开关决定清空或追加输出文件。"""
    if os.path.abspath(training_output) == os.path.abspath(raw_output):
        raise ValueError("TRAINING_OUTPUT 与 RAW_OUTPUT 不能相同")
    for path in (training_output, raw_output):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if clear_output:
            with open(path, "w", encoding="utf-8"):
                pass


def wait_for_initial_observation(
    env: "GazeboEnv", timeout_seconds: float
) -> Optional[np.ndarray]:
    """等待本局第一帧有效 marker，避免用默认值开始 episode。"""
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        has_detection = bool(getattr(env, "_episode_has_detection", False))
        last_detection = getattr(env, "last_yolo_detection_time", None)
        detection_age = float("inf")
        if last_detection is not None:
            try:
                now = last_detection.__class__.now()
                detection_age = float((now - last_detection).to_sec())
            except (AttributeError, TypeError, ValueError):
                detection_age = float("inf")
        if has_detection and detection_age <= YOLO_LOST_TIMEOUT:
            return _vector3(env.get_state(), "initial_observation").astype(np.float32)
        time.sleep(0.05)
    return None


def truth_metadata(
    truth: Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]
) -> Dict[str, Any]:
    drone_position, drone_velocity, drone_yaw, target_position, target_velocity = truth
    return {
        "drone_position": drone_position.astype(float).tolist(),
        "drone_velocity": drone_velocity.astype(float).tolist(),
        "drone_yaw": float(drone_yaw),
        "target_position": target_position.astype(float).tolist(),
        "target_velocity": target_velocity.astype(float).tolist(),
    }


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} 必须是 3 维，当前形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限值: {vector}")
    return vector


def _state_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} 必须是 {STATE_DIM} 维，当前形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限值: {vector}")
    return vector


def _confidence(value: Any) -> float:
    """将环境提供的可选置信度收敛到策略输入范围。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return float(np.clip(confidence, 0.0, 1.0))


def _finite_float_or_none(value: Any) -> Optional[float]:
    """metadata 中无效的可选数值写为 null，避免生成 NaN JSON。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> None:
    """启动仿真并持续采集，直到保存足够的成功回合。"""
    args = parse_args()
    settings = runtime_settings(args.test)
    target_saved = int(settings["target_saved"])
    max_attempts = int(settings["max_attempts"])
    max_steps = int(settings["max_steps"])
    training_output = str(settings["training_output"])
    raw_output = str(settings["raw_output"])

    # ROS/Gazebo 只在真正采集时导入，离线测试无需安装这些依赖。
    from Simulation.env_base import GazeboEnv, TIME_DELTA as ENV_TIME_DELTA
    from Simulation.ship_motion import ShipMotionController

    if abs(float(ENV_TIME_DELTA) - TIME_DELTA) > 1e-9:
        raise RuntimeError(
            f"采集步长 {TIME_DELTA} 与环境步长 {ENV_TIME_DELTA} 不一致"
        )

    np.random.seed(RANDOM_SEED)
    prepare_output_files(
        training_output, raw_output, bool(settings["clear_output"])
    )
    expert = GlobalLandingExpert(EXPERT_CONFIG)
    class_rng = np.random.RandomState(RANDOM_SEED)
    saved_counts = {name: 0 for name in MOTION_CLASSES}

    print("=" * 72)
    print("全局真值专家自动采集" + ("（冒烟测试）" if args.test else ""))
    print(f"目标成功回合数: {target_saved}")
    print(f"最大尝试次数: {max_attempts}")
    print(f"训练数据: {training_output}")
    print(f"原始数据: {raw_output}")
    print("训练 observation: YOLO 位置、速度、加速度和置信度（10维）")
    print("失检策略: 非近地保持水平 action 并用正 z 上升")
    print("=" * 72)

    env = None
    controller = None
    saved_count = 0
    attempt_id = 0

    try:
        env = GazeboEnv(
            LAUNCH_FILE,
            VEHICLE_TYPE,
            VEHICLE_ID,
            max_dist=MAX_WORLD_DISTANCE,
            max_height=MAX_FLIGHT_HEIGHT,
            landing_xy_threshold=LANDING_XY_THRESHOLD,
            visual_x_limit=20.0,
            visual_y_limit=20.0,
            yolo_lost_timeout=YOLO_LOST_TIMEOUT,
            enable_yolo=True,
            system_warmup_seconds=STARTUP_WARMUP_SECONDS,
            roscore_wait_seconds=ROSCORE_WAIT_SECONDS,
            gazebo_wait_seconds=GAZEBO_WAIT_SECONDS,
            configure_rc_loss_exception=False,
            mavros_state_timeout=30.0,
        )
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(SHIP_INITIAL_X, SHIP_INITIAL_Y),
            init_z=SHIP_INITIAL_Z,
            max_speed=MAX_SHIP_SPEED,
        )
        env.landing_target_fn = lambda: controller.get_landing_target(
            marker_offset_z=MARKER_OFFSET_Z,
            marker_offset_x=MARKER_OFFSET_X,
            marker_offset_y=MARKER_OFFSET_Y,
        )
        env.landing_velocity_fn = controller.get_landing_velocity
        if not controller.wait_for_odom(timeout=15.0):
            raise RuntimeError("未收到 WAM-V 模型状态")

        while saved_count < target_saved and attempt_id < max_attempts:
            attempt_id += 1
            motion_class = choose_motion_class(saved_counts, class_rng)
            episode_seed = RANDOM_SEED + attempt_id - 1
            scenario = configure_motion(controller, motion_class, episode_seed)
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(RESET_SETTLE_SECONDS)
            env.pause()
            expert.reset()

            try:
                env.reset()
                initial_position = wait_for_initial_observation(
                    env, INITIAL_DETECTION_WAIT_SECONDS
                )
            except Exception as error:
                print(f"第 {attempt_id:04d} 局重置失败: {error}")
                continue
            if initial_position is None:
                print(f"第 {attempt_id:04d} 局初始未看到 marker，重新开始")
                continue

            observation_builder = VisualMotionObservation(TIME_DELTA)
            observation = observation_builder.initialize(
                initial_position, getattr(env, "yolo_confidence", 0.0)
            )

            episode = []
            success = False
            terminal_reason = "RUNNING"
            marker_visible = True
            lost_steps = 0
            last_action = np.zeros(3, dtype=np.float32)

            try:
                for step_index in range(max_steps):
                    controller.step(step_index * TIME_DELTA)
                    truth_before = current_truth(env, controller)
                    command = expert.compute_action(
                        *truth_before,
                        marker_visible=marker_visible,
                        last_action=last_action,
                    )
                    if command.phase == "SEARCH":
                        lost_steps += 1
                    elif marker_visible:
                        lost_steps = 0

                    raw_next, env_done, env_success, info = env.step(command.action)
                    next_visible = bool(
                        info.get("detection_fresh", info.get("tag_detected", False))
                    )
                    next_observation = observation_builder.update(
                        raw_next,
                        next_visible,
                        info.get("yolo_confidence", 0.0),
                    )
                    truth_after = current_truth(env, controller)

                    reached_limit = step_index + 1 >= max_steps
                    done = bool(env_done or reached_limit)
                    success = bool(info.get("landing_success", env_success))
                    terminal_reason = str(info.get("terminal_reason", "RUNNING"))
                    if reached_limit and not env_done:
                        terminal_reason = "MAX_STEPS"

                    reward = compute_reward(next_observation, done, success)
                    relative_height = float(
                        truth_before[0][2] - truth_before[3][2]
                    )
                    step_data = {
                        "observation": observation.astype(float).tolist(),
                        "action": command.action.astype(float).tolist(),
                        "reward": float(reward),
                        "next_observation": next_observation.astype(float).tolist(),
                        "done": done,
                        "success": success,
                        "episode_id": attempt_id,
                        "step_index": step_index,
                        "collection_mode": "test" if args.test else "normal",
                        "terminal_reason": terminal_reason,
                        "scenario": scenario,
                        "expert": command.to_metadata(),
                        "privileged_state": truth_metadata(truth_before),
                        "next_privileged_state": truth_metadata(truth_after),
                        "env_info": {
                            "marker_visible": bool(marker_visible),
                            "next_marker_visible": bool(next_visible),
                            "observation_confidence": float(observation[-1]),
                            "next_observation_confidence": float(
                                next_observation[-1]
                            ),
                            "lost_steps": int(lost_steps),
                            "relative_height": relative_height,
                            "relative_xy_distance": _finite_float_or_none(
                                info.get("relative_xy_distance")
                            ),
                            "deck_contact": bool(info.get("deck_contact", False)),
                            "landing_success": bool(success),
                        },
                    }
                    episode.append(step_data)

                    observation = next_observation
                    marker_visible = next_visible
                    last_action = command.action.copy()
                    if done:
                        break
            except Exception as error:
                terminal_reason = "EXCEPTION"
                success = False
                if episode:
                    episode[-1]["done"] = True
                    episode[-1]["success"] = False
                    episode[-1]["terminal_reason"] = terminal_reason
                    episode[-1]["error"] = str(error)
                print(f"第 {attempt_id:04d} 局运行异常: {error}")

            if episode:
                append_episode(raw_output, episode)

            trainable, reason = episode_is_trainable(episode)
            if success and trainable:
                append_episode(training_output, episode)
                saved_count += 1
                saved_counts[motion_class] += 1
                result = "已保存"
            else:
                result = f"未保存（{reason}）"

            search_steps = sum(
                int((step.get("expert") or {}).get("phase") == "SEARCH")
                for step in episode
            )
            print(
                f"第 {attempt_id:04d} 局 | {motion_class:16s} | {result} | "
                f"成功={success} | 步数={len(episode):3d} | "
                f"SEARCH={search_steps:3d} | 终止={terminal_reason} | "
                f"进度={saved_count}/{target_saved}"
            )

        if saved_count < target_saved:
            print(
                f"达到最大尝试次数，只保存了 {saved_count}/"
                f"{target_saved} 个成功回合"
            )
        else:
            print(f"采集完成，共保存 {saved_count} 个成功回合")
        print(f"各运动类别数量: {saved_counts}")
    except KeyboardInterrupt:
        print("\n用户中断采集，已经写入的完整回合会保留")
    finally:
        if controller is not None:
            controller.shutdown()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
