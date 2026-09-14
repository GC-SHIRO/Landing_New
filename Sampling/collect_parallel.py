#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多实例并行、无界面的全局真值专家采集，并提前终止注定失败的回合。

每个 worker 拥有独立的 ROS master、Gazebo、PX4 和 YOLO 进程，运行与
``collect_global_expert`` 相同的专家采集循环；主进程负责错开启动 worker、
等待全部结束，再把各 worker 的输出合并到正式文件。运动类别、专家参数、
observation 构造和数据格式全部复用 ``collect_global_expert``。
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from Sampling import collect_global_expert as base
from Sampling.global_expert import GlobalLandingExpert

if TYPE_CHECKING:
    from Simulation.env.env_base import GazeboEnv
    from Simulation.ship_motion import ShipMotionController


# ==================== 并行参数 ====================
NUM_WORKERS = 2
BASE_ROS_PORT = 11311            # worker k 的 ROS master 端口为 BASE_ROS_PORT + k
BASE_GAZEBO_PORT = 11345         # worker k 的 Gazebo master 端口为 BASE_GAZEBO_PORT + k
# 传给 roslaunch 的参数，{worker} 会替换为 worker 编号。launch 文件必须声明
# ID 和 gui 两个 arg，并按 ID 选择端口互不冲突的 PX4/MAVROS 配置（XTDrone 多机写法）；
# gui:=false 只启动 gzserver，不开 gzclient 界面。
WORKER_LAUNCH_ARGS = ("ID:={worker}", "gui:=false")
# YOLO launch 需要按 ID 订阅对应无人机的相机话题；若 yolo_v11.launch 不接受该参数，
# 请改为它实际支持的写法或置空。
WORKER_YOLO_LAUNCH_ARGS = ("ID:={worker}",)
WORKER_VEHICLE_ID = "{worker}"   # 与 launch 里 ID 对应的 MAVROS 命名空间后缀，如 iris_1
WORKER_START_STAGGER_SECONDS = 30.0   # 错开启动，避免多个 Gazebo 和 YOLO 同时加载
WORKER_SEED_STRIDE = 100_000     # worker k 的随机种子为 RANDOM_SEED + k * stride
STATUS_INTERVAL_SECONDS = 60.0   # 主进程打印各 worker 进度的间隔

# ==================== 早停参数 ====================
# 连续 SEARCH 超过该步数说明专家已经找不回 marker，直接放弃本局。
MAX_SEARCH_STREAK = 40
# 相对高度在该窗口内下降不足 MIN_DESCENT_PROGRESS 米视为停滞。窗口要覆盖
# 对准阶段悬停不下降的时间（最大 4 米偏移、1 m/s 限速加稳定判定），不能太短。
STALL_WINDOW_STEPS = 150
MIN_DESCENT_PROGRESS = 0.3

# ==================== 采集参数 ====================
TARGET_SAVED_EPISODES = 300      # 全部 worker 合计目标，平均分配
MAX_ATTEMPTS = 1500              # 全部 worker 合计上限，平均分配
MAX_STEPS = base.MAX_STEPS
RANDOM_SEED = base.RANDOM_SEED
TRAINING_OUTPUT = base.TRAINING_OUTPUT
RAW_OUTPUT = base.RAW_OUTPUT
CLEAR_OUTPUT_ON_START = False
# 各 worker 的独立输出目录；合并后 worker 文件会被删除，日志保留。
WORKER_DIR = os.path.join(base.EXPERT_GLOBAL_DIR, "workers")

# --test：每个 worker 只保存 1 个成功回合，最多尝试 3 局，写入测试文件。
TEST_TARGET_PER_WORKER = 1
TEST_MAX_ATTEMPTS_PER_WORKER = 3


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """--test 为冒烟测试；--worker 由主进程内部使用；--merge 只合并已有 worker 文件。"""
    parser = argparse.ArgumentParser(description="多实例并行全局真值专家采集")
    parser.add_argument("--test", action="store_true", help="每个 worker 采集 1 个成功回合到测试文件")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--merge", action="store_true", help="只合并上次运行遗留的 worker 输出，不启动仿真")
    return parser.parse_args(arguments)


def split_evenly(total: int, parts: int) -> List[int]:
    """把总数尽量平均分给各 worker，前面的 worker 多分 1。"""
    if parts <= 0:
        raise ValueError("worker 数量必须为正数")
    quotient, remainder = divmod(int(total), parts)
    return [quotient + (1 if index < remainder else 0) for index in range(parts)]


def worker_output_paths(worker: int, test_mode: bool) -> Dict[str, str]:
    prefix = "test_" if test_mode else ""
    return {
        "training_output": os.path.join(WORKER_DIR, f"{prefix}worker{worker}.jsonl"),
        "raw_output": os.path.join(WORKER_DIR, f"{prefix}worker{worker}_raw.jsonl"),
        "log": os.path.join(WORKER_DIR, f"{prefix}worker{worker}.log"),
    }


def worker_settings(worker: int, test_mode: bool, num_workers: int = NUM_WORKERS) -> Dict[str, Any]:
    """一个 worker 的端口、launch 参数、随机种子、目标数量和输出文件。"""
    if not 0 <= worker < num_workers:
        raise ValueError(f"worker 编号必须在 0 到 {num_workers - 1} 之间")
    if test_mode:
        target, attempts = TEST_TARGET_PER_WORKER, TEST_MAX_ATTEMPTS_PER_WORKER
    else:
        target = split_evenly(TARGET_SAVED_EPISODES, num_workers)[worker]
        attempts = split_evenly(MAX_ATTEMPTS, num_workers)[worker]
    return {
        "worker": worker,
        "ros_port": BASE_ROS_PORT + worker,
        "gazebo_port": BASE_GAZEBO_PORT + worker,
        "launch_args": [item.format(worker=worker) for item in WORKER_LAUNCH_ARGS],
        "yolo_launch_args": [item.format(worker=worker) for item in WORKER_YOLO_LAUNCH_ARGS],
        "vehicle_id": WORKER_VEHICLE_ID.format(worker=worker),
        "seed": RANDOM_SEED + worker * WORKER_SEED_STRIDE,
        "target_saved": int(target),
        "max_attempts": int(attempts),
        "max_steps": int(MAX_STEPS),
        **worker_output_paths(worker, test_mode),
    }


def worker_environment(settings: Dict[str, Any]) -> Dict[str, str]:
    """子进程环境变量；GazeboEnv 会按 ROS_MASTER_URI 的端口启动 roscore。"""
    return {
        "ROS_MASTER_URI": f"http://localhost:{settings['ros_port']}",
        "GAZEBO_MASTER_URI": f"http://localhost:{settings['gazebo_port']}",
    }


class EarlyAbort:
    """连续 SEARCH 过长或相对高度长期不下降时提前结束回合。"""

    def __init__(self, max_search_streak: int, stall_window_steps: int, min_descent_progress: float) -> None:
        if max_search_streak <= 0 or stall_window_steps <= 0 or min_descent_progress <= 0.0:
            raise ValueError("早停阈值必须为正数")
        self.max_search_streak = int(max_search_streak)
        self.stall_window_steps = int(stall_window_steps)
        self.min_descent_progress = float(min_descent_progress)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.search_streak = 0
        self.best_height: Optional[float] = None
        self.best_step = 0

    def update(self, phase: str, relative_height: float) -> Optional[str]:
        """返回提前终止原因；继续执行返回 None。每步在执行动作前调用一次。"""
        self.step += 1
        self.search_streak = self.search_streak + 1 if phase == "SEARCH" else 0
        if self.search_streak > self.max_search_streak:
            return "EARLY_ABORT_SEARCH"
        height = float(relative_height)
        if self.best_height is None or height <= self.best_height - self.min_descent_progress:
            self.best_height, self.best_step = height, self.step
        elif self.step - self.best_step >= self.stall_window_steps:
            return "EARLY_ABORT_STALL"
        return None


def merge_worker_outputs(sources: Sequence[str], destination: str) -> int:
    """按行追加各 worker 文件到目标文件，成功后删除来源文件，返回合并的回合数。"""
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    merged = 0
    for source in sources:
        if not os.path.isfile(source):
            continue
        if os.path.abspath(source) == os.path.abspath(destination):
            raise ValueError("worker 文件不能与合并目标相同")
        with open(source, "r", encoding="utf-8") as input_file:
            lines = [line for line in input_file if line.strip()]
        with open(destination, "a", encoding="utf-8") as output_file:
            output_file.writelines(lines)
            output_file.flush()
        os.remove(source)
        merged += len(lines)
    return merged


def leftover_worker_files(test_mode: bool, num_workers: int = NUM_WORKERS) -> List[str]:
    """上次运行遗留且非空的 worker 输出文件。"""
    leftovers = []
    for worker in range(num_workers):
        paths = worker_output_paths(worker, test_mode)
        for key in ("training_output", "raw_output"):
            if os.path.isfile(paths[key]) and os.path.getsize(paths[key]) > 0:
                leftovers.append(paths[key])
    return leftovers


def merge_all(test_mode: bool, num_workers: int = NUM_WORKERS) -> None:
    training_output = base.TEST_TRAINING_OUTPUT if test_mode else TRAINING_OUTPUT
    raw_output = base.TEST_RAW_OUTPUT if test_mode else RAW_OUTPUT
    paths = [worker_output_paths(worker, test_mode) for worker in range(num_workers)]
    saved = merge_worker_outputs([item["training_output"] for item in paths], training_output)
    raw = merge_worker_outputs([item["raw_output"] for item in paths], raw_output)
    print(f"合并完成: {saved} 个成功回合 -> {training_output}；{raw} 个原始回合 -> {raw_output}")


def count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def run_worker(settings: Dict[str, Any], test_mode: bool) -> None:
    """单个 worker 的采集循环；与 collect_global_expert.main 一致，仅增加早停。"""
    worker = int(settings["worker"])
    target_saved = int(settings["target_saved"])
    max_attempts = int(settings["max_attempts"])
    max_steps = int(settings["max_steps"])
    training_output = str(settings["training_output"])
    raw_output = str(settings["raw_output"])
    seed = int(settings["seed"])

    # ROS/Gazebo 只在真正采集时导入，离线测试无需安装这些依赖。
    from Simulation.env.env_base import GazeboEnv, TIME_DELTA as ENV_TIME_DELTA
    from Simulation.ship_motion import ShipMotionController

    if abs(float(ENV_TIME_DELTA) - base.TIME_DELTA) > 1e-9:
        raise RuntimeError(f"采集步长 {base.TIME_DELTA} 与环境步长 {ENV_TIME_DELTA} 不一致")

    np.random.seed(seed)
    base.prepare_output_files(training_output, raw_output, True)
    expert = GlobalLandingExpert(base.EXPERT_CONFIG)
    class_rng = np.random.RandomState(seed)
    saved_counts = {name: 0 for name in base.MOTION_CLASSES}
    early_abort = EarlyAbort(MAX_SEARCH_STREAK, STALL_WINDOW_STEPS, MIN_DESCENT_PROGRESS)

    print("=" * 72)
    print(f"worker {worker} 全局真值专家采集" + ("（冒烟测试）" if test_mode else ""))
    print(f"ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI')} GAZEBO_MASTER_URI={os.environ.get('GAZEBO_MASTER_URI')}")
    print(f"launch 参数: {settings['launch_args']}；YOLO 参数: {settings['yolo_launch_args']}；vehicle_id={settings['vehicle_id']}")
    print(f"目标成功回合数: {target_saved}；最大尝试次数: {max_attempts}；随机种子: {seed}")
    print(f"训练数据: {training_output}")
    print(f"原始数据: {raw_output}")
    print(f"早停: 连续 SEARCH>{MAX_SEARCH_STREAK} 步，或 {STALL_WINDOW_STEPS} 步内下降不足 {MIN_DESCENT_PROGRESS} m")
    print("=" * 72, flush=True)

    env = None
    controller = None
    saved_count = 0
    attempt_id = 0

    try:
        env = GazeboEnv(
            base.LAUNCH_FILE,
            base.VEHICLE_TYPE,
            str(settings["vehicle_id"]),
            max_dist=base.MAX_WORLD_DISTANCE,
            max_height=base.MAX_FLIGHT_HEIGHT,
            landing_xy_threshold=base.LANDING_XY_THRESHOLD,
            visual_x_limit=20.0,
            visual_y_limit=20.0,
            yolo_lost_timeout=base.YOLO_LOST_TIMEOUT,
            enable_yolo=True,
            system_warmup_seconds=base.STARTUP_WARMUP_SECONDS,
            roscore_wait_seconds=base.ROSCORE_WAIT_SECONDS,
            gazebo_wait_seconds=base.GAZEBO_WAIT_SECONDS,
            configure_rc_loss_exception=False,
            mavros_state_timeout=30.0,
            launch_args=list(settings["launch_args"]),
            yolo_launch_args=list(settings["yolo_launch_args"]),
        )
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(base.SHIP_INITIAL_X, base.SHIP_INITIAL_Y),
            init_z=base.SHIP_INITIAL_Z,
            max_speed=base.MAX_SHIP_SPEED,
        )
        env.landing_target_fn = lambda: controller.get_landing_target(
            marker_offset_z=base.MARKER_OFFSET_Z,
            marker_offset_x=base.MARKER_OFFSET_X,
            marker_offset_y=base.MARKER_OFFSET_Y,
        )
        env.landing_velocity_fn = controller.get_landing_velocity
        if not controller.wait_for_odom(timeout=15.0):
            raise RuntimeError("未收到 WAM-V 模型状态")

        while saved_count < target_saved and attempt_id < max_attempts:
            attempt_id += 1
            motion_class = base.choose_motion_class(saved_counts, class_rng)
            episode_seed = seed + attempt_id - 1
            scenario = base.configure_motion(controller, motion_class, episode_seed)
            scenario["worker"] = worker
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(base.RESET_SETTLE_SECONDS)
            env.pause()
            expert.reset()
            early_abort.reset()

            try:
                env.reset()
                initial_position = base.wait_for_initial_observation(
                    env, base.INITIAL_DETECTION_WAIT_SECONDS
                )
            except Exception as error:
                print(f"第 {attempt_id:04d} 局重置失败: {error}", flush=True)
                continue
            if initial_position is None:
                print(f"第 {attempt_id:04d} 局初始未看到 marker，重新开始", flush=True)
                continue

            observation_builder = base.VisualMotionObservation(base.TIME_DELTA)
            observation = observation_builder.initialize(
                initial_position, getattr(env, "yolo_confidence", 0.0)
            )

            episode: List[Dict[str, Any]] = []
            success = False
            terminal_reason = "RUNNING"
            marker_visible = True
            lost_steps = 0
            last_action = np.zeros(3, dtype=np.float32)
            try:
                for step_index in range(max_steps):
                    controller.step(step_index * base.TIME_DELTA)
                    truth_before = base.current_truth(env, controller)
                    command = expert.compute_action(
                        *truth_before,
                        marker_visible=marker_visible,
                        last_action=last_action,
                    )
                    if command.phase == "SEARCH":
                        lost_steps += 1
                    elif marker_visible:
                        lost_steps = 0
                    relative_height = float(truth_before[0][2] - truth_before[3][2])
                    # 早停判断在执行动作前完成；触发时本步仍真实执行，并作为终止转移写入。
                    abort_reason = early_abort.update(command.phase, relative_height)

                    raw_next, env_done, env_success, info = env.step(command.action)
                    next_visible = bool(
                        info.get("detection_fresh", info.get("tag_detected", False))
                    )
                    next_observation = observation_builder.update(
                        raw_next,
                        next_visible,
                        info.get("yolo_confidence", 0.0),
                    )
                    truth_after = base.current_truth(env, controller)

                    reached_limit = step_index + 1 >= max_steps
                    done = bool(env_done or reached_limit or abort_reason)
                    success = bool(info.get("landing_success", env_success))
                    terminal_reason = str(info.get("terminal_reason", "RUNNING"))
                    if not env_done:
                        if abort_reason:
                            terminal_reason = abort_reason
                            success = False
                        elif reached_limit:
                            terminal_reason = "MAX_STEPS"

                    reward = base.compute_reward(next_observation, done, success)
                    step_data = {
                        "observation": observation.astype(float).tolist(),
                        "action": command.action.astype(float).tolist(),
                        "reward": float(reward),
                        "next_observation": next_observation.astype(float).tolist(),
                        "done": done,
                        "success": success,
                        "episode_id": attempt_id,
                        "step_index": step_index,
                        "collection_mode": "test" if test_mode else "normal",
                        "terminal_reason": terminal_reason,
                        "scenario": scenario,
                        "expert": command.to_metadata(),
                        "privileged_state": base.truth_metadata(truth_before),
                        "next_privileged_state": base.truth_metadata(truth_after),
                        "env_info": {
                            "marker_visible": bool(marker_visible),
                            "next_marker_visible": bool(next_visible),
                            "observation_confidence": float(observation[-1]),
                            "next_observation_confidence": float(next_observation[-1]),
                            "lost_steps": int(lost_steps),
                            "relative_height": relative_height,
                            "relative_xy_distance": base._finite_float_or_none(
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
                print(f"第 {attempt_id:04d} 局运行异常: {error}", flush=True)

            if episode:
                base.append_episode(raw_output, episode)

            trainable, reason = base.episode_is_trainable(episode)
            if success and trainable:
                base.append_episode(training_output, episode)
                saved_count += 1
                saved_counts[motion_class] += 1
                result = "已保存"
            else:
                result = f"未保存（{reason}）"

            search_steps = sum(
                int((step.get("expert") or {}).get("phase") == "SEARCH") for step in episode
            )
            print(
                f"第 {attempt_id:04d} 局 | {motion_class:16s} | {result} | "
                f"成功={success} | 步数={len(episode):3d} | "
                f"SEARCH={search_steps:3d} | 终止={terminal_reason} | "
                f"进度={saved_count}/{target_saved}",
                flush=True,
            )

        if saved_count < target_saved:
            print(f"达到最大尝试次数，只保存了 {saved_count}/{target_saved} 个成功回合")
        else:
            print(f"采集完成，共保存 {saved_count} 个成功回合")
        print(f"各运动类别数量: {saved_counts}", flush=True)
    except KeyboardInterrupt:
        print("\n用户中断采集，已经写入的完整回合会保留", flush=True)
    finally:
        if controller is not None:
            controller.shutdown()
        if env is not None:
            env.close()


def launch_workers(test_mode: bool) -> None:
    """错开启动全部 worker，等待结束后合并输出。"""
    leftovers = leftover_worker_files(test_mode)
    if leftovers:
        raise FileExistsError(
            "存在上次运行遗留的 worker 输出，请先运行 --merge 合并或手动删除:\n  " + "\n  ".join(leftovers)
        )
    training_output = base.TEST_TRAINING_OUTPUT if test_mode else TRAINING_OUTPUT
    raw_output = base.TEST_RAW_OUTPUT if test_mode else RAW_OUTPUT
    base.prepare_output_files(training_output, raw_output, True if test_mode else CLEAR_OUTPUT_ON_START)
    os.makedirs(WORKER_DIR, exist_ok=True)

    settings = [worker_settings(worker, test_mode) for worker in range(NUM_WORKERS)]
    print("=" * 72)
    print(f"并行采集: {NUM_WORKERS} 个 worker" + ("（冒烟测试）" if test_mode else ""))
    for item in settings:
        print(f"  worker {item['worker']}: ROS {item['ros_port']} | Gazebo {item['gazebo_port']} | "
              f"目标 {item['target_saved']} | 日志 {item['log']}")
    print(f"合并目标: {training_output}")
    print("=" * 72, flush=True)

    processes: List[subprocess.Popen] = []
    logs = []
    try:
        for index, item in enumerate(settings):
            if index > 0:
                time.sleep(WORKER_START_STAGGER_SECONDS)
            environment = {**os.environ, **worker_environment(item)}
            command = [sys.executable, "-m", "Sampling.collect_parallel", "--worker", str(item["worker"])]
            if test_mode:
                command.append("--test")
            log = open(item["log"], "w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                cwd=base.REPO_ROOT, preexec_fn=os.setsid,
            ))
            print(f"worker {item['worker']} 已启动 (pid={processes[-1].pid})", flush=True)

        last_status = time.monotonic()
        while any(process.poll() is None for process in processes):
            time.sleep(1.0)
            if time.monotonic() - last_status >= STATUS_INTERVAL_SECONDS:
                last_status = time.monotonic()
                progress = ", ".join(
                    f"worker {item['worker']}: {count_lines(item['training_output'])}/{item['target_saved']}"
                    for item in settings
                )
                print(f"进度 | {progress}", flush=True)
    except KeyboardInterrupt:
        print("\n用户中断，正在停止全部 worker...", flush=True)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30.0
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    finally:
        for log in logs:
            log.close()

    for item, process in zip(settings, processes):
        print(f"worker {item['worker']} 退出码 {process.returncode}，保存 {count_lines(item['training_output'])} 个成功回合")
    merge_all(test_mode)


def main() -> None:
    args = parse_args()
    if args.worker is not None:
        run_worker(worker_settings(args.worker, args.test), args.test)
    elif args.merge:
        merge_all(args.test)
    else:
        launch_workers(args.test)


if __name__ == "__main__":
    main()
