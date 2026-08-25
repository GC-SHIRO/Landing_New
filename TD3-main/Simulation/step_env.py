#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step1-4 动态平台降落统一评估脚本。

用法示例:
  python step_env.py --step 1 --vx 0.7
  python step_env.py --step 2 --vx 1.0 --v_min 0.4 --v_max 1.0 --seed 42
  python step_env.py --step 3 --curve sine --speed 1.0 --amp 2.0 --wavelen 8.0
  python step_env.py --step 4 --curve circle --v_min 0.4 --v_max 1.0 --seed 42
"""

import os
import sys
import time
import signal
import argparse
from collections import deque

import numpy as np

# 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Simulation.env_base import GazeboEnv, TIME_DELTA
from Simulation.ship_motion import ShipMotionController
from Sampling.collect_global_expert import VisualMotionObservation
from TD3_offline import TD3
from landing_evaluation import (
    LandingEvaluation, EpisodeData, compute_dynamic_target,
    estimate_velocity_from_positions, estimate_accel_from_positions,
    build_relative_dynamics,
)

# ===== 配置 =====
LAUNCH_FILE = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"

# 船体初始位置 (与 launch 文件一致)
SHIP_INIT_X = 10.0
SHIP_INIT_Y = 5.0
SHIP_INIT_Z = 0.1
MARKER_OFFSET_X = -0.20  # marker.dae 几何中心相对 base_link 的船体 x 偏移
MARKER_OFFSET_Y = 0.0
MARKER_OFFSET_Z = 1.3  # marker 距船 base_link 高度 (wamv_gazebo.urdf.xacro)


def _load_norm_stats(ckpt_dir, state_dim):
    """加载归一化参数"""
    mean_path = os.path.join(ckpt_dir, 'state_mean.npy')
    std_path = os.path.join(ckpt_dir, 'state_std.npy')
    if os.path.exists(mean_path) and os.path.exists(std_path):
        state_mean = np.load(mean_path).astype(np.float32)
        state_std = np.load(std_path).astype(np.float32)
        if state_mean.shape[0] != state_dim:
            raise ValueError(
                f"归一化维度不匹配: mean/std={state_mean.shape}/{state_std.shape} "
                f"但 state_dim={state_dim}"
            )
        print("成功加载状态归一化参数!")
        return state_mean, state_std
    print("警告: 未找到归一化参数文件, 使用 mean=0, std=1")
    return np.zeros(state_dim, dtype=np.float32), np.ones(state_dim, dtype=np.float32)


def _motion_name(args):
    if args.step == 1:
        return "匀速直线 (constant)"
    if args.step == 2:
        return f"变速直线 (varspeed, {args.v_min:.1f}-{args.v_max:.1f}m/s)"
    if args.step == 3:
        return f"匀速曲线 ({args.curve})"
    return f"曲线变速 (combined:{args.curve}, {args.v_min:.1f}-{args.v_max:.1f}m/s)"


def _configure_motion(controller, args):
    """按 step 配置船舶运动，返回用于日志展示的模式名称。"""
    if args.step == 1:
        controller.set_mode_constant(args.vx, args.vy)
    elif args.step == 2:
        controller.set_mode_varspeed((args.vx, args.vy),
                                     (args.v_min, args.v_max), args.seed)
    elif args.curve == 'sine':
        controller.set_mode_sine(args.speed, args.amp, args.wavelen,
                                 args.vx, args.vy)
        if args.step == 4:
            controller.set_mode_combined('sine',
                                         (args.v_min, args.v_max), args.seed)
    else:
        controller.set_mode_circle(args.radius, args.period, args.vx, args.vy)
        if args.step == 4:
            controller.set_mode_combined('circle',
                                         (args.v_min, args.v_max), args.seed)
    return _motion_name(args)


def main():
    parser = argparse.ArgumentParser(description="Step1-4 动态平台降落统一评估")
    # ---- 评估控制 ----
    parser.add_argument('--test_episodes', type=int, default=100,
                        help='评估轮数 (默认 100)')
    parser.add_argument('--max_steps', type=int, default=600,
                        help='每轮最大步数')
    parser.add_argument('--dt', type=float, default=TIME_DELTA,
                        help='步长时间 (秒)')

    # ---- 船体运动 ----
    parser.add_argument('--step', type=int, choices=(1, 2, 3, 4), default=3,
                        help='运动难度阶段: 1匀速直线, 2变速直线, 3匀速曲线, 4曲线变速')
    parser.add_argument('--curve', choices=('sine', 'circle'), default='circle',
                        help='Step3/4 曲线类型')
    parser.add_argument('--vx', '--ship_vx', dest='vx', type=float, default=None,
                        help='前进方向 x 分量；Step1 默认0.7，其余默认1.0')
    parser.add_argument('--vy', '--ship_vy', dest='vy', type=float, default=0.0,
                        help='前进方向 y 分量')
    parser.add_argument('--speed', '--v', dest='speed', type=float, default=1.0,
                        help='Step3 曲线速率 m/s')
    parser.add_argument('--v_min', type=float, default=0.4,
                        help='Step2/4 最小速率 m/s')
    parser.add_argument('--v_max', type=float, default=1.0,
                        help='Step2/4 最大速率 m/s')
    parser.add_argument('--seed', type=int, default=None,
                        help='Step2/4 随机变速种子')
    parser.add_argument('--amp', type=float, default=2.0,
                        help='正弦曲线法向摆幅 m')
    parser.add_argument('--wavelen', type=float, default=8.0,
                        help='正弦曲线空间波长 m')
    parser.add_argument('--radius', type=float, default=3.0,
                        help='圆周运动半径 m')
    parser.add_argument('--period', type=float, default=40.0,
                        help='圆周运动周期 s')

    # ---- 模型加载 ----
    parser.add_argument('--ckpt_dir', type=str,
                        default='/home/shiro/Landing_new/checkpoints/TD3/LSTM1',
                        help='模型权重目录')
    parser.add_argument('--load_step', type=int, default=100000,
                        help='加载步数')
    parser.add_argument('--state_dim', type=int, default=10)
    parser.add_argument('--action_dim', type=int, default=3)
    parser.add_argument('--max_action', type=float, default=1.0)
    parser.add_argument('--capacity', type=int, default=65536)

    # ---- LSTM 参数 (必须与训练一致) ----
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attn_hidden_dim', type=int, default=64)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ---- TD3 参数 ----
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--policy_noise', type=float, default=0.0)
    parser.add_argument('--noise_clip', type=float, default=0.5)

    # ---- 判定阈值 ----
    parser.add_argument('--landing_xy_thresh', type=float, default=1.5,
                        help='无人机到降落点最大水平距离 m')
    parser.add_argument('--visual_x_limit', type=float, default=20.0,
                        help='YOLO视觉X越界阈值, 默认 |x|>20m')
    parser.add_argument('--visual_y_limit', type=float, default=20.0,
                        help='YOLO视觉Y越界阈值, 默认 |y|>20m')
    parser.add_argument('--yolo_lost_timeout', type=float, default=2.0,
                        help='YOLO观测新鲜度窗口；失检不直接结束回合')
    parser.add_argument('--settle_seconds', type=float, default=0.0,
                        help='结束后额外等待时间；默认0，避免落地后摩擦/漂移污染判定')

    # ---- 越界阈值 (step1 中无人机在船附近活动, 远离世界原点) ----
    parser.add_argument('--max_dist', type=float, default=25.0,
                        help='无人机距世界原点最大允许距离 m (默认 25)')
    parser.add_argument('--max_height', type=float, default=12.0,
                        help='无人机最大允许高度 m (默认 12)')

    args = parser.parse_args()
    if args.v_min <= 0 or args.v_max < args.v_min:
        parser.error('要求 0 < v_min <= v_max')
    if args.speed <= 0 or args.radius <= 0 or args.period <= 0 or args.wavelen <= 0:
        parser.error('speed/radius/period/wavelen 必须大于 0')
    if args.vx is None:
        args.vx = 0.7 if args.step == 1 else 1.0

    # ==================== 打印配置 ====================
    motion_name = _motion_name(args)
    print("=" * 60)
    print(f"  Step{args.step} 动态平台降落评估")
    print("=" * 60)
    print(f"  Launch:        {LAUNCH_FILE}")
    print(f"  模型:           {args.ckpt_dir}  step={args.load_step}")
    print(f"  评估轮数:       {args.test_episodes}")
    print(f"  每轮最大步数:   {args.max_steps}  (dt={args.dt}s, 最长={args.max_steps * args.dt:.0f}s)")
    print(f"  船运动模式:     {motion_name}")
    print(f"  前进方向:       vx={args.vx:.3f} vy={args.vy:.3f}")
    print(f"  甲板成功:       接触甲板且 rel_xy<={args.landing_xy_thresh}m")
    print(f"  视觉越界:       |x|>{args.visual_x_limit}m 或 |y|>{args.visual_y_limit}m")
    print(f"  YOLO失检:       不直接终止，观测有效期={args.yolo_lost_timeout}s")
    print(f"  越界阈值:       max_dist={args.max_dist:.0f}m  max_height={args.max_height:.0f}m")
    print("=" * 60)

    env = None
    controller = None

    # ==================== 统计 ====================
    success_count = 0
    fail_count = 0
    crash_count = 0
    total_steps = 0
    records = []  # 收集所有 episode 评估记录

    try:
        # ---- 1. 初始化环境 ----
        print("\n[1/4] 启动仿真环境...")
        env = GazeboEnv(LAUNCH_FILE, VEHICLE_TYPE, VEHICLE_ID,
                        max_dist=args.max_dist, max_height=args.max_height)
        print("  ✓ 环境初始化完成")

        # ---- 2. 初始化船舶运动控制器 ----
        print("\n[2/4] 创建船舶运动控制器...")
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(SHIP_INIT_X, SHIP_INIT_Y),
            init_z=SHIP_INIT_Z,
            max_speed=1.0,
        )
        motion_name = _configure_motion(controller, args)
        print(f"  ✓ ShipMotionController 就绪: {motion_name}")

        # 绑定动态目标: 相对 marker 成功边界 (与 LandingEvaluation 一致)
        env.landing_target_fn = lambda: controller.get_landing_target(
            marker_offset_z=MARKER_OFFSET_Z,
            marker_offset_x=MARKER_OFFSET_X,
            marker_offset_y=MARKER_OFFSET_Y,
        )
        env.landing_velocity_fn = controller.get_landing_velocity
        env.landing_xy_threshold = args.landing_xy_thresh
        env.visual_x_limit = args.visual_x_limit
        env.visual_y_limit = args.visual_y_limit
        env.yolo_lost_timeout = args.yolo_lost_timeout
        print(
            f"  ✓ 着陆检测: WAM-V 甲板碰撞 + "
            f"rel_xy<={env.landing_xy_threshold:.3f}m"
        )

        # 控制器从 /gazebo/model_states 读取船当前位姿
        print("  等待船舶模型状态...")
        if not controller.wait_for_odom(timeout=15.0):
            print("  ⚠ 船舶模型状态超时，将使用初始位姿兜底")
        else:
            pos = controller.get_current_pos()
            print(f"  ✓ 船舶模型状态: pos=({pos[0]:.2f}, {pos[1]:.2f})")

        # ---- 3. 加载 TD3 模型 ----
        print(f"\n[3/4] 加载 TD3/LSTM 策略...")
        state_mean, state_std = _load_norm_stats(args.ckpt_dir, args.state_dim)

        def normalize(raw_state):
            raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
            return (raw_state - state_mean) / (state_std + 1e-6)

        # 过滤参数传给 TD3
        td3_allowed = {
            "state_dim", "action_dim", "max_action",
            "gamma", "tau", "policy_delay", "policy_noise", "noise_clip",
            "seq_len", "hidden_dim", "attn_hidden_dim", "dropout_p",
            "lr_actor", "lr_critic", "weight_decay",
            "capacity", "ckpt_dir",
        }
        from types import SimpleNamespace
        td3_kwargs = {k: v for k, v in vars(args).items() if k in td3_allowed}
        td3_args = SimpleNamespace(**td3_kwargs)

        agent = TD3(args.state_dim, args.action_dim, args.max_action,
                    args.capacity, td3_args)

        try:
            agent.load(args.ckpt_dir, step=args.load_step)
            print(f"  ✓ 成功加载模型: {args.ckpt_dir}  step={args.load_step}")
        except TypeError:
            agent.load(args.load_step)
            print(f"  ✓ 成功加载模型: step={args.load_step}")
        except Exception as e:
            print(f"  ✗ 模型加载失败: {e}")
            return

        # ---- 初始化评估器 ----
        evaluator = LandingEvaluation(
            platform_type="dynamic",
            max_dist=args.max_dist,
            max_height=args.max_height,
        )
        print("  ✓ LandingEvaluation 就绪")
        print(f"    平台类型: dynamic  |  水平成功阈值: rel_xy<={args.landing_xy_thresh}m")

        # ---- 4. 评估循环 ----
        print(f"\n[4/4] 开始 {args.test_episodes} 轮评估")
        print("=" * 60)

        for episode_id in range(1, args.test_episodes + 1):
            # ---- 4a. Reset ----
            # 先将船复位，再让无人机按当前甲板位置重置，避免两者状态不同步。
            controller.teleport_to_origin()
            env.unpause()
            time.sleep(0.3)
            env.pause()
            try:
                obs = env.reset()
            except Exception as e:
                print(f"  Ep {episode_id:3d}: ✗ RESET_FAILED: {e}")
                crash_count += 1
                continue

            observation_builder = VisualMotionObservation(args.dt)
            obs = observation_builder.initialize(
                np.asarray(obs, dtype=np.float32).reshape(-1),
                getattr(env, "yolo_confidence", 0.0),
            )
            norm_obs = normalize(obs)

            # ---- LSTM 冷启动 ----
            state_queue = deque(maxlen=args.seq_len)
            for _ in range(args.seq_len):
                state_queue.append(norm_obs)

            # ---- 4b. Rollout ----
            done = False
            step_count = 0
            episode_crashed = False
            crash_msg = ""
            episode_actions = []
            episode_drone_positions = []
            episode_max_height = 0.0
            out_of_bounds = False
            visual_out_of_bounds = False
            landing_success = False
            relative_height = float("nan")
            visual_height = float("nan")
            visual_x = float("nan")
            visual_y = float("nan")
            terminal_reason = "RUNNING"
            lost_detection = False

            try:
                while not done:
                    t = step_count * args.dt   # 当前时刻

                    # 船舶控制: 发布速度命令，由 Gazebo 积分船位置
                    controller.step(t)

                    # 策略推理
                    seq = np.asarray(state_queue, dtype=np.float32)      # (T, D)
                    seq_batch = seq[np.newaxis, :, :]                    # (1, T, D)

                    out = agent.choose_action(seq_batch, noise=0.0)
                    action = out[0] if isinstance(out, tuple) else out
                    action = np.asarray(action, dtype=np.float32).reshape(-1)

                    # 无人机步进
                    raw_next_obs, done, success, info = env.step(action)
                    info = dict(info or {})
                    next_visible = bool(
                        info.get("detection_fresh", info.get("tag_detected", False))
                    )
                    next_obs = observation_builder.update(
                        raw_next_obs,
                        next_visible,
                        info.get("yolo_confidence", 0.0),
                    )
                    if info.get("out_of_bounds"):
                        out_of_bounds = True
                    visual_out_of_bounds = bool(info.get("visual_out_of_bounds", False))
                    landing_success = bool(info.get("landing_success", success))
                    relative_height = float(info.get("relative_height", float("nan")))
                    visual_height = float(info.get("visual_height", float("nan")))
                    visual_x = float(info.get("visual_x", float("nan")))
                    visual_y = float(info.get("visual_y", float("nan")))
                    terminal_reason = str(info.get("terminal_reason", "RUNNING"))
                    if info.get("lost_detection"):
                        lost_detection = True

                    # 收集轨迹数据
                    episode_actions.append(action.tolist())
                    if env.comm.current_position is not None:
                        cp = env.comm.current_position
                        episode_drone_positions.append((float(cp.x), float(cp.y), float(cp.z)))
                        episode_max_height = max(episode_max_height, float(cp.z))

                    obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                    norm_obs = normalize(obs)
                    state_queue.append(norm_obs)

                    step_count += 1
                    if step_count >= args.max_steps:
                        done = True
                        if terminal_reason == "RUNNING":
                            terminal_reason = "MAX_STEPS"

            except Exception as e:
                episode_crashed = True
                crash_msg = str(e)

            # ---- 4c. 物理稳定 + 真实状态 (触地瞬间) ----
            drone_wx = drone_wy = drone_wz = float("nan")
            drone_vx = drone_vy = drone_vz = float("nan")
            ship_sx = ship_sy = ship_sz = float("nan")
            ship_vx = ship_vy = 0.0
            target_x = target_y = target_z = float("nan")
            rel_dyn = {
                "rel_vx": float("nan"), "rel_vy": float("nan"), "rel_vz": float("nan"),
                "rel_ax": float("nan"), "rel_ay": float("nan"), "rel_az": float("nan"),
            }

            try:
                if args.settle_seconds > 0 and not landing_success:
                    env.unpause()
                    time.sleep(args.settle_seconds)
                    env.pause()

                # 无人机世界坐标 (MAVROS)
                cp = env.comm.current_position
                drone_wx = float(cp.x)
                drone_wy = float(cp.y)
                drone_wz = float(cp.z)

                # 无人机速度: 位置差分 (MAVROS 无速度时的统一来源)
                drone_vx, drone_vy, drone_vz = estimate_velocity_from_positions(
                    episode_drone_positions, args.dt
                )
                drone_ax, drone_ay, drone_az = estimate_accel_from_positions(
                    episode_drone_positions, args.dt
                )

                # 船位姿/速度 (/gazebo/model_states)
                ship_state = controller.get_state()
                ship_pos_arr = ship_state["pos"]
                ship_vel_arr = ship_state["vel"]
                ship_sx = float(ship_pos_arr[0])
                ship_sy = float(ship_pos_arr[1])
                ship_sz = float(ship_state["pos_z"])
                ship_vx = float(ship_vel_arr[0])
                ship_vy = float(ship_vel_arr[1])
                ship_yaw = float(ship_state["yaw"])

                # 目标点 = 船 base_link + 旋转后的 marker 几何中心偏移
                target_x, target_y, target_z = compute_dynamic_target(
                    ship_sx,
                    ship_sy,
                    ship_sz,
                    ship_yaw=ship_yaw,
                    marker_offset_x=evaluator.marker_offset_x,
                    marker_offset_y=evaluator.marker_offset_y,
                    marker_offset_z=evaluator.marker_offset_z,
                )

                rel_dyn = build_relative_dynamics(
                    drone_vx, drone_vy, drone_vz,
                    ship_vx, ship_vy, 0.0,
                    drone_ax, drone_ay, drone_az,
                )
            except Exception as e:
                episode_crashed = True
                crash_msg = (crash_msg + " | " if crash_msg else "") + f"settle: {e}"

            # 初始 3D 距离: 轨迹起点相对终点目标 (近似; 动态船目标在变)
            if len(episode_drone_positions) > 0 and all(
                np.isfinite(v) for v in [target_x, target_y, target_z]
            ):
                p0 = episode_drone_positions[0]
                init_dist_3d = float(np.linalg.norm([
                    p0[0] - target_x, p0[1] - target_y, p0[2] - target_z
                ]))
            else:
                init_dist_3d = float("nan")

            # ---- 4d. 填充 EpisodeData 并评估 ----
            ep_data = EpisodeData(
                episode_id=episode_id,
                drone_final_x=drone_wx,
                drone_final_y=drone_wy,
                drone_final_z=drone_wz,
                target_x=target_x if np.isfinite(target_x) else float("nan"),
                target_y=target_y if np.isfinite(target_y) else float("nan"),
                target_z=target_z if np.isfinite(target_z) else float("nan"),
                rel_vx=rel_dyn["rel_vx"],
                rel_vy=rel_dyn["rel_vy"],
                rel_vz=rel_dyn["rel_vz"],
                rel_ax=rel_dyn["rel_ax"],
                rel_ay=rel_dyn["rel_ay"],
                rel_az=rel_dyn["rel_az"],
                steps=step_count,
                dt=args.dt,
                init_dist_3d=init_dist_3d,
                max_height=episode_max_height,
                max_dist_origin=float(np.sqrt(drone_wx**2 + drone_wy**2 + drone_wz**2))
                    if np.isfinite(drone_wx) else float("nan"),
                impact_velocity_z=drone_vz if np.isfinite(drone_vz) else float("nan"),
                actions=episode_actions,
                drone_positions=episode_drone_positions,
                crashed=episode_crashed,
                crash_msg=crash_msg,
                out_of_bounds=out_of_bounds,
                visual_out_of_bounds=visual_out_of_bounds,
                max_steps_reached=(step_count >= args.max_steps),
                lost_detection=lost_detection,
                landing_success=landing_success,
                relative_height=relative_height,
                visual_height=visual_height,
                visual_x=visual_x,
                visual_y=visual_y,
                terminal_reason=terminal_reason,
                ckpt_dir=args.ckpt_dir,
                load_step=args.load_step,
                platform_type="dynamic",
            )

            record = evaluator.evaluate(ep_data)
            record["MaxSteps"] = args.max_steps
            records.append(record)

            result_str = record["Result"]
            if result_str == "SUCCESS":
                success_count += 1
            elif result_str in ("MISSED", "MAX_STEPS", "OUT_OF_BOUNDS", "YOLO_OUT_OF_BOUNDS", "LOST_DETECTION"):
                fail_count += 1
            else:
                crash_count += 1

            total_steps += step_count

            # ---- 4e. 打印 ----
            ship_pos = controller.get_current_pos()
            comp = record.get("CompositeScore", float("nan"))
            rating = record.get("Rating", "N/A")
            herr = record.get("HorizErr", float("nan"))
            verr = record.get("VertErr", float("nan"))
            rel_sp = record.get("RelSpeed", float("nan"))
            print(f"  Ep {episode_id:3d}/{args.test_episodes} | {result_str:16s} "
                  f"score={comp:5.1f} ({rating}) "
                  f"acc={record.get('Score_Accuracy', 0):4.1f} "
                  f"dyn={record.get('Score_Dynamics', 0):4.1f} | "
                  f"steps={step_count:3d} | "
                  f"herr={herr:.3f} verr={verr:.3f} |v_rel|={rel_sp:.3f} | "
                  f"船=({ship_pos[0]:.1f},{ship_pos[1]:.1f})")

        # ==================== 汇总 ====================
        summary = evaluator.summarize(records)
        print("\n" + "=" * 60)
        print("  评估汇总")
        print("=" * 60)
        print(f"  总轮数:         {summary.get('N_total', 0)}")
        print(f"  SUCCESS:        {summary.get('N_SUCCESS', 0)}")
        print(f"  MISSED:         {summary.get('N_MISSED', 0)}")
        print(f"  MAX_STEPS:      {summary.get('N_MAX_STEPS', 0)}")
        print(f"  OUT_OF_BOUNDS:  {summary.get('N_OUT_OF_BOUNDS', 0)}")
        print(f"  YOLO_OOB:       {summary.get('N_YOLO_OUT_OF_BOUNDS', 0)}")
        print(f"  LOST_DETECTION: {summary.get('N_LOST_DETECTION', 0)}")
        print(f"  CRASHED/RESET:  {summary.get('N_CRASHED', 0) + summary.get('N_RESET_FAILED', 0)}")
        print(f"  ---")
        print(f"  SR (含全部):    {summary.get('SR', 0):.1f}%")
        print(f"  SR (可用):      {summary.get('SR_usable', 0):.1f}%")
        if np.isfinite(summary.get('Mean_CompositeScore', float('nan'))):
            print(f"  平均综合分:     {summary['Mean_CompositeScore']:.1f} ± {summary['Std_CompositeScore']:.1f}")
            print(f"    精度(70%): {summary.get('Mean_Score_Accuracy', 0):.1f}  "
                  f"动力学(30%): {summary.get('Mean_Score_Dynamics', 0):.1f}")
            print(f"    平均 herr={summary.get('Mean_HorizErr', float('nan')):.3f}m  "
                  f"verr={summary.get('Mean_VertErr', float('nan')):.3f}m  "
                  f"|v_rel|={summary.get('Mean_RelSpeed', float('nan')):.3f}m/s")
        print(f"  模型:           {args.ckpt_dir}  step={args.load_step}")
        print(f"  Step/模式:      {args.step} / {motion_name}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n用户中断")

    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        if controller is not None:
            print("\n正在关闭船舶控制器...")
            controller.shutdown()
        if env is not None:
            print("正在清理环境...")
            env.stop_yolo()
            if hasattr(env, 'gazebo_process') and env.gazebo_process is not None:
                try:
                    os.killpg(os.getpgid(env.gazebo_process.pid),
                              signal.SIGTERM)
                    print("  Gazebo 已关闭")
                except Exception:
                    pass
        print("清理完成")


if __name__ == "__main__":
    main()
