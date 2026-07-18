#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step3: 正弦/圆周运动 — 动态平台降落评估
=========================================
船舶以正弦曲线或圆周运动, 无人机端到端自主降落。

用法:
  python step3_env.py --test_episodes 100 --curve sine --v 1.0 --amp 2.0 --wavelen 8.0
  python step3_env.py --test_episodes 100 --curve circle --radius 3.0 --period 40.0
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
from TD3_offline import TD3
from landing_evaluation import LandingEvaluation, EpisodeData, compute_dynamic_target

# ===== 配置 =====
LAUNCH_FILE = "/home/wantengyuan/PX4_Firmware/launch/step1_linear.launch"
VEHICLE_TYPE = "iris"
VEHICLE_ID = "0"

# 船体初始位置 (与 launch 文件一致)
SHIP_INIT_X = 10.0
SHIP_INIT_Y = 5.0
SHIP_INIT_Z = 0.1
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


def main():
    parser = argparse.ArgumentParser(description="Step3: 正弦/圆周运动 (sine/circle) 降落评估")
    # ---- 评估控制 ----
    parser.add_argument('--test_episodes', type=int, default=100,
                        help='评估轮数 (默认 100)')
    parser.add_argument('--max_steps', type=int, default=600,
                        help='每轮最大步数')
    parser.add_argument('--dt', type=float, default=TIME_DELTA,
                        help='步长时间 (秒)')

    # ---- 船体运动 ----
    parser.add_argument('--curve', type=str, default='circle',
                        choices=['sine', 'circle'],
                        help='曲线类型: sine 或 circle')
    parser.add_argument('--vx', type=float, default=1.0,
                        help='中心线方向 x 分量')
    parser.add_argument('--vy', type=float, default=0.0,
                        help='中心线方向 y 分量')
    # sine 参数
    parser.add_argument('--v', type=float, default=1.0,
                        help='弧长速率 m/s (sine)')
    parser.add_argument('--amp', type=float, default=2.0,
                        help='法向摆幅 m (sine)')
    parser.add_argument('--wavelen', type=float, default=8.0,
                        help='空间波长 m (sine)')
    # circle 参数
    parser.add_argument('--radius', type=float, default=3.0,
                        help='圆半径 m (circle)')
    parser.add_argument('--period', type=float, default=40.0,
                        help='绕行周期 s (circle)')

    # ---- 模型加载 ----
    parser.add_argument('--ckpt_dir', type=str,
                        default='/home/wantengyuan/Landing_new/checkpoints/TD3/LSTM',
                        help='模型权重目录')
    parser.add_argument('--load_step', type=int, default=60000,
                        help='加载步数')
    parser.add_argument('--state_dim', type=int, default=3)
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
    parser.add_argument('--success_herr_thresh', type=float, default=1.0,
                        help='稳定后水平误差阈值 (m)')
    parser.add_argument('--success_height_thresh', type=float, default=0.6,
                        help='稳定后高度阈值 (m)')
    parser.add_argument('--settle_seconds', type=float, default=0.0,
                        help='结束后额外等待时间；默认0，避免落地后摩擦/漂移污染判定')

    # ---- 越界阈值 (step1 中无人机在船附近活动, 远离世界原点) ----
    parser.add_argument('--max_dist', type=float, default=25.0,
                        help='无人机距世界原点最大允许距离 m (默认 25)')
    parser.add_argument('--max_height', type=float, default=12.0,
                        help='无人机最大允许高度 m (默认 12)')

    args = parser.parse_args()

    # ==================== 打印配置 ====================
    print("=" * 60)
    print(f"  Step3 {'正弦曲线' if args.curve == 'sine' else '圆周运动'} ({args.curve}) 降落评估")
    print("=" * 60)
    print(f"  Launch:        {LAUNCH_FILE}")
    print(f"  模型:           {args.ckpt_dir}  step={args.load_step}")
    print(f"  评估轮数:       {args.test_episodes}")
    print(f"  每轮最大步数:   {args.max_steps}  (dt={args.dt}s, 最长={args.max_steps * args.dt:.0f}s)")
    if args.curve == 'sine':
        print(f"  正弦曲线:       弧长速率={args.v:.1f}m/s  摆幅={args.amp:.1f}m  波长={args.wavelen:.1f}m")
    else:
        print(f"  圆周运动:       半径={args.radius:.1f}m  周期={args.period:.1f}s")
    print(f"  中心方向:       vx={args.vx:.0f} vy={args.vy:.0f}")
    print(f"  成功判据:       herr<{args.success_herr_thresh}m  "
          f"& height<{args.success_height_thresh}m")
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
            init_pos=(SHIP_INIT_X, SHIP_INIT_Y)
        )
        if args.curve == 'sine':
            controller.set_mode_sine(args.v, args.amp, args.wavelen, args.vx, args.vy)
        else:
            controller.set_mode_circle(args.radius, args.period, args.vx, args.vy)
        print(f"  ✓ ShipMotionController 就绪: mode={args.curve}")

        # 绑定动态目标: 无人机到船+marker 的 3D 距离 < 1.0m 判定着陆
        env.landing_target_fn = lambda: (
            controller.get_current_pos()[0],
            controller.get_current_pos()[1],
            SHIP_INIT_Z + MARKER_OFFSET_Z
        )
        env.landing_dist_threshold = 0.4
        print(f"  ✓ 着陆检测: 动态目标模式, 距离阈值={env.landing_dist_threshold:.1f}m")

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
            success_herr_thresh=args.success_herr_thresh,
            success_height_thresh=args.success_height_thresh,
            max_dist=args.max_dist,
            max_height=args.max_height,
        )
        print("  ✓ LandingEvaluation 就绪")
        print(f"    平台类型: dynamic  |  成功阈值: herr<{args.success_herr_thresh}m, h<{args.success_height_thresh}m")

        # ---- 4. 评估循环 ----
        print(f"\n[4/4] 开始 {args.test_episodes} 轮评估")
        print("=" * 60)

        for episode_id in range(1, args.test_episodes + 1):
            # ---- 4a. Reset ----
            try:
                obs = env.reset()
            except Exception as e:
                print(f"  Ep {episode_id:3d}: ✗ RESET_FAILED: {e}")
                crash_count += 1
                continue

            # 船瞬移回原点
            controller.teleport_to_origin()

            # 让 Gazebo 处理瞬移 (物理引擎需要时间窗口)
            env.unpause()
            time.sleep(0.3)
            env.pause()

            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
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
                    next_obs, done, success, info = env.step(action)

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

            except Exception as e:
                episode_crashed = True
                crash_msg = str(e)

            # ---- 4c. 物理稳定 + 真实状态 ----
            drone_wx = drone_wy = drone_wz = float("nan")
            drone_vz = float("nan")
            ship_sx = ship_sy = ship_sz = float("nan")

            try:
                if args.settle_seconds > 0:
                    env.unpause()
                    time.sleep(args.settle_seconds)
                    env.pause()

                # 无人机世界坐标 (MAVROS)
                cp = env.comm.current_position
                drone_wx = float(cp.x)
                drone_wy = float(cp.y)
                drone_wz = float(cp.z)

                # 无人机速度 (尝试多种来源)
                drone_vz = float("nan")
                if hasattr(env.comm, 'current_velocity') and env.comm.current_velocity is not None:
                    drone_vz = float(env.comm.current_velocity.z)
                elif hasattr(env, 'drone_linear_velocity') and env.drone_linear_velocity is not None:
                    drone_vz = float(env.drone_linear_velocity.z)
                elif len(episode_drone_positions) >= 2:
                    # 从位置差分估算触地速度
                    p1 = episode_drone_positions[-1]
                    p0 = episode_drone_positions[-2]
                    drone_vz = (p1[2] - p0[2]) / args.dt

                # 船当前位置 (/gazebo/model_states 反馈)
                ship_pos_arr = controller.get_current_pos()
                ship_sx = float(ship_pos_arr[0])
                ship_sy = float(ship_pos_arr[1])
                ship_sz = 0.1  # ship z 在 launch 中固定为 0.1

                # 计算目标点 (船 base_link + marker 偏移)
                target_x, target_y, target_z = compute_dynamic_target(
                    ship_sx, ship_sy, ship_sz, marker_offset_z=evaluator.marker_offset_z
                )
            except Exception as e:
                episode_crashed = True
                crash_msg = (crash_msg + " | " if crash_msg else "") + f"settle: {e}"

            # ---- 4d. 填充 EpisodeData 并评估 ----
            ep_data = EpisodeData(
                episode_id=episode_id,
                drone_final_x=drone_wx,
                drone_final_y=drone_wy,
                drone_final_z=drone_wz,
                target_x=target_x if np.isfinite(target_x) else float("nan"),
                target_y=target_y if np.isfinite(target_y) else float("nan"),
                target_z=target_z if np.isfinite(target_z) else float("nan"),
                steps=step_count,
                dt=args.dt,
                init_dist_3d=float(np.linalg.norm([drone_wx - target_x, drone_wy - target_y, drone_wz - target_z]))
                    if not episode_crashed and all(np.isfinite(v) for v in [drone_wx, target_x])
                    else float("nan"),
                max_height=episode_max_height,
                max_dist_origin=float(np.sqrt(drone_wx**2 + drone_wy**2 + drone_wz**2))
                    if np.isfinite(drone_wx) else float("nan"),
                impact_velocity_z=drone_vz,
                actions=episode_actions,
                drone_positions=episode_drone_positions,
                crashed=episode_crashed,
                crash_msg=crash_msg,
                out_of_bounds=out_of_bounds,
                max_steps_reached=(step_count >= args.max_steps),
                lost_detection=lost_detection,
                ckpt_dir=args.ckpt_dir,
                load_step=args.load_step,
                platform_type="dynamic",
            )

            record = evaluator.evaluate(ep_data)
            record["MaxSteps"] = args.max_steps  # 补充配置信息
            records.append(record)

            result_str = record["Result"]
            if result_str in ("PERFECT", "GOOD", "ACCEPTABLE"):
                success_count += 1
            elif result_str in ("MISSED", "HARD_LANDING", "OFF_PLATFORM", "MAX_STEPS", "OUT_OF_BOUNDS", "LOST_DETECTION"):
                fail_count += 1
            else:
                crash_count += 1

            total_steps += step_count

            # ---- 4e. 打印 ----
            ship_pos = controller.get_current_pos()
            comp = record.get("CompositeScore", float("nan"))
            rating = record.get("Rating", "N/A")
            herr = record.get("HorizErr", float("nan"))
            print(f"  Ep {episode_id:3d}/{args.test_episodes} | {result_str:20s} "
                  f"score={comp:5.1f} ({rating}) | "
                  f"steps={step_count:3d} | "
                  f"drone=({drone_wx:.2f},{drone_wy:.2f},{drone_wz:.2f}) "
                  f"herr={herr:.3f}m | "
                  f"船=({ship_pos[0]:.1f},{ship_pos[1]:.1f})")

        # ==================== 汇总 ====================
        summary = evaluator.summarize(records)
        print("\n" + "=" * 60)
        print("  评估汇总")
        print("=" * 60)
        print(f"  总轮数:         {summary.get('N_total', 0)}")
        print(f"  PERFECT:        {summary.get('N_PERFECT', 0)}")
        print(f"  GOOD:           {summary.get('N_GOOD', 0)}")
        print(f"  ACCEPTABLE:     {summary.get('N_ACCEPTABLE', 0)}")
        print(f"  MISSED:         {summary.get('N_LANDED', 0) - summary.get('N_PERFECT', 0) - summary.get('N_GOOD', 0) - summary.get('N_ACCEPTABLE', 0)}  (含 HARD_LANDING / OFF_PLATFORM)")
        print(f"  ABORTED:        {summary.get('N_ABORTED', 0)}")
        print(f"  RESET_FAILED + CRASHED: {summary.get('N_RESET_FAILED', 0) + summary.get('N_CRASHED', 0)}")
        print(f"  ---")
        print(f"  SR (含全部):    {summary.get('SR', 0):.1f}%")
        print(f"  SR (可用):      {summary.get('SR_usable', 0):.1f}%")
        if np.isfinite(summary.get('Mean_CompositeScore', float('nan'))):
            print(f"  平均综合分:     {summary['Mean_CompositeScore']:.1f} ± {summary['Std_CompositeScore']:.1f}")
            print(f"    精度: {summary.get('Mean_Score_Accuracy', 0):.1f}  "
                  f"效率: {summary.get('Mean_Score_Efficiency', 0):.1f}  "
                  f"平滑: {summary.get('Mean_Score_Smoothness', 0):.1f}  "
                  f"安全: {summary.get('Mean_Score_Safety', 0):.1f}")
        print(f"  模型:           {args.ckpt_dir}  step={args.load_step}")
        print(f"  模式:           {args.curve}  "
              f"vx={args.vx:.0f} vy={args.vy:.0f}")
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
