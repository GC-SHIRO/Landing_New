#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step0: 船舶运动测试场
=====================
独立船舶速度轨迹验证环境, 无无人机依赖。
每 N 秒自动重置 (瞬移回原点, 速度归零), 支持 Step1 后续几种运动模式。

用法:
  python step0_env.py --mode constant --vx 0.15 --vy 0.0
  python step0_env.py --mode linear   --vx 0.15 --vy 0.0 --max_disp 15
  python step0_env.py --mode varspeed --vx 0.15 --vy 0.0 --v_min 0.1 --v_max 0.4
  python step0_env.py --mode sine     --v 0.2 --amp 2.0 --wavelen 8.0 --vx 0.15 --vy 0.0
  python step0_env.py --mode circle   --radius 3.0 --period 40.0 --vx 0.15 --vy 0.0
  python step0_env.py --mode combined --curve sine --v_min 0.1 --v_max 0.4

参数:
  --mode MODE           constant | linear | varspeed | sine | circle | combined
  --launch FILE         launch 文件 (默认 step1_linear.launch)
  --no-launch           不启动 Gazebo
  --duration N          总运行时间 秒 (默认 300)
  --reset_interval N    重置间隔 秒 (默认 30)
  --dt N                控制步长 秒 (默认 0.1)
  --log_interval N      日志间隔 步数 (默认 10)

  运动参数 (通用):
  --vx VX               目标速度 x 分量 m/s (默认 0.15)
  --vy VY               目标速度 y 分量
  --max_disp N          往复最大位移 m (linear 模式, 默认 15)

  varspeed / combined 变速参数:
  --v_min N             最小速率 m/s (默认 0.1)
  --v_max N             最大速率 m/s (默认 0.4)

  sine 参数:
  --v N                 弧长速率 m/s (默认 0.2)
  --amp N               法向摆幅 m (默认 2.0)
  --wavelen N           空间波长 m (默认 8.0)

  circle 参数:
  --radius N            圆半径 m (默认 3.0)
  --period N            绕行周期 s (默认 40.0)

  combined 参数:
  --curve MODE          底层曲线: sine | circle (默认 sine)
  --seed N              随机种子
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import math

import numpy as np
import rospy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Simulation.ship_motion import ShipMotionController

DEFAULT_LAUNCH = "/home/shiro/PX4_Firmware/launch/step1_linear.launch"
SHIP_INIT_X = 10.0
SHIP_INIT_Y = 5.0


def _launch_gazebo(launch_file):
    """启动 Gazebo 仿真进程"""
    rospy.loginfo(f"启动 Gazebo: {launch_file}")
    proc = subprocess.Popen(
        ["roslaunch", launch_file, "gui:=true"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    rospy.loginfo("等待 Gazebo 加载 (15s)...")
    time.sleep(15)
    return proc


def main():
    parser = argparse.ArgumentParser(description="Step0: 船舶速度轨迹测试场")

    parser.add_argument('--mode', type=str, default='circle',
                        choices=['constant', 'linear', 'varspeed',
                                 'sine', 'circle', 'combined'],
                        help='运动模式')

    # 通用运动参数
    parser.add_argument('--vx', type=float, default=0.15,
                        help='目标速度 x 分量 m/s')
    parser.add_argument('--vy', type=float, default=0.0,
                        help='目标速度 y 分量 m/s')
    parser.add_argument('--max_disp', type=float, default=15.0,
                        help='往复最大位移 m (linear)')

    parser.add_argument('--v_min', type=float, default=0.4,
                        help='最小速率 m/s (varspeed/combined)')
    parser.add_argument('--v_max', type=float, default=1.0,
                        help='最大速率 m/s (varspeed/combined)')

    parser.add_argument('--v', type=float, default=1.0,
                        help='弧长速率 m/s (sine)')
    parser.add_argument('--amp', type=float, default=2.0,
                        help='法向摆幅 m (sine)')
    parser.add_argument('--wavelen', type=float, default=8.0,
                        help='空间波长 m (sine)')

    parser.add_argument('--radius', type=float, default=3.0,
                        help='圆半径 m (circle)')
    parser.add_argument('--period', type=float, default=40.0,
                        help='绕行周期 s (circle)')

    parser.add_argument('--curve', type=str, default='sine',
                        choices=['sine', 'circle'],
                        help='底层曲线类型 (combined)')

    # 测试控制
    parser.add_argument('--launch', type=str, default=DEFAULT_LAUNCH,
                        help='launch 文件路径')
    parser.add_argument('--no-launch', action='store_true',
                        help='不启动 Gazebo (已手动启动时)')
    parser.add_argument('--duration', type=float, default=300.0,
                        help='总运行时间 秒')
    parser.add_argument('--reset_interval', type=float, default=30.0,
                        help='重置间隔 秒')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='控制步长 秒')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志间隔 步数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')

    args = parser.parse_args()

    # ==================== 打印配置 ====================
    print("=" * 60)
    print("  Step0 船舶速度轨迹测试场")
    print("=" * 60)
    spd = math.sqrt(args.vx ** 2 + args.vy ** 2)
    print(f"  模式:         {args.mode}")
    print(f"  速度/方向:    vx={args.vx:.2f} vy={args.vy:.2f}  合速度={spd:.3f} m/s")
    if args.mode == 'linear':
        print(f"  最大位移:     {args.max_disp:.1f} m")
    elif args.mode == 'varspeed':
        print(f"  速率范围:     [{args.v_min:.2f}, {args.v_max:.2f}] m/s")
    elif args.mode == 'sine':
        print(f"  弧长速率:     {args.v:.2f} m/s  摆幅={args.amp:.1f}m  波长={args.wavelen:.1f}m")
    elif args.mode == 'circle':
        print(f"  半径/周期:    radius={args.radius:.1f}m  period={args.period:.1f}s")
    elif args.mode == 'combined':
        print(f"  底层曲线:     {args.curve}  速率范围=[{args.v_min:.2f}, {args.v_max:.2f}] m/s")
    print(f"  总时长:       {args.duration:.0f}s")
    print(f"  重置间隔:     {args.reset_interval:.0f}s")
    print(f"  控制步长:     {args.dt}s")
    print("=" * 60)

    gazebo_proc = None
    controller = None

    try:
        # ---- 1. 启动 Gazebo ----
        if not args.no_launch:
            gazebo_proc = _launch_gazebo(args.launch)

        # ---- 2. 初始化 ROS 节点 ----
        rospy.init_node('step0_ship_test', anonymous=True, disable_signals=True)

        # ---- 3. 创建船舶控制器 ----
        print("\n创建 ShipMotionController...")
        controller = ShipMotionController(
            ship_name="wamv",
            init_pos=(SHIP_INIT_X, SHIP_INIT_Y),
        )

        v_range = (args.v_min, args.v_max)
        if args.mode == 'constant':
            controller.set_mode_constant(args.vx, args.vy)
        elif args.mode == 'linear':
            controller.set_mode_linear(args.vx, args.vy, args.max_disp)
        elif args.mode == 'varspeed':
            controller.set_mode_varspeed((args.vx, args.vy), v_range, args.seed)
        elif args.mode == 'sine':
            controller.set_mode_sine(args.v, args.amp, args.wavelen, args.vx, args.vy)
        elif args.mode == 'circle':
            controller.set_mode_circle(args.radius, args.period, args.vx, args.vy)
        elif args.mode == 'combined':
            if args.curve == 'sine':
                controller.set_mode_sine(args.v, args.amp, args.wavelen, args.vx, args.vy)
            else:
                controller.set_mode_circle(args.radius, args.period, args.vx, args.vy)
            controller.set_mode_combined(args.curve, v_range, args.seed)

        print(f"  ✓ 模式: {args.mode}")

        # 轨迹控制器内部直接维护船位姿
        print("  初始化船舶轨迹状态...")
        if not controller.wait_for_odom(timeout=15.0):
            print("  ⚠ 船舶轨迹状态不可用")
        else:
            state = controller.get_state()
            print(f"  ✓ 船舶轨迹状态: pos=({state['pos'][0]:.2f}, {state['pos'][1]:.2f}), "
                  f"yaw={math.degrees(state['yaw']):.1f}°")

        # ---- 4. 测试循环 ----
        print(f"\n开始测试 ({args.duration:.0f}s, {args.reset_interval:.0f}s 重置)...")
        print("-" * 72)
        print(f"{'Cyc':>4s} {'t_wall':>7s} {'t_cyc':>6s} {'step':>5s} | "
              f"{'x':>7s} {'y':>7s} {'speed':>6s} | "
              f"{'yaw':>7s}")
        print("-" * 72)

        t_start = time.time()
        cycle_count = 0
        total_steps = 0

        while True:
            t_elapsed = time.time() - t_start
            if t_elapsed >= args.duration:
                break

            # --- 瞬移回原点 ---
            controller.teleport_to_origin()
            time.sleep(0.3)

            t_cycle = 0.0
            cycle_steps = 0

            while t_cycle < args.reset_interval:
                if time.time() - t_start >= args.duration:
                    break

                controller.step(t_cycle)

                if cycle_steps % args.log_interval == 0:
                    di = controller.debug_info
                    odom_vel = di['odom_vel']
                    speed = float(np.linalg.norm(odom_vel))
                    print(f"{cycle_count:4d} {time.time() - t_start:7.1f}s "
                          f"{t_cycle:6.1f}s {cycle_steps:5d} | "
                          f"{di['odom_pos'][0]:7.2f} {di['odom_pos'][1]:7.2f} "
                          f"{speed:6.3f} | "
                          f"{math.degrees(di['left_angle']):7.1f}°")

                time.sleep(args.dt)
                t_cycle += args.dt
                cycle_steps += 1
                total_steps += 1

            controller._publish_zero()
            cycle_count += 1

            di = controller.debug_info
            print(f"  -- 周期 {cycle_count - 1} 结束 ({cycle_steps}步, {t_cycle:.1f}s) "
                  f"终点: ({di['odom_pos'][0]:.2f}, {di['odom_pos'][1]:.2f}) --")

        # ==================== 汇总 ====================
        print("\n" + "=" * 60)
        print("  测试完成")
        print("=" * 60)
        print(f"  总周期数:     {cycle_count}")
        print(f"  总步数:       {total_steps}")
        print(f"  总时长:       {time.time() - t_start:.1f}s")
        print(f"  模式:         {args.mode}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n用户中断")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if controller is not None:
            controller.shutdown()
        if gazebo_proc is not None:
            print("\n关闭 Gazebo...")
            try:
                os.killpg(os.getpgid(gazebo_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        print("清理完成")


if __name__ == "__main__":
    main()
