#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
起飞诊断脚本 —— 在仿真运行中监听关键话题，记录 5 秒数据后打印报告。

用法（仿真已在运行，飞机处于 OFFBOARD+armed 状态时执行）：
  python diagnose_takeoff.py

输出:
  - setpoint_raw/local 的实际 type_mask 和 velocity.z 每帧
  - mavros/state 的 mode / armed
  - local_position/pose 的 z 高度变化
  - extended_state 的 landed_state
  - 最终结论
"""

import rospy
import time
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from geometry_msgs.msg import PoseStamped

VEHICLE = "iris_0"
DURATION = 8.0  # 监听秒数

sp_log    = []   # (t, type_mask, coord_frame, vz, pz)
state_log = []   # (t, mode, armed)
pose_log  = []   # (t, z)
ext_log   = []   # (t, landed_state)

t0 = None

def cb_sp(msg):
    t = time.time() - t0
    sp_log.append((round(t,3), msg.type_mask, msg.coordinate_frame,
                   round(msg.velocity.z,4), round(msg.position.z,4)))

def cb_state(msg):
    t = time.time() - t0
    state_log.append((round(t,3), msg.mode, msg.armed))

def cb_pose(msg):
    t = time.time() - t0
    pose_log.append((round(t,3), round(msg.pose.position.z,4)))

def cb_ext(msg):
    t = time.time() - t0
    # landed_state: 0=UNDEFINED, 1=ON_GROUND, 2=IN_AIR, 3=TAKEOFF, 4=LANDING
    NAMES = {0:"UNDEFINED",1:"ON_GROUND",2:"IN_AIR",3:"TAKEOFF",4:"LANDING"}
    ext_log.append((round(t,3), NAMES.get(msg.landed_state, msg.landed_state)))

def main():
    global t0
    rospy.init_node("diagnose_takeoff", anonymous=True)

    rospy.Subscriber(f"/{VEHICLE}/mavros/setpoint_raw/local", PositionTarget, cb_sp)
    rospy.Subscriber(f"/{VEHICLE}/mavros/state",              State,          cb_state)
    rospy.Subscriber(f"/{VEHICLE}/mavros/local_position/pose",PoseStamped,    cb_pose)
    rospy.Subscriber(f"/{VEHICLE}/mavros/extended_state",     ExtendedState,  cb_ext)

    print(f"[diagnose] 监听 {VEHICLE} 共 {DURATION}s ...")
    t0 = time.time()
    rospy.sleep(DURATION)

    # ===== 报告 =====
    print("\n" + "="*65)
    print("  TAKEOFF DIAGNOSIS REPORT")
    print("="*65)

    # --- setpoint_raw 统计 ---
    print(f"\n[setpoint_raw/local]  共收到 {len(sp_log)} 帧")
    vel_frames  = [(t,tm,cf,vz,pz) for t,tm,cf,vz,pz in sp_log if (tm & 0b111) != 0b111]
    pos_frames  = [(t,tm,cf,vz,pz) for t,tm,cf,vz,pz in sp_log if (tm & 0b111) == 0b111]

    # type_mask bit 0-2: IGNORE_VX/VY/VZ; bit 3-5: IGNORE_AFX/AFY/AFZ; bit 6: FORCE; bit 10: IGNORE_PX/PY/PZ
    # 速度模式: IGNORE_PX|PY|PZ 置1 (bits 10-12) => type_mask & 0x1C00 != 0
    vel_sp  = [(t,tm,cf,vz,pz) for t,tm,cf,vz,pz in sp_log if (tm & 0x1C00) and not (tm & 0x7)]
    pos_sp  = [(t,tm,cf,vz,pz) for t,tm,cf,vz,pz in sp_log if not (tm & 0x1C00) and (tm & 0x7)]

    print(f"  位置模式帧 (IGNORE_VX/VY/VZ set): {len(pos_sp)}")
    print(f"  速度模式帧 (IGNORE_PX/PY/PZ set): {len(vel_sp)}")

    if vel_sp:
        vzs = [vz for _,_,_,vz,_ in vel_sp]
        print(f"  速度帧 vz 范围: [{min(vzs):.3f}, {max(vzs):.3f}]  平均: {sum(vzs)/len(vzs):.3f}")
    else:
        print("  !! 警告: 未检测到速度模式 setpoint！PX4 始终收到的是位置模式指令")

    print(f"\n  最近 10 帧 (t, type_mask_hex, coord_frame, vz, pz):")
    for row in sp_log[-10:]:
        t,tm,cf,vz,pz = row
        print(f"    t={t:.2f}  mask=0x{tm:04X}  cf={cf}  vz={vz:+.3f}  pz={pz:.3f}")

    # --- state ---
    print(f"\n[mavros/state]  最近 5 条:")
    for row in state_log[-5:]:
        print(f"    t={row[0]:.2f}  mode={row[1]}  armed={row[2]}")

    # --- pose ---
    if pose_log:
        zs = [z for _,z in pose_log]
        dz = zs[-1] - zs[0]
        print(f"\n[local_position/pose]  z: {zs[0]:.3f} → {zs[-1]:.3f}  Δz={dz:+.3f}m  ({len(pose_log)} 帧)")
        if dz < 0.05:
            print("  !! 警告: 高度几乎没有变化，飞机未离地")
        else:
            print(f"  OK: 高度上升 {dz:.3f}m")

    # --- extended_state ---
    print(f"\n[extended_state]  最近 5 条:")
    for row in ext_log[-5:]:
        print(f"    t={row[0]:.2f}  landed_state={row[1]}")

    # --- 结论 ---
    print("\n" + "-"*65)
    print("  结论:")
    no_vel = len(vel_sp) == 0
    no_rise = len(pose_log) > 1 and (pose_log[-1][1] - pose_log[0][1]) < 0.05
    on_ground = any(s == "ON_GROUND" for _,s in ext_log[-3:]) if ext_log else False

    if no_vel:
        print("  ✗ PX4 从未收到速度模式 setpoint —— start() 线程位置 setpoint 覆盖了速度指令")
        print("    修复: 速度起飞期间必须停止 start() 线程或暂停其发布")
    elif no_rise:
        print("  ✗ 速度 setpoint 已发送但飞机未上升")
        if on_ground:
            print("    landed_state=ON_GROUND: Gazebo 碰撞/地形卡住飞机，检查平台模型 z 和 spawn_z")
        else:
            print("    检查 PX4 log: 是否有 prearm check 未通过 / EKF 未收敛")
    else:
        print("  OK: 速度 setpoint 已发送且飞机正在上升")
    print("="*65)


if __name__ == "__main__":
    main()
