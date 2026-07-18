#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import threading
import subprocess
import sys
import time
import math
import numpy as np
import os
import signal

from mavros_msgs.msg import PositionTarget, ParamValue, State
from mavros_msgs.srv import CommandBool, SetMode, ParamSet
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import PoseStamped, Pose, Twist, TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from apriltag_ros.msg import AprilTagDetectionArray
from pyquaternion import Quaternion
from std_msgs.msg import String

# 定义常量
COLLISION_DIST = 0.35
TIME_DELTA = 0.1
STEP = 260
SYSTEM_WARMUP_SECONDS = 10.0


class GazeboEnv:
    """所有Gazebo环境的基类 (优化版：YOLO 常驻，减少 IO 开销)
    """

    def __init__(self, launchfile, vehicle_type, vehicle_id):
        rospy.loginfo("=== 初始化 GazeboEnv (Persistent YOLO Mode) ===")
        rospy.loginfo(f"参数: launchfile={launchfile}, vehicle={vehicle_type}_{vehicle_id}")

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        rospy.loginfo("ROS 核心已启动，端口 %s", port)

        rospy.init_node("Landing_env", anonymous=True, disable_signals=True)
        time.sleep(5.0)
        rospy.loginfo("正在启动Gazebo环境...")

        self.gazebo_process = subprocess.Popen(["roslaunch", "-p", port, launchfile])
        rospy.loginfo("Gazebo 环境已启动 => %s", launchfile)
        time.sleep(10)

        rospy.loginfo("初始化通信模块...")
        self.comm = Communication(vehicle_type, vehicle_id)

        th = threading.Thread(target=self.comm.start)
        th.daemon = True
        th.start()
        rospy.loginfo("通信线程已启动")

        # Publishers / Subscribers
        rospy.loginfo("正在设置ROS话题...")
        self.set_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)

        velocity_topic = f"{vehicle_type}_{vehicle_id}/mavros/local_position/velocity_local"
        rospy.loginfo(f"订阅速度话题: {velocity_topic}")
        self.velocity_sub = rospy.Subscriber(velocity_topic, TwistStamped, self.velocity_callback)

        yolo_centers_topic = "/yolov11/centers"
        rospy.loginfo(f"订阅YOLO中心点话题: {yolo_centers_topic}")
        self.yolo_centers_sub = rospy.Subscriber(yolo_centers_topic, PointStamped, self.yolo_centers_callback)

        self.multi_cmd_vel_flu_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd_vel_flu', Twist, queue_size=1)
        self.multi_cmd_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd', String, queue_size=1)
        self.multi_cmd_pose_enu_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd_pose_enu', Pose, queue_size=1)
        self.local_pose_pub = rospy.Publisher(vehicle_type+'/mavros/local_position/pose', PoseStamped, queue_size=1)
        # Gazebo services
        rospy.loginfo("正在连接Gazebo服务...")
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=10.0)
            self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
            rospy.loginfo("已连接unpause_physics服务")

            rospy.wait_for_service("/gazebo/pause_physics", timeout=10.0)
            self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
            rospy.loginfo("已连接pause_physics服务")

            rospy.wait_for_service("/gazebo/reset_world", timeout=10.0)
            self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            rospy.loginfo("已连接reset_world服务")
        except rospy.ROSException as e:
            rospy.logerr(f"连接Gazebo服务超时: {e}")
            raise

        # 环境变量与状态
        self.prev_shaping = 0.0
        self.posestampesd = PoseStamped()
        self.twist = Twist()
        self.cmd = String()
        self.pose = Pose()

        # YOLO 相关
        self.yolo_detected = False
        self.yolo_center_x = 0.5
        self.yolo_center_y = 0.5
        self.yolo_distance_z = 10.0
        self.last_yolo_detection_time = rospy.Time.now()

        self.Step = 0
        self.drone_linear_velocity = None
        self.drone_angular_velocity = None

        self.yolo_process = None
        self.yolo_package = "yolov11_ros"
        self.yolo_launch_file = "yolo_v11.launch"

        # ===== Quadrant-balanced reset sampling =====
        self.quadrant_idx = 0  # 0~3 轮流


        # === 修改点 1: 初始化时直接启动 YOLO，之后不再关闭 ===
        self.start_yolo()
        rospy.loginfo(f"系统预热等待 {SYSTEM_WARMUP_SECONDS:.1f}s，等待 ROS/XTDrone/MAVROS 话题稳定...")
        time.sleep(SYSTEM_WARMUP_SECONDS)
        rospy.loginfo("系统预热完成")

    def start_yolo(self):
        if self.yolo_process is not None:
            rospy.logwarn("YOLO进程已经在运行")
            return
        try:
            rospy.loginfo("启动YOLO检测进程...")
            launch_cmd = ["roslaunch", "yolov11_ros", "yolo_v11.launch"]
            self.yolo_process = subprocess.Popen(
                launch_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                text=True
            )
            # 等待稍微久一点，确保显存加载完毕
            time.sleep(8) 
            if self.yolo_process.poll() is not None:
                stdout, stderr = self.yolo_process.communicate()
                rospy.logerr(f"YOLO进程启动失败: {stderr}")
                self.yolo_process = None
                return
            rospy.loginfo("YOLO进程启动成功")
        except Exception as e:
            rospy.logerr(f"启动YOLO进程失败: {e}")
            self.yolo_process = None

    def stop_yolo(self):
        """只在整个脚本退出时调用，Reset时不调用"""
        if self.yolo_process is not None:
            try:
                rospy.loginfo("停止YOLO检测进程...")
                os.killpg(os.getpgid(self.yolo_process.pid), signal.SIGTERM)
                self.yolo_process.wait(timeout=5)
                rospy.loginfo("YOLO进程已停止")
            except subprocess.TimeoutExpired:
                rospy.logwarn("YOLO进程未正常终止，强制杀死...")
                os.killpg(os.getpgid(self.yolo_process.pid), signal.SIGKILL)
            except Exception as e:
                rospy.logerr(f"停止YOLO进程时出错: {e}")
            finally:
                self.yolo_process = None
        else:
            rospy.loginfo("YOLO进程未运行，无需停止")

    def yolo_centers_callback(self, msg):
        try:
            self.yolo_center_x = msg.point.x
            self.yolo_center_y = msg.point.y
            self.yolo_distance_z = msg.point.z
            self.last_yolo_detection_time = rospy.Time.now()
            self.yolo_detected = True
            # rospy.logdebug(f"YOLO检测: ...") # 减少日志打印以节省资源
        except Exception as e:
            rospy.logwarn(f"YOLO回调函数错误: {e}")
            self.yolo_detected = False

    def velocity_callback(self, msg):
        self.drone_linear_velocity = msg.twist.linear
        self.drone_angular_velocity = msg.twist.angular

    def step(self, action):
        # rospy.logdebug(f"===== STEP开始... =====")
        done = False
        target = False

        self.twist.linear.x = action[0]
        self.twist.linear.y = action[1]
        self.twist.linear.z = action[2]

        self.multi_cmd_vel_flu_pub.publish(self.twist)

        # === 修改点 2: 增加 try-except 防止 Timeout 崩溃 ===
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=1.0)
            self.unpause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"Unpause physics failed (Timeout/Error): {e}")

        time.sleep(TIME_DELTA)

        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=1.0)
            self.pause()
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn(f"Pause physics failed (Timeout/Error): {e}")
        # =================================================

        next_state = self.get_state()

        self.Step += 1
        
        # === 修改点 3: 达到最大步数时，只 set done，不杀 YOLO ===
        if self.Step == STEP:
            done = True
            # self.stop_yolo()  <-- 已注释
            self.Step = 0

        # print 减少频率，只打印关键信息
        # print(f"世界位置... 相对位置...") 
        
        if self.yolo_detected and abs(self.comm.current_position.z) < 0.4:
            target = True
            done = True
            self.Step = 0
            rospy.loginfo("成功着陆！")

        if hasattr(self.comm, 'current_position'):
            x0 = self.comm.current_position.x
            y0 = self.comm.current_position.y
            z0 = self.comm.current_position.z
            dist_origin = math.sqrt(x0**2 + y0**2 + z0**2)
            if dist_origin > 15.0 or z0 > 12:
                rospy.logwarn(f"越界结束: Dist={dist_origin:.2f}, Height={z0:.2f}")
                done = True
                self.Step = 0
                # self.stop_yolo() <-- 已注释

        if not self.yolo_detected and hasattr(self.comm, 'current_position'):
            if self.comm.current_position.z < 0.25:
                rospy.logwarn(f"越界结束: 未检测到目标且高度过低")
                done = True
                self.Step = 0
                # self.stop_yolo() <-- 已注释

        success = target and done
        info = {
            "tag_detected": self.yolo_detected,
            "target_reached": target,
            "success": success
        }
        return next_state, done, success, info

    def reset(self):
        # 同样的保护
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=1.0)
            self.unpause()
        except Exception:
            pass

        # === 修改点 4: Reset 时重置内部标志位，而不是重启进程 ===
        self.yolo_detected = False
        # self.stop_yolo() <-- 已注释
        # self.start_yolo() <-- 已注释

        # 启动位置发布循环
        if not hasattr(self, 'pose_publishing_active') or not self.pose_publishing_active:
            self.pose_publishing_active = True
            def _publish_pose_loop():
                rate = rospy.Rate(30)
                while self.pose_publishing_active and not rospy.is_shutdown():
                    if self.comm.current_position is not None:
                        pose = PoseStamped()
                        pose.header.stamp = rospy.Time.now()
                        pose.pose.position = self.comm.current_position
                        self.local_pose_pub.publish(pose)
                    else:
                        pose = PoseStamped()
                        pose.header.stamp = rospy.Time.now()
                        pose.pose.position.x = 0.0
                        pose.pose.position.y = 0.0
                        pose.pose.position.z = 0.0
                        self.local_pose_pub.publish(pose)
                    rate.sleep()
            pose_thread = threading.Thread(target=_publish_pose_loop)
            pose_thread.daemon = True
            pose_thread.start()

        rospy.loginfo("等待当前位置数据...")
        try:
            pose_topic = f"{self.comm.vehicle_type}/mavros/local_position/pose"
            pose_msg = rospy.wait_for_message(pose_topic, PoseStamped, timeout=30.0)
            self.comm.current_position = pose_msg.pose.position
            self.comm.current_yaw = self.comm.q2yaw(pose_msg.pose.orientation)
            rospy.loginfo("当前位置数据已接收")
        except rospy.ROSException as e:
            rospy.logerr(f"等待位置数据超时: {e}")
            raise RuntimeError("无法获取初始位置")

        current_pose = self.comm.construct_target(
            x=self.comm.current_position.x,
            y=self.comm.current_position.y,
            z=self.comm.current_position.z,
            yaw=self.comm.current_yaw
        )

        # 确保 target_motion_pub 连接
        timeout_connection = 10.0
        start_wait = time.time()
        while (time.time() - start_wait) < timeout_connection and not rospy.is_shutdown():
            if self.comm.target_motion_pub.get_num_connections() > 0:
                break
            rospy.sleep(0.1)
        
        # 预发送 setpoints
        t_end = time.time() + 2.0
        while time.time() < t_end and not rospy.is_shutdown():
            self.comm.target_motion_pub.publish(current_pose)
            rospy.sleep(0.05)

        # 切换 OFFBOARD
        self.comm.arm_state = self.comm.arm()
        self.comm.flightModeService(custom_mode="OFFBOARD")
        
        # 等待生效
        wait_start = time.time()
        while time.time() - wait_start < 5.0:
            try:
                mav_state = rospy.wait_for_message(self.comm.vehicle_type+'_'+self.comm.vehicle_id+'/mavros/state', State, timeout=1.0)
                if mav_state.mode == 'OFFBOARD' and mav_state.armed:
                    break
            except Exception:
                pass
            self.comm.target_motion_pub.publish(current_pose)
            rospy.sleep(0.1)

        rospy.sleep(0.5)

        # 随机起始位置（四象限均衡：以矩形中心为“零点”轮流采样）
        self.pose = Pose()

        # 原始范围
        x_min, x_max = -7.0, 2.3
        y_min, y_max = -7.5, 5.0
        z_min, z_max = 8.49, 8.5

        # 以矩形中心作为“零点”
        cx = 0.5 * (x_min + x_max)   # -2.35
        cy = 0.5 * (y_min + y_max)   # -1.25

        # 四象限轮流（保证每个象限出现概率相同）
        # Q1: x>cx, y>cy
        # Q2: x<cx, y>cy
        # Q3: x<cx, y<cy
        # Q4: x>cx, y<cy
        q = getattr(self, "quadrant_idx", 0) % 4
        self.quadrant_idx = getattr(self, "quadrant_idx", 0) + 1

        # 可选：避免采到刚好等于中心点导致边界归属不稳定
        eps = 1e-6

        if q == 0:      # Q1 (+,+)
            rand_x = np.random.uniform(cx + eps, x_max)
            rand_y = np.random.uniform(cy + eps, y_max)
        elif q == 1:    # Q2 (-,+)
            rand_x = np.random.uniform(x_min, cx - eps)
            rand_y = np.random.uniform(cy + eps, y_max)
        elif q == 2:    # Q3 (-,-)
            rand_x = np.random.uniform(x_min, cx - eps)
            rand_y = np.random.uniform(y_min, cy - eps)
        else:           # Q4 (+,-)
            rand_x = np.random.uniform(cx + eps, x_max)
            rand_y = np.random.uniform(y_min, cy - eps)

        rand_z = np.random.uniform(z_min, z_max)

        
        self.pose.position.x = rand_x
        self.pose.position.y = rand_y
        self.pose.position.z = rand_z
        self.pose.orientation.w = 1.0 

        rospy.loginfo(f"OFFBOARD: 移动到起始点 ({rand_x:.1f}, {rand_y:.1f}, {rand_z:.1f})")
        
        start_time = time.time()
        arrival_count = 0
        rate = rospy.Rate(10)
        
        while time.time() - start_time < 20.0 and not rospy.is_shutdown():
            self.multi_cmd_pose_enu_pub.publish(self.pose)
            
            current_x = self.comm.current_position.x
            current_y = self.comm.current_position.y
            current_z = self.comm.current_position.z
            dist = math.sqrt((rand_x - current_x)**2 + (rand_y - current_y)**2 + (rand_z - current_z)**2)
            
            if dist < 0.3:
                arrival_count += 1
                if arrival_count > 15:
                    break
            else:
                arrival_count = 0
            rate.sleep()

        # 悬停稳定
        rospy.loginfo("悬停稳定中...")
        for _ in range(20):
            self.multi_cmd_pose_enu_pub.publish(self.pose)
            rospy.sleep(0.1)

        # === 修改点 5: 不再重启 YOLO，而是直接准备开始 ===
        # self.stop_yolo() <-- 已注释
        # self.start_yolo() <-- 已注释
        
        # 确保模式正确
        if self.comm.flight_mode != "OFFBOARD":
             self.comm.flightModeService(custom_mode="OFFBOARD")

        # 暂停物理
        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=1.0)
            self.pause()
        except Exception:
            pass

        state = self.get_state()
        self.comm.flight_mode = None 
        rospy.loginfo("===== 重置完成 =====")
        return state

    def __del__(self):
        # 只有在最后才会杀掉 YOLO
        self.stop_yolo()

    def get_state(self):
        """
        获取无人机当前状态 [x, y, z, vx, vy, vz]
        维度: 6
        """
        # 1. 获取位置信息 (YOLO)
        # 默认值：如果没有检测到，给一个远离中心的坐标 (防止误判已对准)
        # 注意：这里我们使用你代码里的变量名 yolo_center_x 等
        pos_x, pos_y, pos_z = 0.0, 0.0, 10.0 
        
        # 检查是否检测到目标 (防御性编程)
        if hasattr(self, 'yolo_detected') and self.yolo_detected:
            pos_x = self.yolo_center_x
            pos_y = self.yolo_center_y
            pos_z = self.yolo_distance_z

        # 2. 获取速度信息 (Mavros) [核心修改部分]
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # 必须判断非空，因为ROS回调是异步的，刚启动时可能还没收到速度消息
        if self.drone_linear_velocity is not None:
            vx = self.drone_linear_velocity.x
            vy = self.drone_linear_velocity.y
            vz = self.drone_linear_velocity.z

        # 3. 拼接并返回 6维 数组
        # return np.array([pos_x, pos_y, pos_z, vx, vy, vz])
        return np.array([pos_x, pos_y, pos_z])
    
    def get_real_state(self):
        return np.array([self.comm.current_position.x+2, self.comm.current_position.y+2, self.comm.current_position.z])

    def reward_setup(self, observation, observation_, done, succes):
        delta_x = observation[0]
        delta_y = observation[1]
        height2 = observation[2]
        
        # 奖励函数保持你的逻辑
        shape2 = - ((abs(delta_x) ** 3 + abs(delta_y) ** 3 + abs(height2) ** 3) ** (1 / 3))
        reward = 0.1 * (shape2)

        if done:
            err_x_ = self.comm.current_position.x
            err_y_ = self.comm.current_position.y
            height2 = self.comm.current_position.z
            # 注意：这里的成功判定坐标范围是否需要根据实际降落点调整？
            # 暂时保持原样
            if -1.5 > err_x_ > -2.5 and -1.5 > err_y_ > -2.5 and succes:
                reward = 300
                print("landed successfully", observation, reward, observation_, height2)
                return reward
            else:
                reward = -200
                print('landed else where', observation, reward, observation_, height2)
                return reward
        return reward


class Communication:
    def __init__(self, vehicle_type, vehicle_id):
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id
        self.current_position = None
        self.current_yaw = 0
        self.hover_flag = 0
        self.coordinate_frame = 1

        self.target_motion = PositionTarget()
        self.target_motion.coordinate_frame = self.coordinate_frame

        self.desired_target = PositionTarget()
        self.desired_target.coordinate_frame = self.coordinate_frame
        self.last_cmd_time = rospy.Time.now()
        self.last_publish_time = rospy.Time.now()
        self.publish_rate = 30.0
        self.smoothing_tau = 0.1
        self.enable_rate_limit = True

        self.arm_state = False
        self.motion_type = 0
        self.flight_mode = None
        self.mission = None
        self.last_cmd = None

        mavros_state = rospy.wait_for_message(self.vehicle_type+'_'+self.vehicle_id+"/mavros/state", State)
        if not mavros_state.connected:
            rospy.logwarn(self.vehicle_type+'_'+self.vehicle_id+": No connection to FCU. Check mavros!")
            exit(0)

        self.local_pose_sub = rospy.Subscriber(self.vehicle_type+'_'+self.vehicle_id + "/mavros/local_position/pose",
                                              PoseStamped, self.local_pose_callback, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd",
                                        String, self.cmd_callback, queue_size=3)
        self.cmd_pose_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_pose_flu",
                                                Pose, self.cmd_pose_flu_callback, queue_size=1)
        self.cmd_pose_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_pose_enu",
                                                Pose, self.cmd_pose_enu_callback, queue_size=1)
        self.cmd_vel_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_vel_flu",
                                                Twist, self.cmd_vel_flu_callback, queue_size=3)
        self.cmd_vel_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_vel_enu",
                                                Twist, self.cmd_vel_enu_callback, queue_size=3)
        self.cmd_accel_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_accel_flu",
                                                 Twist, self.cmd_accel_flu_callback, queue_size=1)
        self.cmd_accel_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_accel_enu",
                                                 Twist, self.cmd_accel_enu_callback, queue_size=1)

        self.target_motion_pub = rospy.Publisher(self.vehicle_type+'_'+self.vehicle_id+"/mavros/setpoint_raw/local",
                                                PositionTarget, queue_size=1)

        self.armService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/cmd/arming", CommandBool)
        self.flightModeService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/set_mode", SetMode)
        self.set_param_srv = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/param/set", ParamSet)

        rcl_except = ParamValue(4, 0.0)
        try:
            self.set_param_srv("COM_RCL_EXCEPT", rcl_except)
        except Exception:
            rospy.logwarn("设置 COM_RCL_EXCEPT 失败")

        print(self.vehicle_type+'_'+self.vehicle_id+": communication initialized")

    def start(self):
        rate = rospy.Rate(self.publish_rate)
        self.last_publish_time = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_publish_time).to_sec()
            if dt <= 0:
                dt = 1.0 / self.publish_rate

            if self.enable_rate_limit and self.smoothing_tau > 0:
                alpha = dt / (self.smoothing_tau + dt)
            else:
                alpha = 1.0

            try:
                # 平滑插值逻辑
                self.target_motion.position.x = (1 - alpha) * self.target_motion.position.x + alpha * self.desired_target.position.x
                self.target_motion.position.y = (1 - alpha) * self.target_motion.position.y + alpha * self.desired_target.position.y
                self.target_motion.position.z = (1 - alpha) * self.target_motion.position.z + alpha * self.desired_target.position.z
                
                self.target_motion.velocity.x = (1 - alpha) * self.target_motion.velocity.x + alpha * self.desired_target.velocity.x
                self.target_motion.velocity.y = (1 - alpha) * self.target_motion.velocity.y + alpha * self.desired_target.velocity.y
                self.target_motion.velocity.z = (1 - alpha) * self.target_motion.velocity.z + alpha * self.desired_target.velocity.z
                
                self.target_motion.acceleration_or_force.x = (1 - alpha) * self.target_motion.acceleration_or_force.x + alpha * self.desired_target.acceleration_or_force.x
                self.target_motion.acceleration_or_force.y = (1 - alpha) * self.target_motion.acceleration_or_force.y + alpha * self.desired_target.acceleration_or_force.y
                self.target_motion.acceleration_or_force.z = (1 - alpha) * self.target_motion.acceleration_or_force.z + alpha * self.desired_target.acceleration_or_force.z
                
                self.target_motion.yaw = (1 - alpha) * self.target_motion.yaw + alpha * self.desired_target.yaw
                self.target_motion.yaw_rate = (1 - alpha) * self.target_motion.yaw_rate + alpha * self.desired_target.yaw_rate

                self.target_motion.type_mask = self.desired_target.type_mask
                self.target_motion.coordinate_frame = self.desired_target.coordinate_frame
            except Exception:
                pass

            try:
                self.target_motion_pub.publish(self.target_motion)
            except Exception:
                pass

            self.last_publish_time = now
            rate.sleep()

    def local_pose_callback(self, msg):
        self.current_position = msg.pose.position
        self.current_yaw = self.q2yaw(msg.pose.orientation)

    def construct_target(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, afx=0, afy=0, afz=0, yaw=0, yaw_rate=0):
        target_raw_pose = PositionTarget()
        target_raw_pose.coordinate_frame = self.coordinate_frame

        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z

        target_raw_pose.velocity.x = vx
        target_raw_pose.velocity.y = vy
        target_raw_pose.velocity.z = vz

        target_raw_pose.acceleration_or_force.x = afx
        target_raw_pose.acceleration_or_force.y = afy
        target_raw_pose.acceleration_or_force.z = afz

        target_raw_pose.yaw = yaw
        target_raw_pose.yaw_rate = yaw_rate

        if self.motion_type == 0:
            target_raw_pose.type_mask = (PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                            + PositionTarget.IGNORE_YAW_RATE)
        elif self.motion_type == 1:
            target_raw_pose.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                            + PositionTarget.IGNORE_YAW)
        elif self.motion_type == 2:
            target_raw_pose.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                            + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                            + PositionTarget.IGNORE_YAW)

        return target_raw_pose

    def set_desired_motion(self, position=None, velocity=None, accel=None, yaw=None, yaw_rate=None, coord_frame=None, motion_type=None):
        now = rospy.Time.now()
        self.last_cmd_time = now

        if motion_type is not None:
            self.motion_type = motion_type
        if coord_frame is not None:
            self.desired_target.coordinate_frame = coord_frame

        if position is not None:
            self.desired_target.position.x = position[0]
            self.desired_target.position.y = position[1]
            self.desired_target.position.z = position[2]
        if velocity is not None:
            self.desired_target.velocity.x = velocity[0]
            self.desired_target.velocity.y = velocity[1]
            self.desired_target.velocity.z = velocity[2]
        if accel is not None:
            self.desired_target.acceleration_or_force.x = accel[0]
            self.desired_target.acceleration_or_force.y = accel[1]
            self.desired_target.acceleration_or_force.z = accel[2]
        if yaw is not None:
            self.desired_target.yaw = yaw
        if yaw_rate is not None:
            self.desired_target.yaw_rate = yaw_rate

        if self.motion_type == 0:
            self.desired_target.type_mask = (PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                                + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                                + PositionTarget.IGNORE_YAW_RATE)
        elif self.motion_type == 1:
            self.desired_target.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                                + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                                + PositionTarget.IGNORE_YAW)
        elif self.motion_type == 2:
            self.desired_target.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                                + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                                + PositionTarget.IGNORE_YAW)

    def cmd_pose_flu_callback(self, msg):
        self.coordinate_frame = 9
        self.motion_type = 0
        yaw = self.q2yaw(msg.orientation)
        self.set_desired_motion(position=(msg.position.x, msg.position.y, msg.position.z),
                                velocity=None, accel=None, yaw=yaw,
                                coord_frame=self.coordinate_frame, motion_type=self.motion_type)

    def cmd_pose_enu_callback(self, msg):
        self.coordinate_frame = 1
        self.motion_type = 0
        yaw = self.q2yaw(msg.orientation)
        self.set_desired_motion(position=(msg.position.x, msg.position.y, msg.position.z),
                                velocity=None, accel=None, yaw=yaw,
                                coord_frame=self.coordinate_frame, motion_type=self.motion_type)

    def cmd_vel_flu_callback(self, msg):
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 8
            self.motion_type = 1
            self.set_desired_motion(position=None,
                                     velocity=(msg.linear.x, msg.linear.y, msg.linear.z),
                                     accel=None,
                                     yaw=None,
                                     yaw_rate=msg.angular.z,
                                     coord_frame=self.coordinate_frame,
                                     motion_type=self.motion_type)

    def cmd_vel_enu_callback(self, msg):
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 1
            self.motion_type = 1
            self.set_desired_motion(position=None,
                                     velocity=(msg.linear.x, msg.linear.y, msg.linear.z),
                                     accel=None,
                                     yaw=None,
                                     yaw_rate=msg.angular.z,
                                     coord_frame=self.coordinate_frame,
                                     motion_type=self.motion_type)

    def cmd_accel_flu_callback(self, msg):
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 8
            self.motion_type = 2
            self.set_desired_motion(position=None,
                                     velocity=None,
                                     accel=(msg.linear.x, msg.linear.y, msg.linear.z),
                                     yaw=None,
                                     yaw_rate=msg.angular.z,
                                     coord_frame=self.coordinate_frame,
                                     motion_type=self.motion_type)

    def cmd_accel_enu_callback(self, msg):
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 1
            self.motion_type = 2
            self.set_desired_motion(position=None,
                                     velocity=None,
                                     accel=(msg.linear.x, msg.linear.y, msg.linear.z),
                                     yaw=None,
                                     yaw_rate=msg.angular.z,
                                     coord_frame=self.coordinate_frame,
                                     motion_type=self.motion_type)

    def hover_state_transition(self, x, y, z, w):
        if abs(x) > 0.02 or abs(y) > 0.02 or abs(z) > 0.02 or abs(w) > 0.005:
            self.hover_flag = 0
            self.flight_mode = 'OFFBOARD'
        elif not self.flight_mode == "HOVER":
            self.hover_flag = 1
            self.flight_mode = 'HOVER'
            self.hover()

    def cmd_callback(self, msg):
        if msg.data == self.last_cmd or msg.data == '' or msg.data == 'stop controlling':
            return
        elif msg.data == 'ARM':
            self.arm_state = self.arm()
            print(self.vehicle_type+'_'+self.vehicle_id+": Armed "+str(self.arm_state))
        elif msg.data == 'DISARM':
            self.arm_state = not self.disarm()
            print(self.vehicle_type+'_'+self.vehicle_id+": Armed "+str(self.arm_state))
        elif msg.data[:-1] == "mission" and not msg.data == self.mission:
            self.mission = msg.data
            print(self.vehicle_type+'_'+self.vehicle_id+": "+msg.data)
        else:
            self.flight_mode = msg.data
            self.flight_mode_switch()
        self.last_cmd = msg.data

    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]
        return rotate_z_rad

    def arm(self):
        try:
            return self.armService(True)
        except Exception:
            print(self.vehicle_type+'_'+self.vehicle_id+": arming failed!")
            return False

    def disarm(self):
        try:
            return self.armService(False)
        except Exception:
            print(self.vehicle_type+'_'+self.vehicle_id+": disarming failed!")
            return False

    def hover(self):
        self.coordinate_frame = 1
        self.motion_type = 0
        if self.current_position is not None:
            self.set_desired_motion(position=(self.current_position.x, self.current_position.y, self.current_position.z),
                                    velocity=None, accel=None, yaw=self.current_yaw,
                                    coord_frame=self.coordinate_frame, motion_type=self.motion_type)
        print(self.vehicle_type+'_'+self.vehicle_id+":"+str(self.flight_mode))

    def flight_mode_switch(self):
        if self.flight_mode == 'HOVER':
            self.hover_flag = 1
            self.hover()
        else:
            try:
                if self.flightModeService(custom_mode=self.flight_mode):
                    print(self.vehicle_type+'_'+self.vehicle_id+": "+self.flight_mode)
                    return True
            except Exception as e:
                print(self.vehicle_type+'_'+self.vehicle_id+": "+self.flight_mode+" failed: "+str(e))
            return False
