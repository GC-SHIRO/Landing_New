import rospy
import threading
import subprocess
import sys
import time
import math
import numpy as np
import os
import signal
import subprocess
import time

from mavros_msgs.msg import PositionTarget, ParamValue, State
from mavros_msgs.srv import CommandBool, SetMode, ParamSet
from gazebo_msgs.msg import ModelState,ModelStates
from geometry_msgs.msg import PoseStamped, Pose, Twist,TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from apriltag_ros.msg import AprilTagDetectionArray
from pyquaternion import Quaternion
from std_msgs.msg import String


from Simulation.env.drone import *
# 定义常量
COLLISION_DIST = 0.35
# TIME_DELTA = 0.1  <-- 我们不再依赖固定的时间间隔，而是依赖视觉更新
STEP = 300

# Gazebo环境主类
class GazeboEnv:
    """所有Gazebo环境的基类"""

    def __init__(self, launchfile, vehicle_type, vehicle_id):
        # ---------- 增加调试输出 ----------
        rospy.loginfo("=== 初始化 GazeboEnv (Event-Driven & Metric Mode) ===")
        rospy.loginfo(f"参数: launchfile={launchfile}, vehicle={vehicle_type}_{vehicle_id}")

        # ---------- 启动 ROS 核心 & Gazebo ----------
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        rospy.loginfo("ROS 核心已启动，端口 %s", port)
        # 初始化ROS节点
        rospy.init_node("Landing_env", anonymous=True, disable_signals=True)
        # 等待ROS核心完全启动
        time.sleep(5.0)
        rospy.loginfo("正在启动Gazebo环境...")
        # 启动launch文件
        self.gazebo_process = subprocess.Popen(["roslaunch", "-p", port, launchfile])
        rospy.loginfo("Gazebo 环境已启动 => %s", launchfile)
        time.sleep(10)
        # ---------- 通信 & 模式设置 ----------
        rospy.loginfo("初始化通信模块...")
        self.comm = Communication(vehicle_type, vehicle_id)

        # 启动 Communication 的发布线程
        th = threading.Thread(target=self.comm.start)
        th.daemon = True
        th.start()
        rospy.loginfo("通信线程已启动")

        # ---------- 发布者 / 订阅者 ----------
        rospy.loginfo("正在设置ROS话题...")
        # Gazebo 模型控制
        self.set_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)

        # 订阅速度信息
        velocity_topic = f"{vehicle_type}_{vehicle_id}/mavros/local_position/velocity_local"
        rospy.loginfo(f"订阅速度话题: {velocity_topic}")
        self.velocity_sub = rospy.Subscriber(velocity_topic, TwistStamped, self.velocity_callback)

        # ===== 新增：订阅YOLO中心点话题 =====
        yolo_centers_topic = "/yolov11/centers"
        rospy.loginfo(f"订阅YOLO中心点话题: {yolo_centers_topic}")
        self.yolo_centers_sub = rospy.Subscriber(yolo_centers_topic, PointStamped, self.yolo_centers_callback)

        # 加速度控制模式下的发布者数组
        self.multi_cmd_vel_flu_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd_vel_flu', Twist, queue_size=1)
        self.multi_cmd_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd',String,queue_size=1)
        #位置控制
        self.multi_cmd_pose_enu_pub = rospy.Publisher('/xtdrone/'+vehicle_type+'/cmd_pose_enu', Pose, queue_size=1)
        #订阅位置信息
        self.local_pose_pub = rospy.Publisher(vehicle_type+'/mavros/local_position/pose', PoseStamped, queue_size=1)

        # Gazebo 服务
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

        # 环境维度
        self.prev_shaping = 0.0  # 初始化为0

        self.posestampesd = PoseStamped()
        self.twist = Twist()
        self.cmd= String()
        self.pose = Pose()

        self.drone_linear_velocity = None

        # YOLO检测相关变量初始化
        self.yolo_detected = False
        # 修改：默认值改为 0.0 (米)，代表无偏差
        self.yolo_pos_x = 0.0  # 相对相机的 X 轴距离 (米)
        self.yolo_pos_y = 0.0  # 相对相机的 Y 轴距离 (米)
        self.yolo_pos_z = 10.0  # 相对相机的 Z 轴距离 (深度, 米)

        # ===== 同步控制核心变量 =====
        self.last_yolo_detection_time = rospy.Time(0) # 最新一次YOLO回调的时间戳
        self.last_processed_yolo_time = rospy.Time(0) # 上一次step处理过的YOLO时间戳

        self.Step = 0

        # 初始化速度变量
        self.drone_linear_velocity = None
        self.drone_angular_velocity = None

        # YOLO进程管理
        self.yolo_process = None
        # 修改为分别存储包名和launch文件名
        self.yolo_package = "yolov11_ros"
        self.yolo_launch_file = "yolo_v11.launch"


    def start_yolo(self):
        """启动YOLO检测进程"""
        if self.yolo_process is not None:
            rospy.logwarn("YOLO进程已经在运行")
            return

        try:
            rospy.loginfo("启动YOLO检测进程...")

            # 使用roslaunch启动YOLO
            launch_cmd = ["roslaunch", "yolov11_ros", "yolo_v11.launch"]
            self.yolo_process = subprocess.Popen(
                launch_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                text=True
            )

            # 等待YOLO节点启动
            time.sleep(5)

            # 检查进程是否还在运行
            if self.yolo_process.poll() is not None:
                # 进程已退出，读取错误信息
                stdout, stderr = self.yolo_process.communicate()
                rospy.logerr(f"YOLO进程启动失败: {stderr}")
                self.yolo_process = None
                return

            rospy.loginfo("YOLO进程启动成功")

        except Exception as e:
            rospy.logerr(f"启动YOLO进程失败: {e}")
            self.yolo_process = None


    def stop_yolo(self):
        """停止YOLO检测进程"""
        if self.yolo_process is not None:
            try:
                rospy.loginfo("停止YOLO检测进程...")
                # 终止整个进程组
                os.killpg(os.getpgid(self.yolo_process.pid), signal.SIGTERM)
                self.yolo_process.wait(timeout=5)  # 等待进程结束
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

    # YOLO中心点回调函数
    def yolo_centers_callback(self, msg):
        """
        处理YOLO检测到的目标中心点信息
        注意：msg.point 是相机坐标系下的物理坐标(米)
        """
        try:
            # 存储YOLO检测到的物理坐标 (米)
            self.yolo_pos_x = msg.point.x
            self.yolo_pos_y = msg.point.y
            self.yolo_pos_z = msg.point.z

            # 记录检测时间
            self.last_yolo_detection_time = rospy.Time.now()
            self.yolo_detected = True

        except Exception as e:
            rospy.logwarn(f"YOLO回调函数错误: {e}")
            self.yolo_detected = False


    # 添加速度回调函数
    def velocity_callback(self, msg):
        """速度回调函数"""
        # 更新无人机的线速度和角速度
        self.drone_linear_velocity = msg.twist.linear
        self.drone_angular_velocity = msg.twist.angular


    # 执行动作并返回新状态
    def step(self, action):
        rospy.logdebug(f"===== STEP开始，等待视觉同步 =====")
        done = False
        target = False  # 默认未达到目标

        # self.twist.linear.x = action[0]
        # self.twist.linear.y = action[1]
        # self.twist.linear.z = action[2]

        # self.multi_cmd_vel_flu_pub.publish(self.twist)#（取消控制指令发布）

        # === 1. 解除物理引擎暂停，让世界动起来 ===
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # === 2. 核心修改：等待新的 YOLO 帧 (Vision Blocking) ===
        wait_start = time.time()
        timeout = 3.0 # 2秒超时

        frame_updated = False

        while time.time() - wait_start < timeout:
            # 检查当前最新的YOLO时间戳是否比上次处理的时间戳新
            if self.last_yolo_detection_time > self.last_processed_yolo_time:
                frame_updated = True
                # 更新记录，标记这帧已经用过了
                self.last_processed_yolo_time = self.last_yolo_detection_time
                break

            time.sleep(0.005)

        if not frame_updated:
            rospy.logwarn("Visual Timeout! YOLO没有在3秒内更新。")

        # === 3. 拿到新数据后，立即暂停物理引擎 ===
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        next_state = self.get_state()

        # ===  终止条件判断 ===
        self.Step +=1

        if self.Step == STEP:
            done = True
            rospy.logwarn(f"Step已满")
            self.stop_yolo()
            self.Step = 0

        # 调试输出：注意现在的 State 已经是米为单位了
        # print(f"Step: {self.Step} | Visual Lag: {time.time() - wait_start:.3f}s")
        # print(f"Target Pos (Cam Frame): X={self.yolo_pos_x:.2f}m, Y={self.yolo_pos_y:.2f}m, Z={self.yolo_pos_z:.2f}m")

        # 成功判定：
        # 1. 检测到目标
        # 2. 高度 Z < 0.5m
        # 3. 水平误差 XY 都小于 0.2m (20厘米)
        # 4. 速度 < 0.2
        if self.yolo_detected and abs(self.yolo_pos_z) < 0.5 and \
           abs(self.yolo_pos_x) < 0.2 and abs(self.yolo_pos_y) < 0.2 and \
           abs(self.drone_linear_velocity.z) < 0.2:
                target = True
                done = True
                self.Step = 0
                rospy.loginfo("成功着陆！")

        # === 越界检测 ===
        if hasattr(self.comm, 'current_position'):
            x0 = self.comm.current_position.x
            y0 = self.comm.current_position.y
            z0 = self.comm.current_position.z
            dist_origin = math.sqrt(x0**2 + y0**2 + z0**2)
            if dist_origin > 15.0 or z0 > 12:
                rospy.logwarn(f"越界：距离原点 {dist_origin:.2f}, 高度{z0:.2f} m，结束本回合")
                done = True
                self.Step = 0
                self.stop_yolo()

        if not self.yolo_detected and hasattr(self.comm, 'current_position'):
            if self.comm.current_position.z < 0.5:
                rospy.logwarn(f"越界：未检测到目标 且高度 {self.comm.current_position.z:.2f}m < 0.5m，结束本回合")
                done = True
                self.Step = 0
                self.stop_yolo()

        # === 奖励计算 ===
        success = target and done

        info = {
            "tag_detected": self.yolo_detected,
            "target_reached": target,
            "success": success
        }
        return next_state, done, success, info

    def reset(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # 重置时间戳
        self.last_yolo_detection_time = rospy.Time.now()
        self.last_processed_yolo_time = self.last_yolo_detection_time

        # ===== ：启动位置发布循环 =====
        self.pose_publishing_active = True
        def _publish_pose_loop():
            rate = rospy.Rate(30)  # 30Hz发布频率
            while self.pose_publishing_active and not rospy.is_shutdown():
                if self.comm.current_position is not None:
                    # 如果已有位置数据，发布真实值
                    pose = PoseStamped()
                    pose.header.stamp = rospy.Time.now()
                    pose.pose.position = self.comm.current_position
                    self.local_pose_pub.publish(pose)
                else:
                    # 初始阶段发布零值激活话题
                    pose = PoseStamped()
                    pose.header.stamp = rospy.Time.now()
                    pose.pose.position.x = 0.0
                    pose.pose.position.y = 0.0
                    pose.pose.position.z = 0.0
                    self.local_pose_pub.publish(pose)
                rate.sleep()
        # 启动发布线程
        pose_thread = threading.Thread(target=_publish_pose_loop)
        pose_thread.start()
        # 确保当前位置被初始化：使用阻塞式等待获取初始位置
        rospy.loginfo("等待当前位置数据...")
        try:
            # 明确指定话题名称，确保与Communication类中的订阅一致
            pose_topic = f"{self.comm.vehicle_type}/mavros/local_position/pose"
            pose_msg = rospy.wait_for_message(pose_topic, PoseStamped, timeout=30.0)
            self.comm.current_position = pose_msg.pose.position
            self.comm.current_yaw = self.comm.q2yaw(pose_msg.pose.orientation)
            rospy.loginfo("当前位置数据已接收")
        except rospy.ROSException as e:
            rospy.logerr(f"等待位置数据超时: {e}")
            raise RuntimeError("无法获取初始位置，请检查MAVROS和Gazebo连接状态")
        # 先发送几次当前位置的控制指令，确保OFFBOARD模式能保持
        current_pose = self.comm.construct_target(
            x=self.comm.current_position.x,
            y=self.comm.current_position.y,
            z=self.comm.current_position.z,
            yaw=self.comm.current_yaw
        )
        # 发送当前位置指令几次，确保控制稳定
        for _ in range(10):
            self.comm.target_motion_pub.publish(current_pose)
            rospy.sleep(0.1)

        # 切换到OFFBOARD模式并解锁
        self.comm.arm_state = self.comm.arm()
        offboard_success = self.comm.flightModeService(custom_mode="OFFBOARD")
        if not offboard_success:
            rospy.logwarn("OFFBOARD模式切换失败，重试中...")
            # 继续发送位置控制几次再重试
            for _ in range(5):
                self.comm.target_motion_pub.publish(current_pose)
                rospy.sleep(0.1)
            offboard_success = self.comm.flightModeService(custom_mode="OFFBOARD")
        # 确保模式切换成功
        rospy.sleep(0.5)  # 给系统一些时间来切换模式
        # 准备随机起始位置
        x=self.pose.position.x = np.random.uniform(2.4, 3.0)
        y=self.pose.position.y= np.random.uniform(4.3, 5.0)
        z=self.pose.position.z = np.random.uniform(8.4, 8.5)  # 确保足够高度

        # 移动到目标位置，同时持续发送OFFBOARD指令
        rospy.loginfo(f"移动到起始位置 ({x:.2f}, {y:.2f}, {z:.2f})...")
        # 位置跟踪
        start_time = time.time()
        timeout = 15.0  # 15秒超时
        arrival_count = 0
        rate = rospy.Rate(10)
        while time.time() - start_time < timeout:
            # 发送目标位置指令
            self.multi_cmd_pose_enu_pub.publish(self.pose)
            current_x = self.comm.current_position.x
            current_y = self.comm.current_position.y
            current_z = self.comm.current_position.z
            # 计算到目标的距离
            distance = math.sqrt((x-current_x)**2 + (y-current_y)**2 + (z-current_z)**2)
            if distance < 0.3:
                arrival_count += 1
                if arrival_count > 10:  # 稳定停留超过1秒
                    rospy.loginfo("已稳定到达目标位置")
                    break
            else:
                arrival_count = 0
            rate.sleep()

        # 切换到定点模式（POSCTL）
        mode_switch_response = self.comm.flightModeService(custom_mode="POSCTL")
        if not mode_switch_response.mode_sent:
            rospy.logwarn("切换到定点模式失败，重试中...")
            # 继续发送位置控制几次再重试
            for _ in range(5):
                self.comm.target_motion_pub.publish(current_pose)
                rospy.sleep(0.1)
            mode_switch_response = self.comm.flightModeService(custom_mode="POSCTL")
        # 检查最终结果
        if mode_switch_response.mode_sent:
            rospy.loginfo("无人机已进入定点模式(POSCTL)")
        else:
            rospy.logwarn("切换到定点模式失败，继续尝试")

        # 启动YOLO进程
        self.start_yolo()

        # 暂停物理模拟以获取状态
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        state = self.get_state()
        self.comm.flight_mode = None
        rospy.loginfo("===== 重置完成 =====")

        return state

    def __del__(self):
        """析构函数，确保进程被正确清理"""
        self.stop_yolo()

    def get_state(self):
        """
        获取无人机当前状态 (相对于相机的物理距离 + 机体速度)
        返回维度: 6
        """
        # 1. 准备位置数据 (3维)
        # 如果没有检测到目标，或者还没初始化
        if not hasattr(self, 'yolo_detected') or not self.yolo_detected:
            # 策略：丢失目标时，位置给一个特定值 (比如 0,0,10) 或者保持上一帧
            # 这里给 (0, 0, 10) 让它知道自己在高空且未对准
            pos_x, pos_y, pos_z = 0.0, 0.0, 10.0
        else:
            pos_x = self.yolo_pos_x
            pos_y = self.yolo_pos_y
            pos_z = self.yolo_pos_z

        # 2. 准备速度数据 (3维)
        # 必须做非空检查，因为 velocity_callback 可能还没触发过
        vx, vy, vz = 0.0, 0.0, 0.0
        if self.drone_linear_velocity is not None:
            vx = self.drone_linear_velocity.x
            vy = self.drone_linear_velocity.y
            vz = self.drone_linear_velocity.z

        # 3. 拼接并返回 (6维)
        # state = np.array([pos_x, pos_y, pos_z, vx, vy, vz])
        state = np.array([pos_x, pos_y, pos_z])
        return state

    def reward_setup(self, observation, observation_, done, succes):


        err_x = observation[0]  # 米
        err_y = observation[1]  # 米
        height = observation[2] # 米

        # 2. 计算距离 (Shaping Reward)
        # 使用加权欧氏距离
        # 1米水平误差 的惩罚 > 1米高度下降 的奖励
        w_xy = 2.0  # 水平权重
        w_z = 0.5   # 高度权重

        distance = math.sqrt((w_xy * err_x)**2 + (w_xy * err_y)**2 + (w_z * height)**2)

        # 3. 基础步进奖励
        reward = -0.08 * distance - 0.01

        # 4. 终止奖励
        if done:
            # 使用真实的 yolo 变量进行最终判定
            real_err_x = abs(self.yolo_pos_x)
            real_err_y = abs(self.yolo_pos_y)

            # 成功条件：水平误差 < 0.2m (20cm)
            if succes and real_err_x < 0.2 and real_err_y < 0.2:
                reward += 200.0
                print(f"Landed SUCCESS! Reward: {reward:.2f}, Pos: {observation}")
                return reward
            else:
                reward -= 150.0
                print(f"Landed FAILED. Reward: {reward:.2f}, Pos: {observation}")
                return reward

        return reward

# Communication 类保持不变...
class Communication:
    """
    无人机通信控制主类，负责与MAVROS的通信、指令解析和状态管理
    功能包括：状态订阅、指令发布、飞行模式切换、解锁上锁控制等
    """

    def __init__(self, vehicle_type, vehicle_id):
        """
        类初始化函数
        :param vehicle_type: 无人机类型（通过命令行参数传入）
        :param vehicle_id: 无人机ID（通过命令行参数传入）
        """
        # 初始化无人机基础信息
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id

        # 当前状态变量
        self.current_position = None  # 当前位置（geometry_msgs/Point）
        self.current_yaw = 0          # 当前偏航角（弧度）
        self.hover_flag = 0          # 悬停状态标志（0-非悬停，1-悬停）
        self.coordinate_frame = 1     # 坐标系定义（1-ENU，8-FLU，9-机体坐标系）

        # 目标运动指令容器
        self.target_motion = PositionTarget()
        self.target_motion.coordinate_frame = self.coordinate_frame

        # 系统状态标志
        self.arm_state = False        # 解锁状态
        self.motion_type = 0          # 运动控制类型（0-位置，1-速度，2-加速度）
        self.flight_mode = None       # 当前飞行模式
        self.mission = None           # 当前任务标识
        self.last_cmd = None          # 最后接收的指令缓存

        # MAVROS连接状态检查（阻塞式等待）
        # 订阅mavros/state话题，检查飞控连接状态
        mavros_state = rospy.wait_for_message(self.vehicle_type+'_'+self.vehicle_id+"/mavros/state", State)
        if not mavros_state.connected:
            rospy.logwarn(self.vehicle_type+'_'+self.vehicle_id+": No connection to FCU. Check mavros!")
            exit(0)  # 连接失败则退出程序

        ####################
        ## ROS接口初始化部分 ##
        ####################

        # 订阅者列表
        # 本地位置订阅（高优先级队列）
        self.local_pose_sub = rospy.Subscriber(self.vehicle_type+'_'+self.vehicle_id + "/mavros/local_position/pose",
                                             PoseStamped, self.local_pose_callback, queue_size=1)
        # 通用指令订阅（中等队列深度）
        self.cmd_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd",
                                      String, self.cmd_callback, queue_size=3)
        # 不同坐标系下的控制指令订阅（位置、速度、加速度）
        self.cmd_pose_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_pose_flu",
                                               Pose, self.cmd_pose_flu_callback, queue_size=1)
        self.cmd_pose_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_pose_enu",
                                               Pose, self.cmd_pose_enu_callback, queue_size=1)
        self.cmd_vel_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_vel_flu",
                                              Twist, self.cmd_vel_flu_callback, queue_size=1)
        self.cmd_vel_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_vel_enu",
                                              Twist, self.cmd_vel_enu_callback, queue_size=1)
        self.cmd_accel_flu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_accel_flu",
                                                Twist, self.cmd_accel_flu_callback, queue_size=1)
        self.cmd_accel_enu_sub = rospy.Subscriber("/xtdrone/"+self.vehicle_type+"/cmd_accel_enu",
                                                Twist, self.cmd_accel_enu_callback, queue_size=1)

        # 发布者：用于发送目标位置/速度/加速度指令到飞控
        self.target_motion_pub = rospy.Publisher(self.vehicle_type+'_'+self.vehicle_id+"/mavros/setpoint_raw/local",
                                               PositionTarget, queue_size=1)

        # 服务客户端初始化
        self.armService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/cmd/arming", CommandBool)
        self.flightModeService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/set_mode", SetMode)
        self.set_param_srv = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/param/set", ParamSet)

        # 设置飞控参数COM_RCL_EXCEPT，禁用遥控器失效保护
        rcl_except = ParamValue(4, 0.0)  # 参数类型为整型，值0
        self.set_param_srv("COM_RCL_EXCEPT", rcl_except)

        print(self.vehicle_type+'_'+self.vehicle_id+": "+"communication initialized")

    def start(self):
        """
        主循环函数，持续发布控制指令
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.target_motion_pub.publish(self.target_motion)  # 持续发布当前目标指令
            rate.sleep()  # 维持30Hz发布频率

    def local_pose_callback(self, msg):
        """
        本地位置订阅回调函数
        :param msg: 包含位置和姿态的PoseStamped消息
        功能：更新当前无人机的位置和偏航角
        """
        self.current_position = msg.pose.position  # 提取位置信息
        self.current_yaw = self.q2yaw(msg.pose.orientation)  # 将四元数转换为偏航角

    def construct_target(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, afx=0, afy=0, afz=0, yaw=0, yaw_rate=0):
        """
        构建PositionTarget消息的工厂方法
        :param x,y,z: 位置坐标（米）
        :param vx,vy,vz: 速度分量（米/秒）
        :param afx,afy,afz: 加速度/力分量（米/秒² 或 N/kg）
        :param yaw: 目标偏航角（弧度）
        :param yaw_rate: 偏航角速率（弧度/秒）
        :return: 配置好的PositionTarget消息
        """
        target_raw_pose = PositionTarget()
        target_raw_pose.coordinate_frame = self.coordinate_frame  # 设置坐标系

        # 填充各字段值
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

        # 根据运动类型设置忽略位掩码
        if self.motion_type == 0:  # 位置控制模式
            # 忽略速度、加速度、偏航速率
            target_raw_pose.type_mask = (PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                            + PositionTarget.IGNORE_YAW_RATE)
        elif self.motion_type == 1:  # 速度控制模式
            # 忽略位置、加速度、偏航角
            target_raw_pose.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
                            + PositionTarget.IGNORE_YAW)
        elif self.motion_type == 2:  # 加速度控制模式
            # 忽略位置、速度、偏航角
            target_raw_pose.type_mask = (PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
                            + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ
                            + PositionTarget.IGNORE_YAW)

        return target_raw_pose

    def cmd_pose_flu_callback(self, msg):
        """
        FLU坐标系下的位置指令回调
        :param msg: 包含目标姿态的Pose消息
        功能：设置机体坐标系下的位置控制指令
        """
        self.coordinate_frame = 9  # MAV_FRAME_BODY_FRD坐标系
        self.motion_type = 0       # 位置控制模式
        yaw = self.q2yaw(msg.orientation)  # 提取偏航角
        # 构建位置控制指令
        self.target_motion = self.construct_target(x=msg.position.x, y=msg.position.y,
                                                  z=msg.position.z, yaw=yaw)

    def cmd_pose_enu_callback(self, msg):
        """
        ENU坐标系下的位置指令回调
        :param msg: 包含目标姿态的Pose消息
        功能：设置ENU坐标系下的位置控制指令
        """
        self.coordinate_frame = 1  # MAV_FRAME_LOCAL_NED坐标系（ENU）
        self.motion_type = 0       # 位置控制模式
        yaw = self.q2yaw(msg.orientation)
        self.target_motion = self.construct_target(x=msg.position.x, y=msg.position.y,
                                                 z=msg.position.z, yaw=yaw)

    def cmd_vel_flu_callback(self, msg):
        """
        FLU坐标系下的速度指令回调
        :param msg: 包含线速度和角速度的Twist消息
        功能：处理速度控制指令，可能触发悬停状态转换
        """
        # 检查是否需要进入悬停
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:  # 非悬停状态
            self.coordinate_frame = 8  # MAV_FRAME_BODY_FRD（速度控制）
            self.motion_type = 1       # 速度控制模式
            # 构建速度控制指令
            self.target_motion = self.construct_target(vx=msg.linear.x, vy=msg.linear.y,
                                                      vz=msg.linear.z, yaw_rate=msg.angular.z)

    def cmd_vel_enu_callback(self, msg):
        """
        ENU坐标系下的速度指令回调
        :param msg: 包含线速度和角速度的Twist消息
        功能：处理ENU系速度指令
        """
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 1  # ENU坐标系
            self.motion_type = 1
            self.target_motion = self.construct_target(vx=msg.linear.x, vy=msg.linear.y,
                                                     vz=msg.linear.z, yaw_rate=msg.angular.z)

    def cmd_accel_flu_callback(self, msg):
        """
        FLU坐标系下的加速度指令回调
        :param msg: 包含线加速度和角速度的Twist消息
        功能：处理加速度控制指令
        """
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 8  # 机体坐标系
            self.motion_type = 2       # 加速度控制模式
            self.target_motion = self.construct_target(afx=msg.linear.x, afy=msg.linear.y,
                                                     afz=msg.linear.z, yaw_rate=msg.angular.z)

    def cmd_accel_enu_callback(self, msg):
        """
        ENU坐标系下的加速度指令回调
        :param msg: 包含线加速度和角速度的Twist消息
        """
        self.hover_state_transition(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if self.hover_flag == 0:
            self.coordinate_frame = 1  # ENU坐标系
            self.motion_type = 2
            self.target_motion = self.construct_target(afx=msg.linear.x, afy=msg.linear.y,
                                                    afz=msg.linear.z, yaw_rate=msg.angular.z)

    def hover_state_transition(self, x, y, z, w):
        """
        悬停状态转换逻辑
        :param x,y,z: 线运动分量
        :param w: 角速度分量
        功能：根据控制输入判断是否进入悬停状态
        """
        # 判断各分量是否超过阈值（微小量）
        if abs(x) > 0.02 or abs(y) > 0.02 or abs(z) > 0.02 or abs(w) > 0.005:
            self.hover_flag = 0  # 有有效控制指令，退出悬停
            self.flight_mode = 'OFFBOARD'
        elif not self.flight_mode == "HOVER":  # 无有效指令且当前非悬停模式
            self.hover_flag = 1
            self.flight_mode = 'HOVER'
            self.hover()  # 执行悬停操作

    def cmd_callback(self, msg):
        """
        通用指令回调函数
        :param msg: 包含指令的String消息
        处理指令包括：ARM/DISARM、任务切换、飞行模式切换等
        """
        # 过滤重复或无效指令
        if msg.data == self.last_cmd or msg.data == '' or msg.data == 'stop controlling':
            return

        # 处理解锁指令
        elif msg.data == 'ARM':
            self.arm_state = self.arm()  # 调用解锁服务
            print(self.vehicle_type+'_'+self.vehicle_id+": Armed "+str(self.arm_state))

        # 处理上锁指令
        elif msg.data == 'DISARM':
            self.arm_state = not self.disarm()  # 调用上锁服务
            print(self.vehicle_type+'_'+self.vehicle_id+": Armed "+str(self.arm_state))

        # 处理任务切换指令
        elif msg.data[:-1] == "mission" and not msg.data == self.mission:
            self.mission = msg.data
            print(self.vehicle_type+'_'+self.vehicle_id+": "+msg.data)

        # 处理其他飞行模式指令
        else:
            self.flight_mode = msg.data
            self.flight_mode_switch()  # 切换飞行模式

        self.last_cmd = msg.data  # 记录最后有效指令

    def q2yaw(self, q):
        """
        四元数转偏航角工具函数
        :param q: 四元数（可以是Quaternion对象或geometry_msgs/Quaternion）
        :return: 偏航角（弧度，-π到π）
        """
        if isinstance(q, Quaternion):  # 直接使用pyquaternion对象
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:  # 处理geometry_msgs/Quaternion类型
            q_ = Quaternion(q.w, q.x, q.y, q.z)  # 转换为pyquaternion对象
            rotate_z_rad = q_.yaw_pitch_roll[0]  # 提取偏航角

        return rotate_z_rad

    def arm(self):
        """调用解锁服务，返回执行结果"""
        if self.armService(True):
            return True
        else:
            print(self.vehicle_type+'_'+self.vehicle_id+": arming failed!")
            return False

    def disarm(self):
        """调用上锁服务，返回执行结果"""
        if self.armService(False):
            return True
        else:
            print(self.vehicle_type+'_'+self.vehicle_id+": disarming failed!")
            return False

    def hover(self):
        """执行悬停操作，保持当前位置和偏航角"""
        self.coordinate_frame = 1  # 使用ENU坐标系
        self.motion_type = 0       # 位置控制模式
        # 构建当前位置的悬停指令
        self.target_motion = self.construct_target(x=self.current_position.x,
                                                  y=self.current_position.y,
                                                  z=self.current_position.z,
                                                  yaw=self.current_yaw)
        print(self.vehicle_type+'_'+self.vehicle_id+":"+self.flight_mode)

    def flight_mode_switch(self):
        """
        飞行模式切换处理
        支持HOVER模式直接切换，其他模式通过服务调用切换
        """
        if self.flight_mode == 'HOVER':
            self.hover_flag = 1
            self.hover()  # 进入悬停模式
        elif self.flightModeService(custom_mode=self.flight_mode):  # 调用模式切换服务
            print(self.vehicle_type+'_'+self.vehicle_id+": "+self.flight_mode)
            return True
        else:
            print(self.vehicle_type+'_'+self.vehicle_id+": "+self.flight_mode+" failed")
            return False
