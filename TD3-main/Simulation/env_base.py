#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用途:
- Simulation/ 目录下所有 step 脚本的基础环境, 与 landing_env.py 保持一致。
- 修复点: __init__ 末尾增加系统预热延迟(SYSTEM_WARMUP_SECONDS), 等待
  ROS/MAVROS/XTDrone 话题加载稳定后再开始, 从根本上解决首次 reset
  飞机起飞失败(电机怠速、慢速转旋翼)的问题。

用法:
- from env_base import GazeboEnv
- 典型流程: env = GazeboEnv(...); obs = env.reset(); obs2, done, success, info = env.step(action)。

依赖关系:
- 被 Simulation/ 下各 step_*.py 脚本导入。
- 依赖 ROS(mavros/gazebo_msgs/geometry_msgs)、Gazebo 服务与外部视觉节点。
"""

import rospy
import rosgraph
import threading
import subprocess
import time
import math
import numpy as np
import os
import signal

from mavros_msgs.msg import PositionTarget, ParamValue, State
from mavros_msgs.srv import CommandBool, SetMode, ParamSet
from geometry_msgs.msg import PoseStamped, Pose, PointStamped, TwistStamped
from yolov11_ros_msgs.msg import BoundingBoxes
from gazebo_msgs.msg import ContactsState
from std_srvs.srv import Empty
from pyquaternion import Quaternion

# 定义常量
TIME_DELTA = 0.1
SYSTEM_WARMUP_SECONDS = 10.0

DEFAULT_LANDING_XY_THRESHOLD = 1.5
DEFAULT_VISUAL_X_LIMIT = 8.0
DEFAULT_VISUAL_Y_LIMIT = 6.0
DEFAULT_YOLO_LOST_TIMEOUT = 2.0


class GazeboEnv:
    """所有Gazebo环境的基类 (优化版：YOLO 常驻，减少 IO 开销)
    """

    def __init__(self, launchfile, vehicle_type, vehicle_id,
                 max_dist=15.0, max_height=12.0, landing_z_threshold=0.5,
                 landing_target_fn=None,
                 landing_xy_threshold=DEFAULT_LANDING_XY_THRESHOLD,
                 visual_x_limit=DEFAULT_VISUAL_X_LIMIT,
                 visual_y_limit=DEFAULT_VISUAL_Y_LIMIT,
                 yolo_lost_timeout=DEFAULT_YOLO_LOST_TIMEOUT,
                 enable_yolo=True,
                 system_warmup_seconds=SYSTEM_WARMUP_SECONDS,
                 roscore_wait_seconds=5.0,
                 gazebo_wait_seconds=10.0,
                 configure_rc_loss_exception=True,
                 mavros_state_timeout=30.0,
                 mavros_param_timeout=5.0):
        self.roscore_process = None
        self.gazebo_process = None
        self.yolo_process = None
        self.enable_yolo = False
        self._closed = False

        rospy.loginfo("=== 初始化 GazeboEnv (Persistent YOLO Mode) ===")
        rospy.loginfo(f"参数: launchfile={launchfile}, vehicle={vehicle_type}_{vehicle_id}")
        rospy.loginfo(f"越界阈值: max_dist={max_dist:.1f}m  max_height={max_height:.1f}m "
                      f"landing_z={landing_z_threshold:.2f}m")
        if landing_target_fn is not None:
            rospy.loginfo(
                f"动态甲板成功条件: 接触 WAM-V 甲板且水平误差"
                f"<={landing_xy_threshold:.3f}m"
            )

        port = "11311"
        if rosgraph.is_master_online():
            rospy.loginfo("检测到已有 ROS master，直接复用: %s", os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
        else:
            self.roscore_process = subprocess.Popen(
                ["roscore", "-p", port], preexec_fn=os.setsid
            )
            deadline = time.monotonic() + float(roscore_wait_seconds)
            while time.monotonic() < deadline and not rosgraph.is_master_online():
                time.sleep(0.1)
            if not rosgraph.is_master_online():
                raise RuntimeError(
                    f"ROS master 在 {roscore_wait_seconds:.1f}s 内未启动"
                )
            rospy.loginfo("ROS 核心已启动，端口 %s", port)

        rospy.init_node("Landing_env", anonymous=True, disable_signals=True)
        rospy.loginfo("正在启动Gazebo环境...")

        self.gazebo_process = subprocess.Popen(
            ["roslaunch", "-p", port, launchfile], preexec_fn=os.setsid
        )
        rospy.loginfo("Gazebo 环境已启动 => %s", launchfile)
        time.sleep(float(gazebo_wait_seconds))

        rospy.loginfo("初始化通信模块: 等待 MAVROS/FCU 状态...")
        self.comm = Communication(
            vehicle_type,
            vehicle_id,
            configure_rc_loss_exception=configure_rc_loss_exception,
            state_timeout=mavros_state_timeout,
            param_timeout=mavros_param_timeout,
        )
        rospy.loginfo("通信模块初始化完成")

        th = threading.Thread(target=self.comm.start)
        th.daemon = True
        th.start()
        rospy.loginfo("通信线程已启动")

        # Publishers / Subscribers
        rospy.loginfo("正在设置ROS话题...")

        self.enable_yolo = bool(enable_yolo)
        self.system_warmup_seconds = float(system_warmup_seconds)
        self.yolo_centers_sub = None
        self.yolo_boxes_sub = None
        if self.enable_yolo:
            yolo_centers_topic = "/yolov11/centers"
            rospy.loginfo(f"订阅YOLO中心点话题: {yolo_centers_topic}")
            self.yolo_centers_sub = rospy.Subscriber(
                yolo_centers_topic, PointStamped, self.yolo_centers_callback
            )
            yolo_boxes_topic = "/yolov11/BoundingBoxes"
            rospy.loginfo(f"订阅YOLO检测框话题: {yolo_boxes_topic}")
            self.yolo_boxes_sub = rospy.Subscriber(
                yolo_boxes_topic, BoundingBoxes, self.yolo_boxes_callback
            )
        else:
            rospy.loginfo("专家采样模式: 跳过 YOLO 订阅与进程启动")

        velocity_topic = f"{vehicle_type}_{vehicle_id}/mavros/local_position/velocity_local"
        rospy.loginfo(f"订阅无人机速度话题: {velocity_topic}")
        self.velocity_sub = rospy.Subscriber(
            velocity_topic, TwistStamped, self.velocity_callback
        )

        contact_topic = "/benchmarker/collision"
        rospy.loginfo(f"订阅起落架碰撞话题: {contact_topic}")
        self.contact_sub = rospy.Subscriber(
            contact_topic, ContactsState, self.contact_callback
        )

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
        self.pose = Pose()

        # YOLO 相关
        self.yolo_detected = False
        self.yolo_center_x = 0.5
        self.yolo_center_y = 0.5
        self.yolo_distance_z = 10.0
        self.yolo_confidence = 0.0
        self.last_yolo_detection_time = rospy.Time.now()
        self.drone_linear_velocity = None
        self.deck_contact = False

        self.Step = 0

        # 越界阈值 (可被子类或调用方覆盖)
        self.max_dist = max_dist      # 无人机距世界原点最大允许距离 (m)
        self.max_height = max_height  # 无人机最大允许高度 (m)
        self.landing_z_threshold = landing_z_threshold  # 静态模式绝对 z / 失检高度阈值 (m)

        # 动态甲板只使用起落架与 WAM-V 甲板的真实碰撞判定落地。
        # 相对位置和速度仅用于记录与判断是否落在目标区域。
        self.landing_target_fn = landing_target_fn
        self.landing_xy_threshold = float(landing_xy_threshold)
        self.visual_x_limit = float(visual_x_limit)
        self.visual_y_limit = float(visual_y_limit)
        self.yolo_lost_timeout = float(yolo_lost_timeout)
        self.landing_velocity_fn = None
        self._episode_has_detection = False
        self._episode_start_time = rospy.Time.now()

        self.yolo_package = "yolov11_ros"
        self.yolo_launch_file = "yolo_v11.launch"

        # ===== Quadrant-balanced reset sampling =====
        self.quadrant_idx = 0  # 0~3 轮流


        # === 修改点 1: 初始化时直接启动 YOLO，之后不再关闭 ===
        if self.enable_yolo:
            self.start_yolo()
        rospy.loginfo(
            f"系统预热等待 {self.system_warmup_seconds:.1f}s，等待 ROS/XTDrone/MAVROS 话题稳定..."
        )
        time.sleep(self.system_warmup_seconds)
        rospy.loginfo("系统预热完成")

    def start_yolo(self):
        if not self.enable_yolo:
            return
        if self.yolo_process is not None:
            rospy.logwarn("YOLO进程已经在运行")
            return
        try:
            rospy.loginfo("启动YOLO检测进程...")
            launch_cmd = ["roslaunch", "yolov11_ros", "yolo_v11.launch"]
            self.yolo_process = subprocess.Popen(
                launch_cmd,
                preexec_fn=os.setsid,
                text=True
            )
            # 等待稍微久一点，确保显存加载完毕
            time.sleep(8)
            if self.yolo_process.poll() is not None:
                rospy.logerr(
                    f"YOLO进程启动失败, returncode={self.yolo_process.returncode}"
                )
                self.yolo_process = None
                return
            rospy.loginfo("YOLO进程启动成功")
        except Exception as e:
            rospy.logerr(f"启动YOLO进程失败: {e}")
            self.yolo_process = None

    def stop_yolo(self):
        """只在整个脚本退出时调用，Reset时不调用"""
        yolo_process = getattr(self, "yolo_process", None)
        if yolo_process is not None:
            try:
                rospy.loginfo("停止YOLO检测进程...")
                os.killpg(os.getpgid(yolo_process.pid), signal.SIGTERM)
                yolo_process.wait(timeout=5)
                rospy.loginfo("YOLO进程已停止")
            except subprocess.TimeoutExpired:
                rospy.logwarn("YOLO进程未正常终止，强制杀死...")
                os.killpg(os.getpgid(yolo_process.pid), signal.SIGKILL)
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
            self._episode_has_detection = True
            # rospy.logdebug(f"YOLO检测: ...") # 减少日志打印以节省资源
        except Exception as e:
            rospy.logwarn(f"YOLO回调函数错误: {e}")
            self.yolo_detected = False

    def yolo_boxes_callback(self, msg):
        """保存与中心点话题同一检测顺序的 YOLO 置信度。"""
        boxes = getattr(msg, "bounding_boxes", ())
        if not boxes:
            # 空检测帧应立即反映为无置信度，不能沿用上一帧数值。
            self.yolo_confidence = 0.0
            return

        try:
            # YOLO 节点按同一顺序发布中心点；环境最终接收的是最后一个中心点。
            confidence = float(boxes[-1].probability)
        except (AttributeError, TypeError, ValueError):
            rospy.logwarn("YOLO检测框缺少有效置信度")
            self.yolo_confidence = 0.0
            return

        if not math.isfinite(confidence):
            rospy.logwarn("YOLO检测框置信度不是有限数值")
            self.yolo_confidence = 0.0
            return
        self.yolo_confidence = float(np.clip(confidence, 0.0, 1.0))

    def velocity_callback(self, msg):
        self.drone_linear_velocity = msg.twist.linear

    def contact_callback(self, msg):
        self.deck_contact = False
        for state in msg.states:
            collision_names = (
                state.collision1_name.lower(),
                state.collision2_name.lower(),
            )
            if any(
                "wamv" in name
                and ("top_base" in name or "deck_collision" in name)
                for name in collision_names
            ):
                self.deck_contact = True
                break

    def _check_dynamic_landing(self):
        result = {
            "landed": False,
            "deck_contact": bool(self.deck_contact),
            "position_ok": False,
            "relative_x": float("nan"),
            "relative_y": float("nan"),
            "relative_height": float("nan"),
            "relative_xy_distance": float("nan"),
            "relative_vx": float("nan"),
            "relative_vy": float("nan"),
            "relative_vz": float("nan"),
            "relative_xy_speed": float("nan"),
        }

        if self.comm.current_position is None or self.landing_target_fn is None:
            return result

        try:
            target_x, target_y, target_z = self.landing_target_fn()
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"读取甲板状态失败: {e}")
            return result

        relative_x = float(self.comm.current_position.x) - float(target_x)
        relative_y = float(self.comm.current_position.y) - float(target_y)
        relative_height = float(self.comm.current_position.z) - float(target_z)
        relative_xy_distance = math.hypot(relative_x, relative_y)
        position_ok = relative_xy_distance <= self.landing_xy_threshold

        if self.deck_contact:
            result.update({
                "landed": True,
                "position_ok": position_ok,
                "relative_x": relative_x,
                "relative_y": relative_y,
                "relative_height": relative_height,
                "relative_xy_distance": relative_xy_distance,
            })
            return result

        if self.drone_linear_velocity is None or self.landing_velocity_fn is None:
            result.update({
                "position_ok": position_ok,
                "relative_x": relative_x,
                "relative_y": relative_y,
                "relative_height": relative_height,
                "relative_xy_distance": relative_xy_distance,
            })
            return result

        try:
            target_vx, target_vy, target_vz = self.landing_velocity_fn()
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"读取甲板速度失败: {e}")
            return result

        relative_vx = float(self.drone_linear_velocity.x) - float(target_vx)
        relative_vy = float(self.drone_linear_velocity.y) - float(target_vy)
        relative_vz = float(self.drone_linear_velocity.z) - float(target_vz)
        relative_xy_speed = math.hypot(relative_vx, relative_vy)

        result.update({
            "landed": False,
            "position_ok": position_ok,
            "relative_x": relative_x,
            "relative_y": relative_y,
            "relative_height": relative_height,
            "relative_xy_distance": relative_xy_distance,
            "relative_vx": relative_vx,
            "relative_vy": relative_vy,
            "relative_vz": relative_vz,
            "relative_xy_speed": relative_xy_speed,
        })
        return result

    def step(self, action):
        # rospy.logdebug(f"===== STEP开始... =====")
        done = False
        target = False

        self._publish_action_velocity(action)

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

        # print 减少频率，只打印关键信息
        # print(f"世界位置... 相对位置...")

        # 动态平台只使用 WAM-V 甲板碰撞判定落地。
        # YOLO 与相对运动只用于导航、越界和评估记录。
        landing_detected = False
        landing_success = False
        deck_contact = False
        out_of_bounds = False
        visual_out_of_bounds = False
        lost_detection = False
        terminal_reason = "RUNNING"
        visual_height = float(self.yolo_distance_z)
        relative_height = float("nan")
        relative_xy_distance = float("nan")
        position_ok = False
        visual_x = float(self.yolo_center_x)
        visual_y = float(self.yolo_center_y)
        rel_vx = rel_vy = rel_vz = float("nan")
        rel_xy_speed = float("nan")
        now = rospy.Time.now()
        detection_age = max(0.0, (now - self.last_yolo_detection_time).to_sec())
        episode_age = max(0.0, (now - self._episode_start_time).to_sec())
        # 只依据最后一次有效检测的时间抗抖。YOLO 短时断帧期间继续使用
        # 最近一次有效观测，超过 timeout 后才声明真正失检。
        detection_fresh = (
            self._episode_has_detection
            and detection_age <= self.yolo_lost_timeout
        )

        if self.comm.current_position is not None:
            if self.landing_target_fn is not None:
                landing_state = self._check_dynamic_landing()
                landing_detected = landing_state["landed"]
                deck_contact = landing_state["deck_contact"]
                position_ok = landing_state["position_ok"]
                landing_success = landing_detected and position_ok
                relative_height = landing_state["relative_height"]
                relative_xy_distance = landing_state["relative_xy_distance"]
                rel_vx = landing_state["relative_vx"]
                rel_vy = landing_state["relative_vy"]
                rel_vz = landing_state["relative_vz"]
                rel_xy_speed = landing_state["relative_xy_speed"]

                if landing_detected:
                    terminal_reason = (
                        "DECK_CONTACT_LANDED"
                        if landing_success
                        else "DECK_CONTACT_OFF_TARGET"
                    )
                    rospy.loginfo(
                        f"检测到落地: contact={deck_contact}, success={landing_success}, "
                        f"rel_h={relative_height:.3f}m, "
                        f"rel_xy={relative_xy_distance:.3f}m, "
                        f"rel_vxy={rel_xy_speed:.3f}m/s, rel_vz={rel_vz:.3f}m/s"
                    )
                elif detection_fresh and (
                    abs(visual_x) > self.visual_x_limit
                    or abs(visual_y) > self.visual_y_limit
                ):
                    visual_out_of_bounds = True
                    out_of_bounds = True
                    done = True
                    terminal_reason = "YOLO_OUT_OF_BOUNDS"
                    self.Step = 0
                    rospy.logwarn(
                        f"YOLO视觉越界: x={visual_x:.3f} (limit={self.visual_x_limit:.2f}), "
                        f"y={visual_y:.3f} (limit={self.visual_y_limit:.2f})"
                    )
            else:
                if abs(self.comm.current_position.z) < self.landing_z_threshold:
                    landing_detected = True
                    landing_success = True
                    rospy.loginfo("成功着陆! (静态模式)")

        if landing_detected:
            target = landing_success
            done = True
            self.Step = 0

        # 已成功触地则不再标越界/失检, 避免 SUCCESS 被覆盖
        if not done and not landing_detected and self.comm.current_position is not None:
            x0 = self.comm.current_position.x
            y0 = self.comm.current_position.y
            z0 = self.comm.current_position.z
            dist_origin = math.sqrt(x0**2 + y0**2 + z0**2)
            # z < 0: 水下无碰撞体, 必须立即终止, 否则会继续穿水飞行
            if dist_origin > self.max_dist or z0 > self.max_height or z0 < 0.0:
                reason = []
                if dist_origin > self.max_dist:
                    reason.append(f"Dist={dist_origin:.2f}>{self.max_dist:.1f}")
                if z0 > self.max_height:
                    reason.append(f"Height={z0:.2f}>{self.max_height:.1f}")
                if z0 < 0.0:
                    reason.append(f"Underwater z={z0:.2f}<0")
                rospy.logwarn(f"越界结束: {', '.join(reason)}")
                done = True
                out_of_bounds = True
                terminal_reason = "WORLD_OUT_OF_BOUNDS"
                self.Step = 0

        if done:
            self._publish_action_velocity(np.zeros(3, dtype=np.float32))

        success = target and done
        yolo_confidence = (
            float(self.yolo_confidence) if detection_fresh else 0.0
        )
        info = {
            "tag_detected": detection_fresh,
            "detection_fresh": detection_fresh,
            "target_reached": target,
            "success": success,
            "out_of_bounds": out_of_bounds,
            "visual_out_of_bounds": visual_out_of_bounds,
            "lost_detection": lost_detection,
            "terminal_reason": terminal_reason,
            "position_ok": bool(position_ok),
            "deck_contact": bool(deck_contact),
            "landed": bool(landing_detected),
            "landing_success": bool(success),
            "visual_height": visual_height,
            "relative_height": relative_height,
            "relative_xy_distance": relative_xy_distance,
            "visual_x": visual_x,
            "visual_y": visual_y,
            "rel_vx": rel_vx,
            "rel_vy": rel_vy,
            "rel_vz": rel_vz,
            "rel_xy_speed": rel_xy_speed,
            "detection_age": detection_age,
            "yolo_confidence": yolo_confidence,
            "xy_threshold": self.landing_xy_threshold,
        }
        return next_state, done, success, info

    def _publish_action_velocity(self, action):
        """策略动作保持为速度指令, 与离线训练/原环境语义一致。"""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 3:
            raise ValueError(f"action 至少需要 3 维, 当前为 {action.shape}")
        self.comm.set_velocity_target(action[0], action[1], action[2])

    def reset(self):
        # 同样的保护
        try:
            rospy.wait_for_service("/gazebo/unpause_physics", timeout=1.0)
            self.unpause()
        except Exception:
            pass

        # === 修改点 4: Reset 时重置内部标志位，而不是重启进程 ===
        self.yolo_detected = False
        self.yolo_confidence = 0.0
        self._episode_has_detection = False
        self.drone_linear_velocity = None
        self.deck_contact = False
        self._episode_start_time = rospy.Time.now()
        self.last_yolo_detection_time = self._episode_start_time
        # self.stop_yolo() <-- 已注释
        # self.start_yolo() <-- 已注释

        rospy.loginfo("等待当前位置数据...")
        try:
            pose_topic = f"{self.comm.vehicle_type}_{self.comm.vehicle_id}/mavros/local_position/pose"
            pose_msg = rospy.wait_for_message(pose_topic, PoseStamped, timeout=30.0)
            self.comm.current_position = pose_msg.pose.position
            self.comm.current_yaw = self.comm.q2yaw(pose_msg.pose.orientation)
            rospy.loginfo("当前位置数据已接收")
        except rospy.ROSException as e:
            rospy.logerr(f"等待位置数据超时: {e}")
            raise RuntimeError("无法获取初始位置")

        current_pose = self.comm.set_pose_target(
            self.comm.current_position.x,
            self.comm.current_position.y,
            self.comm.current_position.z,
            yaw=self.comm.current_yaw,
            publish=False,
        )

        # 确保位置 setpoint 发布器连接
        timeout_connection = 10.0
        start_wait = time.time()
        while (time.time() - start_wait) < timeout_connection and not rospy.is_shutdown():
            if self.comm.pose_target_pub.get_num_connections() > 0:
                break
            rospy.sleep(0.1)

        # 预发送 setpoints
        t_end = time.time() + 2.0
        while time.time() < t_end and not rospy.is_shutdown():
            self.comm.publish_pose_setpoint(current_pose)
            rospy.sleep(0.05)

        # 切换 OFFBOARD
        self.comm.arm_state = self.comm.arm()
        self.comm.set_mode("OFFBOARD")

        # 等待生效
        wait_start = time.time()
        while time.time() - wait_start < 5.0:
            try:
                mav_state = rospy.wait_for_message(self.comm.vehicle_type+'_'+self.comm.vehicle_id+'/mavros/state', State, timeout=1.0)
                if mav_state.mode == 'OFFBOARD' and mav_state.armed:
                    break
            except Exception:
                pass
            self.comm.publish_pose_setpoint(current_pose)
            rospy.sleep(0.1)

        rospy.sleep(0.5)

        # 随机起始位置（四象限均衡：以船位置 (10,5) 为中心轮流采样）
        self.pose = Pose()

        # 范围: 以 WAM-V 船 (10, 5) 为中心, ±4m, 确保 YOLO 能检测到甲板 marker
        x_min, x_max = 6.0, 14.0
        y_min, y_max = 1.0, 9.0
        z_min, z_max = 7.5, 9.0

        # 以矩形中心作为“零点”
        cx = 0.5 * (x_min + x_max)   # 10.0
        cy = 0.5 * (y_min + y_max)   # 5.0

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
            self.comm.publish_pose_setpoint(self.pose)

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
            self.comm.publish_pose_setpoint(self.pose)
            rospy.sleep(0.1)

        # === 修改点 5: 不再重启 YOLO，而是直接准备开始 ===
        # self.stop_yolo() <-- 已注释
        # self.start_yolo() <-- 已注释

        # 确保模式正确
        self.comm.set_mode("OFFBOARD")

        # 暂停物理
        try:
            rospy.wait_for_service("/gazebo/pause_physics", timeout=1.0)
            self.pause()
        except Exception:
            pass

        state = self.get_state()
        rospy.loginfo("===== 重置完成 =====")
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.stop_yolo()
        self._terminate_process_group("Gazebo", "gazebo_process")
        self._terminate_process_group("ROS master", "roscore_process")

    def _terminate_process_group(self, label, attribute):
        process = getattr(self, attribute, None)
        if process is None:
            return
        try:
            if process.poll() is None:
                rospy.loginfo(f"停止{label}进程...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            setattr(self, attribute, None)

    def get_state(self):
        """
        获取视觉观测 [x, y, z], 其中 z 为目标距离/高度估计。
        """
        pos_x, pos_y, pos_z = 0.0, 0.0, 10.0

        if hasattr(self, 'yolo_detected') and self.yolo_detected:
            pos_x = self.yolo_center_x
            pos_y = self.yolo_center_y
            pos_z = self.yolo_distance_z

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
    def __init__(self, vehicle_type, vehicle_id,
                 configure_rc_loss_exception=True,
                 state_timeout=30.0,
                 param_timeout=5.0):
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id
        self.current_position = None
        self.current_yaw = 0
        self.last_publish_time = rospy.Time.now()
        self.publish_rate = 30.0
        self.target_pose = PoseStamped()
        self.has_target_pose = False
        self.target_velocity = PositionTarget()
        self.has_target_velocity = False

        self.arm_state = False

        state_topic = self.vehicle_type+'_'+self.vehicle_id+"/mavros/state"
        rospy.loginfo(f"等待 FCU 状态话题: {state_topic} (timeout={state_timeout:.1f}s)")
        deadline = time.monotonic() + float(state_timeout)
        mavros_state = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            remaining = deadline - time.monotonic()
            try:
                mavros_state = rospy.wait_for_message(
                    state_topic, State, timeout=min(1.0, max(remaining, 0.01))
                )
            except rospy.ROSException:
                continue
            if mavros_state.connected:
                break
            rospy.loginfo_throttle(1.0, "已收到 MAVROS 状态，等待 FCU heartbeat...")
        if mavros_state is None or not mavros_state.connected:
            raise RuntimeError(
                f"等待 FCU 连接超时: {state_topic}; 请检查 MAVROS fcu_url 与 PX4 SITL"
            )
        rospy.loginfo("FCU 已连接")

        self.local_pose_sub = rospy.Subscriber(self.vehicle_type+'_'+self.vehicle_id + "/mavros/local_position/pose",
                                              PoseStamped, self.local_pose_callback, queue_size=1)
        self.pose_target_pub = rospy.Publisher(self.vehicle_type+'_'+self.vehicle_id+"/mavros/setpoint_position/local",
                                               PoseStamped, queue_size=1)
        self.velocity_target_pub = rospy.Publisher(self.vehicle_type+'_'+self.vehicle_id+"/mavros/setpoint_raw/local",
                                                   PositionTarget, queue_size=1)

        self.armService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/cmd/arming", CommandBool)
        self.flightModeService = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/set_mode", SetMode)
        self.set_param_srv = rospy.ServiceProxy(self.vehicle_type+'_'+self.vehicle_id+"/mavros/param/set", ParamSet)

        if configure_rc_loss_exception:
            self._configure_rc_loss_exception(float(param_timeout))
        else:
            rospy.loginfo("专家采集模式: 跳过 COM_RCL_EXCEPT 参数设置")

        print(
            self.vehicle_type+'_'+self.vehicle_id+": communication initialized",
            flush=True,
        )

    def _configure_rc_loss_exception(self, timeout):
        service_name = self.vehicle_type+'_'+self.vehicle_id+"/mavros/param/set"
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
        except rospy.ROSException:
            rospy.logwarn(
                f"参数服务 {service_name} 在 {timeout:.1f}s 内不可用，跳过 COM_RCL_EXCEPT"
            )
            return

        result = {"error": None}

        def set_parameter():
            try:
                self.set_param_srv("COM_RCL_EXCEPT", ParamValue(4, 0.0))
            except Exception as error:
                result["error"] = error

        worker = threading.Thread(target=set_parameter)
        worker.daemon = True
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            rospy.logwarn(
                f"设置 COM_RCL_EXCEPT 超过 {timeout:.1f}s，继续启动"
            )
        elif result["error"] is not None:
            rospy.logwarn(f"设置 COM_RCL_EXCEPT 失败: {result['error']}")
        else:
            rospy.loginfo("COM_RCL_EXCEPT 设置完成")

    def start(self):
        rate = rospy.Rate(self.publish_rate)
        self.last_publish_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.has_target_velocity:
                try:
                    self.publish_velocity_setpoint(self.target_velocity)
                except Exception:
                    pass
            elif self.has_target_pose:
                try:
                    self.publish_pose_setpoint(self.target_pose)
                except Exception:
                    pass
            self.last_publish_time = rospy.Time.now()
            rate.sleep()

    def local_pose_callback(self, msg):
        self.current_position = msg.pose.position
        self.current_yaw = self.q2yaw(msg.pose.orientation)

    def set_pose_target(self, x, y, z, yaw=0.0, publish=True):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
        pose.pose.orientation.w = math.cos(float(yaw) * 0.5)

        self.target_pose = pose
        self.has_target_pose = True
        self.has_target_velocity = False
        if publish:
            self.pose_target_pub.publish(pose)
        return pose

    def publish_pose_setpoint(self, pose):
        if isinstance(pose, PoseStamped):
            msg = pose
        else:
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.pose = pose
        msg.header.stamp = rospy.Time.now()
        self.target_pose = msg
        self.has_target_pose = True
        self.has_target_velocity = False
        self.pose_target_pub.publish(msg)
        return msg

    def set_velocity_target(self, vx, vy, vz, yaw_rate=0.0):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = 8
        msg.type_mask = (
            PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ
            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ
            + PositionTarget.IGNORE_YAW
        )
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw_rate = float(yaw_rate)
        self.target_velocity = msg
        self.has_target_velocity = True
        self.has_target_pose = False
        self.velocity_target_pub.publish(msg)
        return msg

    def publish_velocity_setpoint(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.target_velocity = msg
        self.has_target_velocity = True
        self.has_target_pose = False
        self.velocity_target_pub.publish(msg)
        return msg

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

    def set_mode(self, mode):
        try:
            ok = self.flightModeService(custom_mode=mode)
            if ok:
                print(self.vehicle_type+'_'+self.vehicle_id+": "+mode)
                return True
        except Exception as e:
            print(self.vehicle_type+'_'+self.vehicle_id+": "+mode+" failed: "+str(e))
        return False
