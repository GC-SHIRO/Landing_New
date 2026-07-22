"""
用途:
- 封装 PX4/MAVROS 的基础无人机控制接口, 提供解锁、起飞、悬停、降落与 waypoint 控制。

用法:
- 作为工具模块被环境脚本导入, 通常不单独运行。
- 典型调用: Drone().arm() / takeoff() / hover() / land()。

实现方式:
- 通过 ROS topic 发布位置设定点, 并调用 MAVROS 服务切换 OFFBOARD 与解锁状态。
- 维护当前位姿与飞控状态, 用于闭环控制与状态判断。

依赖关系:
- 被 landing_env.py / landing_env_listen.py / pre_set_works_for_ROS.py 间接使用。
- 依赖 rospy、geometry_msgs、mavros_msgs。
"""
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State 
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np
from numpy.linalg import norm
import time


class Drone:
    def __init__(self):
        self.pose = None
        self.yaw = 0
        self.sp = None
        self.hz = 10
        self.rate = rospy.Rate(self.hz)

        self.current_state = State()
        self.prev_request = None
        self.prev_state = None
        self.state = None

        self.setpoint_publisher = rospy.Publisher('/iris_0/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy('/iris_0/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/iris_0/mavros/set_mode', SetMode)
        rospy.Subscriber('/iris_0/mavros/state', State, self.state_callback)
        rospy.Subscriber('/iris_0/mavros/local_position/pose', PoseStamped, self.drone_pose_callback)

    def state_callback(self, state):
        self.current_state = state

    def drone_pose_callback(self, pose_msg):
        self.pose = np.array([ pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z ])

    def arm(self):
        for i in range(self.hz):
            self.publish_setpoint([0,0,-1])
            self.rate.sleep()
    
        # wait for FCU connection
        while not self.current_state.connected:
            print('Waiting for FCU connection...')
            self.rate.sleep()

        prev_request = rospy.get_time()
        prev_state = self.current_state
        while not rospy.is_shutdown():
            now = rospy.get_time()
            if self.current_state.mode != "OFFBOARD" and (now - prev_request > 2.):
                self.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
                prev_request = now 
            else:
                if not self.current_state.armed and (now - prev_request > 2.):
                   self.arming_client(True)
                   prev_request = now 

            # older versions of PX4 always return success==True, so better to check Status instead
            if prev_state.armed != self.current_state.armed:
                print("Vehicle armed: %r" % self.current_state.armed)

            if prev_state.mode != self.current_state.mode: 
                print("Current mode: %s" % self.current_state.mode)
            prev_state = self.current_state

            if self.current_state.armed:
                break
            # Update timestamp and publish sp 
            self.publish_setpoint([0,0,-1])
            self.rate.sleep()

    @staticmethod
    def get_setpoint(x, y, z,yaw=np.pi/2):
        set_pose = PoseStamped()
        set_pose.pose.position.x = x
        set_pose.pose.position.y = y
        set_pose.pose.position.z = z
        set_pose.pose.orientation.x = 0
        set_pose.pose.orientation.y = 0
        set_pose.pose.orientation.z = 0.8
        set_pose.pose.orientation.w = 1.5
        return set_pose
        
    def publish_setpoint(self, sp, yaw=np.pi/2):
        setpoint = self.get_setpoint(sp[0], sp[1], sp[2], yaw)
        setpoint.header.stamp = rospy.Time.now()
        self.setpoint_publisher.publish(setpoint)
    def takeoff(self, height):
        print("Takeoff...")
        self.sp = self.pose
        while self.pose[2] < height:
            print(self.pose[2])
            self.sp[0] = 0
            self.sp[1] = 0
            self.sp[2] += 0.5
            self.publish_setpoint(self.sp)
            self.rate.sleep()

    def hover(self, t_hold):
        print('Position holding...')
        t0 = time.time()
        self.sp = self.pose
        while not rospy.is_shutdown():
            t = time.time()
            if t - t0 > t_hold and t_hold > 0: break
            # Update timestamp and publish sp 
            self.publish_setpoint(self.sp)
            self.rate.sleep()

    def land(self):
        print("Landing...")
        self.sp = self.pose
        while self.sp[2] > - 1.0:
            self.sp[2] -= 0.5
            self.publish_setpoint(self.sp)
            self.rate.sleep()
        # self.stop()

    def stop(self):
        while self.current_state.armed or self.current_state.mode == "OFFBOARD":
            if self.current_state.armed:
                self.arming_client(False)
            if self.current_state.mode == "OFFBOARD":
                self.set_mode_client(base_mode=0, custom_mode="MANUAL")
            self.rate.sleep()

    @staticmethod
    def transform(pose):
        # transformation: x - froward, y - left, z - up (ENU - MoCap frame)
        pose_new = np.zeros(3)
        pose_new[0] = - pose[1]
        pose_new[1] = pose[0]
        pose_new[2] = pose[2]
        return pose_new

    def goTo(self, wp, mode='global', tol=0.05):
        wp = self.transform(wp)
        if mode=='global':
            goal = wp
        elif mode=='relative':
            goal = self.pose + wp
        print ("Going to a waypoint...")
        self.sp = self.pose
        while norm(goal - self.pose) > tol:
            n = (goal - self.sp) / norm(goal - self.sp)
            self.sp += 0.03 * n
            self.publish_setpoint(self.sp)
            self.rate.sleep()
    def get_drone_pose(self):
        time.sleep(0.1)
        for i in range(5):
            drone_pose = self.pose[2]
            self.rate.sleep()
        return drone_pose
