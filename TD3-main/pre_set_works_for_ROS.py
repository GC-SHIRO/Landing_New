

"""
用途:
- 提供 ROS 侧预设流程与视觉/位姿相关辅助函数, 用于降落任务调试与流程验证。

用法:
- 主要作为实验辅助脚本运行或被其他 ROS 脚本复用, 不属于 TD3 主训练闭环。

实现方式:
- 订阅图像、IMU、Gazebo 模型状态等话题, 提供目标跟踪、误差计算和动作发布辅助。
- 包含若干预设流程函数, 用于快速验证传感器和控制链路是否可用。

依赖关系:
- 依赖 drone.py 与 ROS 消息类型。
- 与 landing_env 系列脚本共享仿真通信基础设施。
"""

import cv2
import torch
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import TwistStamped

from gazebo_msgs.msg import ModelStates, ModelState
# from landing.msg import center
from pyquaternion import Quaternion
from drone import *
import time
import argparse

parser = argparse.ArgumentParser()
device = 'cpu'
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)

parser.add_argument("--config", type=str, help="Path to the config file.")
args = parser.parse_args()
takeoffheight = 5
imu = None
gps = None
local_pose = None
current_state = None
# self.current_heading = None
current_heading = None
takeoff_height = 5
local_enu_position = None

cur_target_pose = None
global_target = None

received_new_task = False
arm_state = False
offboard_state = False
received_imu = False
frame = "BODY"

# state = None
# arm_state = False
# offboard_state = False
# current_heading = None
# local_pose = None
# cur_target_pose = None
def drone_pose_callback(pose_msg):
    global drone_pose
    drone_pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])

def q2yaw(q):
    if isinstance(q, Quaternion):
        rotate_z_rad = q.yaw_pitch_roll[0]
    else:
        q_ = Quaternion(q.w, q.x, q.y, q.z)
        rotate_z_rad = q_.yaw_pitch_roll[0]

    return rotate_z_rad

def imu_callback(msg):
    global global_imu, current_heading
    imu = msg

    current_heading = q2yaw(imu.orientation)

    received_imu = True
def timetable(tim):
    toc = time.time()
    consume_time = tim - toc
    return consume_time
def actionpubilsh(action):
        vel.twist.linear.x = action[0]
        vel.twist.linear.y = action[1]
        vel.twist.linear.z = action[2]
        x = 0
        while x<8000:
            action_pub.publish(vel)
            x +=1

def center_cb(msg):
     global landmark
     landmark = msg
def Imu_cb(msg):
    global carvel
    carvel = msg
def Imu_cb1(msg):
    global dronevel
    dronevel = msg
def callback(msg):
    global current_state
    current_state = msg
def local_pose_callback(msg):
    local_pose = msg
    local_enu_position = msg

# This contains the position of a point in free space
# MSG: geometry_msgs / Quaternion
# This represents an orientation in free space in quaternion form.
current_act = TwistStamped()


# center_sub = rospy.Subscriber('center', center, center_cb)
dronevel_sub = rospy.Subscriber('mavros/imu/data', Imu, Imu_cb1)
carvel_sub = rospy.Subscriber('imu/data', Imu, Imu_cb)
imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, imu_callback)
UAV_state = rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
action_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
# landmark = center()  # tarcker.py give info(pad's hight,width to the center.msg ,landmark read from it.
current_state1 = State()  # state=mode such as armed,landing,etc
ms = ModelState()
ms.model_name = 'husky'
carvel = Imu()
dronevel = Imu()
pose = PoseStamped()  # MSG: geometry_msgs/Point
# last_time = rospy.Time.now()
current_state = ModelStates()
vel = TwistStamped()

def callback(Image):
    img = np.fromstring(Image.data, np.uint8)
    img = img.reshape(720, 720, 3)
    track(img, Image.width, Image.height)



    
    
def move_husky():
    global  ms
    # des_x = random.uniform(drone_position.x - rectx, drone_position.x + rectx)
    des_x = -2
    # des_y = random.uniform(drone_position.y - recty, drone_position.y + recty)
    des_y = 0
    angle = 2
    ms.pose.position.x = des_x
    ms.pose.position.y = des_y
    ms.pose.orientation.z = 0
    ms.pose.orientation.w = 1
    pub.publish(ms)

def listener():
    rospy.Subscriber('/iris_fpv_cam/usb_cam/image_raw', Image, callback)
    rospy.spin()


def track(frame, width, height):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # for i in range(0,101):
    #     cv2.imwrite(str(i)+'.jpg',img)
    #     time.sleep(0.7)
    # time_now = time.time_ns()
    # cv2.imwrite(str(time_now)+'.jpg',img)
    # time.sleep(0.2)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    centers = []
    z = np.double(1.0)
    for contour in contours[1]:
        if cv2.contourArea(contour) > 307200 or cv2.contourArea(contour) < 1500:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] == 4 and cv2.isContourConvex(approx):
            rects.append(approx)
            centers.append((approx[0] + approx[1] + approx[2] + approx[3]).squeeze() / 4.0)
            z = 0.554 * (388800 / (cv2.contourArea(approx))) ** 0.5

    cv2.polylines(frame, rects, True, (0, 0, 255), 2)
    cv2.imshow('w', frame)
    # for i in range(0,101):
    #     cv2.imwrite(str(i)+'.jpg',frame)
    #     time.sleep(0.7)
    cv2.waitKey(1)



def get_error_from_apriltag():

       husky_x =current_state.pose[2].position.x
       husky_y =current_state.pose[2].position.y
       drone_x =current_state.pose[1].position.x
       drone_y =current_state.pose[1].position.y
       delta_x = drone_x - husky_x
       delta_y = drone_y - husky_y
       time.sleep(0.1)
       print(float(delta_x-1.5),float(delta_y))
       return  float(delta_x-1.5),float(delta_y)
def drone_landing_preset_works():
    listener()
def reward_setup(observation, observation_,height1):
    drone = Drone()
    delta_x,delta_y = get_error_from_apriltag()
    height2 = drone.get_drone_pose()
    shape2 = - ((abs(delta_x) ** 3 + abs(delta_y) ** 3 + height2 ** 3) ** (1 / 3))
    reward = 0.1 * (shape2)
    if drone.get_drone_pose() < 0.9:
            err_x_ , err_y_ = get_error_from_apriltag()
            height2 = drone.get_drone_pose()
            if 0.5 > err_x_ > -0.5 and 0.2 > err_y_ > -0.2:
                reward = 300

                print("landed successfully", observation, reward, observation_, height2)
                return reward
            elif 1 > abs(err_x_) > 0.5 or 0.8>abs(err_y_) > 0.2 :
                reward = -1
                print("landed,but unsuccessfully", observation, reward, observation_, height2)
                return reward
            else:
                reward = -200
                print('landed else where', observation, reward, observation_, height2)
                return reward
    return reward


def timetable(tim):
        toc = time.time()
        consume_time = tim - toc
        return consume_time
def store_noise_record(action,action_list,takeoffheight):
    drone = Drone()
    h = drone.get_drone_pose() / takeoffheight
    if len(action_list)>3:
     action1 = np.array(action)
     action_list1 = np.array(action_list)
     action = h*(action1 +np.random.beta(2,5)*(action_list1[-1]+action_list1[-2])*0.5)
     action_list.append(action)
     print(action)
     return action
    else:
     action_list.append(action)
     return action

# def noise_added_to_action(action,max_action):
#     if action is complex:
#         return [0,0,0]
#     else:
#         action = torch.Tensor(action)
#         noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(device)
#         noise = noise.clamp(-args.noise_clip, args.noise_clip)
#         next_action = (action + noise)
#         next_action = next_action.clamp(-max_action, max_action)
#         next_action = next_action.numpy().tolist()
#         print(next_action)
#         return next_action
def get_k1(ep,max_ep,batch_size):
    if 4*ep < max_ep :
        return int(batch_size/4),0.25
    elif 2*ep < max_ep :
        return int(batch_size/2),0.5
    elif 4*ep < 3*max_ep:
        return int(3*batch_size/4),0.75
def observation_confirm():
    altitude = Drone().get_drone_pose()
    delta_x,delta_y = get_error_from_apriltag()
    observation_ = np.array([delta_x, delta_y,altitude])
    return observation_
def get_best_score(acc_reward_history):
    acc_reward_history = acc_reward_history.sort()
    average = 0
    for i in (1,len(acc_reward_history)/4):
        average += acc_reward_history[-i]
    average = average/(int(len(acc_reward_history)/2))
    return average
# def get_drone_height()

