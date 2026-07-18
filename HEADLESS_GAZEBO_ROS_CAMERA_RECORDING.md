# 无头 Gazebo 第三人称录像固定流程

适用场景：在服务器上不启动 `gzclient`（Gazebo GUI），通过 Gazebo 内部相机模型发布 ROS 图像话题，再录制为 MP4。该流程用于离线观察无人机、降落平台、WAM-V 和 marker 的相对位置；**不向 PID、KF+PID 或 STAR-TD3 控制器注入 Gazebo truth（仿真真值）**。

## 固定数据流

```text
gzserver + PX4 + MAVROS (+ 可选 YOLO)
    |
    +-- landing_review_third_person_cam
    |       `-- /landing_review/third_person/image_raw
    |                    `-- record_landing_gazebo_views.py --> H.264 MP4
    |
    +-- /gazebo/get_model_state --> 仅用于第三人称相机跟随和 CSV 审计
    +-- /marker_yolo_detector/point --> 控制器的真实视觉输入（若控制器运行）
    `-- rosbag --> 轨迹、姿态、视觉输入审计
```

## 文件与默认参数

在开始前，请按实际机器填写以下环境变量；不要将服务器用户名、IP、密码或 token 写入仓库：

```bash
export LANDING_NEW_ROOT=/path/to/Landing_New
export PX4_FIRMWARE_ROOT=/path/to/PX4_Firmware
export CATKIN_WS_ROOT=/path/to/catkin_ws
```

已验证文件位于：

```text
recordings/step1_scene_20260718/
├── landing_review_third_person_cam.sdf
├── record_landing_gazebo_views.py
├── run_step1_ship_motion.py                 # Step1 演示时可选
└── limit_online_cpus.c                       # Gazebo/OGRE CPU 数量兼容层
```

默认相机模型与话题：

| 项目 | 值 |
| --- | --- |
| 模型 | `landing_review_third_person_cam` |
| 图像话题 | `/landing_review/third_person/image_raw` |
| SDF 设置 | 960×540、目标更新率 15 Hz |
| 当前 Step1 目标 | `wamv`（移动目标）和 `iris_1`（无人机/平台侧） |
| MAVROS 命名空间 | `/iris_1/mavros/...` |

`15 Hz` 是相机的目标仿真更新率。最终 MP4 会按实际收到的帧率转码；服务器负载高时实际帧率会低于 15 Hz，但视频时长仍与真实录制时长一致。

## 1. 启动隔离的无头仿真

先进入环境，并确认隔离端口、显示号、磁盘和现有用户进程不会冲突：

```bash
cd "$LANDING_NEW_ROOT/recordings/step1_scene_20260718"
source ~/.bashrc.landing_new
ss -ltnp | grep -E ':(11317|11351) ' || true
pgrep -af 'Xvfb :109' || true
df -h "$LANDING_NEW_ROOT"
```

OGRE 在高核心数服务器上可能创建异常多的渲染线程。以下兼容层将 `gzserver` 看到的在线 CPU 数限制为 32；它只作用于当前 `gzserver`，不影响其他用户进程：

```bash
gcc -shared -fPIC limit_online_cpus.c -o /tmp/codex_limit_roscam.so -ldl
exec 9</tmp/codex_limit_roscam.so
rm -f /tmp/codex_limit_roscam.so

export DISPLAY=:109
export ROS_MASTER_URI=http://127.0.0.1:11317
export GAZEBO_MASTER_URI=http://127.0.0.1:11351

Xvfb :109 -ac -screen 0 1280x720x24 -nolisten tcp > xvfb_roscam.log 2>&1 & echo $! > xvfb_roscam.pid
roscore -p 11317 > roscore_roscam.log 2>&1 & echo $! > roscore_roscam.pid

source "$PX4_FIRMWARE_ROOT/Tools/setup_gazebo.bash" \
  "$PX4_FIRMWARE_ROOT" \
  "$PX4_FIRMWARE_ROOT/build/px4_sitl_default"

LD_PRELOAD=/proc/self/fd/9 gzserver -e ode \
  "$CATKIN_WS_ROOT/devel/share/vrx_gazebo/worlds/example_course.world" \
  -s "$CATKIN_WS_ROOT/devel/lib/libgazebo_ros_paths_plugin.so" \
  -s "$CATKIN_WS_ROOT/devel/lib/libgazebo_ros_api_plugin.so" \
  > gzserver_roscam.log 2>&1 & echo $! > gzserver_roscam.pid

roslaunch -p 11317 "$PX4_FIRMWARE_ROOT/launch/step1_linear.launch" \
  start_gazebo:=false ID:=1 ID_in_group:=1 \
  mavlink_udp_port:=18571 mavlink_tcp_port:=4561 \
  fcu_url:=udp://:24541@localhost:34581 gui:=false \
  > models_roscam.log 2>&1 & echo $! > launch_roscam.pid
```

`exec 9<...` 打开的是本次 shell 专用文件描述符；在 `gzserver` 结束前不要关闭它。

## 2. 生成 Gazebo ROS Camera

不要启动 GUI。待 `/gazebo/spawn_sdf_model` 可用后，执行一次：

```bash
rosrun gazebo_ros spawn_model -sdf \
  -model landing_review_third_person_cam \
  -file landing_review_third_person_cam.sdf \
  -x 12.5 -y -25 -z 30 -R 0 -P 0.78539816339 -Y 1.57079632679

rostopic list | grep '^/landing_review/third_person/'
timeout 8 rostopic hz /landing_review/third_person/image_raw -w 15
```

必须看到 `image_raw` 和 `camera_info`。同一 `gzserver` 会话中不要反复删除并以**相同名称**重建相机；`libgazebo_ros_camera.so` 的动态参数服务可能重名。需要重建时请使用新模型/相机名称，或重启本次隔离仿真。

## 3. 可选：启动 YOLO

当前 Step1 双目输入来自 `iris_1`。此节点的输出映射到控制器惯用的话题名：

```bash
CUDA_VISIBLE_DEVICES=0 rosrun yolov11_ros yolo_v11.py \
  _use_gpu:=true \
  _weight_path:="$CATKIN_WS_ROOT/src/yolov11_ros/weights/best.pt" \
  _left_image_topic:=/iris_1/stereo_camera/left/image_raw \
  _right_image_topic:=/iris_1/stereo_camera/right/image_raw \
  _centers_topic:=/marker_yolo_detector/point \
  _visualize:=false \
  > yolo_roscam.log 2>&1 & echo $! > yolo_roscam.pid
```

YOLO 没有检测到 marker 不会阻塞第三人称录像。若运行控制器，控制器仍应只订阅 `/marker_yolo_detector/point`，不要把 `/gazebo/get_model_state` 的结果接入控制输入。

## 4. 启动录像与相机跟随

下面示例录制 300 秒。`--primary-model`、`--target-model` 用于相机跟随和审计；可按实际模型名替换。`--raw-only` 输出干净画面，不叠加坐标和文字。

```bash
OUT_DIR=ros_camera_follow_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

python record_landing_gazebo_views.py \
  --out-dir "$OUT_DIR" \
  --controller step1 \
  --run-s 300 --fps 15 --sample-hz 10 \
  --third-topic /landing_review/third_person/image_raw \
  --camera-topic '' \
  --primary-model wamv --target-model iris_1 \
  --visual-topic /marker_yolo_detector/point \
  --state-topic /iris_1/mavros/state \
  --pose-topic /iris_1/mavros/local_position/pose \
  --follow-camera-model landing_review_third_person_cam \
  --follow-back 16 --follow-z 9 --follow-pitch 0.85 \
  --look-at-mode midpoint3d --raw-only \
  > "$OUT_DIR/recorder.out" 2> "$OUT_DIR/recorder.err" &
echo $! > recorder_roscam.pid
```

录制器会：

- 订阅第三人称 ROS 图像；
- 周期性读取两个模型状态，并移动第三人称相机；
- 每帧先写入 MJPG 临时 AVI，结束后转为 H.264 MP4；
- 生成 CSV、JSONL 和汇总 JSON；
- 使用 `-nostdin`、写入锁和实际帧率转码，适合在 SSH 后台运行。

## 5. 可选：独立录制 rosbag

```bash
rosbag record -O "$OUT_DIR/step1_ep01.bag" \
  /gazebo/model_states \
  /iris_1/mavros/local_position/pose \
  /iris_1/mavros/imu/data \
  /iris_1/mavros/setpoint_raw/local \
  /iris_1/mavros/state \
  /marker_yolo_detector/point \
  > "$OUT_DIR/rosbag.out" 2> "$OUT_DIR/rosbag.err" &
echo $! > rosbag_roscam.pid
```

rosbag 用于轨迹、姿态和视觉输入审计，**不用于生成第三人称视频**。

## 6. 产物与验证

每次正常录制应至少产生：

```text
<controller>_third_person_gazebo_raw.mp4
<controller>_third_person_gazebo.mp4
<controller>_gazebo_view_samples.csv
<controller>_gazebo_view_samples.jsonl
<controller>_gazebo_view_record_summary.json
```

验证命令：

```bash
cat "$OUT_DIR/step1_gazebo_view_record_summary.json"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 "$OUT_DIR/step1_third_person_gazebo_raw.mp4"
ffmpeg -nostdin -v error -i "$OUT_DIR/step1_third_person_gazebo_raw.mp4" -f null -
rosbag info "$OUT_DIR/step1_ep01.bag"
```

`third_*_effective_fps`（汇总 JSON）代表实际收到图像的帧率。若该值明显偏低，优先降低分辨率或暂停 YOLO，再考虑减少 Gazebo 场景负载；不要人为把视频补帧后当作新增观测信息。

## 7. 仅清理本次隔离任务

不要使用广泛的 `pkill`，避免影响服务器其他用户。只根据本次 PID 文件清理：

```bash
ROS_MASTER_URI=http://127.0.0.1:11317 rosnode kill -a || true
for f in recorder_roscam.pid rosbag_roscam.pid yolo_roscam.pid \
         launch_roscam.pid gzserver_roscam.pid roscore_roscam.pid xvfb_roscam.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
done
exec 9<&-
```

检查 `ss -ltnp | grep -E ':(11317|11351) '` 没有输出，即说明本次隔离端口已关闭。
