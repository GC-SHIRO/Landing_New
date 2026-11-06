# ROS2 迁移方案：动态船载降落新仓库

> 状态：方案，未实施（2026-11-06）。
> 适用范围：新建干净仓库，替换 ROS1 Noetic + Gazebo Classic + MAVROS + XTDrone 工具链。
> 本仓库（`Landing_New`）保留为旧工具链与历史数据的参照，不在原地改造。

## 1. 目标与原则

### 1.1 目标

- 仿真栈换成 **ROS2 Jazzy + Gazebo Harmonic + PX4（uXRCE-DDS）**，去掉 XTDrone 和 MAVROS。
- 场景从"静止 + 直线移动的平台"扩展为**带升沉、纵摇、横摇的船甲板**。
- 训练侧（`model/`、`scripts/train/`）和数据格式保持不变，旧训练代码可以直接复用。
- 新仓库从一开始就带部署说明，换一台机器能按文档复现环境。

### 1.2 推进原则：先写完整体代码，再做适配和修复

1. **先定接口契约**：话题、消息、坐标系、时间步长、数据字段，写清楚再写代码。
2. **纯 Python 核心先行**：专家、observation 构造、SEARCH 逻辑、数据写入与校验都不依赖 ROS，
   可在 Windows 上跑离线测试。
3. **Mock 后端打通全链路**：用纯 Python 运动学模拟替代 Gazebo/PX4/YOLO，让
   "采集 → 校验 → 训练 → 评估"在没有仿真环境时就能完整跑通。
4. **最后接真实仿真栈**：逐项替换 mock，遇到的问题记录到适配清单（第 9 节），集中修复。

这样做的好处是：仿真环境没搭好时，代码工作不会被卡住；接真实仿真时，出了问题可以确定在适配层，
而不是核心逻辑。

## 2. 目标技术栈

| 组件 | 选型 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 24.04 | Jazzy 与 Harmonic 的官方配套平台 |
| ROS | ROS2 Jazzy（LTS） | 官方配套 Gazebo 版本为 Harmonic |
| 仿真器 | Gazebo Harmonic（LTS） | 原生支持 PX4 SITL，无需 XTDrone |
| 飞控 | PX4 SITL，固定一个稳定 tag | 撰写时建议 v1.16 或更新的稳定版；部署时锁定版本号写进文档 |
| 飞控通信 | uXRCE-DDS（`MicroXRCEAgent` + `px4_msgs`） | PX4 官方 ROS2 接口，替代 MAVROS；`px4_msgs` 分支必须与 PX4 版本一致 |
| 仿真桥接 | `ros_gz_bridge` | 相机图像、时钟、接触、位姿 |
| 视觉检测 | Ultralytics YOLO 的 ROS2 节点（自写） | 输出自定义 `MarkerDetection` 消息 |
| 船体场景 | 自建简化船模型（默认）；VRX 作为可选高保真场景 | 见第 5 节 |
| 部署 | Docker（NVIDIA Container Toolkit） | 宿主机为 Windows 时走 WSL2 或直接装 Ubuntu |
| 训练 | Python ≥ 3.10、PyTorch ≥ 1.13 | Ubuntu 24.04 系统 Python 为 3.12，不再受 Noetic 的 3.8 限制 |

版本号以部署时实际验证通过的组合为准，验证后写进新仓库的 `doc/environment.md`。

## 3. 新仓库结构

```text
landing_ros2/
├── core/                      纯 Python，不依赖 ROS（可在 Windows 测试）
│   ├── expert.py              全局真值专家（从 Sampling/global_expert.py 原样迁移）
│   ├── observation.py         10 维 observation 构造、差分、限幅、失检保持
│   ├── frames.py              坐标系换算：FLU/ENU ↔ FRD/NED、机体系 ↔ 世界系
│   ├── ship_motion.py         甲板运动模型（直线 + 升沉/纵摇/横摇）
│   ├── episode.py             单回合缓冲、终止判定、SEARCH 记账
│   ├── dataset.py             JSONL 写入、raw/training 分流、类别平衡
│   └── validate.py            数据校验（从 validate_expert_data.py 迁移）
├── backends/
│   ├── base.py                仿真后端接口（唯一的抽象层）
│   ├── mock.py                纯 Python 运动学后端：一阶速度响应 + 甲板运动 + 伪检测
│   └── ros2_gz.py             真实后端：封装 ROS2 节点，供采集循环调用
├── model/                     从旧仓库原样复制（td3_offline.py、moe_td3.py）
├── train/                     从旧仓库 scripts/train/ 和 train_offline.py 复制
├── collect.py                 采集入口：顶部参数 + --test，后端可选 mock / ros2_gz
├── evaluate.py                闭环评估入口
├── ros2_ws/src/
│   ├── landing_interfaces/    自定义消息：MarkerDetection、DeckState
│   ├── landing_sim/           world SDF、船/甲板/marker 模型、launch、甲板运动插件
│   ├── landing_perception/    YOLO 检测节点、真值伪检测节点（调试用）
│   └── landing_control/       PX4 offboard 封装节点
├── docker/                    Dockerfile、compose 文件、入口脚本
├── tests/                     离线测试（core、backends/mock、frames）
├── data/  checkpoints/        均加入 .gitignore
└── doc/                       environment.md、部署说明、接口契约
```

约定沿用旧仓库 `AGENTS.md`：参数放在脚本顶部按用途分组，只保留 `--test` 等少量开关，
注释和提示使用中文。

**唯一的抽象层**是 `backends/base.py`。存在两个后端（mock 与真实），这层抽象是必要的；
除此之外不再引入其他抽象。

```python
class Backend:
    def reset(self, scenario) -> None: ...         # 按场景参数复位甲板和无人机
    def step(self, action_flu) -> StepInfo: ...    # 下发一步动作，推进 TIME_DELTA 仿真时间
    def truth(self) -> Truth: ...                  # 无人机与甲板的全局真值（只给专家和判定用）
    def detection(self) -> Detection: ...          # 视觉检测结果（给 observation 用）
    def close(self) -> None: ...
```

## 4. 接口契约

### 4.1 必须保持不变的语义（旧数据与旧模型兼容的前提）

| 项 | 旧实现 | 新实现要求 |
| --- | --- | --- |
| 控制周期 | `TIME_DELTA=0.1s`，pause/unpause 伪锁步 | 仍为 0.1s，但按**仿真时间**计时（见 4.4） |
| 动作 | 3 维，`[-1,1]`，MAVROS `setpoint_raw/local` 中 `coordinate_frame=8`（机体系速度）；**偏航固定约为 0**，机头朝 Gazebo 世界 +x，因此水平动作实际上等同于世界 ENU 的 x/y；z 向上为正（SEARCH 用正 z 上升） | 语义不变；新栈同样把偏航锁在机头朝世界 +x，换算见 4.5 节 |
| observation | 10 维：YOLO 相对位置 3 + 差分速度 3 + 差分加速度 3 + 置信度 1 | 不变；限幅 `MAX_VISUAL_SPEED=10`、`MAX_VISUAL_ACCELERATION=100` 不变 |
| YOLO 相对位置 | 来自 `/yolov11/centers`（`PointStamped`）：**x、y 为检测框中心的像素坐标，z 为估计距离（米）** | 新检测节点输出同一语义；像素坐标依赖相机分辨率和视场，**新相机的分辨率、视场角必须与旧相机一致**（见第 6 节） |
| 失检 | `YOLO_LOST_TIMEOUT=0.5s` 内保持；超时后位置保持、其余清零 | 不变，逻辑放在 `core/observation.py` |
| 数据格式 | 五个核心字段，每行一个完整成功 episode | 不变；每个 episode 额外记录 `sim_stack`、`scenario` 元数据 |

旧数据与新数据**默认不混用**。新栈的动力学和检测特性都会变化，是否混合训练要等数据分布对比（M3）后再决定。

### 4.2 PX4 话题（uXRCE-DDS，多机时带 `/px4_k` 命名空间前缀）

| 方向 | 话题 | 消息 | 用途 |
| --- | --- | --- | --- |
| 发布 | `fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | 以 ≥ 2 Hz 持续发布，声明速度控制 |
| 发布 | `fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | NED 世界系速度；位置字段填 NaN |
| 发布 | `fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | 解锁、切 OFFBOARD |
| 订阅 | `fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | 位姿与速度（NED/FRD） |
| 订阅 | `fmu/out/vehicle_status` | `px4_msgs/VehicleStatus` | 解锁状态与飞行模式 |

注意：PX4 的 offboard 需要在切模式前就开始持续发布 setpoint，且发布不能中断。
新版 PX4 的 `fmu/out` 话题可能带版本后缀（如 `vehicle_status_v1`），以所锁定版本为准。

### 4.3 自定义话题

| 话题 | 消息 | 发布者 | 说明 |
| --- | --- | --- | --- |
| `/landing/slot_k/detection` | `MarkerDetection` | 检测节点 | `header`、`visible`、`position[3]`、`confidence`；空检测也要发布 |
| `/landing/slot_k/deck_state` | `DeckState` | 甲板运动插件 | 甲板中心与 marker 的世界位姿、速度，作为真值 |
| `/landing/slot_k/contact` | `ros_gz_interfaces/Contacts` | `ros_gz_bridge` | 甲板接触传感器，替代 `/benchmarker/collision` |
| `/clock` | `rosgraph_msgs/Clock` | `ros_gz_bridge` | 所有节点 `use_sim_time:=true` |

`MarkerDetection` 在没有 marker 时也必须照常发布（`visible=false`、`confidence=0`），
用来区分"看不到 marker"和"没有收到视觉帧"，与旧实现一致。

### 4.4 时间与步进

- 旧实现：每一步 unpause → 墙钟 sleep 0.1s → pause。仿真实际推进的时长不确定。
- 新实现：PX4 与 Gazebo 锁步运行，采集循环**按 `/clock` 等待 0.1s 仿真时间**后读取状态，
  不暂停世界。
- 无界面运行时可把 real-time factor 调到大于 1 加速采集；observation 差分始终使用仿真时间。

### 4.5 坐标系

| 来源 | 坐标系 |
| --- | --- |
| ROS / Gazebo | 世界 ENU，机体 FLU |
| PX4 | 世界 NED，机体 FRD |
| 策略动作 | 偏航固定、机头朝世界 +x 时的机体系速度，等同于世界 ENU 的 `(vx, vy, vz)`，z 向上为正 |

由于偏航固定，动作换算为 PX4 `TrajectorySetpoint` 只需要轴交换：

```text
v_north = a_y
v_east  = a_x
v_down  = -a_z
yaw     = +π/2      # NED 中机头朝东，即 Gazebo 世界 +x
```

所有换算只放在 `core/frames.py`，并用离线测试覆盖：

- 动作 `(1, 0, 0)` 对应 NED 的 `(0, 1, 0)`，即向世界 +x 飞；
- 正 z 动作对应 NED 中负的 `v_down`（上升）；
- 世界系下甲板与无人机的相对位置在 ENU 和 NED 之间往返换算后保持不变。

**仍需核对**：PX4 的 Gazebo 桥把 Gazebo 世界的 +y 当作北，这一点在 M2 第一次飞行时
用"发 `(1,0,0)` 动作，看是否沿 Gazebo +x 飞"验证。偏航锁定由 offboard 节点每次发布
setpoint 时填写 `yaw` 保证，不发偏航角速度。

## 5. 船与甲板运动

### 5.1 默认方案：自建简化船 + 运动学甲板

- 模型：简化船体 + 甲板碰撞体 + marker 贴图 + 甲板接触传感器。
- 运动：甲板作为运动学刚体，由一个 Gazebo system 插件（C++，约 150 行）在每个物理步按仿真时间
  设置位姿。运动参数通过 SDF 或话题下发。
- 运动模型（`core/ship_motion.py` 与插件使用同一套公式），**水平运动与姿态运动叠加**：
  - 水平：静止、匀速直线、变速直线（沿用旧采集器的三类，航向限制在 ±120°）；
  - 姿态：**升沉 / 纵摇 / 横摇为必选项**，用若干正弦分量叠加，振幅和周期按海况等级参数化；
    海况 0 表示无摇荡，用于和旧数据对照。
  - 场景标签由"水平类别 × 海况等级"组成，例如 `line_constant/sea3`，类别平衡按这个组合计算。
- 插件同时发布 `DeckState`（位置、姿态、线速度、角速度），真值和实际运动严格一致。

甲板有姿态后，需要同步调整的地方：

- **marker 真值位置**：marker 相对甲板中心的偏移（旧参数 `MARKER_OFFSET_*`）要随甲板姿态旋转，
  不能再按固定偏移直接相加；纵摇、横摇还会让 marker 处的升沉速度大于甲板中心。
- **专家**：`core/expert.py` 跟踪的是 marker 点的位置和速度，因此要使用旋转后的 marker 真值；
  专家控制律本身先不改，M2 后看成功率再决定是否加入"等待甲板姿态平稳再触地"的逻辑。
- **触地判定**：除了位置误差和相对速度，新增触地瞬间的甲板倾角和无人机与甲板的相对法向速度，
  写入 episode 元数据，作为论文里的安全性指标。
- **mock 后端**：同样实现姿态运动和 marker 旋转，保证 M1 阶段就能覆盖这部分逻辑。

选择运动学甲板而不是物理浮力的原因：实验需要可复现、可按海况精确控制的甲板运动，
这正是对比实验和消融实验所需要的。

**风险点**：运动学物体每步被直接设置位姿时，与无人机的接触冲量可能不真实，
需要在 M2 中检查触地时的反弹和穿透情况。

### 5.2 可选方案：VRX（WAM-V + 波浪）

VRX 提供 WAM-V 船模、波浪和浮力，画面真实，适合做演示和少量泛化测试，
但甲板运动由物理仿真决定，不便精确控制。部署时需确认 VRX 支持 Jazzy/Harmonic 的分支。

已确认老师的要求是"有摇荡的甲板即可，船最好有姿态运动"，不要求真实海面画面。
5.1 节的运动学甲板已覆盖姿态运动，因此 VRX 不在主线里程碑内，只在论文需要演示画面时再考虑。

## 6. 视觉检测

- 相机：与旧场景一致，**固定下视、无云台**，随机体一起倾斜。使用 PX4 的 Gazebo 机型中
  带下视相机的 x500 变体，或自建机型；具体机型名以所锁定的 PX4 版本为准。
- **相机参数必须与旧相机一致**：observation 前两维是像素坐标，分辨率或视场角一变，
  同样的相对位置就会对应不同的数值，旧数据和旧模型都不再适用。M2 之前需要从旧机型 SDF
  中抄出分辨率、水平视场角和安装位置，写进新机型。
- 由于相机随机体倾斜，像素坐标会受滚转和俯仰影响；船甲板摇荡后，
  这种耦合比旧场景更明显。这是数据语义的一部分，保持不变，但要在论文中说明。
- `landing_perception/yolo_node.py`：订阅 `ros_gz_bridge` 转出的图像，运行 Ultralytics YOLO，
  输出检测框中心像素坐标、估计距离和置信度，发布 `MarkerDetection`。
  距离的估计方法与旧节点保持一致。
- `landing_perception/truth_detector_node.py` 与 `backends/mock.py` 的伪检测：
  用真值位姿和**同一套针孔相机模型**（含机体姿态）把 marker 投影成像素坐标，
  超出画面即判为不可见，再叠加噪声。用于 M2 早期在 YOLO 之前先打通控制链路，
  也可以作为对比实验的"理想感知"上限。
- YOLO 权重和标注数据集单独管理，不放进 Git，路径写在节点顶部参数中。

## 7. 复位策略

旧实现依赖 `/gazebo/reset_world`。新栈中重置世界会打断 PX4 的状态估计，因此改为**相对复位**：

1. 无人机不重置：切到位置控制，飞到新的随机起始点并悬停；
2. 甲板插件把船移到新的起始位置并重置运动相位；
3. 等待首个有效检测（沿用 `INITIAL_DETECTION_WAIT_SECONDS`），然后开始计步。

只有在 PX4 出现异常（如 failsafe、EKF 发散）时才重启该 slot 的 PX4 实例。

## 8. 并行采集

延续 `doc/archive/multi_vehicle_parallel_collection_plan.md` 的 slot 思路，在新栈下实现更简单：

- 一个 Gazebo 世界中放 N 个 slot，每个 slot 包含一艘船、一块甲板和一架无人机，彼此间隔足够远；
- 每架无人机对应一个 PX4 实例 `-i k`，使用 standalone 模式连接已启动的 Gazebo；
- 一个 `MicroXRCEAgent` 服务全部实例，话题自动带 `/px4_k` 命名空间前缀；
- 每个 slot 一个采集 worker 进程，输出单独的 worker 文件，最后合并。

参考启动方式（以所锁定 PX4 版本的文档为准）：

```bash
# 第 0 架：启动 Gazebo 和 PX4
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=<相机机型> ./build/px4_sitl_default/bin/px4 -i 0
# 第 k 架：连接已有的 Gazebo
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=<相机机型> \
  PX4_GZ_MODEL_POSE="<x>,<y>" ./build/px4_sitl_default/bin/px4 -i <k>
```

## 9. 里程碑

| 阶段 | 内容 | 依赖环境 | 验收标准 |
| --- | --- | --- | --- |
| **M0 核心迁移** | 新仓库骨架；`core/` 迁移专家、observation、数据写入与校验；`model/`、`train/` 复制 | 无，Windows 即可 | 旧仓库的离线测试迁移后全部通过；`frames.py` 测试覆盖 4.5 节三项 |
| **M1 全部代码写完** | `backends/mock.py`；全部 ROS2 包、消息、launch、甲板插件、检测节点、offboard 节点按契约写完；`collect.py`、`evaluate.py` | 无（mock） | mock 后端下 `collect.py --test` 产出通过校验的数据；三阶段训练在 mock 数据上能跑完；ROS2 包能 `colcon build` |
| **M2 接入真实仿真** | Docker 镜像；按 PX4 → 甲板运动 → 接触 → 真值伪检测 → YOLO 的顺序逐个接入 | Ubuntu + GPU | 单 slot 用真值伪检测完成 10 个成功回合；再换成 YOLO 完成 10 个成功回合 |
| **M3 数据验收与并行** | 多 slot 并行；新旧数据分布对比（动作分布、回合长度、失检率、SEARCH 比例） | 同上 | 并行稳定运行 1 小时以上；对比结论写进文档，决定新旧数据是否混用 |
| **M4 正式实验** | 多海况采集、训练、闭环评估、对比与消融 | 同上 | 出论文所需的闭环指标表 |

M1 结束时代码应当"完整但未适配"：所有模块都存在且能在 mock 下运行，
真实仿真相关的部分允许有已知问题，统一记录在下面的适配清单中。

### 9.1 适配清单（M2 中逐项确认，接入后补充）

- [ ] `px4_msgs` 分支与 PX4 版本一致，`fmu/out` 话题名（是否带版本后缀）
- [ ] offboard 切换时序：先发 setpoint 再切模式，心跳不中断
- [ ] 动作 `(1,0,0)` 沿 Gazebo 世界 +x 飞行，偏航保持朝 +x（4.5 节）
- [ ] 锁步下 0.1s 仿真时间步长的实际抖动
- [ ] 运动学甲板的接触冲量、触地判定阈值
- [ ] 下视相机分辨率、视场角、安装位置与旧机型 SDF 一致；同一相对位置下新旧像素坐标一致
- [ ] 新 YOLO 节点的距离估计与旧节点方法一致
- [ ] YOLO 在新渲染画面下的检测率（可能需要补充训练数据）
- [ ] 相对复位后 PX4 状态正常，不触发 failsafe
- [ ] 多实例下 uXRCE-DDS 命名空间和 Gazebo 模型名不冲突
- [ ] real-time factor 大于 1 时 PX4 锁步和 YOLO 推理是否跟得上

## 10. 部署方法（新仓库 `doc/deployment.md` 的提纲）

1. **宿主机**：Ubuntu 24.04 + NVIDIA 驱动 + Docker + NVIDIA Container Toolkit。
   Windows 宿主机使用 WSL2（Ubuntu 24.04），并确认 WSLg 或无头模式下 Gazebo 能渲染相机。
2. **镜像构建**：基于 ROS2 Jazzy 官方镜像，安装 Gazebo Harmonic、`ros_gz`、PX4 依赖，
   在镜像中编译锁定 tag 的 PX4 与 `MicroXRCEAgent`，再 `colcon build` 本仓库的 `ros2_ws`。
3. **训练环境**：与仿真镜像分开，`pip install -r requirements.txt` 即可，Windows 也可以。
4. **启动顺序**：`MicroXRCEAgent udp4 -p 8888` → Gazebo 世界（含 slot）→ 各 PX4 实例 →
   检测节点 → `python collect.py`。用一个 launch 文件或 compose 文件统一编排。
5. **冒烟测试**：`python collect.py --test`，写入独立测试文件，不污染正式数据。
6. **版本记录**：每次验证通过后，把 PX4 tag、`px4_msgs` 提交号、镜像 tag、YOLO 权重哈希
   写进 `doc/environment.md`。

## 11. 旧文件迁移对照

| 旧文件 | 处理方式 |
| --- | --- |
| `Sampling/global_expert.py` | 原样迁移为 `core/expert.py` |
| `Sampling/collect_global_expert.py` | 拆分：observation、SEARCH、回合缓冲、类别平衡、写入进 `core/`；ROS 循环由 `collect.py` + 后端替代 |
| `Sampling/collect_parallel.py` | 不迁移，由第 8 节的单世界多 slot 方案替代 |
| `Sampling/validate_expert_data.py` | 迁移为 `core/validate.py` |
| `Sampling/tests/*` | 迁移到 `tests/`，去掉对 ROS 包装层的依赖 |
| `Simulation/env/env_base.py` | 不迁移代码，按第 4 节契约在 `backends/ros2_gz.py` 和 `landing_control` 中重写 |
| `Simulation/ship_motion.py` | 运动公式迁移到 `core/ship_motion.py` 并扩展摇荡；ROS 部分由甲板插件替代 |
| `Simulation/dynamic_episode_runner.py`、`landing_evaluation.py` | 评估逻辑参考后在 `evaluate.py` 中重写 |
| `model/*`、`scripts/train/*`、`scripts/train_offline.py` | 原样复制 |
| `scripts/check_data_dist.py`、`analyze_expert_episode.py` | 按需复制，用于 M3 数据对比 |
| `landing_env_old.py`、`landing_env_listen.py`、`step_env.py`、`step0_env.py`、`evaluate_iros.py`、`train_listen.py`、`*.bak` | 不迁移 |
| `data/expert_data/expert_data_lstm.json` | 不迁移；旧数据留在本仓库作为对照 |
