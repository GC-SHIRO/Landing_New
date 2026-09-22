# 单 Gazebo 世界的多飞行器并行采集方案

## 1. 目标与结论

将当前“每个 worker 独立启动 ROS master、gzserver、PX4 和 YOLO”的结构，改为：

- 只启动 **一个** ROS master 和 **一个** Gazebo 世界；
- 在该世界中预生成 `N` 个相互隔离的采集槽位（slot）；
- 每个 slot 有一架 `iris_k`、一艘 `wamv_k`、一块 `landing_pad_k` 和一套 PX4/MAVROS/YOLO 命名空间；
- 每个 Python worker 仅控制自己的 slot，不再启动或暂停整个世界；
- 所有 worker 共享仿真时钟，但观测、控制、碰撞、YOLO 输出和文件均按 slot 隔离。

这样仍可并行采集，但只存在一个 gzserver 和一套相机渲染器，避开当前多 gzserver 时后启动实例虽然有图像帧、却无法稳定识别 marker 的问题。

## 2. 为什么不能只把多架 iris 放进当前场景

若多架无人机共用当前 `wamv`、起降平台和 `/benchmarker/collision`，则以下状态会互相干扰：

- 一架无人机 reset 或起飞会改变其他 worker 所依赖的世界暂停状态；
- 多架机落在同一甲板上会产生碰撞串扰；
- 同一个 WAM-V 的运动、目标相对位置和 reset 无法同时满足多个 episode；
- YOLO、碰撞和 Gazebo model-state 话题无法区分所属无人机。

因此并行单位必须是完整的独立 slot，而不是仅增加无人机数量。

## 3. 目标拓扑

以两个 slot 为例：

```text
roscore + gzserver（仅一套，物理持续运行）
│
├─ slot 0：iris_0 + wamv_0 + landing_pad_0
│  ├─ /iris_0/mavros/*
│  ├─ /iris_0/stereo_camera/*
│  ├─ /slot_0/yolov11/{centers,BoundingBoxes}
│  └─ /slot_0/collision
│
└─ slot 1：iris_1 + wamv_1 + landing_pad_1
   ├─ /iris_1/mavros/*
   ├─ /iris_1/stereo_camera/*
   ├─ /slot_1/yolov11/{centers,BoundingBoxes}
   └─ /slot_1/collision

采集主进程
├─ world coordinator：启动/关闭共享世界，持续 unpause，汇总 worker 结果
├─ worker 0：只控制 slot 0，写 worker0.jsonl
└─ worker 1：只控制 slot 1，写 worker1.jsonl
```

## 4. 槽位布局与仿真 launch

### 4.1 空间隔离

设 `SLOT_SPACING = 80.0` 米；slot `k` 的世界原点为：

```text
slot_origin(k) = (k * SLOT_SPACING, 0, 0)
```

以下模型都以 slot 原点平移：

- `landing_pad_k`；
- `wamv_k` 与其起始位置；
- `iris_k`；
- 水面/静态地面可由整个世界共享，不需要重复生成。

80 米大于飞行边界、船体运动范围和视觉检测范围的总和，避免相邻 slot 的无人机、船体和视觉 marker 进入同一场景。

### 4.2 新增共享 launch

在仓库外 PX4 工作区新增 `launch/step1_multi_slot.launch`，它只启动一次 Gazebo，并通过 `num_slots` 循环或显式 include 生成 slot。每个 slot 的 include 必须传入：

- `ID:=k`；
- `slot_x:=k * SLOT_SPACING`；
- `vehicle:=iris`、`model_name:=iris_k`；
- `ship_name:=wamv_k`、`landing_pad_name:=landing_pad_k`；
- 独立 PX4/MAVLink 端口、`fcu_url`、`tgt_system`；
- `gui:=false`。

现有 `single_vehicle_spawn_xtd.launch` 已能按 `ID` 注入 MAVLink 端口。需要扩展其模型位置参数，使 `iris_k` 出现在 `slot_x` 附近；同时所有 `spawn_model -model` 名称都必须包含 slot 编号。

### 4.3 相机与 YOLO

保留每架 `iris_k` 的相机话题：

```text
/iris_k/stereo_camera/left/image_raw
/iris_k/stereo_camera/right/image_raw
```

每个 YOLO 节点运行在独立 ROS namespace，且输出参数显式指定为：

```text
/slot_k/yolov11/centers
/slot_k/yolov11/BoundingBoxes
```

YOLO launch 需增加 `namespace`、`pub_topic` 与 `centers_topic` 参数。不要继续让多个节点在各自 ROS master 中复用根话题名；共享 master 下这会直接冲突。

## 5. Python 采集器改造

### 5.1 进程职责

新增 `Sampling/collect_multi_vehicle.py` 作为入口，职责如下：

1. 主进程启动一次 `step1_multi_slot.launch`，等待所有 MAVROS、相机与 YOLO ready；
2. 主进程不在 worker 内调用 `roscore`、`roslaunch` 或 `gzserver`；
3. 主进程启动 `N` 个 Python worker；每个 worker 只接收 `slot_id` 和独立输出路径；
4. 全部 worker 退出后沿用现有合并规则写入正式 JSONL；
5. 主进程统一关闭共享 launch。

`collect_parallel.py` 保留为旧的多世界实验入口，不再作为正式并行采集入口。

### 5.2 GazeboEnv 的共享世界模式

给 `GazeboEnv` 增加最小化参数：

```python
shared_world: bool = False
slot_id: int = 0
yolo_centers_topic: str = "/yolov11/centers"
yolo_boxes_topic: str = "/yolov11/BoundingBoxes"
collision_topic: str = "/benchmarker/collision"
```

`shared_world=True` 时：

- 不启动/关闭 roscore、Gazebo、YOLO；
- 不调用全局 `/gazebo/reset_world`；
- 不调用全局 pause/unpause；物理始终由 coordinator 保持 unpause；
- 只订阅自身 MAVROS、YOLO、碰撞话题；
- `close()` 仅注销当前 worker 的订阅和发布，不得终止共享进程。

现有单实例行为必须保持默认值不变。

### 5.3 无全局暂停的 reset

当前 `env.reset()` 使用全局 pause/unpause，不能用于共享世界。新增 `reset_slot()`：

1. 清理本 worker 的 episode 状态和视觉缓存；
2. 将 `wamv_k` 复位到本 slot 初始位姿，并停止其推进器；
3. 通过 `/iris_k/mavros/setpoint_position/local` 将 `iris_k` 飞到本 slot 的随机起点；
4. 等待该机到位、YOLO 针对该 slot 收到新鲜检测；
5. 不影响其他 slot 的模型、控制指令或物理时间。

不使用 `/gazebo/reset_world`、`/gazebo/pause_physics`、`/gazebo/unpause_physics`。如需 Gazebo 级模型复位，使用按模型名的 `set_model_state`，且模型名必须是 `iris_k` 或 `wamv_k`。

### 5.4 坐标与专家真值

专家继续使用相对坐标，不改变 `state_dim=3`、`action_dim=3`、`seq_len=8` 与 JSONL transition 语义。

每个 slot 的世界坐标须在生成 `privileged_state` 前转换为 slot 局部坐标：

```text
local_drone = world_drone - slot_origin(k)
local_target = world_target - slot_origin(k)
```

因奖励、专家控制和 observation 都是无人机相对 WAM-V 的量，局部化后各 slot 的数据分布与单实例保持一致。

### 5.5 碰撞隔离

最可靠的实现是为每架 iris 的起落架碰撞插件发布独立话题 `/slot_k/collision`。如果插件只能发布全局 ContactsState，则 worker 必须同时匹配：

- 自身无人机标识 `iris_k`；
- 自身甲板标识 `wamv_k`；
- 目标甲板顶面碰撞名称。

只按 `wamv` 过滤是不够的，会将其他 slot 的成功落地错误记到当前 episode。

## 6. 参数建议

集中放在 `collect_multi_vehicle.py` 顶部：

```python
NUM_SLOTS = 2
SLOT_SPACING = 80.0
TARGET_SAVED_EPISODES = 454
MAX_ATTEMPTS = 1500
WORKER_START_STAGGER_SECONDS = 5.0
YOLO_READY_TIMEOUT_SECONDS = 45.0
INITIAL_DETECTION_WAIT_SECONDS = 10.0
```

初始只启用两个 slot。确认 GPU 显存、相机帧率、YOLO 延迟和成功率后，再提升 `NUM_SLOTS`。

## 7. 实施顺序

1. 在 PX4/catkin 工作区实现 `step1_multi_slot.launch`，先只生成 `iris_0`、`iris_1` 和两个相机；
2. 确认两架机的 MAVROS、MAVLink、相机话题与 PX4 instance ID 全部唯一；
3. 将 WAM-V、landing pad、碰撞插件改为按 slot 命名和位移；
4. 扩展 YOLO launch，确认两个 namespace 的输出均有稳定 marker 检测；
5. 为 `GazeboEnv` 实现 `shared_world=True` 与 `reset_slot()`；
6. 实现 `collect_multi_vehicle.py` coordinator 与 worker；
7. 完成离线单元测试后，由使用者执行双 slot `--test`；
8. 只有两个 worker 均能保存完整 episode，才允许启动正式采集。

## 8. 验收标准

双 slot `--test` 必须同时满足：

- 只有一个 `gzserver`、一个 ROS master；
- 两个 PX4、两套 MAVROS、两个 YOLO 节点都存活且命名空间不重叠；
- 两个相机话题都持续达到不少于 15 Hz；
- 两个 YOLO 输出都有新鲜 marker 检测；
- 两个 worker 各保存 1 条完整成功 episode；
- 每条 episode 保持 `next_observation == 下一条 observation`；
- 一个 worker reset、早停或退出不会改变另一个 worker 的 episode；
- 正式文件只在全部 worker 结束后由主进程合并。

## 9. 风险与边界

- 单世界的物理负载仍会随 slot 数量上升；首先观察 real-time factor，低于 0.7 时不要继续增加 slot。
- 多 slot 必须使用独立 WAM-V，不能共享移动甲板。
- 共享世界模式不得复用当前全局暂停/reset 逻辑。
- 仓库外 PX4 launch、SDF、碰撞插件和 YOLO launch 都是实现该方案的必要改动，需纳入版本控制或导出补丁。
- 本文是实施设计，不代表多飞行器版本已实现或已通过仿真验证。
