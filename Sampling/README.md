# Sampling：全局真值专家自动采集

当前推荐入口是：

```bash
python -m Sampling.collect_global_expert
```

专家使用无人机和动态甲板的全局真值生成动作，训练 observation 只使用实际 YOLO 状态及其差分：

```text
[position_x, position_y, position_z,
 velocity_x, velocity_y, velocity_z,
 acceleration_x, acceleration_y, acceleration_z,
 yolo_confidence]
```

输出文件每行保存一个完整成功 episode，可直接交给 `scripts/train_offline.py`。

## 文件说明

- `global_expert.py`：全局真值 PD 专家、分阶段下降和 SEARCH 控制。
- `collect_global_expert.py`：无参数自动采集入口。
- `collect_parallel.py`：多实例并行、无界面采集入口，带失败回合早停。
- `validate_expert_data.py`：训练前的轻量数据检查。
- `tests/test_global_expert.py`：专家动作方向和 SEARCH 测试。
- `tests/test_policy_observation.py`：失检 observation 保持测试。
- `tests/test_dataset_continuity.py`：相邻帧和 TD3 数据格式测试。
- `tests/test_collect_parallel.py`：并行 worker 配置、早停判断和输出合并测试。

旧采集器和旧质量门测试已经删除；需要追溯时直接使用 Git 历史。

### 旧三维数据转换

可用交互脚本将旧格式“每行一个 episode”的三维 JSONL 原路径转换为十维：

```bash
python -m scripts.convert_legacy_3d_dataset
```

输入文件路径并键入 `YES` 后，脚本会先创建同目录的 `.pre_10d_backup` 备份，再原子替换原文件。旧数据没有真实 YOLO 置信度，转换得到的置信度只能是 `marker_visible` 的 `1/0` 代理值，适合迁移和链路冒烟验证，不应用于正式十维模型训练。

## 修改参数

第一版不使用复杂 CLI。直接打开 `collect_global_expert.py`，修改文件顶部的分组常量：

```text
采集参数
仿真参数
专家参数
动态甲板参数
```

最常修改的是：

```python
TARGET_SAVED_EPISODES = 300
MAX_ATTEMPTS = 1500
MAX_STEPS = 600
RANDOM_SEED = 42
TRAINING_OUTPUT = ".../global_expert.jsonl"
RAW_OUTPUT = ".../global_expert_raw.jsonl"
CLEAR_OUTPUT_ON_START = False
```

代码注释、运行提示和错误信息均使用中文。

## 数据规则

核心字段严格匹配 `model/td3_offline.py` 的数据接口：

```text
observation
action
reward
next_observation
done
```

固定约束：

- `observation` 和 `next_observation` 都是十维数组；前三维是 YOLO 相对位置，随后三维是相对速度、三维是相对加速度，最后一维是 YOLO 置信度。
- `action` 是三维数组。
- action 每一维都在 `[-1, 1]`。
- 每行是一个完整 episode。
- 只把成功且长度不少于 15 的完整 episode 写入训练文件。
- episode 最后一条 `done=True`，其他步骤均为 `False`。
- `step[i].next_observation` 与 `step[i+1].observation` 完全相同。
- 不在 Sampling 中归一化；`scripts/train_offline.py` 计算并保存 mean/std。
- 不删除任何中间 transition。

所有尝试回合写入 raw 文件；只有成功完整回合写入 training 文件。

MoE 数据消费说明：`model/moe_td3.py` 使用 32 帧历史，保留完整回合最后的真实终止
transition，因此 N 步回合产生 `max(0, N-32+1)` 个训练窗口。成功奖励 `+300`、
终止动作和 `done=True` 会进入训练；不需要 Sampling 复制终止帧或修改奖励。
各 Stage 共用的 `prepare_offline_data` 按 episode 划分训练/验证集，仅从训练集计算
归一化统计。MoE 三阶段训练入口已放在 `scripts/train/`，依次执行：

```bash
python -m scripts.train.stage0_single_head
python -m scripts.train.stage1_pretrain
python -m scripts.train.stage2_joint
```

先修改各脚本顶部的数据、实验目录和训练参数，详见 [MoE 训练说明](../scripts/train/README.md)。
后两阶段复用上游存档的数据划分和归一化统计；现有 LSTM 入口和数据消费行为保持现状。

## marker 丢失处理

环境使用 `detection_fresh` 判断 marker 是否可见。允许短暂漏帧的时间由脚本顶部
`YOLO_LOST_TIMEOUT` 控制，默认 `0.5s`。

marker 可见时：

- observation 使用当前新鲜 YOLO 位置、按 `TIME_DELTA=0.1s` 差分出的速度和加速度，以及 `/yolov11/BoundingBoxes` 的检测置信度。
- 每局第一帧速度、加速度均为零；第二帧只计算速度；第三帧起才计算加速度。
- 专家使用全局真值执行对准、跟踪、下降和触地。

非近地失检时：

- observation 的位置保持最近一次有效 YOLO 位置；速度、加速度和置信度全部置零。
- 不删帧、不插值、不写入真值冒充视觉状态。
- 专家进入 `SEARCH`。
- action 的水平分量保持上一条实际 action 不变。
- z action 设置为 `+search_climb_speed`，持续上升寻找 marker。

重新看到 marker 后立即退出 `SEARCH`，恢复正常全局专家动作。

近地失检时不进入 `SEARCH`。专家继续使用全局真值低速触地，避免因为相机近距离遮挡而重新爬升。

模型默认使用 `state_dim=10`。后续推理端必须使用与采集器相同的差分初始化、限幅和失检清零规则。

## 动态场景

采集器自动平衡三类成功数据：

```text
static
line_constant
line_variable
```

直线运动的航向在船头正前方（世界 `+x`）左右各 `MAX_HEADING_DEVIATION_DEG`（默认 120 度）内均匀随机，避免甲板向后方障碍物行驶；每个 episode 使用独立 seed。
正弦和圆周场景暂不采集，等直线数据训练验证后再决定是否恢复。

类别平衡只按成功写入训练文件的 episode 数量计算。

## 采集

默认向输出文件追加。开始一批全新数据前，可以修改输出文件名，或将：

```python
CLEAR_OUTPUT_ON_START = True
```

运行：

```bash
python -m Sampling.collect_global_expert
```

采集过程中每局打印：运动类别、是否保存、成功状态、总步数、SEARCH 步数和当前进度。

### 少量样本冒烟测试

正式采集前可以使用唯一的测试开关：

```bash
python -m Sampling.collect_global_expert --test
```

测试模式固定为：

- 保存 2 个成功 episode。
- 最多尝试 5 局。
- 每局仍允许 600 步，保证专家有足够时间完成降落。
- 写入 `global_expert_test.jsonl` 和 `global_expert_test_raw.jsonl`。
- 每次启动测试模式都会清空这两个测试文件。
- 不会读取、覆盖或追加正式的 `global_expert.jsonl`。

## 并行无界面采集

`collect_parallel.py` 同时开 `NUM_WORKERS` 套独立的 ROS master、Gazebo、PX4 和 YOLO，
每套跑与 `collect_global_expert.py` 相同的采集循环，结束后把各 worker 的输出合并到
正式文件。它复用单实例采集器的运动类别、专家参数、observation 构造和数据格式，
只额外做三件事：

- 每个 worker 使用 `BASE_ROS_PORT + k`、`BASE_GAZEBO_PORT + k`，通过
  `ROS_MASTER_URI` 和 `GAZEBO_MASTER_URI` 隔离；随机种子为 `RANDOM_SEED + k * WORKER_SEED_STRIDE`。
- 以 `gui:=false` 只启动 gzserver，不开 gzclient 界面；每个 worker 自动启动独立的
  Xvfb 显示器，并设定 `LIBGL_ALWAYS_SOFTWARE=1`，使 Gazebo 相机在没有桌面或多个
  实例并行时使用相互隔离的 Mesa 离屏渲染上下文。
- worker 在开始计入回合尝试前，会最多等待 `YOLO_READY_TIMEOUT_SECONDS` 秒以取得首个
  YOLO `BoundingBoxes` 消息；空检测也代表相机与模型链路已就绪，避免较晚启动的模型加载期间
  耗尽 `--test` 的尝试次数。
- 并行无头模式使用 `PARALLEL_INITIAL_DETECTION_WAIT_SECONDS=20` 秒等待每局起始位置的
  marker，给 Mesa 双实例的相机和推理频率留出稳定时间；不改变单实例采集器的默认等待。
- 默认 `YOLO_CPU_WORKERS=(1,)`：worker 0 使用 GPU，worker 1 隐藏 CUDA、使用 CPU 推理，
  避免当前单 GPU 上双 YOLO 进程互相导致空检测。多 GPU 环境确认稳定后可改为空元组。
- 连续 `SEARCH` 超过 `MAX_SEARCH_STREAK` 步，或 `STALL_WINDOW_STEPS` 步内相对高度下降
  不足 `MIN_DESCENT_PROGRESS` 米，立即结束本局。这类回合只写入 raw 文件，
  `terminal_reason` 为 `EARLY_ABORT_SEARCH` 或 `EARLY_ABORT_STALL`。

### 仿真侧前置条件

多套 PX4 SITL 不能共用端口，所以每个 worker 以不同的 `ID` 启动 launch 文件，
无人机命名空间为 `iris_{ID}`。使用前请确认 Ubuntu 机器上的 launch 文件满足：

- `LAUNCH_FILE` 声明 `ID` 和 `gui` 两个 arg，并按 `ID` 选择端口互不冲突的
  PX4 实例、机型 SDF 和 MAVROS `fcu_url`（XTDrone 多机 launch 的写法）。
- `yolo_v11.launch` 能按 `ID` 订阅对应无人机的相机话题；如果它不接受该参数，
  修改脚本顶部的 `WORKER_YOLO_LAUNCH_ARGS`。
- `YOLO_PYTHON_EXECUTABLE` 指向同时安装 PyTorch 与 ROS Python 模块的解释器；当前默认
  使用本机 `lab_env`，避免 roslaunch 通过系统 Python 启动 YOLO。
- 无头采集依赖系统包 `xvfb`。脚本为 worker `k` 自动创建 `:(BASE_XVFB_DISPLAY + k)`，
  结束时关闭；不依赖桌面 `DISPLAY`，也不启动 gzclient。默认启用 Mesa 软件渲染以避免
  多个 Gazebo 相机争用同一个 GPU/X 上下文；若后续验证 GPU 多实例稳定，可将
  `FORCE_SOFTWARE_RENDERING` 改为 `False`。
- GPU 显存能容纳 `NUM_WORKERS` 份 YOLO。

这些 launch 文件不在本仓库内，脚本无法自行检查。

### 运行

先在 `collect_parallel.py` 顶部设置 `NUM_WORKERS` 和 `TARGET_SAVED_EPISODES`
（全部 worker 合计，平均分配）。当前默认是 2 个 worker、剩余 454 个成功回合；与本轮已写入的
46 个成功回合合计为 500 个。然后：

```bash
python -m Sampling.collect_parallel
```

主进程按 `WORKER_START_STAGGER_SECONDS` 错开启动 worker，每 `STATUS_INTERVAL_SECONDS`
打印一次各 worker 进度。worker 的完整输出在 `data/expert_global/workers/worker{k}.log`，
临时数据在同目录的 `worker{k}.jsonl` 和 `worker{k}_raw.jsonl`。全部 worker 退出后自动
合并到 `global_expert.jsonl` 和 `global_expert_raw.jsonl`，合并后删除 worker 数据文件。

冒烟测试每个 worker 只保存 1 个成功回合，写入测试文件：

```bash
python -m Sampling.collect_parallel --test
```

测试模式的 worker 使用同一已验证的随机起点序列，仅用于检查多进程无头渲染、ROS/PX4
隔离和数据链路；正式采集仍为每个 worker 使用不同随机种子。

如果主进程异常退出导致没有合并，目录里会遗留 worker 文件，再次启动会拒绝运行。
先手动合并：

```bash
python -m Sampling.collect_parallel --merge
```

## 验证

先在 `validate_expert_data.py` 顶部确认 `DATA_PATH`，然后运行：

```bash
python -m Sampling.validate_expert_data
```

验证器检查：

- 五个核心字段、十维形状、有限数值和置信度范围。
- action 范围。
- step_index、done 和相邻 observation 链。
- 失检帧是否保持上一有效位置，并将速度、加速度和置信度清零。
- 非近地失检是否进入 SEARCH。
- SEARCH 是否保持水平 action 且 z 为正。
- 近地失检是否错误进入 SEARCH。
- 按 `seq_len=8` 计算的窗口总数是否至少为 64。

离线测试：

```bash
python3 -m unittest \
  Sampling.tests.test_global_expert \
  Sampling.tests.test_policy_observation \
  Sampling.tests.test_dataset_continuity \
  Sampling.tests.test_collect_parallel -v
```

## 训练

验证通过后只使用现有训练脚本：

```bash
python -m scripts.train_offline \
  --data_path data/expert_global/global_expert.jsonl \
  --ckpt_dir checkpoints/TD3/global_expert \
  --training_steps 100000 \
  --state_dim 10 \
  --action_dim 3 \
  --max_action 1.0 \
  --seq_len 8 \
  --batch_size 64
```

每批数据使用独立 checkpoint 目录，确保权重、`state_mean.npy`、`state_std.npy` 和日志来自同一次训练。
