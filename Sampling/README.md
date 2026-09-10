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
- `validate_expert_data.py`：训练前的轻量数据检查。
- `tests/test_global_expert.py`：专家动作方向和 SEARCH 测试。
- `tests/test_policy_observation.py`：失检 observation 保持测试。
- `tests/test_dataset_continuity.py`：相邻帧和 TD3 数据格式测试。

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

采集器自动平衡七类成功数据：

```text
static
line_constant
line_variable
sine_constant
sine_variable
circle_constant
circle_variable
```

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
  Sampling.tests.test_dataset_continuity -v
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
