# Sampling：全局真值专家自动采集

当前推荐入口是：

```bash
python TD3-main/Sampling/collect_global_expert.py
```

专家使用无人机和动态甲板的全局真值生成动作，训练 observation 仍使用实际三维 YOLO 状态：

```text
[marker_x, marker_y, marker_z]
```

输出文件每行保存一个完整成功 episode，可直接交给 `TD3_offline.py`。

## 文件说明

- `global_expert.py`：全局真值 PD 专家、分阶段下降和 SEARCH 控制。
- `collect_global_expert.py`：无参数自动采集入口。
- `validate_expert_data.py`：训练前的轻量数据检查。
- `tests/test_global_expert.py`：专家动作方向和 SEARCH 测试。
- `tests/test_policy_observation.py`：失检 observation 保持测试。
- `tests/test_dataset_continuity.py`：相邻帧和 TD3 数据格式测试。

旧采集器和旧质量门测试已经删除；需要追溯时直接使用 Git 历史。

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

核心字段严格匹配 `TD3_offline.py`：

```text
observation
action
reward
next_observation
done
```

固定约束：

- `observation`、`next_observation` 和 `action` 都是三维数组。
- action 每一维都在 `[-1, 1]`。
- 每行是一个完整 episode。
- 只把成功且长度不少于 15 的完整 episode 写入训练文件。
- episode 最后一条 `done=True`，其他步骤均为 `False`。
- `step[i].next_observation` 与 `step[i+1].observation` 完全相同。
- 不在 Sampling 中归一化；`TD3_offline.py` 自己计算并保存 mean/std。
- 不删除任何中间 transition。

所有尝试回合写入 raw 文件；只有成功完整回合写入 training 文件。

## marker 丢失处理

环境使用 `detection_fresh` 判断 marker 是否可见。允许短暂漏帧的时间由脚本顶部
`YOLO_LOST_TIMEOUT` 控制，默认 `0.5s`。

marker 可见时：

- observation 使用当前新鲜 YOLO 状态。
- 专家使用全局真值执行对准、跟踪、下降和触地。

非近地失检时：

- observation 保持最近一次有效 YOLO 状态。
- 不删帧、不插值、不写入真值冒充视觉状态。
- 专家进入 `SEARCH`。
- action 的水平分量保持上一条实际 action 不变。
- z action 设置为 `+search_climb_speed`，持续上升寻找 marker。

重新看到 marker 后立即退出 `SEARCH`，恢复正常全局专家动作。

近地失检时不进入 `SEARCH`。专家继续使用全局真值低速触地，避免因为相机近距离遮挡而重新爬升。

由于 `TD3_offline.py` 固定 `state_dim=3`，数据中没有额外增加可见性维度。LSTM 通过连续保持的视觉状态和对应的正 z 专家动作学习搜寻行为。因此后续推理端也必须在失检时保持上一有效 observation。

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
python TD3-main/Sampling/collect_global_expert.py
```

采集过程中每局打印：运动类别、是否保存、成功状态、总步数、SEARCH 步数和当前进度。

### 少量样本冒烟测试

正式采集前可以使用唯一的测试开关：

```bash
python TD3-main/Sampling/collect_global_expert.py --test
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
python TD3-main/Sampling/validate_expert_data.py
```

验证器检查：

- 五个核心字段、三维形状和有限数值。
- action 范围。
- step_index、done 和相邻 observation 链。
- 失检帧是否保持上一有效 observation。
- 非近地失检是否进入 SEARCH。
- SEARCH 是否保持水平 action 且 z 为正。
- 近地失检是否错误进入 SEARCH。
- 按 `seq_len=8` 计算的窗口总数是否至少为 64。

离线测试：

```bash
PYTHONPATH=TD3-main python3 -m unittest \
  Sampling.tests.test_global_expert \
  Sampling.tests.test_policy_observation \
  Sampling.tests.test_dataset_continuity -v
```

## 训练

验证通过后只使用现有训练脚本：

```bash
python TD3-main/TD3_offline.py \
  --data_path expert_data_dynamic/global_expert.jsonl \
  --ckpt_dir checkpoints/TD3/global_expert \
  --training_steps 100000 \
  --state_dim 3 \
  --action_dim 3 \
  --max_action 1.0 \
  --seq_len 8 \
  --batch_size 64
```

每批数据使用独立 checkpoint 目录，确保权重、`state_mean.npy`、`state_std.npy` 和日志来自同一次训练。
