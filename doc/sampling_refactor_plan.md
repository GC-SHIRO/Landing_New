# Sampling 重构规划：全局真值专家自动采集

## 当前目录说明

仓库目录整理后，模型数据接口位于 `model/td3_offline.py`，离线训练入口为
`scripts/train_offline.py`，Sampling 输出统一位于 `data/expert_global/`。
本文其余部分保留 Sampling 初次重构时的设计背景和阶段性参数记录。

## 1. 目标

重构 `Sampling`，提供一套简单、稳定、可重复的自动专家数据采集流程。

最终只解决一件事：

> 使用能够读取仿真全局真值的专家稳定降落，并生成可被现有 `model/td3_offline.py` 直接读取和正常训练的连续 JSONL 专家数据。

本次规划遵循以下约束：

- 采集主循环尽量接近 `train_listen.py`：reset、逐步记录 transition、成功后整局保存。
- 专家可以读取无人机与甲板的全局真值，不要求专家只依赖 YOLO。
- 训练 observation 与实际运行时的三维视觉输入保持一致；目标丢失帧也连续记录。
- 不删除单个 transition，不拼接跨时间缺口。
- 保持 `state_dim=3`、`action_dim=3` 和当前 JSON/JSONL 字段兼容。
- `model/td3_offline.py` 是唯一的数据接口规范；Sampling 主动适配它。
- 不修改 `model/td3_offline.py`，也不要求它适配新的采集器。
- 旧 Sampling 代码在新流程实现并通过离线测试后删除，需要时通过 Git 历史追溯。

## 2. 不做的事情

本轮重构不包含：

- YOLO 质量门、逐帧 accepted/rejected 筛选。
- `min_valid_ratio`。
- 失检帧插值、删帧或 episode 分段。
- 在线强化学习或在线微调。
- 修改 TD3 网络结构、归一化方式和训练损失。
- 在采集阶段人为添加观测噪声。
- 为旧版所有参数和入口提供永久兼容层。
- 复杂 CLI、配置文件系统、自动恢复和过度的文件保护逻辑。

## 2.1 `model/td3_offline.py` 固定契约

规划和实现均以当前 `model/td3_offline.py` 的实际代码为准，而不是根据旧采集脚本推测格式。

当前默认训练配置为：

```text
state_dim       = 3
action_dim      = 3
max_action      = 1.0
seq_len         = 8
batch_size      = 64
capacity        = 200000
training_steps  = 100000
state_noise_std = 0.0
```

Sampling 必须遵守的输入契约：

- 每个 step 必须包含准确命名的五个字段：`observation`、`action`、`reward`、`next_observation`、`done`。
- `observation` 和 `next_observation` 必须是长度 3 的数值数组。
- `action` 必须是长度 3 的数值数组，范围与 `max_action=1.0` 一致。
- `reward` 必须可转换为有限 `float`。
- `done` 必须能够转换为布尔值，并且只在 episode 最后一条为 `true`。
- 可以保存额外诊断字段，但训练器会全部忽略，不能依赖这些字段参与训练。

当前训练器的数据行为也视为固定条件：

- JSON 和 JSONL 都能读取；新采集器统一输出“每行一个完整 episode”的 JSONL。
- 训练器只根据 `done=True` 切分 episode，不执行训练集/验证集随机切分。
- 均值和标准差只使用所有 step 的原始 `observation` 计算。
- 同一组均值和标准差同时用于 `observation` 与 `next_observation`。
- Sampling 必须写原始状态，不能预先归一化；归一化由训练器完成并保存到 checkpoint 目录。
- `seq_len=8` 时，长度为 `N` 的 episode 最多生成 `N-8` 个训练窗口。
- 回放池至少需要 64 个窗口才能按默认 `batch_size=64` 开始训练。
- 训练器用连续 `observation` 自己构造下一状态序列；数据中的 `next_observation` 会被读取和归一化，但不会直接进入回放池窗口。因此仍要求它与下一条 observation 严格一致，用于保证数据本身正确且兼容未来检查。
- 当前窗口循环截止到 `N-2`，最后一条 transition 的 `reward` 和 `done` 不进入回放池。Sampling 不伪造额外步骤，也不通过改写 `done` 绕过这一行为。

由此，新 Sampling 不负责数据集切分、归一化或序列补齐，只负责生成真实、连续且满足上述契约的完整 episode。

## 2.2 实现风格

Sampling 以方便直接打开代码修改为优先，不建立复杂的命令行参数系统。所有经常调整的重要参数集中写在采集脚本顶部，例如：

```python
# ==================== 采集参数 ====================
目标成功回合数 = 300
最大尝试回合数 = 1500
单回合最大步数 = 600
随机种子 = 42
输出文件 = "data/expert_global/global_expert.jsonl"

# ==================== 丢失目标处理 ====================
近地高度阈值 = 0.60
搜寻上升速度 = 0.25

# ==================== 专家控制参数 ====================
最大水平速度 = 1.0
最大下降速度 = 0.50
```

实际变量名可以使用清晰的英文，但所有代码注释、阶段说明、运行提示和异常信息使用中文。参数按“采集、目标丢失、专家控制、动态场景、输出路径”分组，避免散落在函数内部。

正式采集运行时不要求传入参数：

```bash
python -m Sampling.collect_global_expert
```

如以后确实需要临时覆盖，只保留极少数必要参数；第一版不实现通用 CLI 配置层。

唯一例外是少量样本冒烟测试开关：

```bash
python -m Sampling.collect_global_expert --test
```

测试模式只保存 2 个成功 episode、最多尝试 5 局，并使用独立且启动时清空的 test JSONL，不影响正式数据。

## 3. 核心设计决定

### 3.1 专家输入使用全局真值

专家每一步读取：

- 无人机世界坐标位置 `drone_position_world`。
- 无人机世界坐标速度 `drone_velocity_world`。
- 无人机 yaw。
- 甲板 marker 世界坐标位置 `target_position_world`。
- 甲板世界坐标速度 `target_velocity_world`。

这些信息只用于生成专家动作和采集诊断，不要求后续策略在推理时直接获得。

### 3.2 训练 observation 使用实际策略看到的三维视觉状态

全局真值只供专家决策，不能直接替代训练 observation。写入训练文件的状态必须与后续策略运行时收到的三维视觉状态语义一致：

```text
observation = [visual_relative_x, visual_relative_y, visual_height]
```

具体规则：

- marker 可见时，直接记录当前环境提供的新鲜三维视觉 observation。
- marker 丢失时，不删除该物理 step，也不改用全局真值冒充策略输入。
- 由于 `model/td3_offline.py` 固定 `state_dim=3`，不能额外增加 `visible` 第四维；丢失时沿用最近一次有效 observation，形成连续的“保持值”序列。
- 第 `i` 步保存的 `next_observation` 原样作为第 `i+1` 步的 `observation`，保证严格连续。
- 全局真值、`marker_visible` 和丢失持续步数只写入 metadata，用于专家控制和数据检查，不进入模型输入。

这种处理与当前三维模型兼容。LSTM 可以根据连续多个重复视觉状态及对应的上升动作学习丢失后的搜寻行为。它的限制是：没有显式可见性维度时，第一次失检和目标短暂停止可能无法从单帧区分，因此采集数据必须保留完整连续序列，推理端也必须采用相同的“失检保持上一有效状态”规则。

### 3.3 视觉状态与真值只做一次语义核对

正式采集前只做一个简短核对：在 marker 可见时同时打印视觉状态和真值相对位置，确认 x/y 轴、正负方向、高度和数值量级没有写反。核对完成后不建立复杂标定流程；正式训练字段始终使用实际视觉 observation，真值只用于专家动作和 metadata。

### 3.4 action 是实际下发的三维速度指令

训练字段固定为：

```text
action = [vx_body, vy_body, vz]
```

约束：

- 与 `Simulation.env.env_base` 的 `set_velocity_target` 完全相同。
- 每个维度限制在 `[-1, 1]`。
- z 轴沿用当前实测语义：负值下降，正值上升。
- 数据中的 `action` 必须是本步真正发送给环境的命令，而不是发送后的测量速度。

另外保存诊断字段：

```text
expert_world_velocity
measured_drone_velocity
target_velocity
```

这些诊断字段不进入 `model/td3_offline.py`。

### 3.5 每个成功 episode 原样完整保存

采集过程中维护一个 `current_episode_data`，每个物理 step 都追加一条 transition。

禁止：

- 删除中间帧。
- 把第 10 步和第 20 步直接拼接。
- 修改历史 transition 的 observation/action。
- 因附加质量标签而改变时间序列。

训练文件只保存成功完整 episode；失败和异常 episode 只保存到 raw 文件。

## 4. 建议的新目录结构

保持结构精简，只建立三个主要实现文件：

```text
Sampling/
├── global_expert.py
├── collect_global_expert.py
├── validate_expert_data.py
├── README.md
└── tests/
    ├── test_global_expert.py
    ├── test_policy_observation.py
    └── test_dataset_continuity.py
```

职责划分：

### `global_expert.py`

- `ExpertConfig`
- `GlobalLandingExpert`
- 世界速度到动作坐标的转换
- 简单的分阶段降落和目标丢失控制

### `collect_global_expert.py`

- 文件顶部集中参数
- 动态甲板场景随机化
- reset 与 episode 循环
- 调用专家并与环境交互
- raw/training JSONL 写盘
- 控制采集数量和场景平衡

### `validate_expert_data.py`

- 只做训练前必要检查：JSONL 能否读取、字段维度、相邻帧连续、动作范围和可用窗口数量。
- 不建设通用数据治理或复杂报告系统。

旧的 `collect_dynamic_expert.py`、`privileged_pd_expert.py` 和 `random_dynamic_expert.py` 已由新流程替代并删除，新入口不依赖旧质量门逻辑。

## 5. 全局专家控制逻辑

专家保持确定性和简单性，不再包含大量特殊分支。

### 5.1 水平控制

世界坐标水平速度：

```text
position_error_xy = target_position_xy - drone_position_xy
velocity_error_xy = target_velocity_xy - drone_velocity_xy

world_velocity_xy =
    target_velocity_xy
    + kp_xy * position_error_xy
    + kd_xy * velocity_error_xy
```

最后限制水平模长不超过 `max_xy_speed`。

目标速度前馈用于跟随动态甲板，PD 校正用于消除相对位置和相对速度误差。

### 5.2 垂直控制

只保留五个简单阶段：

```text
SEARCH      非近地失检时保持原水平速度并上升，直到重新看到 marker
ALIGN       水平误差较大，下降到或保持安全跟踪高度
TRACK       已接近甲板，先降低水平相对速度
DESCEND     水平位置和速度满足要求后下降
TOUCHDOWN   接近甲板后低速下降直到接触
```

建议初始参数：

```text
tracking_height       = 2.0 m
align_xy_threshold    = 1.0 m
descend_xy_threshold  = 0.30 m
descend_vxy_threshold = 0.25 m/s
flare_height          = 0.60 m
touchdown_height      = 0.15 m

max_descent_speed     = 0.50 m/s
flare_descent_speed   = 0.20 m/s
touchdown_speed       = 0.08 m/s
max_climb_speed       = 0.40 m/s
near_ground_height    = 0.60 m
search_climb_speed    = 0.25 m/s
```

专家只依据当前全局状态和少量稳定计数做判断。phase 只用于诊断，不作为训练 observation。

### 5.3 目标丢失时的搜寻动作

每一步额外读取 YOLO 的 `marker_visible`，控制规则固定为：

```text
if marker_visible:
    使用全局真值专家正常对准、跟踪和下降
elif relative_height > near_ground_height:
    vx = 上一步实际下发的 vx
    vy = 上一步实际下发的 vy
    vz = +search_climb_speed
else:
    近地失检，不上升，继续使用全局真值专家完成 TOUCHDOWN
```

这里“保持原速度上升”具体定义为：保持上一步实际下发的水平 `vx/vy`，将垂直速度设为固定正值 `+search_climb_speed`。z 轴正值必须先通过一次实机/仿真小测试确认确实表示上升。

SEARCH 状态一直持续到获得一帧新的有效 marker 检测。重新看到 marker 后，下一步立即恢复正常全局专家控制。为避免把近地遮挡误判为需要搜寻，只有真值相对高度大于 `near_ground_height` 时才允许进入 SEARCH；该高度只供专家判断，不加入三维训练 observation。

SEARCH 期间：

- 每个物理 step 都照常写入 transition。
- observation 使用最近一次有效视觉状态，不用真值替换。
- action 记录实际发送的“保持水平速度并上升”命令。
- metadata 记录 `marker_visible=false`、`phase="SEARCH"` 和 `lost_steps`。
- 不删除、插值或切断 episode。

这样数据中会形成“视觉状态连续保持 → 专家持续上升 → marker 重新出现 → 恢复降落”的完整时序，用于让长度为 8 的 LSTM 窗口学习找 marker 的动作。

### 5.4 动作平滑

为避免动态专家产生大量突变离散标签，对最终动作增加简单变化率限制：

```text
abs(action_t - action_t-1) <= max_action_delta_per_step
```

建议起始值：

```text
xy 每步最大变化 0.15
z  每步最大变化 0.10
```

这使自动专家数据更接近 `train_listen.py` 中实际测量速度的平滑分布，同时仍保留真正发送的速度命令语义。

SEARCH 是唯一例外：水平分量直接沿用上一步 action，垂直分量直接设为 `+search_climb_speed`，不再叠加额外复杂的平滑状态机。

## 6. 场景生成

继续覆盖当前七类场景，但统一通过一个简单入口配置：

```text
static
line_constant
line_variable
sine_constant
sine_variable
circle_constant
circle_variable
```

基本规则：

- 甲板水平速度不超过 `1.0 m/s`。
- 每类按“已成功保存数量”补齐，最终数量差不超过 1。
- 每个 episode 使用独立 seed。
- 无人机初始位置相对甲板随机，而不是相对固定世界原点随机。
- 初始相对高度和水平距离限制在专家可以稳定完成的范围内。

第一版不做复杂速度分桶。等基本数据训练成功后，再评估是否增加困难场景比例。

## 7. 简化后的采集主循环

主循环保持与 `train_listen.py` 相似：

```python
for episode_id in range(max_attempts):
    configure_scenario()
    reset_ship_and_drone()
    expert.reset()

    truth = read_global_truth()
    observation, marker_visible = read_policy_observation()
    if not marker_visible:
        # 初始没有有效视觉状态时不开始记录，重新 reset
        continue

    last_action = np.zeros(3, dtype=np.float32)
    lost_steps = 0
    current_episode = []

    for step_index in range(max_steps):
        update_ship_motion(step_index * dt)

        if not marker_visible and truth.relative_height > near_ground_height:
            action = np.array([
                last_action[0],
                last_action[1],
                search_climb_speed,
            ])
            phase = "SEARCH"
            lost_steps += 1
        else:
            action, phase = expert.compute_action(truth)
            lost_steps = 0

        visual_after, env_done, env_success, info = env.step(action)

        next_truth = read_global_truth()
        next_visible = info["marker_visible"]
        if next_visible:
            next_observation = visual_after
        else:
            # 失检帧保留上一有效视觉状态，不删帧、不写入真值
            next_observation = observation.copy()

        done = env_done or reached_max_steps
        success = info["landing_success"]
        reward = compute_reward(next_observation, done, success)

        current_episode.append(
            build_transition(
                observation,
                action,
                reward,
                next_observation,
                done,
                metadata,
            )
        )

        observation = next_observation
        truth = next_truth
        marker_visible = next_visible
        last_action = action
        if done:
            break

    append_raw_episode(current_episode)

    if success and structurally_valid(current_episode):
        append_training_episode(current_episode)
```

不存在 `accepted` 子序列，也不存在逐帧质量删除。

`marker_visible` 必须来自当前视觉检测结果，不能根据 observation 是否变化推断。近地失检仍然记录保持值 observation，但专家通过全局真值继续低速触地。

## 8. 数据格式

保持每行一个完整 episode：

```json
[
  {
    "observation": [0.1, -0.2, 7.5],
    "action": [0.3, -0.1, -0.5],
    "reward": -0.75,
    "next_observation": [0.08, -0.18, 7.45],
    "done": false,
    "success": false,
    "episode_id": 1,
    "step_index": 0,
    "scenario": {},
    "expert": {},
    "truth": {}
  }
]
```

`model/td3_offline.py` 只使用上述五个核心字段，因此额外 metadata 不影响现有训练。

训练文件要求：

- 每个 episode 最后一条 `done=True`，其余步骤全部为 `done=False`。
- 每个训练 episode 最终 `success=True`。
- action 和 observation 均为三维有限数值。
- 数据保持原始物理/策略量纲，不在 Sampling 内执行标准化。

虽然 `next_observation` 不直接参与当前回放池的滑动窗口构造，仍必须满足：

```text
step[i].next_observation == step[i + 1].observation
```

这样可以确保连续数据没有错位，并避免采集文件绑定当前训练器的一个实现细节。

## 9. 奖励

第一版使用连续的距离 shaping：

```text
step_reward = -0.1 * L3_norm(next_observation)
terminal_success_reward = +300
terminal_failure_reward = -200
```

奖励只根据写入数据的状态和最终环境结果计算，不使用固定世界坐标成功框。

需要特别注意：当前 `model/td3_offline.py` 的窗口构造不会把 episode 最后一条 transition 放进回放池，因此 terminal 的 `done` 和 `±300/200` 不会成为实际训练样本。不能把“学会下降和对准”寄托在最后的成功奖励上。终止前每一步的连续距离奖励、专家动作以及完整状态轨迹才是当前训练器真正使用的监督信号。

terminal reward 仍按环境真实结果记录，保证文件语义正确；Sampling 不复制终止奖励到前一步，也不增加虚假 transition。小规模训练验收若证明 critic 的奖励信号不足，再单独评估训练侧问题，不在采集数据中做隐式补偿。

本轮不调整 `model/td3_offline.py` 对终止 transition 的使用方式；这是训练侧独立事项，不在 Sampling 重构中绕过或伪造数据。

2026-09-10 MoE 训练侧更新：`model/moe_td3.py` 已单独保留真实终止 transition，
使用记录的 `next_observation` 构造下一状态窗口，终止样本的 Q 目标直接取即时奖励。
以上关于终止步被排除的说明仍适用于 LSTM 基线；Sampling 的奖励、动作和输出文件均不改变。

## 10. 写盘策略

### training 文件

只包含：

- 成功触地。
- 长度至少 15 步，与 `train_listen.py` 的成功数据筛选保持一致，并高于当前 `seq_len + 1 = 9` 的理论最低值。
- 所有核心字段结构正确。
- episode 时间连续。

### raw 文件

包含所有完整尝试，用于失败分析。

### 输出方式

输出路径直接写在脚本顶部。第一版只保留 training 和 raw 两个文件，每完成一局立即追加一行，不实现时间戳生成、`--overwrite`、自动恢复或复杂 summary 系统。

建议默认输出：

```text
data/expert_global/global_expert.jsonl
data/expert_global/global_expert_raw.jsonl
```

开始新一批数据前由使用者直接修改顶部文件名，避免不小心把不同配置的数据混在一起。脚本启动时打印最终输出路径即可。

## 11. 离线验证器

`validate_expert_data.py` 保持为一个简单的训练前检查脚本，数据路径同样写在文件顶部。检查失败时打印中文原因并退出，不引入额外配置框架。

### 硬性检查

- JSONL 每行可解析。
- episode 非空。
- `step_index` 从头到尾连续递增 1。
- 对所有相邻步骤：

```text
step[i].next_observation == step[i+1].observation
```

- observation/action 都是 `(3,)`。
- 所有数值有限。
- action 每维在 `[-1,1]`。
- `done=True` 只出现在 episode 最后一步。
- training 文件中的 episode 全部成功。
- 没有跨 episode 拼接。
- 原始 observation 的每一维标准差均明显大于 `1e-6`，避免训练器归一化后某一维退化。
- 按当前 `seq_len=8` 预计算总窗口数 `sum(max(0, N-8))`，必须不少于默认 batch size 64。

### 简单统计

- episode 数量、总 transition 数和可用 LSTM 窗口数。
- observation/action 的 mean、std、min、max。
- marker 丢失 transition 数量和 SEARCH transition 数量。
- SEARCH 数据中 z action 为正的比例。
- 近地失检数据中错误进入 SEARCH 的数量，必须为 0。

## 12. 测试规划

### `test_policy_observation.py`

- marker 可见时使用新鲜视觉 observation。
- marker 丢失时保持上一有效 observation。
- marker 重新出现时立即恢复新视觉 observation。
- observation/next_observation 连续性。

### `test_global_expert.py`

- 目标在前方时动作指向目标。
- 动态甲板速度前馈正确。
- 高空下降动作 z 为负。
- 接近甲板时下降速度减小。
- 非近地失检时保持上一 action 的水平分量并输出正 z。
- 近地失检时不进入 SEARCH，继续低速触地。
- 重新检测到 marker 后退出 SEARCH。
- 动作范围不超过 1。
- 动作变化率满足限制。

### `test_dataset_continuity.py`

- 不允许 step_index 跳变。
- 不允许 next_observation 链断裂。
- 不允许中间 done。
- 不允许无效形状和非有限值。
- SEARCH transition 不得被删除。

## 13. 实施阶段

### 阶段 A：视觉状态语义核对

- 同步采集少量 YOLO 与真值样本。
- 确认视觉 observation 的轴、符号、高度和数值量级。
- 确认失检时推理端与采集端都保持上一有效 observation。

核对完成后直接进入专家实现，不增加复杂标定模块。

### 阶段 B：全局专家

- 实现水平跟踪、正常垂直控制和 SEARCH 控制。
- 在静态甲板上测试 20 局。
- 目标成功率不低于 95%。

### 阶段 C：动态场景

- 接入七类甲板运动。
- 每类先测试 10 局。
- 检查失败原因和动作饱和，不立即扩大数据量。

### 阶段 D：采集器与验证器

- 实现与 `train_listen.py` 类似的完整 episode 采集循环。
- 完成 JSONL 写盘和离线硬性检查。

### 阶段 E：小规模训练验收

- 先保存 10～20 个成功 episode。
- 运行验证器。
- 使用独立 checkpoint 目录进行短训练。
- 检查 actor 在专家状态上的 action MSE、z 符号一致率和初始动作。

### 阶段 F：正式采集

- 通过小规模训练后再采集 300 个成功 episode。
- 七类场景成功数量保持均衡。
- 保留 training 和 raw 两个数据文件。

## 14. 验收标准

Sampling 重构完成需要同时满足：

### 数据正确性

- 0 个相邻帧缺口。
- 0 个 `next_observation` 链断裂。
- 0 个非有限 observation/action。
- 0 个超范围 action。
- 0 个失败 episode 进入 training 文件。
- 所有训练 episode 长度至少为 15。
- 按 `seq_len=8` 计算的可用窗口总数不少于 64。

### 专家能力

- 静态场景成功率不低于 95%。
- 直线动态场景成功率不低于 90%。
- 曲线场景先以不低于 80% 为第一阶段目标。
- z 动作方向与无人机真实垂直响应一致率不低于 99%。
- 非近地失检后 1 个控制 step 内进入 SEARCH。
- SEARCH 时水平 action 与上一实际 action 相同，z action 为正。
- 重新看到 marker 后 1 个控制 step 内退出 SEARCH。
- 近地失检进入 SEARCH 的次数为 0。

### 可训练性

- 归一化 mean/std 有限且各维 std 不接近 0。
- actor 在专家训练窗口上的 z 符号一致率不低于 95%。
- 初始高空状态下，专家要求下降时 actor 不输出饱和上升动作。
- 包含连续失检历史的窗口中，actor 应能拟合专家的正 z 搜寻动作。
- checkpoint、归一化统计和 TensorBoard 日志来自同一次训练。

## 15. 运行方式

Sampling 的参数直接修改脚本顶部，然后无参数运行：

```bash
python -m Sampling.collect_global_expert
```

验证器的数据路径同样写在脚本顶部：

```bash
python -m Sampling.validate_expert_data
```

验证通过后，唯一训练入口为：

```bash
python -m scripts.train_offline \
  --data_path data/expert_global/global_expert.jsonl \
  --ckpt_dir checkpoints/TD3/global_expert \
  --training_steps 100000 \
  --state_dim 3 \
  --action_dim 3 \
  --max_action 1.0 \
  --seq_len 8 \
  --batch_size 64
```

每次使用新的 checkpoint 目录，使模型权重、`state_mean.npy`、`state_std.npy` 和 TensorBoard 日志严格对应同一份数据及同一次训练。

## 16. 最终交付物

重构实施完成后应交付：

- `global_expert.py`
- `collect_global_expert.py`
- `validate_expert_data.py`
- 三组离线测试
- 更新后的 Sampling README
- 一份 10～20 局的小规模验证数据
- 验证器的控制台检查结果
- 一次短训练的动作拟合报告

新实现已完成离线检查；正式 300 局采集由使用者完成仿真验证后启动。

## 17. MoE 阶段转移对齐（2026-09-06）

MoE 转移表按当前 Sampling 的连续稳定两步配置修正，允许跟踪直接进入近地阶段，
以及下降、近地阶段重新对准/跟踪和近地阶段随相对高度增大恢复下降。
采集器控制律、phase 标签与数据输出保持现状，不追加全面的数据审核流程。
具体转移表和配置前提见 `doc/moe_td3_implementation_spec.md`；新增离线回归测试
直接调用全局专家验证转移、replay 装载和 Router mask，不启动仿真。
