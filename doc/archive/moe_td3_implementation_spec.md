# MoE-TD3 实施规格

## 1. 目标与边界

本规格记录将 `model/moe_td3.py` 从原有 TD3-BC/LSTM 副本改造成
**Causal Transformer + 阶段 Router + 五专家 Actor** 的离线 TD3-BC 模型。

第一版的目标是验证“阶段监督的多专家 Actor”是否优于同样十维输入的单头模型，
而不是一次性改动现有所有训练和仿真入口。

本轮必须保持的约束：

- 策略 observation 固定为 10 维，不能混入全局真值、phase、检测年龄或上一动作。
- action 固定为 3 维且范围为 `[-1, 1]`。
- 一个 JSONL 行仍是一个完整 episode，五个核心 transition 字段不变。
- `model/td3_offline.py` 和已有 `scripts/train_offline.py` 保持为 LSTM 基线，不修改其数据接口。
- MoE 使用独立的模型、训练入口、checkpoint 目录和后续仿真入口，不能覆盖 LSTM 基线权重。
- 不自动启动 ROS、Gazebo、PX4 或 YOLO；本规格中的验证均为离线验证。

本轮不做的内容：在线微调、HOLD/ABORT 第六专家、动作残差基线、真实飞行、
学习式 detection-age 估计和 Critic 的 phase embedding。它们只有在第一版 MoE
数据与离线结果成立后才单独讨论。

## 2. 改造前基线与实现目标

改造前的 `model/moe_td3.py` 只有文件头说明不同，网络和训练逻辑来自
`model/td3_offline.py` 的 LSTM + feature attention TD3-BC：

```text
10D state sequence (T=8) -> LSTM Actor/Critic -> TD3-BC
```

改造后的第一版固定为：

```text
10D state sequence (T=32)
        |
        v
共享 Causal Transformer Encoder
        |
        +--> Router -> 5 个阶段概率 w_t
        |
        +--> 5 个 Actor 专家头 -> a_all[t, phase]
        |
        v
训练: soft mixture action；部署: top-1 action

两个 Causal Transformer Critic: Q_i(state sequence, action)
```

`T=32` 对应当前采集周期 `0.1s` 的 3.2 秒历史。短于 32 步的**推理**历史通过
重复首个可用 state 左填充；离线 replay 从长度至少 32 的完整 episode 生成窗口，
恰好 32 步时仅生成终止样本，不伪造训练 transition。

## 3. 固定数据契约

### 3.1 策略可见输入

每个 time step 的网络输入严格为：

```text
s_t = [dx, dy, dz,
       dvx, dvy, dvz,
       dax, day, daz,
       yolo_confidence]
shape = (10,)
```

位置、速度、加速度均来自同一视觉相对坐标系；失检时采用当前采集器的“保持位置，
运动量和置信度清零”规则。Transformer 只能读取 `s_(t-31) ... s_t`，不能读取未来帧。

下列字段只可作为离线教师信号或诊断信息，绝不能拼接进 `s_t`：

- `privileged_state`、`next_privileged_state`；
- 专家全局位置、速度、目标真值；
- `expert.phase`；
- `env_info` 中的真实相对高度、丢失步数或接触结果。

### 3.2 阶段标签

采集器现有 metadata 已保存 `step["expert"]["phase"]`。第一版只接受以下五类：

| 采集器标签    | MoE 内部标签  | 索引 | 语义                         |
| ------------- | ------------- | ---: | ---------------------------- |
| `ALIGN`     | `APPROACH`  |    0 | 横向误差较大，优先靠近       |
| `TRACK`     | `MATCH`     |    1 | 继续跟踪并等待位置、速度同步 |
| `DESCEND`   | `DESCEND`   |    2 | 已稳定同步，正常下降         |
| `TOUCHDOWN` | `TOUCHDOWN` |    3 | 近地低速下降与触地           |
| `SEARCH`    | `SEARCH`    |    4 | 非近地失检后的上升搜寻       |

映射必须在 `model/moe_td3.py` 中定义为唯一常量，不允许不同脚本各自解释。
`FLARE` 不是当前采集器输出标签，第一版不得臆造或将其作为第六个专家。

缺失、空值或未知 phase 的正式训练数据必须报错；不能静默回退为 `ALIGN` 或以
特权状态重算标签。这样可以防止 Router 在错误标签上看似“正常训练”。

### 3.3 Replay window 的 phase 对齐

设一个 replay 样本最后一个原始 step 为 `t`，其窗口为：

```text
s_seq      = s[t-T+1 : t+1]
a_seq      = a[t-T+1 : t+1]
s2_seq     = next_observation[t-T+1 : t+1]
phase_prev = phase[t-1]
phase      = phase[t]
phase_next = phase[t+1]（非终止步）；-1（终止步，不存在下一阶段）
reward     = reward[t]
done       = done[t]
```

`phase` 是当前 Actor 专家头的 BC 标签和 Router CE 标签；`phase_prev` 用于当前
路由的阶段转移掩码。非终止步的 target action 使用当前 `phase` 作为上一阶段；
`phase_next` 仅用于真实相邻转移校验，不作为 target 的上一阶段。标签只来自同一
episode；终止步的 `phase_next=-1` 是缺省标记，不能送入 Router 或当作第六类标签。

2026-09-10 修正：MoE replay 保留最后的 terminal transition，直接读取真实
`next_observation`，不复制 terminal step 或制造下一阶段。普通连续窗口的 `s2_seq`
仍等价于状态序列右移一帧。终止样本的奖励、done、动作和当前 phase 均参与训练，
Q 目标直接等于即时奖励，目标 Actor 和目标 Critic 只处理非终止样本。
完整 N 步回合产生 `max(0, N-T+1)` 个窗口；没有下一阶段的非终止尾帧仍跳过。
LSTM 基线 `model/td3_offline.py` 保持原行为。

归一化仅处理 observation 和 next_observation，必须显式保留
`phase_prev`、`phase`、`phase_next`、action、reward、done。不能再使用会直接丢弃
metadata 的旧 `normalize_episodes` 结果作为 MoE buffer 输入。

### 3.4 各 Stage 共用的数据准备

`prepare_offline_data(episodes, args)` 按完整 episode 固定划分训练/验证集，
默认 `validation_fraction=0.2`、使用 `args.seed`，至少需两个 episode。
只用训练集 observation 计算一套 mean/std，训练集和验证集都用它归一化。
返回 `PreparedOfflineData`，保留划分索引和实际窗口末端的五阶段样本量；
阶段缺样仅体现为计数 0，不追加严格的数据审核或自动重采样。

调用 `prepared.save(directory)` 保存 `state_mean.npy`、`state_std.npy` 和
`data_split.json`（episode 索引、序列长度、阶段计数）。后续 Stage 复用这份统计与
划分，不重新拟合归一化。Stage 0/1/2 独立入口已在 `scripts/train/` 接入，
后两阶段从上游存档恢复这份数据记录，并核对源文件 SHA256 以免套用错误索引。

## 4. 模型结构

### 4.1 配置

在 `model/moe_td3.py` 的 `Args` 顶部集中新增或修改以下默认参数：

```python
state_dim = 10
action_dim = 3
max_action = 1.0
seq_len = 32
hidden_dim = 256
transformer_layers = 2
transformer_heads = 4
transformer_ffn_dim = 512
dropout_p = 0.1
n_phases = 5
router_hidden_dim = 64
router_temperature = 1.0
```

保留 TD3 的 `gamma`、`tau`、双 Critic、target policy smoothing、延迟 Actor 更新、
梯度裁剪和 TD3-BC 自适应 lambda。checkpoint 默认目录使用
`checkpoints/MoE_TD3/global_expert`，不得使用 `checkpoints/TD3/...`。

### 4.2 Causal Transformer Encoder

新增 `CausalTransformerEncoder`：

1. 用线性层将 `(B, T, 10)` 投影为 `(B, T, 256)`；
2. 加入固定 sinusoidal 位置编码，位置编码注册为 buffer，不参与学习；
3. 使用 `batch_first=True` 的 TransformerEncoder，传入上三角 causal mask；
4. `forward` 返回所有 token 的 `(B, T, 256)`，调用方取最后一个 token；
5. mask 的定义必须保证第 `i` 个 token 不能注意 `j > i` 的 token。

Actor 与两个 Critic **各自拥有** encoder；第一版不共享 Actor/Critic 参数，避免
target network 软更新、优化器所有权和梯度路径变得不清晰。

### 4.3 Router 与专家 Actor

新增：

- `PhaseRouter(h)`：`256 -> 64 -> 5`，输出 mask 前 logits；
- `ExpertHeads(h)`：五个独立的 `256 -> 64 -> 3` MLP，堆叠为
  `a_all.shape == (B, 5, 3)`，每个输出经 `tanh * max_action` 限幅；
- `MoEActor(state_seq, previous_phase, mode)`：返回 `action`、`router_logits`、
  `weights`、`all_actions` 与 `last_hidden`。

训练 `mode="soft"`：

```text
w_t = softmax(masked_logits / temperature)
a_t = sum_p w_t[p] * a_all[p]
```

部署 `mode="hard"`：选择 `argmax(w_t)` 对应的单个专家动作。hard mode 只用于
推理和单元测试；训练与 TD3 target 均使用 soft mode 以保证可微。

### 4.4 阶段转移安全掩码

第一版先实现可测试的有限状态转移约束。行是上一个阶段，列是当前允许阶段：

| previous -> current | APPROACH | MATCH | DESCEND | TOUCHDOWN | SEARCH |
| ------------------- | -------: | ----: | ------: | --------: | -----: |
| APPROACH            |       ✓ |    ✓ |       - |         - |     ✓ |
| MATCH               |       ✓ |    ✓ |      ✓ |        ✓ |     ✓ |
| DESCEND             |       ✓ |    ✓ |      ✓ |        ✓ |     ✓ |
| TOUCHDOWN           |       ✓ |    ✓ |      ✓ |        ✓ |     ✓ |
| SEARCH              |       ✓ |    ✓ |       - |         - |     ✓ |

2026-09-06 最小修正：转移表以当前 Sampling 的 `stable_steps_required=2`、
`descend_xy_threshold < align_xy_threshold` 为依据。跟踪稳定后可直接进入近地阶段；
下降与近地阶段遇到误差增大时可重新对准或跟踪，近地阶段也可随相对高度增大恢复下降。
保留 APPROACH/SEARCH 必须先经过 MATCH 才能下降或触地的限制；
若以后将稳定步数改为 1，需要重新对齐该限制。此次不改变采集器控制律、动作或标签。
checkpoint 新增 `phase_transition_version`，拒绝加载旧转移版本，防止权重中的
Router buffer 恢复旧表。

实现 `allowed_phase_mask(previous_phase)`，对不允许的 logits 加一个足够小的有限值
（例如 `torch.finfo(dtype).min`），再 softmax。函数必须验证每一行至少有一个允许项。

这是**阶段跳转**安全约束，而不是基于特权真值的部署规则。第一版不根据训练 metadata
中的真实高度或速度修改 mask。对低置信度强制 SEARCH、近地失检例外和 HOLD/ABORT
需要先用可部署的 10D 坐标语义制定并验证后，才能加入；不能现在凭全局真值实现。

窗口末端检查 `(phase_prev, phase)`，非终止样本再检查 `(phase, phase_next)`；
终止步没有下一阶段，不对 `phase_next=-1` 查询转移表。
当前按用户确认，以已有 Sampling 阶段语义修正转移表，不增加全面的数据审核流程；
保留 replay 窗口末端的现有转移检查，用专家输出的离线回归测试验证新增路径。
以后出现表外跳转时先核对采集参数与阶段语义，不在训练时直接关闭 mask。

### 4.5 Twin Critic

2026-09-06 实施决定：当前保留代码中的 `Q(s_seq, a_seq)`，将逐帧 state/action
拼接后送入 Transformer；下述仅拼接当前动作的方案留待后续消融实验，本次不修改 Critic。

`CriticTransformer(state_seq, action)` 使用独立 Causal Transformer 的最后 token，
将其与 `(B, 3)` action 拼接，再由 MLP 输出单个 Q 值。

第一版 Critic 不接收 phase、Router 权重或特权信息：

```text
Q1(s_seq, a), Q2(s_seq, a)
```

这样可直接保留 TD3 的双 Q 取最小值、target smoothing 与软更新逻辑。phase-aware
Critic 是单独消融，不应混进第一版。

## 5. 训练流程

### 5.1 Stage 0：单头 Transformer 基线

先使用同一 Causal Transformer 和一个单头 Actor 训练 TD3-BC，不启用 Router 和
专家头。目的不是复用 LSTM 权重（两者结构不兼容），而是取得 Transformer 的可比较
基线，以及可复制到五个专家头的 action head 初始化。

当前独立脚本将 checkpoint 放在 `checkpoints/MoE_TD3/<实验名>/stage0`，
Stage 1/2 分别保存到同一实验下的 `stage1`、`stage2`，共用统计保存到 `shared`。

### 5.2 Stage 1：阶段监督预训练

从 Stage 0 初始化共享 Actor encoder 和单头 action MLP：

- 将单头 action MLP 的参数复制到五个专家头；
- Router 使用 `phase` 做加权交叉熵；类别权重按正式训练集的 phase 频率计算；
- 每个专家只对其标签样本计算 BC：`MSE(a_all[phase], a_expert)`；
- Critic 暂不参与更新，Actor encoder 可以冻结或使用更小学习率，具体由 Args 显式控制。

该阶段必须记录每类样本量、Router accuracy、混淆矩阵和五个专家的 BC loss。
按当前约定，某阶段没有样本时记录计数 0 并提示，不增加训练前强制失败检查；
该专家的对应阶段 BC 能力不能据此认定已训练完成。

### 5.3 Stage 2：联合 MoE TD3-BC

解冻 Stage 1 需要训练的模块，保持 Router 监督。对一个 batch：

1. 仅非终止样本调用 `actor_target(s2_seq, previous_phase=phase, mode="soft")` 产生 target action；
2. 对 target action 加入现有 TD3 clipped Gaussian smoothing，并裁到 action 范围；
3. 非终止样本使用 `reward + gamma * min(Q1_target, Q2_target)`；终止样本直接使用 `reward`，不调用目标网络；
4. 两个 Critic 最小化各自对 `target_q` 的 MSE；
5. 每隔 `policy_delay` 步更新 Actor。

Actor 的第一版总损失：

```text
L_actor = L_td3
        + beta_bc * MSE(a_all[phase], a_expert)
        + beta_router * CE(masked_logits, phase)
        + beta_switch * mean(||w_t - w_(t-1)||^2)
```

其中 `L_td3 = -lambda * mean(Q1(s_seq, a_soft))`，`lambda` 延用现有 TD3-BC
自适应定义。`w_(t-1)` 从同一个 causal encoder 的倒数第二 token 计算，不能用未来 token。
所有系数置于 Args 顶部；第一版 `beta_switch` 默认为较小的非零值，负载均衡正则默认
关闭。类别不平衡优先由 Stage 1 的加权 CE 和采样统计处理，避免强制均匀路由破坏真实
阶段比例。

### 5.4 部署约束

推理对象维护自己的 `previous_phase`：每次 hard Router 选择后再更新它。episode reset
时初值固定为 `APPROACH`，同时以首帧左填充 32 步 state 窗口。训练标签、真值 phase
和 `env_info` 均不可传入部署模型。

MoE checkpoint 与 LSTM checkpoint 不兼容。加载时必须保存并检查至少：state_dim、
seq_len、hidden_dim、layers、heads、n_phases、标签映射版本和归一化统计维度。

## 6. 计划修改的文件

| 文件                                      | 修改                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `model/moe_td3.py`                      | 重写为 Transformer MoE 模型、phase-aware replay、数据读取与 checkpoint。    |
| `scripts/train/stage0_single_head.py`   | 单头 Transformer TD3-BC 训练入口。 |
| `scripts/train/stage1_pretrain.py`      | Router 与五专家监督预训练入口。 |
| `scripts/train/stage2_joint.py`         | 联合 MoE TD3-BC 训练入口。 |
| `scripts/train/common.py`              | 数据恢复、离线指标、训练日志、阶段衔接和存档/续训。 |
| `Sampling/validate_expert_data.py`      | 保留原 10D 基础检查，按当前约定不增加全面审核。 |
| `model/tests/test_moe_phase_transitions.py` | 专家转移、mask 和版本检查。 |
| `model/tests/test_moe_replay_data.py`   | 终止窗口、TD3 目标和数据划分测试。 |
| `model/tests/test_moe_training_stages.py` | 单头复制、阶段冻结、三脚本串联、顺序评估与续训测试。 |
| `Sampling/README.md`                    | MoE 数据前置条件与独立训练命令在实现完成后同步。                            |
| `Simulation/step_env_moe.py`            | **后续单独新增**；只有离线训练验证后才实现，旧 `step_env.py` 不改。 |

## 7. 离线验收标准

完成每个实现阶段后，只运行不依赖 ROS 的检查：

1. `python -m compileall -q model scripts Sampling`；
2. `python -m unittest discover -s Sampling/tests -v`；
3. `python -m unittest discover -s model/tests -v`；
4. `git diff --check`。

新增测试至少证明：

- 改变未来输入不会改变之前 token 的 Transformer 输出；
- 输入 `(B, 32, 10)` 时动作、router 权重、专家动作和 Q 值的 shape 正确且有限；
- soft/hard 动作在 `[-1, 1]`；Router 权重按行求和为 1；
- 五阶段转移掩码禁止表外阶段，并允许正式标签的相邻转移；
- phase 的 prev/current/next 与 state/action 窗口末尾严格对齐且不跨 episode；
- 缺失、未知 phase 给出中文错误；空阶段数据只记录计数 0；
- 终止窗口保留真实 reward/action/next_observation，目标网络仅处理非终止样本；
- episode 划分可复现且不重叠，修改验证数据不会改变训练集归一化统计；
- 一个合成小 batch 可完成 Critic 与 Actor 更新，所有 loss 和梯度有限；
- checkpoint 保存/加载后相同输入给出相同 eval mode 输出。

这些检查只证明数据接口和计算图正确，不证明正式数据质量、训练收敛、ROS 话题、
Gazebo 行为或动态平台降落安全。完成离线训练后，再由使用者手动启动仿真进行验证。

## 8. 实施顺序

1. 先实现 phase 映射、MoE 数据校验与 phase-aware buffer 的离线测试；
2. 实现 Causal Transformer、单头 Transformer 基线和对应测试；
3. 实现 Router、五专家头、转移 mask 与 Stage 1 监督；
4. 实现 Stage 2 的 TD3 target、联合损失、日志和独立训练入口；
5. 用正式 10D 数据依次训练 LSTM、单头 Transformer、MoE，比较相同协议下的离线指标；
6. 离线结果和 checkpoint 完整性通过后，另行计划仿真推理适配与人工仿真验证。

## 9. 当前实现进度（2026-09-10）

三个独立入口和共用工具已经实现。Stage 0 使用单头 Actor；Stage 1 复制 encoder 和
五动作头，默认冻结 encoder 并保留 Critic；Stage 2 解冻并重新配置优化器、同步目标网络。
同阶段续训恢复目标网络、优化器与随机状态，不等同于新阶段初始化。
JSONL 指标包含逐专家 BC、Router 混淆矩阵，以及 soft/hard 和自身阶段顺序评估。
实现细节和运行命令见 `scripts/train/README.md`。已用合成数据验证阶段衔接，
正式数据训练、收敛比较和仿真验证仍待执行。
