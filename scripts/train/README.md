# MoE 三阶段离线训练

三个入口从仓库根目录执行，常调参数集中在各脚本顶部。依赖 NumPy、PyTorch；
日志采用 JSONL，不需要 TensorBoard。这里的脚本均不启动 ROS、Gazebo、PX4 或 YOLO。

## 运行顺序

先在 `stage0_single_head.py` 的 `TRAINING_ARGS.data_path` 指定正式 Sampling 文件，
并在三个脚本顶部设置同一个新的 `EXPERIMENT_DIR`。默认使用 `experiment_01`。

```bash
python -m scripts.train.stage0_single_head
python -m scripts.train.stage1_pretrain
python -m scripts.train.stage2_joint
```

| 脚本 | 输入 | 更新内容 | 默认训练步数 |
| --- | --- | --- | ---: |
| `stage0_single_head.py` | Sampling JSON/JSONL | 单头 Transformer Actor、双 Critic | 100000 |
| `stage1_pretrain.py` | 指定 Stage 0 的 `final.pt` | Router 和五动作头，encoder 默认冻结 | 20000 |
| `stage2_joint.py` | 指定 Stage 1 的 `final.pt` | MoE Actor、encoder、Router、双 Critic | 100000 |

Stage 0 使用单头 TD3-BC。Stage 1 复制单头 encoder 和动作 MLP 到五个专家，
用加权交叉熵监督 Router，用阶段对应动作头计算 BC；双 Critic 原样保留。
将 `pretrain_freeze_encoder=False` 可允许 encoder 按 `pretrain_encoder_lr_scale`
指定的较低学习率更新。缺样类别权重为零，仅提示该专家缺少对应监督。
Stage 2 使用现有联合 TD3-BC 损失，重新建立优化器、解冻参数并同步目标网络。
Critic 始终使用当前 `Q(s_seq, a_seq)` 结构，单动作 Critic 留待消融实验。

## 数据与存档

Stage 0 默认按完整 episode 做 80%/20% 划分，仅用训练集拟合均值和标准差。
所有阶段包含真实终止样本：终止 Q 目标直接使用奖励，目标网络只处理非终止行。
回放池按实际训练窗口数量分配，不使用容量截断回合。短于 32 步的回合没有训练窗口。

```text
checkpoints/MoE_TD3/experiment_01/
├── shared/
│   ├── state_mean.npy
│   ├── state_std.npy
│   ├── data_split.json
│   └── source.json
├── stage0/
├── stage1/
└── stage2/
```

每个阶段目录包含：

- `config.json`：本阶段实际参数。
- `train.jsonl`：训练损失与更新状态；延迟 Actor 更新时，Actor 指标沿用最近一次值。
- `validation.jsonl`：每次离线评估的指标。
- `step_N.pt`：按 `save_every` 保存的周期存档。
- `best.pt`：验证指标最优存档。
- `final.pt`：完成指定总步数时的存档，默认作为下一阶段输入。

训练存档内包含模型结构参数、网络权重、目标网络、优化器、已完成步数、随机状态，
以及同一套归一化统计、episode 索引和数据文件 SHA256。后续阶段直接读取存档中的
划分与统计，不重新计算；数据文件发生变化时会拒绝复用索引，请为新数据建立新实验。
这是防止阶段间错用数据的检查，不是全面数据审核。`shared` 额外保存便于查阅的副本。

新训练不会覆盖已有阶段目录。需要换用 `best.pt` 时，明确修改下一脚本的
`INPUT_CHECKPOINT`，不会自动搜索“最新”或“最佳”文件。

这些 `.pt` 是完整训练存档，与旧的 `actor_N.pth` 单网络权重文件格式不同。
使用 `scripts.train.common.load_checkpoint` 读取；根据存档 `args` 构造模型，
Stage 0 设置 `single_head=True`，再加载 `actor`，并使用 `data.mean/std` 归一化输入。
MoE 推理每回合将上一阶段设为 0，随后把每次预测的阶段传入下一次调用。

## 同阶段续训

将对应脚本顶部 `RESUME_CHECKPOINT` 设为该阶段目录内的 `step_N.pt` 或 `final.pt`，
将 `training_steps` 改为更大的**目标总步数**，然后重新运行同一脚本。

续训恢复存档中的目标网络、优化器、随机状态和阶段超参数，只有目标总步数允许更新；
不把其他新填写的学习率或冻结配置覆盖到旧优化器。日志继续追加；若显式从更早存档
重跑，日志可能出现重复步数，分析时应取同一步的最后一条记录。
进入下一阶段则使用 `INPUT_CHECKPOINT`，不使用 `RESUME_CHECKPOINT`。

## 验证指标

Stage 0 记录动作 MSE、三轴 MSE/MAE 和 z 符号一致率。Stage 1/2 额外记录：

- `soft`：真实上一阶段辅助下，多个专家动作的加权平均误差。
- `hard`：真实上一阶段辅助下，最大权重专家的完整动作误差。
- `router_accuracy`、`confusion_matrix`：当前窗口阶段分类；矩阵行是真实类、列是预测类。
- `expert_bc_mse`：标签对应专家的动作误差，缺样阶段为 `null`。
- `rollout_soft`、`rollout_hard`、`rollout_router_accuracy`：逐回合使用模型自己的上一阶段，
  从首帧开始左填充历史并更新路由；只在具有完整历史的窗口上计分，与批量指标对齐。

阶段索引依次为 APPROACH、MATCH、DESCEND、TOUCHDOWN、SEARCH。
`best.pt` 的选择指标分别为 Stage 0 的动作 MSE、Stage 1 的 `hard.mse`、
Stage 2 的 `rollout_hard.mse`。顺序评估使用记录的 observation，没有真实动作反馈，
不代表仿真闭环成功率。温度和 soft/hard 差异应结合这些指标分析。

## 离线验证

```bash
python -m unittest discover -s model/tests -v
python -m unittest discover -s Sampling/tests -v
python -m compileall -q model scripts Sampling
git diff --check
```

新增集成测试在临时目录创建五阶段合成数据，连续运行三个脚本，并比较 Stage 1/2
中断恢复与连续训练的参数。小网络离线测试证明阶段衔接与计算流程，不证明正式数据
上的收敛、GPU 运行或降落效果。正式训练由使用者准备数据后执行。
