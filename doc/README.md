# Landing

## 项目概览

本项目围绕无人机自主降落任务展开，核心目标是在 **ROS + Gazebo + PX4/MAVROS** 仿真环境中，完成从专家数据采集、离线强化学习训练、在线微调，到批量评估与结果分析的完整实验闭环。

从当前工程结构看，项目主体代码集中在 [`TD3-main`](TD3-main)，外层目录主要保存数据分析脚本、专家数据、训练权重与评估结果。整体技术路线可概括为：

1. 使用 [`train_listen.py`](TD3-main/train_listen.py) 在事件驱动环境中采集专家轨迹；
2. 使用 [`TD3_offline.py`](TD3-main/TD3_offline.py) 基于专家数据进行离线 TD3-BC + LSTM + Attention 训练；
3. 使用 [`train_test.py`](TD3-main/train_test.py) 或 [`evaluate_iros_new.py`](TD3-main/evaluate_iros_new.py) 加载离线模型进行测试与评估；
4. 使用 [`TD3_online_finetune.py`](TD3-main/TD3_online_finetune.py) 在离线模型基础上继续在线微调；
5. 使用 [`analyze_expert_episode.py`](analyze_expert_episode.py) 等分析脚本对数据与结果做二次统计。

## 目录结构总览

```text
Landing_new/
├── analyze_expert_episode.py          # 专家数据离线分析脚本
├── expert_data_lstm.json              # 专家数据集
├── checkpoints/                       # 离线训练与在线微调权重
│   └── TD3/LSTM/
├── Landing_new/evaluation_data/       # 评估结果输出目录
│   ├── AWAC/
│   └── TD3_LSTM/
├── TD3-main/                          # 核心训练、环境与评估代码
│   ├── drone.py
│   ├── landing_env.py
│   ├── landing_env_listen.py
│   ├── TD3_offline.py
│   ├── TD3_online_finetune.py
│   ├── train_listen.py
│   ├── train_test.py
│   ├── evaluate_iros_new.py
│   ├── pre_set_works_for_ROS.py
│   └── README.md
└── doc/
    └── README.md                      # 本文档
```

## 项目工作流

### 1. 专家数据采集

[`train_listen.py`](TD3-main/train_listen.py) 使用 [`GazeboEnv`](TD3-main/landing_env_listen.py) 与 ROS/Gazebo 环境交互，将每一步的：

- `observation`
- `action`
- `reward`
- `next_observation`
- `done`
- `episode_final_reward`

写入 JSON/JSONL 风格的数据集中。

该流程具有几个鲜明特点：

- 环境使用事件驱动版本 [`landing_env_listen.py`](TD3-main/landing_env_listen.py)，强调视觉回调同步；
- 采样动作直接来自飞控速度信息，而不是学习策略输出；
- 只保存“长度足够且成功降落”的回合，用于提升离线训练数据质量。

这说明本项目是 **先模仿专家行为，再做策略优化** 的路线，而不是从零开始在线探索。

### 2. 离线训练

[`TD3_offline.py`](TD3-main/TD3_offline.py) 是整个项目最核心的训练脚本。根据文件头注释与实现结构，可以确定它具备如下能力：

- 采用 **TD3-BC** 风格的离线强化学习；
- Actor / Critic 均引入 **LSTM**，用于处理时序状态；
- 使用 **Attention** 机制做特征增强；
- 支持从 JSON / JSONL 数据中自动解析 episode；
- 自动计算并保存 `state_mean.npy` 与 `state_std.npy`，供后续测试和微调统一归一化使用；
- 将训练结果保存为 `actor_*.pth`、`critic1_*.pth`、`critic2_*.pth`。

从目录可见，当前已有一组现成训练结果位于 [`checkpoints/TD3/LSTM`](checkpoints/TD3/LSTM)，包括 10000 到 60000 step 的多个权重文件，以及 TensorBoard 日志。

### 3. 在线微调

[`TD3_online_finetune.py`](TD3-main/TD3_online_finetune.py) 在离线模型基础上继续与仿真环境交互，实现 offline-to-online 训练链路。其关键机制包括：

- 加载离线阶段保存的归一化参数和模型权重；
- 支持将离线专家数据重新预填充到 replay buffer；
- 通过 [`landing_env.py`](TD3-main/landing_env.py) 或 [`landing_env_listen.py`](TD3-main/landing_env_listen.py) 与 Gazebo 环境交互；
- 定期输出在线微调后的 checkpoint 与 TensorBoard 日志。

已有微调结果位于 [`checkpoints/TD3/LSTM/online_finetune`](checkpoints/TD3/LSTM/online_finetune)。

### 4. 快速测试与批量评估

项目提供两类推理验证入口：

#### 快速测试

[`train_test.py`](TD3-main/train_test.py) 用于快速加载模型并执行测试，主要作用是：

- 验证模型与环境接口是否正常；
- 检查归一化配置是否一致；
- 通过硬降落逻辑降低末端抖动，快速查看策略效果。

#### 批量评估

[`evaluate_iros_new.py`](TD3-main/evaluate_iros_new.py) 用于论文口径评估，具有以下特点：

- 支持批量测试多个 episode；
- 每回合实时写入 `jsonl/csv`，避免中断丢失数据；
- 支持自动或手动断点续跑；
- 统计稳定后末态误差、时间效率、动作平滑性等指标。

当前已有评估结果位于 [`Landing_new/evaluation_data/TD3_LSTM`](Landing_new/evaluation_data/TD3_LSTM)，包括：

- [`episode_metrics_td3lstm_60000.jsonl`](Landing_new/evaluation_data/TD3_LSTM/episode_metrics_td3lstm_60000.jsonl)
- [`detailed_trajectories_td3lstm_60000.csv`](Landing_new/evaluation_data/TD3_LSTM/detailed_trajectories_td3lstm_60000.csv)

这表明当前仓库已经保留了一轮可复现实验结果。

### 5. 数据质量与结果分析

[`analyze_expert_episode.py`](analyze_expert_episode.py) 是与训练主链条解耦的辅助分析工具，用于：

- 分析专家数据回合长度、奖励分布、初末位置误差；
- 统计动作均值、RMS、最大幅值；
- 输出 `summary.csv`、`episodes.csv`、`steps.csv` 等结构化结果；
- 对专家数据做数据质量检查。

这个脚本非常适合在离线训练前先检查数据是否存在异常，例如：

- 某些回合过短；
- 动作分布异常；
- 终止条件不一致；
- 初始位姿或最终误差不合理。

## 核心模块说明

### 环境层

#### [`landing_env.py`](TD3-main/landing_env.py)
标准环境接口，主要服务于测试、评估与在线微调流程。其职责包括：

- 启动 `roscore` 与 `roslaunch`；
- 管理 Gazebo 服务，例如物理暂停、恢复和世界重置；
- 订阅无人机速度、YOLO 中心点等 ROS 话题；
- 提供 `reset()`、`step()`、`reward_setup()` 等 RL 风格接口；
- 管理 YOLO 检测进程，并维护降落任务判定逻辑。

从实现可看出，该版本强调 **YOLO 常驻**，用于减少视觉进程频繁启停带来的额外开销。

#### [`landing_env_listen.py`](TD3-main/landing_env_listen.py)
事件驱动版本环境，更适合专家数据采集。它与标准环境的区别在于：

- 更依赖视觉回调触发状态更新；
- 更强调采样时刻与感知结果同步；
- 用于支持监听式、采集式的数据记录流程。

### 控制与通信层

#### [`drone.py`](TD3-main/drone.py)
这是底层飞控辅助模块，封装了 PX4/MAVROS 常见操作：

- 解锁、切换 OFFBOARD；
- 起飞、悬停、降落；
- waypoint 控制；
- 位姿与飞控状态维护。

虽然该文件不是主训练入口，但它是环境层正常工作的基础组件之一。

### 学习与推理层

#### [`TD3_offline.py`](TD3-main/TD3_offline.py)
项目的主训练脚本，承担：

- 数据读取与规范化；
- 序列 replay buffer 构建；
- LSTM Actor/Critic 定义；
- TD3-BC 训练逻辑；
- 模型保存与日志记录。

#### [`TD3_online_finetune.py`](TD3-main/TD3_online_finetune.py)
承接离线模型，继续做在线训练。

#### [`train_test.py`](TD3-main/train_test.py)
用于单次或小规模验证模型表现。

#### [`evaluate_iros_new.py`](TD3-main/evaluate_iros_new.py)
用于批量、可恢复、面向论文统计口径的正式评估。

## 当前仓库中的重要数据与产物

### 1. 专家数据

- [`expert_data_lstm.json`](expert_data_lstm.json)

这是离线训练的主要数据来源，体积较大，说明仓库中已经保留真实采样结果而不仅仅是代码骨架。

### 2. 训练权重

位于 [`checkpoints/TD3/LSTM`](checkpoints/TD3/LSTM)，包含：

- 多个训练 step 的 actor / critic 权重；
- 状态归一化统计量；
- TensorBoard 运行日志。

### 3. 在线微调权重

位于 [`checkpoints/TD3/LSTM/online_finetune`](checkpoints/TD3/LSTM/online_finetune)，说明项目已经执行过离线后继续在线优化的实验。

### 4. 评估结果

位于 [`Landing_new/evaluation_data/TD3_LSTM`](Landing_new/evaluation_data/TD3_LSTM)，说明项目已经完成至少一组评估并保存了：

- 回合级指标；
- 轨迹级明细；
- 可供后处理的数据文件。

## 建议的使用顺序

### 1. 检查专家数据质量

```bash
python analyze_expert_episode.py --input ./expert_data_lstm.json --out_dir ./analysis_out
```

### 2. 进行离线训练

```bash
python TD3-main/TD3_offline.py --data_path ./expert_data_lstm.json --ckpt_dir ./checkpoints/TD3/LSTM
```

### 3. 进行快速测试

```bash
python TD3-main/train_test.py --ckpt_dir ./checkpoints/TD3/LSTM --load_step 60000
```

### 4. 进行批量评估

```bash
python TD3-main/evaluate_iros_new.py
```

### 5. 进行在线微调

```bash
python TD3-main/TD3_online_finetune.py --ckpt_dir ./checkpoints/TD3/LSTM --load_step 60000
```

## 依赖环境

从源码可以推断，本项目运行依赖至少包括：

### Python 依赖

- `numpy`
- `pandas`
- `torch`
- `tensorboard`

### ROS / 仿真依赖

- `rospy`
- `gazebo`
- `mavros`
- `geometry_msgs`
- `nav_msgs`
- `gazebo_msgs`
- `apriltag_ros`
- `pyquaternion`
- `yolov11_ros`
- PX4 对应 launch 文件与仿真资源

## 当前项目状态判断

根据当前目录内容，可判断该工程不是初始模板，而是一个已经跑通过若干实验阶段的工作仓库，具有以下特征：

1. 已有专家数据集；
2. 已有离线训练权重；
3. 已有在线微调产物；
4. 已有评估结果文件；
5. 代码中已补充较完整的中文头注释，便于后续维护。

因此，当前仓库最适合的定位是：

> 一个面向无人机仿真降落任务的、以 TD3-BC + LSTM + Attention 为核心的离线强化学习实验工程，并已包含从采集到评估的完整实验资产。

## 后续整理建议

为了让项目更易于交接和复现，建议后续继续补充以下内容：

1. 在仓库根目录新增统一入口型 README；
2. 增加 `requirements.txt` 或 Conda 环境说明；
3. 明确 ROS 工作空间、PX4 固件路径和启动顺序；
4. 补充评估指标定义文档；
5. 将 [`train_listen.py`](TD3-main/train_listen.py) 中硬编码保存路径与当前仓库路径对齐。

## 文档说明

本文档依据当前仓库实际文件结构与核心源码内容整理，重点反映“项目有什么、各部分做什么、如何串起来使用”。它适合作为新成员上手、项目交接或论文复现实验的快速索引文档。
