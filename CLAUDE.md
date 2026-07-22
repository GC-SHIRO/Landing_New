# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

无人机自主降落的强化学习实验工程，核心算法为 **TD3-BC + LSTM + Attention**。运行在 ROS + Gazebo + PX4/MAVROS 仿真环境中，完整实验链路：专家数据采集 → 离线训练 → 在线微调 → 批量评估 → 结果分析。

所有核心代码在 `TD3-main/`，外层目录存放数据、权重与分析脚本。

## 常用命令

```bash
# 1. 检查专家数据质量（不需要 ROS）
python analyze_expert_episode.py --input ./expert_data_lstm.json --out_dir ./analysis_out

# 2. 离线训练（不需要 ROS，硬件推荐 GPU）
python TD3-main/TD3_offline.py --data_path ./expert_data_lstm.json --ckpt_dir ./checkpoints/TD3/LSTM

# 3. 监控训练进度
tensorboard --logdir ./checkpoints/TD3/LSTM/runs

# 4. 快速测试（需要 ROS + Gazebo 环境）
python TD3-main/train_test.py --ckpt_dir ./checkpoints/TD3/LSTM --load_step 60000

# 5. 批量评估（需要 ROS + Gazebo 环境，支持断点续跑）
python TD3-main/evaluate_iros_new.py

# 6. 在线微调（需要 ROS + Gazebo 环境）
python TD3-main/TD3_online_finetune.py --ckpt_dir ./checkpoints/TD3/LSTM --load_step 60000

# 7. 采集专家数据（需要 ROS + Gazebo 环境）
python TD3-main/train_listen.py --max_episodes 1000
```

## 架构与数据流

### 模块分层

```
drone.py              ← PX4/MAVROS 底层控制封装
    ↓
landing_env.py        ← 标准环境接口（测试/评估/在线微调用）
landing_env_listen.py ← 事件驱动版本（专家数据采集用）
    ↓
TD3_offline.py        ← 离线训练主脚本（不依赖 ROS）
TD3_online_finetune.py← 在线微调（加载离线权重，继续与仿真交互）
    ↓
train_test.py         ← 快速测试
evaluate_iros_new.py  ← 批量评估（输出 JSONL + CSV）
    ↓
analyze_expert_episode.py / analyze_eval_results.py  ← 离线统计分析
```

### 归一化一致性（关键约束）

`TD3_offline.py` 训练时自动计算并保存 `state_mean.npy` / `state_std.npy` 到 `ckpt_dir`。所有后续脚本（`train_test.py`、`evaluate_iros_new.py`、`TD3_online_finetune.py`）必须从**同一 `ckpt_dir`** 加载这两个文件，否则归一化不一致会导致策略完全失效。

### Checkpoint 结构

```
checkpoints/TD3/LSTM/
├── actor_<step>.pth       # Actor 网络权重
├── critic1_<step>.pth     
├── critic2_<step>.pth     
├── state_mean.npy         # 状态归一化均值
├── state_std.npy          # 状态归一化标准差
├── runs/                  # TensorBoard 日志
└── online_finetune/       # 在线微调的输出权重
```

### 状态与动作空间

- `state_dim = 3`，`action_dim = 3`，`max_action = 1.0`（见 `TD3_offline.py:Args`）
- 状态来自 YOLO 检测中心点 + 速度信息；动作为速度指令

### 离线算法要点（`TD3_offline.py`）

- TD3 双 Q + 延迟策略更新（`policy_delay=2`）
- TD3-BC 正则项：BC 权重从 1.0 线性退火至 0.2（`bc_anneal_steps=150000`）
- LSTM Actor/Critic，序列长度 `seq_len=8`
- 支持 JSON 和 JSONL 两种专家数据格式，自动按 `done=True` 切分 episode

## 依赖环境

**纯离线脚本**（无需 ROS）：`TD3_offline.py`、`analyze_expert_episode.py`
- Python 3.8+，numpy、torch、tensorboard、pandas

**需要 ROS + Gazebo** 的脚本：`landing_env.py`、`landing_env_listen.py`、`train_listen.py`、`train_test.py`、`evaluate_iros_new.py`、`TD3_online_finetune.py`
- ROS（rospy、geometry_msgs、nav_msgs、gazebo_msgs、mavros_msgs）
- Gazebo + PX4 仿真资源
- apriltag_ros、pyquaternion、yolov11_ros 节点

## 现有实验产物

- 专家数据：`expert_data_lstm.json`（约 6.3 MB）
- 离线权重：`checkpoints/TD3/LSTM/actor_*.pth`，step 10000–60000
- 在线微调权重：`checkpoints/TD3/LSTM/online_finetune/`
- 评估结果：`Landing_new/evaluation_data/TD3_LSTM/`（60000 step 的 JSONL + CSV）
