# TD3 动态平台降落

本仓库用于无人机在 ROS/Gazebo 动态平台场景中的专家数据采集、
TD3-BC 离线训练、在线微调和仿真评估。

## 目录结构

```text
model/                  模型、回放池和离线数据接口
scripts/                训练、评估和数据处理入口
Simulation/             动态平台仿真与评估逻辑
Simulation/env/         仿真环境及飞控通信封装
Sampling/               全局真值专家采集器与离线测试
data/expert_data/       传统专家数据
data/expert_global/     全局真值专家采集数据
data/evaluation/        评估输出
checkpoints/            权重、归一化统计和训练日志
doc/                    设计与重构文档
```

旧版 `landing_env.py` 已迁移为
`Simulation/env/landing_env_old.py`；当前 Sampling 和动态评估统一使用
`Simulation/env/env_base.py`。

## 主要入口

采集全局专家数据：

```bash
python -m Sampling.collect_global_expert
```

少量样本采集模式：

```bash
python -m Sampling.collect_global_expert --test
```

验证专家数据：

```bash
python -m Sampling.validate_expert_data
```

离线训练：

```bash
python -m scripts.train_offline \
  --data_path data/expert_global/global_expert.jsonl \
  --ckpt_dir checkpoints/TD3/global_expert
```

在线微调：

```bash
python -m scripts.train_online_finetune \
  --ckpt_dir checkpoints/TD3/global_expert \
  --load_step 80000
```

旧版环境评估：

```bash
python -m scripts.evaluate_iros
```

## 依赖

离线训练和检查依赖 Python 3.8+、NumPy、PyTorch、Pandas 和 TensorBoard。
仿真入口还依赖 ROS、Gazebo、MAVROS、PX4 相关消息以及 YOLO 检测节点。

代理默认只运行不依赖 ROS、Gazebo、PX4 和 YOLO 的离线测试；仿真由用户手动启动。
