# TD3 Landing (ROS/Gazebo)

本目录用于无人机仿真降落任务, 主要流程为:
1. 采集专家数据
2. 离线训练 TD3-BC(LSTM+Attention)
3. 仿真测试与论文口径评估
4. 评估结果二次分析

## 运行依赖

基础依赖:
- Python 3.8+
- numpy, pandas, torch, tensorboard

仿真依赖:
- ROS (rospy)
- Gazebo
- MAVROS 相关消息/服务
- 几何与导航消息包 (geometry_msgs, nav_msgs, gazebo_msgs 等)
- 视觉检测节点 (yolov11_ros)

## 核心脚本与关系

- `train_listen.py`
	- 用途: 采集专家演示数据并追加写入 JSONL。
	- 依赖: `landing_env_listen.py`。
	- 输出: 专家数据文件, 供 `TD3_offline.py` 训练。

- `TD3_offline.py`
	- 用途: 离线训练 TD3-BC(LSTM+Attention), 保存模型和归一化统计。
	- 输入: 专家数据 JSON/JSONL。
	- 输出: `actor_*.pth`, `critic1_*.pth`, `critic2_*.pth`, `state_mean.npy`, `state_std.npy`。

- `TD3_online_finetune.py`
	- 用途: 加载离线权重后继续仿真在线训练(offline -> online)。
	- 输入: 离线 checkpoint 与归一化统计, 可选离线数据预填充。
	- 输出: 在线微调 checkpoint 与 TensorBoard 日志。

- `train_test.py`
	- 用途: 加载离线模型进行仿真测试, 支持硬降落末段策略。
	- 依赖: `TD3_offline.py`, `landing_env.py`。

- `evaluate_iros_new.py`
	- 用途: 批量评估并按回合落盘, 支持断点续跑。
	- 输出: `episode_metrics_*.jsonl`, `detailed_trajectories_*.csv`。

- `analyze_eval_results.py`
	- 用途: 汇总评估结果, 生成成功率/时间/误差等统计表。
	- 输入: `evaluate_iros_new.py` 输出文件。

- `landing_env.py`
	- 用途: 标准仿真环境接口, 被测试/评估调用。

- `landing_env_listen.py`
	- 用途: 事件驱动版本环境接口, 主要用于数据采集流程。

- `drone.py`
	- 用途: PX4/MAVROS 基础控制封装。

- `pre_set_works_for_ROS.py`
	- 用途: ROS 预设流程与视觉/传感器调试辅助脚本。

- `../analyze_expert_episode.py`
	- 用途: 对专家数据做离线统计分析与数据质量检查。

## 推荐执行顺序

1. 采集专家数据
```bash
python train_listen.py --max_episodes 1000
```

2. 离线训练
```bash
python TD3_offline.py --data_path <expert.jsonl> --ckpt_dir <ckpt_dir>
```

3. 在线微调（可选）
```bash
python TD3_online_finetune.py --ckpt_dir <offline_ckpt_dir> --load_step 80000
```

4. 快速测试
```bash
python train_test.py --ckpt_dir <ckpt_dir> --load_step 80000
```

5. 批量评估
```bash
python evaluate_iros_new.py
```

6. 结果汇总
```bash
python analyze_eval_results.py
```

## 说明

- 各脚本开头已补充用途、用法、实现方式与依赖关系注释。
- 如需做“离线后在线微调”, 建议新增 `TD3_online_finetune.py`, 复用 `TD3_offline.py` 的模型与归一化统计。
