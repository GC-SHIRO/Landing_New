# 采集效率改造与训练闭环检查说明（2026-09-14）

本文记录本次工作的检查结论、代码改动、运行方法和遗留事项，供后续在 Ubuntu 仿真机上
执行时对照。

## 1. 训练闭环检查结论

模型构建到三阶段离线训练的代码闭环已经完成，正式数据尚未采集。

已验证：

- `python -m compileall -q model scripts Sampling` 通过。
- `model/tests` 18 个测试通过，其中 `test_all_stages_and_same_stage_resume` 真实调用
  stage0、stage1、stage2 三个脚本并做续训，产出 `final.pt`、`best.pt`、`train.jsonl`、
  `validation.jsonl`。
- `Sampling/tests` 全部通过，`git diff --check` 通过。
- 采集器写入的 `expert.phase`（ALIGN、TRACK、DESCEND、TOUCHDOWN、SEARCH）与
  `model/moe_td3.py` 的 `EXPERT_PHASE_TO_INDEX` 一致。

缺失或需注意：

| 项目 | 现状 | 影响 |
| --- | --- | --- |
| `data/expert_global/global_expert.jsonl` | 不存在 | 三阶段训练无法启动，需先采集 |
| `data/expert_data/expert_data_lstm.json` | 无 `phase` 字段 | 只能喂旧版 LSTM 入口，不能用于 MoE |
| 本机默认 Python 3.14 | 无 torch | 需使用 conda 环境 `pytorch`（Python 3.11，torch 2.13） |
| `tensorboard` | 两个 conda 环境均未安装 | 旧版 `scripts/train_offline.py` 启动即失败；MoE 脚本不依赖 |
| `Simulation/step_env_moe.py` | 未实现 | MoE 存档只能离线评估，训练到仿真部署尚未闭环 |

三阶段训练命令（三个脚本顶部 `EXPERIMENT_DIR` 必须一致，且目录为空）：

```bash
/opt/miniconda3/envs/pytorch/bin/python -m Sampling.validate_expert_data
/opt/miniconda3/envs/pytorch/bin/python -m scripts.train.stage0_single_head
/opt/miniconda3/envs/pytorch/bin/python -m scripts.train.stage1_pretrain
/opt/miniconda3/envs/pytorch/bin/python -m scripts.train.stage2_joint
```

数据硬条件：至少 2 个 episode 才能划分训练和验证集；短于 32 步的 episode 不产生训练窗口。

## 2. 已提交的改动

### 2.1 README 清理（`32c29d2`）

删除 `README.md` 中引用的 `scripts.train_online_finetune`，该脚本不在仓库内。
第 4 行项目简介仍保留"在线微调"字样，未改。

### 2.2 采集场景收缩（`a5f2fc0`）

`Sampling/collect_global_expert.py`：

- `MOTION_CLASSES` 缩减为 `static`、`line_constant`、`line_variable`，删除正弦、圆周和
  combined 分支，未知类别抛中文错误。
- 新增 `MAX_HEADING_DEVIATION_DEG = 120.0`，直线航向只在船头正前方（世界 `+x`，
  `teleport_to_origin` 后 yaw=0）左右各 120 度内均匀随机，避免甲板驶向后方障碍物。
  若实际船头朝向不是 `+x`，改这一处基准即可。

`Simulation/ship_motion.py` 的正弦、圆周模式保留，`step_env.py`、`step0_env.py` 和
`dynamic_episode_runner.py` 的评估流程仍在使用。

`Sampling/README.md` 与 `doc/sampling_refactor_plan.md` 已同步。

## 3. 未提交的改动：并行无界面采集

### 3.1 动机

单实例采集每局约 40 到 90 秒，时间主要花在两处：

| 阶段 | 位置 | 墙钟耗时 |
| --- | --- | --- |
| 预发 setpoint、解锁、切 OFFBOARD | `env_base.py` reset | 约 3 到 8 秒 |
| 位置模式飞回 7.5 到 9 米随机起点并悬停 | `env_base.py` reset | 约 5 到 22 秒 |
| 等首帧 marker | `wait_for_initial_observation` | 最多 5 秒 |
| 每步 `sleep(0.1)` 加两次 pause/unpause 服务调用 | `env_base.py` step | 每步约 0.12 到 0.15 秒 |

采纳的方案：多实例并行、只启动 gzserver、失败回合早停。未采纳缩短 reset 的方案：
瞬移无人机会让 PX4 估计器不稳定，压缩等待常量收益有限。

### 3.2 新增 `Sampling/collect_parallel.py`

复用单实例采集器的运动类别、专家参数、observation 构造、奖励和数据格式，只额外做三件事。

**多实例并行。** 主进程按 `WORKER_START_STAGGER_SECONDS` 错开启动 `NUM_WORKERS` 个子进程，
每个 worker：

- ROS master 端口 `BASE_ROS_PORT + k`，Gazebo master 端口 `BASE_GAZEBO_PORT + k`，
  通过 `ROS_MASTER_URI` 和 `GAZEBO_MASTER_URI` 隔离。
- 随机种子 `RANDOM_SEED + k * WORKER_SEED_STRIDE`。
- 目标数量和最大尝试次数由合计值平均分配，余数分给靠前的 worker。
- 独立输出 `data/expert_global/workers/worker{k}.jsonl`、`worker{k}_raw.jsonl`
  和 `worker{k}.log`。
- 每条 step 的 `scenario` 额外记录 `worker` 编号。

全部 worker 退出后自动合并到 `global_expert.jsonl` 和 `global_expert_raw.jsonl`，合并后删除
worker 数据文件，日志保留。主进程每 `STATUS_INTERVAL_SECONDS` 打印一次各 worker 进度。

**无界面。** 通过 roslaunch 参数 `gui:=false` 只启动 gzserver。

**失败回合早停。** `EarlyAbort` 在每步执行动作前判断：

- 连续 `SEARCH` 超过 `MAX_SEARCH_STREAK`（默认 40）步，返回 `EARLY_ABORT_SEARCH`。
- `STALL_WINDOW_STEPS`（默认 150）步内相对高度下降不足 `MIN_DESCENT_PROGRESS`
  （默认 0.3 米），返回 `EARLY_ABORT_STALL`。

触发时本步仍真实执行并作为终止转移写入 raw 文件，`done=True`、`success=False`。
停滞窗口取 150 步是因为对准阶段专家会悬停不下降，窗口太短会误杀正常回合。

**命令行。** 只有两个开关：

- `--test`：每个 worker 保存 1 个成功回合，最多尝试 3 局，写入测试文件。
- `--merge`：主进程异常退出后手动合并遗留的 worker 文件。目录里有非空遗留文件时，
  正常启动会拒绝运行，避免数据丢失或重复合并。

### 3.3 对 `Simulation/env/env_base.py` 的最小改动

多实例绕不开环境类，改了两处，不传参数时行为与原来一致：

- roscore 端口从 `ROS_MASTER_URI` 读取，原来硬编码 11311。
- `GazeboEnv.__init__` 新增 `launch_args` 和 `yolo_launch_args`，分别追加到 Gazebo 和
  YOLO 的 roslaunch 命令。

### 3.4 仿真侧前置条件

多套 PX4 SITL 不能共用端口，所以每个 worker 以不同 `ID` 启动，无人机命名空间为
`iris_{ID}`。以下 launch 文件不在仓库内，脚本无法自行检查，使用前需在 Ubuntu 机器上确认：

1. `LAUNCH_FILE`（`step1_linear.launch`）声明 `ID` 和 `gui` 两个 arg，并按 `ID` 选择
   端口互不冲突的 PX4 实例、机型 SDF 和 MAVROS `fcu_url`。XTDrone 多机 launch 是这种写法。
2. `yolo_v11.launch` 能按 `ID` 订阅对应无人机的相机话题。不行的话修改脚本顶部的
   `WORKER_YOLO_LAUNCH_ARGS`。
3. GPU 显存能容纳 `NUM_WORKERS` 份 YOLO。

### 3.5 运行方法

先在 `Sampling/collect_parallel.py` 顶部设置 `NUM_WORKERS` 和 `TARGET_SAVED_EPISODES`
（全部 worker 合计）。

冒烟测试：

```bash
python -m Sampling.collect_parallel --test
```

正式采集：

```bash
python -m Sampling.collect_parallel
```

异常退出后合并遗留文件：

```bash
python -m Sampling.collect_parallel --merge
```

采集完成后照常验证：

```bash
python -m Sampling.validate_expert_data
```

### 3.6 验证

- 新增 `Sampling/tests/test_collect_parallel.py` 8 个测试：端口和种子隔离、目标平均分配、
  测试模式独立文件、早停两条规则和重置、合并追加与删除来源、拒绝同名文件。
- Sampling 25 个离线测试通过，`compileall` 和 `git diff --check` 通过。
- 仿真侧未运行，本机没有 ROS、Gazebo、PX4 和 YOLO。

### 3.7 涉及文件

| 文件 | 改动 |
| --- | --- |
| `Sampling/collect_parallel.py` | 新增 |
| `Sampling/tests/test_collect_parallel.py` | 新增 |
| `Simulation/env/env_base.py` | 端口读 `ROS_MASTER_URI`，新增两个 launch 参数 |
| `Sampling/README.md` | 新增"并行无界面采集"一节 |
| `doc/sampling_refactor_plan.md` | 追加并行采集说明 |

## 4. 遗留事项

- 在 Ubuntu 机器上确认 3.4 节的三个前置条件，跑通 `--test`。
- 采集正式数据后按第 1 节命令训练，检查 `validation.jsonl` 中 Router 准确率和逐专家 BC。
- `Simulation/step_env_moe.py` 尚未实现，MoE 存档还不能接入仿真推理。
- `README.md` 第 4 行简介仍写有"在线微调"，可视情况删除。
- 若想让代理直接创建 PR，需安装并登录 `gh`。
