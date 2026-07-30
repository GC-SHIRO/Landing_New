# 动态甲板特权 PD 专家

`Sampling/privileged_pd_expert.py` 使用 Gazebo/MAVROS 真值状态计算专家速度动作，
但训练数据中的 `observation` 与旧数据保持一致，保存 YOLO 三维输出：

```text
[marker_x, marker_y, marker_z]
```

因此专家负责上帝视角控制，学生策略仍学习从 YOLO 观测到速度动作的映射。

## 首轮 Step1 验证

先运行 5 轮固定场景：

```bash
python TD3-main/Sampling/collect_dynamic_expert.py \
  --step 1 \
  --episodes 5 \
  --vx 0.5 \
  --max_steps 600
```

成功数据默认写入：

```text
expert_data_dynamic/step1_privileged_pd.jsonl
```

所有成功和失败回合默认写入：

```text
expert_data_dynamic/step1_privileged_pd_all.jsonl
```

每行是一个完整 episode，核心字段与 `TD3_offline.py` 兼容：

```text
observation, action, reward, next_observation, done
```

额外保存 `privileged_state`、`expert`、`scenario` 和 `env_info`，这些字段用于调参与
失败分析，不会被当前离线训练数据读取器使用。

## 精度与速度默认值

- 下降前要求水平误差不超过 `0.25m`。
- 成功数据要求甲板接触时水平误差不超过 `0.25m`。
- 低空阶段使用更强的水平 PD 增益持续锁定 marker 中心。
- `marker.dae` 的图案几何中心相对 WAM-V `base_link` 为船体系
  `(-0.20, 0.0, 1.3) m`；采集、落地判定与评估均使用该中心，且 XY 偏移会随船 yaw 旋转。
- Step1 默认船速提高到 `0.5m/s`。
- 正常下降速度提高到 `0.70m/s`，减速段为 `0.22m/s`。
- 起始跟踪高度从 `3.0m` 降低到 `2.5m`，减少无效悬停时间。

离线点质量测试覆盖 48 组初始位置，触地水平误差最大值低于 `0.01m`。Gazebo
中的实际误差仍取决于 PX4 响应、碰撞模型和坐标转换。

## 启动与YOLO

采集器默认：

- 启动并订阅 YOLO，使训练观测与旧专家数据一致。
- ROS core 等待 `2s`。
- Gazebo 等待 `5s`。
- MAVROS/PX4 额外预热 `3s`。

如果机器启动较慢，可增加：

```bash
--roscore_wait 4 --gazebo_wait 10 --startup_warmup 6
```

采集模式会跳过 PX4 `COM_RCL_EXCEPT` 参数写入，避免 MAVROS 已连接后长期阻塞。
启动时依次打印：

```text
[启动 1/3] 启动 Gazebo、PX4 与 MAVROS
[启动 2/3] GazeboEnv 初始化完成
[启动 3/3] WAM-V 状态就绪，开始采集
```

如果上一次异常退出后仍有 ROS/Gazebo/PX4 进程，应先关闭残留进程再重新运行，避免
ROS 端口和仿真模型冲突。

## 坐标系检查

当前采集器固定按 BODY_NED 接口的 z 轴向上为正生成动作，不再提供
`--body_z_up` 命令行参数。

## Step1 批量采集

固定速度专家数据：

```bash
python TD3-main/Sampling/collect_dynamic_expert.py \
  --step 1 \
  --episodes 200 \
  --vx 0.5
```

在 0.3–0.7 m/s 范围随机船速：

```bash
python TD3-main/Sampling/collect_dynamic_expert.py \
  --step 1 \
  --episodes 500 \
  --vx 1.0 \
  --randomize_step1_speed \
  --v_min 0.3 \
  --v_max 0.7
```

专家 `max_xy_speed` 必须高于船速，否则无人机在船后方时没有追赶余量。

## 扩展到 Step2–4

专家使用实时目标位置与速度，因此同一个控制器可以用于变速和曲线轨迹：

```bash
python TD3-main/Sampling/collect_dynamic_expert.py --step 2 --episodes 100
python TD3-main/Sampling/collect_dynamic_expert.py --step 3 --curve sine --episodes 100
python TD3-main/Sampling/collect_dynamic_expert.py --step 4 --curve sine --episodes 100
```

Step2–4 应在 Step1 成功率和接触速度稳定后逐级启用，不建议直接混合采集。

## 随机 Step0-4 采集

`random_dynamic_expert.py` 用于生成覆盖静止、直线、变速直线、曲线和变速曲线的
混合专家数据。默认采集 300 个回合，并以分层随机方式保证 Step0-4 各有 60 个回合；
每个回合的方向、轨迹形状和速度独立随机，固定 `--seed` 时可复现。

```bash
python TD3-main/Sampling/random_dynamic_expert.py
```

船速会在约 0-1.2 m/s 内随机生成；专家的 XY 指令上限保留为 2.0 m/s，以便在追赶
移动甲板时留出闭合误差余量。Step0 是执行完整无人机降落过程的静止或近静止甲板
基线，不是 `step0_env.py` 的船体单独验证程序。抽到接近静止的曲线时，场景会降级为
静止甲板，避免生成无意义的超长圆周周期；圆周半径限制在 1.5-3.0m，保持目标在采集
视野附近。

成功回合默认写入：

```text
expert_data_dynamic/random_dynamic_privileged_pd.jsonl
```

所有非空回合默认写入：

```text
expert_data_dynamic/random_dynamic_privileged_pd_all.jsonl
```

每个 transition 的 `scenario` 字段保存 Step、轨迹模式、曲线类型、回合种子、方向、
速度/速度区间和曲线几何参数，方便之后按场景切分或审计数据。该脚本仅生成专家数据；
它会以 2.0 m/s 配置本次采集的专家动作上限，但不会修改现有离线训练的动作尺度或
checkpoint。

## 离线测试

```bash
python TD3-main/Sampling/test_privileged_pd_expert.py
```

该测试使用点质量模型覆盖 Step1 多组初始偏差，只验证控制律方向、限幅和收敛性，
不能替代 PX4、Gazebo、MAVROS 与碰撞模型的实际验证。
