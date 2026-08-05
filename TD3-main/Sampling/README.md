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

### 奖励（与 Simulation 动态落地判定对齐，Sampling 内本地复刻）

`reward` 由 `Sampling/privileged_pd_expert.compute_transition_reward` 生成，**不 import Simulation**，避免 ROS 耦合：

- 非终止：视觉 L3 shaping  
  `0.1 * -(|x|^3 + |y|^3 + |z|^3)^(1/3)`，使用**步进前** `observation`
- 终止：成功判定与 `Simulation/env_base.step()` 的权威落地判定一致  
  `landing_success = deck_contact AND relative_xy_distance <= landing_xy_threshold`
  - 使用**相对甲板水平距离**（`relative_xy_distance`，步进后无人机与甲板目标的世界系水平距离），而非旧静态世界框 `(-2.5,-1.5)^2`
  - `success=True` 且相对距离 ≤ 阈值 → `+300`，否则 `-200`
  - 未传入距离时直接信任 `success` 标志（env 已按相对甲板 + 接触判定）

注意：旧版曾用静态世界框，动态甲板上成功落点常远离该框而误判 `-200`，已改为相对甲板判定。

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

## 训练样本质量门控

触底阶段相机过近/遮挡时 YOLO 经常冻结或失检，这是物理上预期的现象，
不再把整条成功轨迹一票否决。

当前策略：

- 只把 `sample_is_usable=True` 的 transition 写入训练 jsonl。
- `shape` / `non_finite` / `action_limit` 视为致命质量问题，整回合 reject。
- `FLARE` / `TOUCHDOWN` 或 `relative_height <= near_ground_height` 时：
  失检、重复帧只裁剪，不否决整回合。
- 成功保存还要求：
  - 落地成功
  - 有效步数 `>= min_success_steps`（默认 15）
  - 有效占比 `>= min_valid_ratio`（默认 0.85）
- raw jsonl 逐步记录 `quality_accepted` / `quality_reason` / `quality_fatal`。

离线回归（不需要 ROS；建议使用 `lab_env`）：

```bash
cd /home/shiro/Landing_new
PYTHONPATH=TD3-main /home/shiro/anaconda3/envs/lab_env/bin/python \
  -m unittest Sampling.tests.test_sample_quality_gate -v
```

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

`random_dynamic_expert.py` 用于生成覆盖 7 类运动的混合专家数据：

| 类别 | 对应 Step | 说明 |
|------|-----------|------|
| `static` | 0 | 静止甲板 |
| `line_constant` | 1 | 直线匀速 |
| `line_varspeed` | 2 | 直线变速 |
| `sine_constant` | 3 | 正弦曲线匀速 |
| `sine_varspeed` | 4 | 正弦曲线变速 |
| `circle_constant` | 3 | 圆圈匀速 |
| `circle_varspeed` | 4 | 圆圈变速 |

- **停止条件**：默认直到**保存满 300 条有效回合**才停止（`--target_saved`），
  而不是按尝试次数；每条保存仍要求成功落地且通过质量门控。
- **均匀分布**：每局从“已保存最少”的类别中随机挑选（失败/被拒不计入），
  因此最终 7 类各约 1/7（300 条时各 42–43 条），避免前期某类扎堆。
- **安全上限**：`--max_attempts`（默认 1500）防止某类长期采不到时无限运行；
  触达上限会保留现有数据并打印警告。
- **速度随机**：每局方向、速度（或速度区间）、曲线几何参数独立随机，
  固定 `--seed` 时可复现；`--episodes` 作为 `--target_saved` 的废弃别名兼容。

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
