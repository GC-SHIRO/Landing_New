# 无人机自主降落评估规范

> 版本: v2.0 | 日期: 2026-06-24 | 适用于: 静态平台 (sandisland) / 动态平台 (step1_linear)

---

## 一、概述

本规范定义无人机自主降落任务的评估体系, 包含 **结果分类** 与 **多维度评分** 两部分。评估对象为单次 episode 的完整降落轨迹, 适用于:

- 离线训练后的批量评估 (`evaluate_iros_new.py` / `step1_env.py`)
- 在线微调中的实时评估
- 专家数据质量分析 (`analyze_expert_episode.py`)

---

## 二、坐标系与目标点定义

### 2.1 检测目标

YOLO v11 检测的目标是 **WAM-V 无人船甲板上的 AR marker**, 定义在 WAM-V URDF 中 (`wamv_gazebo.urdf.xacro`):

```xml
<link name="marker_link">
  <visual>
    <origin xyz="0 0 1.3" rpy="0 -1.5708 0"/>
    <geometry>
      <mesh filename="model://marker/meshes/marker.dae" scale="0.8 0.8 0.8"/>
    </geometry>
  </visual>
</link>
<joint name="marker_joint" type="fixed">
  <parent link="base_link"/>
  <child link="marker_link"/>
</joint>
```

**关键参数**:

- marker 相对于船 `base_link` 的偏移: `(dx=0, dy=0, dz=1.3)` 米
- marker 旋转: `rpy="0 -1.5708 0"` → pitch -90°, 面朝上 (供无人机俯视检测)
- marker 模型: `model://marker` (AR marker, 缩放 0.8×)

### 2.2 降落目标点

| 平台类型                      | 目标点公式                                                       | 说明                                                  |
| ----------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------- |
| **静态** (sandisland)   | 世界坐标原点附近 (由`get_real_state()` +2 偏移校准)            | ship 在 (5, 5) 不参与判定; 降落目标为固定 marker 位置 |
| **动态** (step1_linear) | `(ship_x, ship_y, ship_z + 1.3)` — 船 base_link + marker 偏移 | 船在 (10, 5) 起始, 由 ShipMotionController 驱动移动   |

> **注意**: 静态平台的坐标系统存在历史遗留的 +2 偏移 (`get_real_state()` 返回 `[x+2, y+2, z]`), 在重构评估脚本时应统一为直接使用世界坐标与目标点比较, 消除该偏移。

### 2.3 各 launch 文件中的模型位置

| 模型                   | sandisland (静态) | step1_linear (动态)     |
| ---------------------- | ----------------- | ----------------------- |
| 无人机 iris (HOME)     | (0, 0, 0.5)       | (0, 0, 0.5)             |
| landing_pad (起降平台) | (0, 0, 0), 6×6 m | (0, 0, 0), 6×6 m       |
| WAM-V ship (base_link) | (5, 5, 0.1)       | (10, 5, 0.1)            |
| marker (甲板上)        | (5, 5, 1.4)       | (10, 5, 1.4)            |
| 无人机 reset 范围      | 世界原点 ±5m     | 以船 (10,5) 为中心 ±4m |

---

## 三、结果分类 (Classification)

### 3.1 分类决策树

```
Episode 结束
├─ [RESET_FAILED]   reset() 异常 / 起飞失败 / 超时
├─ [CRASHED]        step() 过程中异常崩溃
├─ [ABORTED]        主动中止
│   ├─ OUT_OF_BOUNDS    无人机飞出安全区域
│   ├─ MAX_STEPS         达到最大步数仍未着陆
│   └─ LOST_DETECTION    低空 (< 0.25m) 且无目标检测
├─ [LANDED]         触地 (高度 < 阈值 或 检测到着陆)
│   ├─ SUCCESS           着陆成功
│   │   ├─ PERFECT       精度优秀 (herr < 0.3m & h < 0.15m & 冲击 < 0.5m/s)
│   │   ├─ GOOD          精度良好 (herr < 0.6m & h < 0.3m)
│   │   └─ ACCEPTABLE    精度可接受 (herr < 1.0m & h < 0.6m)
│   └─ FAILED            着陆失败
│       ├─ MISSED        未落在目标上 (偏离过大)
│       ├─ HARD_LANDING  着陆冲击过大 (触地垂速 > 1.5 m/s)
│       └─ OFF_PLATFORM  动态平台: 落点距船 > 2.0m
```

### 3.2 分类判定规则

| 结果码             | 判定条件                                                                  | 优先级   |
| ------------------ | ------------------------------------------------------------------------- | -------- |
| `RESET_FAILED`   | `reset()` 抛出异常或超时 (20s 内未到达起始点)                           | 0 (最高) |
| `CRASHED`        | rollout 中任何`step()` 或 settle 阶段抛出异常                           | 1        |
| `OUT_OF_BOUNDS`  | `dist_origin > max_dist` 或 `z > max_height`                          | 2        |
| `LOST_DETECTION` | `z < 0.25m` 且 `yolo_detected == False`                               | 2        |
| `MAX_STEPS`      | `step_count >= max_steps` 且未触发上述条件                              | 3        |
| `PERFECT`        | 触地 &`herr < 0.3m` & `final_z_abs < 0.15m` & `impact_vz < 0.5 m/s` | 4        |
| `GOOD`           | 触地 &`herr < 0.6m` & `final_z_abs < 0.3m`                            | 4        |
| `ACCEPTABLE`     | 触地 &`herr < 1.0m` & `final_z_abs < 0.6m`                            | 4        |
| `MISSED`         | 触地但`herr >= 1.0m`                                                    | 5        |
| `HARD_LANDING`   | 触地但`impact_vz >= 1.5 m/s` (冲击过大)                                 | 5        |
| `OFF_PLATFORM`   | 动态平台: 触地但`herr` 相对船 > 2.0m                                    | 5        |

> **注**: 动态平台的 `herr` 以船 `base_link` 为参考。高度判定使用 `z_drone - (z_ship + 1.3)` (marker 高度)。

### 3.3 汇总统计口径

| 统计量                      | 定义                                                   |
| --------------------------- | ------------------------------------------------------ |
| **SR (Success Rate)** | `N_SUCCESS / N_total` (含 RESET_FAILED)              |
| **SR_usable**         | `N_SUCCESS / N_usable` (仅含 LANDED 类型)            |
| **SR_attempted**      | `N_SUCCESS / (N_total - N_RESET_FAILED - N_CRASHED)` |
| **CR (Crash Rate)**   | `(N_RESET_FAILED + N_CRASHED) / N_total`             |

---

## 四、多维度评测指标 (四大维度)

每个成功或失败的着陆 episode 计算以下指标 (RESET_FAILED / CRASHED 仅记录, 不计算)。

### 4.1 终点精度 (Terminal Accuracy) — 权重 35%

评估无人机最终位置与目标点的接近程度。

| 指标           | 符号               | 定义                                                               | 单位 |
| -------------- | ------------------ | ------------------------------------------------------------------ | ---- |
| 水平误差       | $E_{xy}$         | $\sqrt{(x_{drone} - x_{target})^2 + (y_{drone} - y_{target})^2}$ | m    |
| 垂直误差       | $E_z$            | $\vert z_{drone} - z_{target} \vert$                             | m    |
| 三维误差       | $E_{3D}$         | $\sqrt{E_{xy}^2 + E_z^2}$                                        | m    |
| 归一化水平误差 | $\tilde{E}_{xy}$ | $E_{xy} / D_{init}$ (初始 3D 距离归一化)                         | —   |

**精度评分** ($S_{acc}$, 0~100):

$$
S_{acc} = 100 \cdot \max\left(0,\; 1 - \frac{E_{3D}}{E_{threshold}}\right)
$$

其中 $E_{threshold} = 2.0$ m (三维误差容忍上限)。$E_{3D} \ge 2.0$ 得 0 分。

### 4.2 效率指标 (Efficiency) — 权重 25%

评估降落的时空效率。

| 指标     | 符号       | 定义                                                    | 单位   |
| -------- | ---------- | ------------------------------------------------------- | ------ |
| 降落耗时 | $T$      | `step_count × dt` (settle 时间不计入)                | s      |
| 降落步数 | $N$      | `step_count`                                          | step   |
| 每米耗时 | $T_{pm}$ | $T / D_{init}$                                        | s/m    |
| 每米步数 | $N_{pm}$ | $N / D_{init}$                                        | step/m |
| 路径比   | $\rho$   | $L_{actual} / D_{init}$ (实际路径长度 / 初始直线距离) | —     |

**效率评分** ($S_{eff}$, 0~100):

$$
S_{eff} = 100 \cdot \max\left(0,\; 1 - \frac{T_{pm} - T_{pm}^{ref}}{T_{pm}^{max}}\right)
$$

其中:

- $T_{pm}^{ref} = 1.0$ s/m (参考值: 专家策略中位数)
- $T_{pm}^{max} = 5.0$ s/m (容忍上限)
- $T_{pm} \le T_{pm}^{ref}$ 得满分 100

### 4.3 平滑性指标 (Smoothness) — 权重 20%

评估动作序列的平滑程度, 反映控制品质。

| 指标              | 符号         | 定义                                                                                    |
| ----------------- | ------------ | --------------------------------------------------------------------------------------- |
| 动作平滑度        | $J_{act}$  | $\frac{1}{N-1}\sum_{t=1}^{N-1} \lVert a_t - a_{t-1} \rVert^2$                         |
| 加速度波动 (Jerk) | $J_{jerk}$ | $\frac{1}{N-2}\sum_{t=2}^{N-1} \lVert (a_t - a_{t-1}) - (a_{t-1} - a_{t-2}) \rVert^2$ |
| 方向变更率        | $F_{dir}$  | 相邻动作夹角 > 90° 的频率 (次/step)                                                    |

**平滑性评分** ($S_{smooth}$, 0~100):

$$
S_{smooth} = 100 \cdot \max\left(0,\; 1 - \frac{J_{act}}{J_{max}}\right)
$$

其中 $J_{max} = 0.5$ (动作平滑度容忍上限)。

### 4.4 安全性指标 (Safety) — 权重 20%

评估降落过程中的安全边界遵守情况及着陆冲击。

| 指标         | 符号         | 定义                                 |
| ------------ | ------------ | ------------------------------------ |
| 最大高度     | $Z_{max}$  | episode 中无人机的最大高度 (m)       |
| 最大偏移     | $D_{max}$  | episode 中距世界原点的最大距离 (m)   |
| 着陆冲击速度 | $V_{imp}$  | 触地前最后一帧的垂向速度绝对值 (m/s) |
| 高度超限比   | $R_{over}$ | $Z_{max} / Z_{max\_allowed}$       |

**安全性评分** ($S_{safe}$, 0~100):

$$
S_{safe} = 100 \cdot \left(0.5 \cdot \min\left(1, \frac{Z_{max\_allowed}}{Z_{max}}\right) + 0.5 \cdot \max\left(0, 1 - \frac{V_{imp}}{V_{max}}\right)\right)
$$

其中:

- $Z_{max\_allowed} = 12.0$ m
- $V_{max} = 1.5$ m/s (着陆冲击速度上限)
- 越界 ($D_{max} > max\_dist$ 或 $Z_{max} > max\_height$) 直接 $S_{safe} = 0$

---

## 五、综合评分 (Composite Score)

### 5.1 加权总分

$$
S_{total} = 0.35 \cdot S_{acc} + 0.25 \cdot S_{eff} + 0.20 \cdot S_{smooth} + 0.20 \cdot S_{safe}
$$

分值范围: **0 ~ 100**。

### 5.2 评级映射

| 总分区间 | 评级                     | 含义                       |
| -------- | ------------------------ | -------------------------- |
| 90 ~ 100 | **A (Excellent)**  | 精准、高效、平滑、安全     |
| 75 ~ 89  | **B (Good)**       | 总体良好, 某维度有提升空间 |
| 60 ~ 74  | **C (Acceptable)** | 基本可接受, 多个维度需改进 |
| 40 ~ 59  | **D (Poor)**       | 明显缺陷, 不建议部署       |
| 0 ~ 39   | **F (Fail)**       | 严重失败, 策略需重新训练   |

### 5.3 特殊情况的评分

| 结果类型           | $S_{total}$ 处理                       |
| ------------------ | ---------------------------------------- |
| `RESET_FAILED`   | 不评分, 标记为 NaN                       |
| `CRASHED`        | 不评分, 标记为 NaN                       |
| `OUT_OF_BOUNDS`  | $S_{safe} = 0$, 其他维度按实际轨迹计算 |
| `MAX_STEPS`      | 按实际终点计算所有维度                   |
| `LOST_DETECTION` | 按实际轨迹计算所有维度                   |

---

## 六、输出数据规范

### 6.1 每 Episode 输出字段 (JSONL)

```json
{
  "Episode": 1,
  "Result": "PERFECT",
  "CompositeScore": 92.5,

  "Score_Accuracy": 95.0,
  "Score_Efficiency": 88.0,
  "Score_Smoothness": 91.0,
  "Score_Safety": 94.0,

  "FinalX": -0.28,
  "FinalY": 0.02,
  "FinalZ": 0.11,
  "TargetX": 0.0,
  "TargetY": 0.0,
  "TargetZ": 0.0,
  "HorizErr": 0.28,
  "VertErr": 0.11,
  "Err3D": 0.30,
  "NormHorizErr": 0.025,

  "Steps": 156,
  "TimeSec": 15.6,
  "InitDist3D": 10.67,
  "TimePerMeter3D": 1.46,
  "StepsPerMeter3D": 14.6,
  "PathRatio": 1.12,

  "ActionSmoothness": 0.015,
  "Jerk": 0.002,
  "DirectionChangeRate": 0.03,

  "MaxHeight": 9.5,
  "MaxDistOrigin": 11.2,
  "ImpactVelocityZ": 0.35,

  "CkptDir": "...",
  "LoadStep": 60000,
  "Dt": 0.1,
  "MaxSteps": 600,
  "PlatformType": "static",
  "Timestamp": "2026-06-24T12:00:00"
}
```

### 6.2 汇总输出字段 (CSV)

| 字段                                          | 含义                            |
| --------------------------------------------- | ------------------------------- |
| `N_total`                                   | 总 episode 数                   |
| `N_RESET_FAILED`                            | reset 失败数                    |
| `N_CRASHED`                                 | 崩溃数                          |
| `N_ABORTED`                                 | 中止数 (越界+超步数+低空无检测) |
| `N_LANDED`                                  | 着陆数 (SUCCESS + FAILED)       |
| `N_PERFECT` / `N_GOOD` / `N_ACCEPTABLE` | 各级成功数                      |
| `SR`                                        | 成功率 (含 RESET_FAILED)        |
| `SR_usable`                                 | 可用成功率                      |
| `Mean_CompositeScore`                       | 平均综合评分 (± Std)           |
| `Mean_Score_Accuracy`                       | 平均精度分                      |
| `Mean_Score_Efficiency`                     | 平均效率分                      |
| `Mean_Score_Smoothness`                     | 平均平滑性分                    |
| `Mean_Score_Safety`                         | 平均安全分                      |
| `Mean_HorizErr`                             | 平均水平误差 (仅 LANDED)        |
| `Mean_TimePerMeter3D`                       | 平均每米耗时                    |
| `Mean_ImpactVelocityZ`                      | 平均着陆冲击速度                |
| `Mean_PathRatio`                            | 平均路径比                      |

---

## 七、阈值配置 (可调参数)

| 参数名                    | 默认值        | 说明                             |
| ------------------------- | ------------- | -------------------------------- |
| `success_herr_thresh`   | 1.0 m         | 水平误差成功阈值                 |
| `success_height_thresh` | 0.6 m         | 高度成功阈值                     |
| `perfect_herr_thresh`   | 0.3 m         | PERFECT 水平误差阈值             |
| `perfect_height_thresh` | 0.15 m        | PERFECT 高度阈值                 |
| `good_herr_thresh`      | 0.6 m         | GOOD 水平误差阈值                |
| `good_height_thresh`    | 0.3 m         | GOOD 高度阈值                    |
| `impact_vz_limit`       | 1.5 m/s       | 硬着陆垂向速度阈值               |
| `settle_seconds`        | 3.0 s         | 触地后物理稳定等待时间           |
| `max_dist`              | 15.0 / 25.0 m | 越界距离 (静态/动态)             |
| `max_height`            | 12.0 m        | 越界高度                         |
| `marker_offset_z`       | 1.3 m         | marker 距船 base_link 的高度偏移 |

---

## 八、动态平台特化

对于动态平台 (船舶), 在静态平台基础上增加以下约束:

1. **目标点动态化**: 每步从 `/gazebo/model_states` 读取船 `base_link` 位置, 目标点 = `(ship_x, ship_y, ship_z + 1.3)`。
2. **跟踪误差**: 记录每一步无人机与船的实时距离序列, 用于评估是否持续靠近目标。
3. **OFF_PLATFORM**: 触地时若无人机距船 > 2.0 m, 判定为"落在平台外"。
4. **相对运动补偿**: 船速矢量与无人机速度矢量的夹角, 衡量策略对平台运动的补偿能力。
5. **安全性中 max_dist 放宽**: 因为无人机在船周围活动 (约 (10,5) 附近), 距原点本就较远, 默认 25m。

---

## 九、实施建议

1. **第一阶段**: 创建独立 `compute_metrics.py` 工具脚本, 支持对已有 JSONL 重算评分 (无需重新跑仿真)。
2. **第二阶段**: 在 `evaluate_iros_new.py` 中实现新的分类体系 (PERFECT/GOOD/ACCEPTABLE) 和四维评分。
3. **第三阶段**: 在 `step1_env.py` 中同步实现, 增加动态平台特化指标和目标点计算。
4. **第四阶段**: 消除 `get_real_state()` 中的 +2 偏移, 统一使用世界坐标与目标点直接比较。

---

*本规范与 `CLAUDE.md` 中的项目架构、归一化约定保持一致。*
*坐标系定义基于 `wamv_gazebo.urdf.xacro` 中的 `marker_link` 实际建模参数。*
