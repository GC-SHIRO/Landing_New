# Simulation 降落判定统一设计

> 日期: 2026-07-22 | 范围: 仅 `TD3-main/Simulation/`

## 1. 目标

统一 Simulation 内「是否成功」与「连续打分」为**单一权威源**，消除 env 在线终止阈值与离线评估阈值分裂。

## 2. 已确认需求

| 项 | 选择 |
|----|------|
| 成功时机 | 触地瞬间判定（无随船观察窗） |
| 成功边界 | 相对 marker：`herr < 1.0 m` 且 `verr < 0.5 m` |
| 目标点 | `ship_base + (0, 0, marker_offset_z)`，默认 `marker_offset_z=1.3` |
| 打分 | 连续 0~100，两维：精度 + 相对船动力学 |
| 失败 | 一律 0 分 |
| 动力学量 | 相对船速度 + 加速度 |
| 权重 | 精度 70% / 动力学 30% |
| 范围 | 仅 Simulation |

## 3. 架构（方案 A）

```
env_base.step()     → 终止时机（与成功半径同一套阈值）
step_env.py         → 通过 --step 1~4 控制场景，采集终点 drone/ship 位姿速度 → EpisodeData
landing_evaluation  → SUCCESS 判定 + 两维打分 + 汇总（权威）
```

- `env.success` 与评估器使用同一 `herr_max/verr_max`，但 **SR / Result 以评估器为准**。
- 动力学（相对船静止程度）**只影响分数**，不作为成功硬门槛。

## 4. 成功与结果标签

```
SUCCESS  ⇔  finite(herr,verr) ∧ herr < 1.0 ∧ verr < 0.5
            ∧ 非 CRASHED / OUT_OF_BOUNDS / LOST_DETECTION
```

| Result | 条件 |
|--------|------|
| `SUCCESS` | 满足成功边界 |
| `MISSED` | 正常结束但未满足成功边界 |
| `OUT_OF_BOUNDS` | 越界结束 |
| `LOST_DETECTION` | 低空无检测结束 |
| `MAX_STEPS` | 满步且未 SUCCESS（满步但已成功 → 仍 `SUCCESS`） |
| `CRASHED` / `RESET_FAILED` | 异常 |

取消 PERFECT / GOOD / ACCEPTABLE 离散精度档；精度由连续分体现。

**分类优先级：**

1. `CRASHED`
2. `OUT_OF_BOUNDS`
3. `LOST_DETECTION`
4. 位置成功 → `SUCCESS`（即使满步）
5. `max_steps_reached` → `MAX_STEPS`
6. 其余 → `MISSED`

## 5. 两维打分

仅 `SUCCESS` 计分；否则两维与综合分均为 0。

### 5.1 精度分 \(S_{acc}\)（权重 0.70）

\[
E_{3D}=\sqrt{E_{xy}^2+E_z^2},\quad
S_{acc}=100\cdot\max\bigl(0,\,1-E_{3D}/E_{th}\bigr)
\]

默认 \(E_{th}=1.2\) m（略大于成功边界 \(\sqrt{1^2+0.5^2}\approx1.12\)，成功边界上约得低分而非直接 0）。

### 5.2 动力学分 \(S_{dyn}\)（权重 0.30）

相对船速度（船 \(v_z\approx0\)）：

\[
\mathbf{v}_{rel}=\mathbf{v}_{drone}-\mathbf{v}_{ship},\quad
v=\lVert\mathbf{v}_{rel}\rVert
\]

相对加速度：优先用终点提供的 \(\mathbf{a}_{rel}\)；否则用末两步 \(\mathbf{v}_{rel}\) 差分 \(\mathbf{a}_{rel}=\Delta\mathbf{v}_{rel}/\Delta t\)。

\[
S_v=100\cdot\max(0,\,1-v/v_{max}),\quad
S_a=100\cdot\max(0,\,1-a/a_{max}),\quad
S_{dyn}=0.5\,S_v+0.5\,S_a
\]

默认 \(v_{max}=1.5\) m/s，\(a_{max}=5.0\) m/s²。

### 5.3 综合分

\[
S=0.70\,S_{acc}+0.30\,S_{dyn}
\]

评级（可选，仅展示）：A≥90, B≥75, C≥60, D≥40, 否则 F。

## 6. env 在线终止

动态目标（`landing_target_fn` 非空）：

```
done_success ⇔ herr < landing_dist_threshold
              ∧ verr < landing_height_threshold
```

默认与成功边界一致：`1.0 / 0.5`。**禁止硬编码 0.4/0.1。**

仍保留：满步、越界（`max_dist`/`max_height`）、低空失检。

`info` 至少包含：`success`, `target_reached`, `out_of_bounds`, `lost_detection`, `tag_detected`。

## 7. step 脚本接线

- `env.landing_dist_threshold = success_herr_thresh`（默认 1.0）
- `env.landing_height_threshold = success_height_thresh`（默认 0.5）
- 终点采样：`drone` 位姿/速度、`ship` 位姿/速度 → 相对速度/加速度
- `out_of_bounds` / `lost_detection` 从 `info` 回填
- `max_steps_reached = (steps >= max_steps)`；分类时 SUCCESS 优先于 MAX_STEPS
- `init_dist_3d` 用轨迹起点相对当时目标（若可得），否则 NaN；本轮打分不依赖它

## 8. 测试

无 ROS 单测 `landing_evaluation.py`：

- 边界内 → SUCCESS 且分 > 0
- 边界外 → MISSED 且分 = 0
- 满步但边界内 → SUCCESS
- 越界 / 失检 / 崩溃 标签
- 相对速度、加速度对 \(S_{dyn}\) 的单调影响

## 9. 非目标

- 不改 `landing_env.py` / `evaluate_iros_new.py`
- 不做随船共速硬门槛
- 不恢复四维评分（效率/平滑性）——本轮仅精度 + 动力学
