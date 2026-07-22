# 落到甲板上 判定设计

> 日期: 2026-07-22 | 范围: `TD3-main/Simulation/`

## 问题

旧成功球过大：`target_z≈1.4` 且 `|dz|<0.5` 时，约 1.9 m 高度就会判成功，飞机并未贴甲板。

## 成功定义（在线终止 = 评估 SUCCESS）

须**同时**满足，并连续保持 `hold_seconds`（默认 0.5 s）：

1. **几何贴甲板**
   - `herr = sqrt((x-tx)^2+(y-ty)^2) < 1.0 m`（marker 圆心半径）
   - `0 ≤ (z_drone − z_deck) < h`，默认 `h=0.10 m`（仅上方，不穿甲板）
   - `z_deck = ship_base_z + marker_offset_z`，默认 `0.1 + 1.3 = 1.4`

2. **相对船共速**
   - `||vxy_rel|| < 0.30 m/s`
   - `|vz_rel| < 0.20 m/s`
   - `v_rel = v_drone − v_ship`

3. **保持**
   - 上述 1+2 连续累计 ≥ 0.5 s（`dt=0.1` → 约 5 步）
   - 任一条件断开则计数清零

## 实现

| 组件 | 职责 |
|------|------|
| `env_base.GazeboEnv.step` | 在线状态机：几何+共速累计 hold，满则 `done/success` |
| `Communication` | 订阅 `local_position/velocity_local`；无则位置差分 |
| `ship_velocity_fn` | step 脚本注入船速度 |
| `landing_evaluation` | 同一套硬条件；`on_deck_success` 优先；打分仍 70/30 |

失败：未满足 → 继续飞直到 OOB / 水下 z<0 / 失检 / 满步。  
`z < 0` 仍立即 OUT_OF_BOUNDS。

## CLI（step1–4）

```
--success_herr_thresh 1.0
--success_height_thresh 0.06   # 0<=dz<h
--vxy_rel_max 0.30
--vz_rel_max 0.20
--hold_seconds 0.50
```

## 模型备注

- marker 相对 WAM-V base_link：`xyz="0 0 1.3"`
- Iris 贴地时 `base_link` 约在甲板上方 ~0.10–0.15 m（起落架）
- **h=0.06 可能偏紧**：若真实贴地难触发 SUCCESS，可把 `--success_height_thresh` 调到 0.12–0.15

## 非目标

- 不改外层 `landing_env.py` / `evaluate_iros_new.py`
- 不用 Gazebo 接触传感器
