# 文档索引

## 当前文档

| 文档 | 内容 |
| --- | --- |
| [environment.md](environment.md) | 离线训练与仿真采集的环境依赖、版本约束和外部资源 |

运行入口和数据约定以仓库根目录 `README.md`、`Sampling/README.md`、
`scripts/train/README.md` 为准。

## 归档文档（2026-11-06 归档）

`archive/` 保存此前的设计与改造记录，仅供追溯，不再随代码同步更新。
与当前代码冲突时以代码和上述 README 为准。

| 文档 | 内容 | 状态 |
| --- | --- | --- |
| [sampling_refactor_plan.md](archive/sampling_refactor_plan.md) | 全局真值专家自动采集的重构规划与历次架构决定 | 已实施 |
| [observation_10d_refactor_plan.md](archive/observation_10d_refactor_plan.md) | observation 从三维视觉位置改为十维视觉运动状态 | 已实施 |
| [moe_td3_implementation_spec.md](archive/moe_td3_implementation_spec.md) | MoE-TD3 结构、阶段映射与三阶段训练规格 | 已实施 |
| [parallel_sampling_change_notes.md](archive/parallel_sampling_change_notes.md) | 多实例无头并行采集改造与训练闭环检查记录 | 已实施 |
| [multi_vehicle_parallel_collection_plan.md](archive/multi_vehicle_parallel_collection_plan.md) | 单 Gazebo 世界多飞行器并行采集方案 | 未实施 |
