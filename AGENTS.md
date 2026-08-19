# 项目开发约定

本文件适用于整个 `/home/shiro/Landing_new` 仓库。

## 总体原则

- 优先使用简单、直接、容易现场修改的实现，不为暂时不存在的需求增加复杂抽象。
- 避免过度工程化，不主动增加配置框架、插件系统、自动恢复流程或多层兼容逻辑。
- 修改应聚焦当前任务；旧实现已经被替代时可以删除，需要追溯时使用 Git 历史。
- 除非用户明确要求，不修改与当前任务无关的训练、评估或仿真代码。

## 参数与命令行

- 经常调整的重要参数集中放在脚本顶部，并按用途分组，例如“采集参数、仿真参数、专家参数、输出路径”。
- 参数不得散落在主循环和多个辅助函数中。
- 默认通过直接修改脚本顶部参数完成配置，避免复杂 CLI。
- 只为高频且明确的运行模式保留少量 CLI 开关。
- Sampling 采集器保留 `--test` 作为唯一的少量样本冒烟测试开关；测试数据必须使用独立文件，不得污染正式数据。

## 注释与输出

- 新增或修改的代码注释使用中文。
- docstring、阶段说明、运行提示和异常信息优先使用中文。
- 变量、函数和类名可以使用语义清晰的英文。
- 注释解释设计原因、坐标语义和数据约束，避免重复描述代码表面行为。

## Sampling 数据约定

- `TD3-main/TD3_offline.py` 是离线训练数据接口的唯一标准；除非用户明确要求，否则不要修改它。
- Sampling 输出必须包含：`observation`、`action`、`reward`、`next_observation`、`done`。
- 默认保持 `state_dim=3`、`action_dim=3`、`max_action=1.0` 和 `seq_len=8` 的兼容性。
- Sampling 写原始 observation，不提前归一化；归一化由 `TD3_offline.py` 完成。
- 每行保存一个完整 episode，不删除中间 transition，不拼接不相邻帧。
- 必须满足 `step[i].next_observation == step[i+1].observation`。
- 只把完整成功 episode 写入正式训练文件；失败尝试可以写入独立 raw 文件。
- marker 失检时保持上一有效三维视觉 observation，不使用全局真值冒充策略输入。
- 非近地失检进入 `SEARCH`：保持上一实际 action 的水平分量，并使用正 z 上升。
- 近地失检不进入 `SEARCH`，继续使用全局真值专家完成低速触地。
- 重新检测到 marker 后立即退出 `SEARCH`，恢复正常专家控制。

## 测试约定

- 默认只运行不依赖 ROS、Gazebo、PX4 和 YOLO 的离线测试。
- 不要由代理自动启动仿真、PX4、Gazebo、ROS 或 YOLO；仿真测试由用户手动执行。
- `--test` 仅表示少量样本采集模式，不授权代理自行启动仿真。
- 离线测试至少覆盖动作范围、SEARCH 方向、近地失检例外、视觉 observation 保持和 episode 连续性。
- 完成修改后运行相关离线单元测试、Python 静态编译检查和 `git diff --check`。

## 文档同步

- Sampling 的入口、参数、输出文件或数据语义变化时，同步更新 `TD3-main/Sampling/README.md`。
- 重要架构决定同步更新 `doc/sampling_refactor_plan.md`。
