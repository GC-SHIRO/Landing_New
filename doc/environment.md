# 环境依赖

项目分为两套运行环境：离线训练/测试可在 Windows 或 Linux 上运行；仿真采集与闭环评估
只能在装有 ROS1 的 Ubuntu 上运行。两套环境的 PyTorch 版本应保持一致，避免 `.pt` 存档
跨版本读取失败。

## 1. 离线训练与测试

安装：

```bash
pip install -r requirements.txt
```

| 依赖 | 版本约束 | 原因 |
| --- | --- | --- |
| Python | >= 3.8 | 代码未使用 3.9 以后才有的运行时语法 |
| PyTorch | >= 1.13 | `scripts/train/common.py` 使用 `torch.load(..., weights_only=False)` |
| NumPy | >= 1.20 | 全部模块；1.x 与 2.x 均可 |
| TensorBoard | 任意 | 仅 LSTM 版 `scripts/train_offline.py` 使用；MoE 三阶段训练用 JSONL 日志 |
| Pandas | 任意 | 仅旧评估入口 `scripts/evaluate_iros.py` 使用 |

已验证组合（2026-11-06）：Python 3.8.20 + PyTorch 1.13.1+cu117 + NumPy，
`model/tests` 18 个测试与 `Sampling/tests` 28 个测试全部通过。

离线测试命令：

```bash
python -m unittest discover -s model/tests -v
python -m unittest discover -s Sampling/tests -v
python -m compileall -q model scripts Sampling
```

## 2. 仿真采集与评估（Ubuntu）

### 系统与 ROS

- ROS1（代码使用 `rospy`、`roslaunch`、`rosgraph`），按 ROS Noetic / Ubuntu 20.04 /
  系统 Python 3.8 准备。
- Gazebo、PX4 SITL、MAVROS，以及 XTDrone 风格的多机 launch。
- ROS 消息包：`geometry_msgs`、`nav_msgs`、`std_msgs`、`std_srvs`、`gazebo_msgs`、
  `mavros_msgs`。
- 并行无界面采集额外需要系统包 `xvfb` 和 Mesa 软件渲染。

Python 包（ROS 消息包通过 apt / catkin 提供，不在 pip 中）：

```bash
pip install -r requirements-sim.txt
```

Noetic 的系统 Python 为 3.8，PyTorch 2.5 起不再支持 Python 3.8，因此仿真侧
PyTorch 版本上限受此限制；训练可在更新的环境中进行，但要确认存档能在仿真侧加载。

### 视觉检测

- ROS 包 `yolov11_ros` 与消息包 `yolov11_ros_msgs`，发布 `/yolov11/BoundingBoxes`
  和 `/yolov11/centers`，通过 `roslaunch yolov11_ros yolo_v11.launch` 启动。
- YOLO 所在 Python 环境需同时安装 PyTorch 和 ROS Python 模块。
- 并行采集时显存需能容纳 `NUM_WORKERS` 份 YOLO。

### 仓库外资源

以下文件不在仓库中，换机器时需准备并修改脚本顶部的路径：

| 资源 | 引用位置 | 要求 |
| --- | --- | --- |
| `/home/shiro/PX4_Firmware/launch/step1_linear.launch` | `Sampling/collect_global_expert.py` 的 `LAUNCH_FILE` | 声明 `ID`、`gui` 参数；包含 `wamv`、降落平台和 `/benchmarker/collision` |
| `yolo_v11.launch` 与模型权重 | `Simulation/env/env_base.py` | 能按 `ID` 订阅对应无人机相机话题 |
| `/home/shiro/anaconda3/envs/lab_env/bin/python` | `Sampling/collect_parallel.py` 的 `YOLO_PYTHON_EXECUTABLE` | 同时安装 PyTorch 与 ROS Python 模块 |

### 仅旧环境需要

`apriltag_ros` 只被 `Simulation/env/landing_env_old.py` 和
`Simulation/env/landing_env_listen.py` 使用，当前 Sampling 与动态评估不需要。
