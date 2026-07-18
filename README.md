# Landing_New

无人机自主降落仿真与 TD3-BC + LSTM + Attention 实验工程。

仓库保留了训练、评估、专家数据、模型权重和仿真辅助脚本；运行时录像、rosbag、日志、缓存、服务器连接信息和本地工具配置不会提交。

## 主要目录

- `TD3-main/`：训练、在线微调、评估及 ROS/Gazebo 环境代码。
- `checkpoints/TD3/LSTM/`：离线训练和在线微调的 `.pth` 权重，以及状态归一化文件。
- `expert_data_lstm.json`：专家数据。
- `Landing_new/evaluation_data/`：已有评估结果。
- `tools/headless_recording/`：无头 Gazebo ROS Camera 第三人称录像辅助文件。
- `HEADLESS_GAZEBO_ROS_CAMERA_RECORDING.md`：无头录像流程说明。

## 注意事项

运行 ROS、Gazebo、PX4 与 YOLO 前，请按本机/服务器实际路径配置环境。模型权重与状态归一化文件必须来自同一 checkpoint 目录，否则策略输入分布会不一致。

本仓库不包含密码、API token、SSH 密钥、运行日志、rosbag 或录制视频。
