# 持久化环境设计方案

## 一、核心需求分析

你的需求是：**环境不变，通过运行不同的脚本改变无人机的运行轨迹或方法**

这意味着：
- Gazebo 仿真只启动**一次**，持续运行
- roscore 和 YOLO 也只启动**一次**
- 通过启动不同的脚本来**复用同一个环境**，改变策略执行结果

**好处**：
1. 避免反复启动/关闭仿真的时间开销（每次 10-15 秒）
2. 环境状态连贯，便于对比不同策略的效果
3. 支持并行运行多个脚本（同时进行多个评估任务）

---

## 二、整体架构设计

```
系统分层结构：
┌─────────────────────────────────────────────────────────┐
│  脚本层（可启动多个）                                    │
│  ┌──────────────┬──────────────┬──────────────┐        │
│  │ train_test.py│evaluate.py   │finetune.py   │ ...   │
│  │(客户端模式)  │(客户端模式)  │(客户端模式)  │        │
│  └──────┬───────┴──────┬───────┴──────┬───────┘        │
└─────────┼──────────────┼──────────────┼────────────────┘
          │ 连接         │ 连接         │ 连接
┌─────────▼──────────────▼──────────────▼────────────────┐
│ 环境服务层（只启动一次）                                │
│  ┌─────────────────────────────────────────┐           │
│  │  env_server.py                          │           │
│  │  - 启动 roscore（端口 11311）           │           │
│  │  - 启动 Gazebo + PX4（Launch 文件）     │           │
│  │  - 启动 YOLO 视觉节点（常驻）           │           │
│  │  - 提供 ROS 话题/服务接口               │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
          │ ROS 主题/服务
┌─────────▼──────────────────────────────────────────────┐
│ 物理仿真层（Gazebo + PX4）                             │
│  - 无人机动力学                                        │
│  - 传感器模型                                          │
│  - 环境（海面、目标区域）                               │
└────────────────────────────────────────────────────────┘
```

---

## 三、实施步骤详解

### 步骤 1：启动环境服务器（一次性，后台运行）

**终端 1**（保持运行）：
```bash
cd /home/shiro/Landing_new/TD3-main
python env_server.py --launch_file ~/PX4_Firmware/launch/Landing_with_boat.launch
```

**env_server.py 做的事情**：
1. 启动 `roscore`（ROS 主服务，所有节点通过它通信）
2. 启动 `roslaunch` 加载 Gazebo + PX4 仿真
3. 启动 YOLO 视觉检测节点（用于目标识别）
4. 持续运行，等待客户端连接

**输出日志应该是这样**：
```
【环境服务器启动】
[步骤1] 启动 ROS Core（端口: 11311）
[步骤2] 初始化 ROS 节点
[步骤3] 启动 Gazebo + PX4 仿真
[步骤4] 启动 YOLO 检测节点
[步骤5] 注册服务
【环境服务器运行中】
等待客户端连接...
```

---

### 步骤 2：客户端脚本连接环境（可启动多个）

**终端 2**（可以有多个）：
```bash
python train_test.py --use_env_server True --load_step 60000
```

**或者**：
```bash
python evaluate_iros_new.py --use_env_server True --test_episodes 100
```

**或者**：
```bash
python TD3_online_finetune.py --use_env_server True --load_step 60000
```

**客户端脚本做的事情**：
1. 不启动 roscore 和 Gazebo（节省时间）
2. 连接到**已有的** ROS Master（由 env_server 运行）
3. 订阅 ROS 话题（获取传感器数据）
4. 发布控制命令
5. 执行策略（加载不同的模型权重改变行为）

**特点**：
- 客户端是**无状态的**，环境状态由服务器维护
- 可以同时启动多个客户端（例如同时评估两个模型）
- 客户端退出后，环境继续运行

---

## 四、改变无人机轨迹的三种方式

### 方式 1：加载不同的模型权重

```bash
# 客户端脚本 1：加载 10000 步的权重（早期模型，性能较差）
python train_test.py --use_env_server True --load_step 10000

# 客户端脚本 2：加载 60000 步的权重（训练完成，性能好）
python train_test.py --use_env_server True --load_step 60000

# 客户端脚本 3：加载在线微调的权重（经过实时环境优化）
python train_test.py --use_env_server True --ckpt_dir ./checkpoints/TD3/LSTM/online_finetune --load_step 89965
```

**效果**：同一个环境，加载不同的模型，输出完全不同的降落轨迹

---

### 方式 2：改变控制策略

在脚本中添加参数来改变策略：

```bash
# 脚本 1：使用 TD3 策略
python train_test.py --use_env_server True --strategy td3

# 脚本 2：使用纯行为克隆策略
python train_test.py --use_env_server True --strategy bc

# 脚本 3：使用随机探索（用于基准对比）
python train_test.py --use_env_server True --strategy random

# 脚本 4：使用 PID 控制（传统方法对比）
python train_test.py --use_env_server True --strategy pid
```

---

### 方式 3：改变环境参数和起始条件

```bash
# 脚本 1：起始位置在北方
python train_test.py --use_env_server True --start_pos "north"

# 脚本 2：起始位置在东方
python train_test.py --use_env_server True --start_pos "east"

# 脚本 3：有风的情况
python train_test.py --use_env_server True --wind_speed 5.0

# 脚本 4：无风的情况
python train_test.py --use_env_server True --wind_speed 0.0

# 脚本 5：目标移动速度 1 m/s
python train_test.py --use_env_server True --target_velocity 1.0

# 脚本 6：目标不动
python train_test.py --use_env_server True --target_velocity 0.0
```

---

## 五、技术实现要点

### 5.1 现有代码修改点

**landing_env.py 中的改动**：
- 添加 `use_env_server` 参数到 `GazeboEnv.__init__()`
- 当 `use_env_server=True` 时：
  - **不启动** `roscore`（因为 env_server 已启动）
  - **不启动** `Gazebo`（因为 env_server 已启动）
  - **不启动** `YOLO`（因为 env_server 已启动）
  - 直接连接到已有的 ROS Master
- 当 `use_env_server=False` 时（默认）：
  - 保持现有行为（向后兼容）
  - 启动完整的仿真环境

**伪代码**：
```python
class GazeboEnv:
    def __init__(self, launchfile, vehicle_type, vehicle_id, 
                 use_env_server=False, ros_port="11311"):
        
        if not use_env_server:
            # 原有逻辑：启动 roscore、Gazebo、YOLO
            subprocess.Popen(["roscore", "-p", ros_port])
            time.sleep(3)
            rospy.init_node("Landing_env", ...)
            self.gazebo_process = subprocess.Popen(["roslaunch", ...])
            self.start_yolo()
        else:
            # 新增逻辑：客户端模式，只连接
            rospy.loginfo("客户端模式：连接到现有 ROS Master")
            rospy.init_node("Landing_env_client", ...)
            time.sleep(2)
            # 其他通信模块照常初始化，但不启动外部进程
```

### 5.2 客户端脚本改动点

**train_test.py、evaluate_iros_new.py 中的改动**：
```python
# 添加参数
parser.add_argument('--use_env_server', type=bool, default=False,
                    help='是否使用环境服务器（True=客户端模式）')

# 在初始化环境时
env = GazeboEnv(
    launchfile=args.launch_file,
    vehicle_type="iris",
    vehicle_id="0",
    use_env_server=args.use_env_server  # ← 传入参数
)
```

### 5.3 优雅关闭处理

**env_server.py 中**：
- 使用 signal handler 捕捉 `Ctrl+C`
- 按顺序关闭 YOLO、Gazebo、roscore
- 确保不留下僵尸进程

**客户端脚本中**：
- 客户端退出时**不关闭** roscore 和 Gazebo
- 只清理自己的 ROS 节点和发布者/订阅者
- 这样其他客户端继续访问同一环境

---

## 六、使用流程示意

### 场景 1：对比两个模型的性能

```bash
# 终端 1：启动环境服务器（一次）
python env_server.py

# 等待 30 秒，确保环境完全启动

# 终端 2：评估模型 A（早期模型）
python evaluate_iros_new.py --use_env_server True --load_step 10000 --test_episodes 50

# 终端 3（并行运行）：评估模型 B（最终模型）
python evaluate_iros_new.py --use_env_server True --load_step 60000 --test_episodes 50

# 结果：两个模型在同一环境下的对比数据一起输出
```

**时间节省**：
- 原方法：50 + 50 = 100 个 episode，每个启动环境 15 秒 = 1500 秒总启动时间
- 新方法：只需 30 秒启动一次，100 个 episode 并行运行 = 节省 1470 秒！

---

### 场景 2：不同起始位置的鲁棒性测试

```bash
# 终端 1：启动环境
python env_server.py

# 终端 2-5：分别从四个方向进行降落
python train_test.py --use_env_server True --start_pos "north"
python train_test.py --use_env_server True --start_pos "south"
python train_test.py --use_env_server True --start_pos "east"
python train_test.py --use_env_server True --start_pos "west"

# 一次性收集四个方向的数据，对比不同初始位置的影响
```

---

### 场景 3：在线微调工作流

```bash
# 终端 1：启动环境
python env_server.py

# 终端 2：加载预训练模型并进行在线微调
python TD3_online_finetune.py --use_env_server True --load_step 60000

# 在微调过程中，权重不断更新，可以实时观察改进
# 环境一直在运行，连贯性好
```

---

## 七、关键 ROS 话题和服务

env_server 启动的 ROS 节点会发布/订阅以下话题：

**客户端需要订阅的话题**（获取传感器数据）：
- `/iris_0/mavros/local_position/velocity_local` → 无人机速度
- `/yolov11/centers` → YOLO 检测结果（目标中心坐标）
- `/gazebo/model_states` → 模型状态（位置、速度）

**客户端需要发布的话题**（发送控制命令）：
- `/xtdrone/iris/cmd_vel_flu` → 速度命令
- `/xtdrone/iris/cmd` → 文本命令（起飞、着陆等）

**客户端需要调用的服务**（控制仿真）：
- `/gazebo/pause_physics` → 暂停物理引擎
- `/gazebo/unpause_physics` → 恢复物理引擎
- `/gazebo/reset_world` → 重置仿真场景

---

## 八、文件结构总结

```
Landing_new/
├── env_server.py                    ← 环境服务器（启动一次）
├── TD3-main/
│   ├── landing_env.py               ← 改进版（支持 use_env_server 参数）
│   ├── train_test.py                ← 改进版（加 --use_env_server 参数）
│   ├── evaluate_iros_new.py         ← 改进版（加 --use_env_server 参数）
│   ├── TD3_online_finetune.py       ← 改进版（加 --use_env_server 参数）
│   └── ...（其他脚本）
└── doc/
    └── PERSISTENT_ENV_PLAN.md       ← 本文档
```

---

## 九、优势总结

| 方面 | 原方法 | 新方法 |
|------|-------|-------|
| **环境启动** | 每个脚本启动一次 | 共享一个环境 |
| **总启动时间** | N × 15 秒 | 1 × 30 秒 |
| **并行能力** | 受限 | 支持多个客户端同时运行 |
| **脚本灵活性** | 改变脚本参数需要重启环境 | 快速切换脚本无需等待 |
| **调试方便性** | 低（每次重启浪费时间） | 高（保持环境连贯） |
| **对比实验** | 不同环境状态的对比 | 同一环境状态的严格对比 |

---

## 十、实现建议

**第 1 阶段**（必须做）：
1. 创建 `env_server.py`
2. 修改 `landing_env.py` 添加 `use_env_server` 参数
3. 修改脚本参数传递

**第 2 阶段**（优化）：
1. 添加环境参数配置文件，支持动态改变风速、目标速度等
2. 创建统一的"客户端模板"供其他脚本参考
3. 添加错误处理和自动重连机制

**第 3 阶段**（增强）：
1. 实现客户端间的"协调"机制（例如多个客户端知道彼此的存在）
2. 添加环境状态的实时监控面板
3. 支持"环境快照"和"还原"功能（保存/加载特定仿真状态）

---

## 十一、常见问题

**Q1: 两个客户端同时改变环境参数会不会冲突？**
A: 会。需要添加"锁机制"或"队列机制"来协调。建议每个 episode 结束后才允许改变参数。

**Q2: 如何在环境服务器运行时动态改变参数？**
A: 可以通过 ROS Service 或 ROS Parameter Server 实现，但现有代码未支持，需要扩展。

**Q3: 客户端异常退出会不会影响环境？**
A: 不会。只是该客户端的节点断开，环境继续运行。其他客户端不受影响。

**Q4: 如何监控环境服务器的运行状态？**
A: 可以添加一个"监控脚本"订阅特定话题，实时打印环境信息。

---

## 十二、后续扩展思路

1. **Web 界面**：提供实时查看无人机位置、轨迹的可视化
2. **多机支持**：同一环境中运行多个无人机
3. **参数配置系统**：通过 YAML 文件配置不同的测试场景
4. **自动化测试套件**：定义一组测试用例，自动运行并输出报告
5. **性能监测**：记录 CPU/GPU 使用率、通信延迟等指标

