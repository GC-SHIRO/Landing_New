"""
用途:
- 在仿真环境中采集专家演示数据(在线飞行轨迹), 以 JSONL 形式写入离线训练数据集。

用法:
- python -m scripts.train_listen --max_episodes N
- 输出文件默认为 expert_data_lstm_left_up.json(每行一个 episode)。

实现方式:
- 每步读取飞控速度作为专家动作, 与环境交互后记录 (s, a, r, s', done)。
- 回合结束按条件筛选(成功且长度足够)并追加写盘, 同时回填 episode 总奖励。

依赖关系:
- 依赖 Simulation/env/landing_env_listen.py 提供 GazeboEnv 交互接口。
- 输出数据通常作为 scripts/train_offline.py 的训练输入。
"""
import numpy as np
import argparse
import json
import os
import time
from pathlib import Path

# 确保这里引用的环境是你修改过那个带 Vision Blocking 的版本
from Simulation.env.landing_env_listen import GazeboEnv

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
# 状态维度 3 (位置)
parser.add_argument('--state_dim', default=3, type=int) 
parser.add_argument('--action_dim', default=3, type=int)
parser.add_argument('--max_action', default=1, type=float)
parser.add_argument('--capacity', default=16384, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=1, type=int)
# ... 其他参数保持默认即可 ...
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--tau',  default=0.005, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--max_episode', default=2000, type=int)

args = parser.parse_args()

# ==========================================
# 1. 设置保存路径
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
save_dir = PROJECT_ROOT / 'data' / 'expert_data'
expert_data_file = save_dir / 'expert_data_lstm_left_up.json'

def main():
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化环境
    env = GazeboEnv("/home/shiro/PX4_Firmware/launch/sandisland.launch", "iris", '0')

    # 注意：在追加模式下，我们不需要全局列表 all_episodes_data 了
    print(f"===== 开始采集 =====")
    print(f"数据将追加保存至: {expert_data_file}")
    print(f"筛选条件: 长度>=15 且 降落成功(Success=True)")

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        succes = False # 注意：环境中的变量名是 succes (少一个s)
        
        # 重置环境
        observation = env.reset()
        
        # 当前回合的临时数据列表
        current_episode_data = []
        
        # 打印回合开始
        print(f"Episode {episode}: 开始飞行...", flush=True)
        
        while not done:
            # === 防御性编程：防止刚启动时速度为 None 报错 ===
            vx, vy, vz = 0.0, 0.0, 0.0
            if env.drone_linear_velocity is not None:
                vx = env.drone_linear_velocity.x
                vy = env.drone_linear_velocity.y
                vz = env.drone_linear_velocity.z
            
            # 获取专家动作 (飞控的速度指令)
            action = np.array([vx, vy, vz])
            
            # 执行动作
            observation_, done, succes, info = env.step(action)
            
            # 计算这一步的奖励 (Reward Step)
            reward = env.reward_setup(observation, observation_, done, succes)
            
            # === 保存单步数据 ===
            step_data = {
                'observation': observation.tolist(),
                'action': action.tolist(),
                'reward': reward,                 # <--- 这一步的奖励
                'next_observation': observation_.tolist(),
                'done': done,
                'episode_final_reward': 0.0       # <--- 先填0占位，等回合成功结束后回填
            }
            current_episode_data.append(step_data)

            total_reward += reward
            observation = observation_

        # ==========================================
        # 2. 回合结束：筛选 & 保存
        # ==========================================
        
        # 筛选条件：长度足够 (LSTM需要) AND 降落成功 (只学好的)
        if len(current_episode_data) >= 15 and succes:
            
            # [核心逻辑]：把计算好的 total_reward 填入这一回合的所有步骤中
            for step in current_episode_data:
                step['episode_final_reward'] = total_reward
            
            print(f">>> [保存] 回合 {episode} | 长度: {len(current_episode_data)} | 总分: {total_reward:.2f} | 结果: 成功")
            
            # [核心修改]：追加写入文件 (Append Mode)
            # 每一行都是一个完整的 List (代表一个回合)
            try:
                with expert_data_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(current_episode_data) + "\n") 
            except Exception as e:
                print(f"[Error] 写入文件失败: {e}")
                
        else:
            # 打印丢弃原因
            reason = "未知"
            if len(current_episode_data) < 15:
                reason = "数据过短"
            elif not succes:
                reason = "降落失败"
                
            print(f">>> [丢弃] 回合 {episode} | 长度: {len(current_episode_data)} | 原因: {reason}")

if __name__ == '__main__':
    main()
