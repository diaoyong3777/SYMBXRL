'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks"
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import sys
import os
# 获取脚本所在目录的两级上级目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)  # 将项目根目录添加到Python路径
from constants import PROJ_ADDR  # 从常量文件导入项目地址
import numpy as np
import gymnasium as gym
from DQNAgent import *  # 导入DQN智能体类
import h5py
import time
import torch

# 再次设置路径，指向A2-MIMOResourceScheduler目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # 上一级目录：A2-MIMOResourceScheduler
sys.path.insert(0, parent_dir)
from custom_mimo_env import MimoEnv  # 导入自定义的MIMO环境
import matplotlib.pyplot as plt

# 设置设备：如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
# 读取HDF5格式的信道数据集文件
H_file = h5py.File(f'{PROJ_ADDR}/A2-MIMOResourceScheduler/Datasets/LOS_highspeed2_64_7.hdf5', 'r')
H = np.array(H_file.get('H'))  # 获取信道矩阵数据（64天线×7用户）
se_max_ur = np.array(H_file.get('se_max'))  # 获取最大频谱效率数据

# 初始化环境和智能体
env = MimoEnv(H, se_max_ur)  # 创建Massive MIMO环境实例
print('Environment initialized')
# 创建DQN智能体实例，参数说明：
# alpha=0.0003: 学习率
# input_dims=21: 输入维度（7用户×3特征=21）
# n_actions=127: 动作空间大小（7个用户的所有非空组合）
# batch_size=256: 批处理大小
# device: 计算设备（GPU/CPU）
agent = DQNAgent(alpha=0.0003, input_dims=21, n_actions=127, batch_size=256, device=device)

# 加载预训练模型
# 从文件加载已经训练好的DQN模型权重
agent.load_model(f'{PROJ_ADDR}/A2-MIMOResourceScheduler/models/DQN_956.59_300_dtLOS_HS2_final.pth')

# 评估模型性能
observation, info = env.reset()  # 重置环境，获取初始状态
done = False  # 回合结束标志
score = 0  # 累计总奖励
step_rewards = []  # 存储每一步的奖励值
mean_rew = []  # 存储平均奖励值（用于绘图）

# 测试循环：在环境中运行智能体直到回合结束【智能体和环境交互】
while not done:
    # 智能体根据当前观察选择动作（np.squeeze移除单维度）
    action = agent.choose_action(np.squeeze(observation))
    # 在环境中执行动作，获取下一步状态、奖励等信息
    next_obs, reward, done, _, info = env.step(action)
    score += reward  # 累加总奖励
    step_rewards.append(reward)  # 记录当前步奖励

    # 计算到当前步为止的平均奖励
    mean_reward = np.mean(step_rewards)
    mean_rew.append(mean_reward)

    # 打印回合信息
    test_print = f'Step: {info["current_step"]} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score:.3f}\n'
    print(test_print)

    observation = next_obs  # 更新当前状态为下一步状态

# 可视化结果
plt.figure(figsize=(10, 5))  # 创建图形窗口
plt.plot(step_rewards, label='Step Reward')  # 绘制每一步的奖励曲线
plt.plot(mean_rew, label='Mean Reward')  # 绘制平均奖励曲线
plt.xlabel('Steps')  # x轴标签：步数
plt.ylabel('Reward')  # y轴标签：奖励值
plt.title('DQN Agent Performance')  # 图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图表