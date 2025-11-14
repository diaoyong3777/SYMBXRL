'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import os
import sys

# 获取脚本所在目录的两级上级目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)  # 将项目根目录添加到Python路径
from constants import PROJ_ADDR  # 从常量文件导入项目地址
import numpy as np
import gymnasium as gym
import h5py
from SACArgs import SACArgs  # 导入SAC算法的参数配置类
from sac import SAC  # 导入Soft Actor-Critic算法实现
from replay_memory import ReplayMemory  # 导入经验回放内存
from smartfunc import sel_ue  # 导入用户选择函数（动作解码）
import torch
import time

# 再次设置路径，指向A2-MIMOResourceScheduler目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # 上一级目录：A2-MIMOResourceScheduler
sys.path.insert(0, parent_dir)
from custom_mimo_env import MimoEnv  # 导入自定义的MIMO环境
import os

# 设置设备：如果有GPU则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 加载数据集
# 读取HDF5格式的信道数据集文件（LOS_highspeed2场景）
H_file = h5py.File(f'{PROJ_ADDR}/A2-MIMOResourceScheduler/Datasets/LOS_highspeed2_64_7.hdf5', 'r')
H = np.array(H_file.get('H'))  # 获取信道矩阵数据（64天线×7用户）
se_max_ur = np.array(H_file.get('se_max'))  # 获取最大频谱效率数据
print('Data loaded successfully')

# 设置训练参数
max_episode = None  # 如果输入无效则默认为300
args = SACArgs(H, max_episode=max_episode)  # 创建SAC算法参数对象

# 设置随机种子以确保结果可重现
torch.manual_seed(args.seed)  # 设置PyTorch随机种子
np.random.seed(args.seed)  # 设置NumPy随机种子

# 初始化环境
env = MimoEnv(H, se_max_ur)  # 创建Massive MIMO环境实例
print('Environment initialized')

# 获取环境参数
num_states = env.observation_space.shape[0]  # 状态空间维度（21）
num_actions = len([env.action_space.sample()])  # 动作空间维度（1）
max_actions = env.action_space.n  # 最大动作数量（127）

# 初始化SAC智能体
# 参数说明：状态维度，动作维度，最大动作数，参数配置，学习率，alpha学习率
agent = SAC(num_states, num_actions, max_actions, args, args.lr, args.alpha_lr)
memory = ReplayMemory(args.replay_size, args.seed)  # 初始化经验回放内存

# 加载预训练模型
agent.load_checkpoint(f'{PROJ_ADDR}/A2-MIMOResourceScheduler/models/SACG_884.53_551_dtLOS_HS2_checkpointed.pth_')
print('SAC build finished')

# 评估模型性能
print(
    "###############################################################EVALUATION STARTS ############################################################################################################")
print("Evaluation started...")

# 初始化评估指标
step_rewards = []  # 存储每一步的奖励值
acn_str = []  # 存储动作策略信息（可选）
grp_str = []  # 存储分组策略信息（可选）
mean_rew = []  # 存储平均奖励值

# 重置环境，开始新的测试回合
observation, info = env.reset()  # 获取初始状态和环境信息
done = False  # 回合结束标志
score = 0  # 累计总奖励

# 回合循环：在环境中运行智能体直到回合结束【交互】
while not done:
    # 智能体根据当前观察选择动作
    # action: 原始动作表示, final_action: 最终选择的动作（离散动作编号）
    action, final_action = agent.select_action(observation)

    # 将动作解码为选择的用户索引
    ue_select, idx = sel_ue(final_action[0])  # ue_select: 选择的用户列表, idx: 选择的用户数量

    # 在环境中执行动作，获取下一步状态、奖励等信息
    next_obs, reward, done, _, info = env.step(final_action[0])

    # 更新分数和奖励记录
    score += reward  # 累加总奖励
    step_rewards.append(reward)  # 记录当前步奖励

    # 计算到当前步为止的平均奖励
    mean_reward = np.mean(step_rewards)
    mean_rew.append(mean_reward)

    # 打印评估信息（使用\r实现进度条效果）
    test_print = f'Step: {info["current_step"]} / {env.total_steps - 1} |Action taken: {ue_select} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score:.3f}'
    print(test_print, end='\r')  # 使用回车符覆盖上一行输出

    observation = next_obs  # 更新当前状态为下一步状态