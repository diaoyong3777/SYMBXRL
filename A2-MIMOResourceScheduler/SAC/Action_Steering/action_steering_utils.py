'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import ast
import random
import numpy as np
import pandas as pd

def extract_decision_from_suggested(suggested_decision):
    # # 符号建议来自决策图分析
    # suggested_decision = [
    #     "[[0, 2], [1, 3, 4, 5, 6]]",  # 分组0建议调度用户0,2
    #     "[[1, 3], [0, 2, 4, 5, 6]]",  # 分组1建议调度用户1,3
    #     "[[4, 5], [0, 1, 2, 3, 6]]"  # 分组2建议调度用户4,5
    # ]
    #
    # # 提取后：
    # extracted_decision = (0, 1, 2, 3, 4, 5)  # 所有被调度的用户
    """
    从建议决策列表中提取并排序决策。
    该函数接收一个建议决策列表，其中每个决策可以是列表的字符串表示或列表本身。
    它将任何字符串表示转换为列表，从每个列表中提取第一个元素，然后排序并返回这些元素作为元组。

    Args:
        suggested_decision (list): 建议决策列表，每个决策是列表的字符串表示或列表本身

    Returns:
        tuple: 提取的决策的排序元组
    """
    extracted_decision = []
    for item in suggested_decision:
        converted_decision = ast.literal_eval(item) if type(item) == str else item
        extracted_decision.extend(converted_decision[0])  # 提取调度用户列表

    return tuple(sorted(extracted_decision))  # 返回排序后的元组

# 【IAS】
def do_action_steering_this_timestep(curr_state_df, history_df, rt_decision_graph):
    """
    基于当前状态、历史和实时决策图执行当前时间步的动作引导。

    Parameters:
    curr_state_df (pd.DataFrame): 包含当前状态信息的数据框
    history_df (pd.DataFrame): 包含历史状态信息的数据框
    rt_decision_graph (dict): 包含每个分组实时决策图的字典

    Returns:
    tuple: 包含以下内容的元组：
        - pd.Series: 动作引导时间步的调度成员
        - float: 动作引导时间步的奖励
        如果未找到共同时间步，返回 (False, None)
    """
    # 获取上一个时间步的状态
    prev_state_df = history_df[history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]]
    groups = curr_state_df['group'].unique()  # 获取所有分组
    common_timesteps = set(history_df['timestep'])  # 初始化共同时间步集合

    # 遍历每个分组
    for group in groups:
        if not prev_state_df[prev_state_df['group'] == group].empty:
            # 获取该分组的决策图
            G = rt_decision_graph[group].get_graph(mode="networkX")
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]  # 上一个决策
            neighbors = list(G.neighbors(node_id))  # 获取邻居节点（可能的下一步决策）

            # 收集邻居及其平均奖励
            neighbors_with_mean_rewards = []
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node_id, neighbor)
                mean_reward = edge_data.get('mean_reward', 0)  # 获取转移的平均奖励
                neighbors_with_mean_rewards.append((neighbor, mean_reward))

            # 按平均奖励降序排序邻居
            neighbors_with_mean_rewards.sort(key=lambda x: x[1], reverse=True)

            # 获取前3个邻居（如果少于3个则取全部）
            top_neighbors = neighbors_with_mean_rewards[:3]

            # 构建动作列表：前3个最推荐动作 + 当前动作
            actions = [x[0] for x in top_neighbors] + [curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]]

            # 过滤具有相同状态且智能体做出相同决策的先前时间步
            conditioned_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) &
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) &
                (history_df['decision'].isin(actions))  # 决策在候选动作中
            )]['timestep'])

            common_timesteps &= conditioned_timesteps  # 取交集
        else:
            # 过滤共同时间步中包含该分组的时间步
            group_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) &
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
            )]['timestep'])
            common_timesteps &= group_timesteps

        # 如果没有共同时间步，提前返回
        if len(common_timesteps) == 0:
            return False, None

    # 选择奖励最高的时间步
    action_steered_timestep = history_df[history_df['timestep'].isin(common_timesteps)].groupby('timestep').first().reset_index().nlargest(1, 'reward')['timestep'].iloc[0]
    return history_df[history_df['timestep'] == action_steered_timestep]['sched_members'], history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]

def softmax(x):
    """计算softmax函数，将奖励转换为概率分布"""
    e_x = np.exp(x - np.max(x))  # 数值稳定性处理
    return e_x / e_x.sum()

def do_action_steering_this_timestep_randomized(curr_state_df, history_df, rt_decision_graph, agent_expected_reward):
    """
    基于历史数据和随机化方法执行当前时间步的动作引导。

    Parameters:
    curr_state_df (pd.DataFrame): 包含当前时间步信息的当前状态数据框
    history_df (pd.DataFrame): 包含先前时间步信息的历史数据框
    rt_decision_graph (dict): 每个分组的决策图字典，每个图以networkX格式表示
    agent_expected_reward (float): 智能体的预期奖励

    Returns:
    tuple: 包含以下内容的元组：
        - sched_members (pd.Series): 引导动作时间步的调度成员
        - reward (float): 与引导动作时间步相关的奖励
    如果未找到合适的时间步，返回 (False, None)
    """
    prev_state_df = history_df[history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]]

    groups = curr_state_df['group'].unique()
    common_timesteps = set(history_df['timestep'])

    # 与第一个函数相同的过滤逻辑
    for group in groups:
        if not prev_state_df[prev_state_df['group'] == group].empty:
            G = rt_decision_graph[group].get_graph(mode="networkX")
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]
            neighbors = list(G.neighbors(node_id))

            neighbors_with_mean_rewards = []
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node_id, neighbor)
                mean_reward = edge_data.get('mean_reward', 0)
                neighbors_with_mean_rewards.append((neighbor, mean_reward))

            neighbors_with_mean_rewards.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = neighbors_with_mean_rewards[:3]
            actions = [x[0] for x in top_neighbors] + [curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]]

            conditioned_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) &
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) &
                (history_df['decision'].isin(actions))
            )]['timestep'])

            common_timesteps &= conditioned_timesteps
        else:
            group_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) &
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
            )]['timestep'])
            common_timesteps &= group_timesteps

        if len(common_timesteps) == 0:
            return False, None

    # 过滤奖励优于或等于智能体预期奖励的时间步
    better_timesteps = history_df[(history_df['timestep'].isin(common_timesteps)) & (history_df['reward'] >= agent_expected_reward)]

    if better_timesteps.empty:
        return False, None

    # 收集动作及其奖励
    action_rewards = better_timesteps.groupby('decision', as_index=False)['reward'].mean()

    # 对奖励应用softmax计算权重
    action_rewards['weight'] = softmax(action_rewards['reward'])

    # 基于权重随机选择动作
    chosen_action = random.choices(
        population=action_rewards['decision'].tolist(),
        weights=action_rewards['weight'].tolist(),
        k=1
    )[0]

    action_steered_timestep = better_timesteps[better_timesteps['decision'] == chosen_action].sort_values(by='reward', ascending=False).iloc[0]['timestep']

    return history_df[history_df['timestep'] == action_steered_timestep]['sched_members'], history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]


def transform_action(action, high=1, low=-1, tot_act=127):
    """将连续动作转换为离散动作编号"""
    k = (high - low) / (tot_act - 1)
    return round((action - low) / k)

def process_buffer(buff, transform_action, sel_ue, mode, timestep=0, agent_type='SAC'):
    """
    处理转换缓冲区并返回两个数据框：一个用于状态，一个用于动作和奖励。

    Args:
        buff (list): 转换列表，每个转换是元组 (state, action)
        transform_action (function): 如果代理类型为'SAC'，用于转换动作的函数
        sel_ue (function): 从动作中选择用户设备(UE)的函数
        mode (str): 处理模式，'buffer'或其他模式
        timestep (int, optional): 如果模式不是'buffer'，使用的时间步，默认为0
        agent_type (str, optional): 代理类型，默认为'SAC'

    Returns:
        tuple: 包含两个数据框的元组：
            - states_df (pd.DataFrame): 包含处理状态的数据框
            - actions_rewards_df (pd.DataFrame): 包含处理动作和奖励的数据框
    """
    # 状态列定义
    buff_state_columns = ["MSEUr0", "MSEUr1", "MSEUr2", "MSEUr3", "MSEUr4", "MSEUr5", "MSEUr6",
                 "DTUr0", "DTUr1", "DTUr2", "DTUr3", "DTUr4", "DTUr5", "DTUr6",
                 "UGUr0", "UGUr1", "UGUr2", "UGUr3", "UGUr4", "UGUr5", "UGUr6"]

    buff_states = []
    buff_actions_rewards = []

    # 处理每个转换
    for transition in buff:
        state, action = transition

        state_1d = state.flatten()  # 展平状态
        buff_states.append(state_1d)

        action_reward = [action[0]]
        buff_actions_rewards.append(action_reward)

    # 创建状态数据框
    states_df = pd.DataFrame(buff_states, columns=buff_state_columns)

    # 添加时间步列
    if mode == 'buffer':
        states_df['timestep'] = states_df.index + 1
    else:
        states_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in states_df.columns if col != 'timestep']
    states_df = states_df[cols]

    # 创建动作奖励数据框
    actions_rewards_df = pd.DataFrame(buff_actions_rewards, columns=["action"])
    if agent_type == 'SAC':
        actions_rewards_df["action"] = actions_rewards_df["action"].apply(transform_action)  # 转换SAC动作
    actions_rewards_df["action"] = actions_rewards_df["action"].apply(lambda x: sel_ue(x)[0])  # 解码为用户选择
    if mode == 'buffer':
        actions_rewards_df['timestep'] = actions_rewards_df.index + 1
    else:
        actions_rewards_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in actions_rewards_df.columns if col != 'timestep']
    actions_rewards_df = actions_rewards_df[cols]

    return states_df, actions_rewards_df