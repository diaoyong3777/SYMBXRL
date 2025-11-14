'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

########################### IMPORTS ######################################################
import numpy as np
import gymnasium as gym
import random
from collections import Counter
import numpy.matlib
from itertools import combinations

########################### OPEN AI GYM CLASS DEFINATIONS ######################################################

# 定义Massive MIMO环境类，继承自gym.Env
class MimoEnv(gym.Env):
    # 初始化Massive MIMO环境
    def __init__(self, H, se_max):
        super(MimoEnv, self).__init__()
        """
        Initialize the 7 User MIMO environment.
        Args:
            H (numpy.ndarray): Channel matrix(CSI). 信道状态信息矩阵
            se_max (numpy.ndarray): Maximum achievable spectral efficiency of 7 users. 7个用户的最大可达频谱效率
        """
        self.H = H  # 信道矩阵
        self.se_max = se_max  # 最大频谱效率
        self.num_ue = H.shape[2]  # 用户数量，从信道矩阵的第三维获取
        self.current_step = 0  # 当前时间步
        self.total_steps = H.shape[0]  # 总时间步数，从信道矩阵的第一维获取
        ue_history = np.zeros((H.shape[2], ))  # 用户历史数据，初始化为零
        self.ue_history = ue_history
        self.obs_state = []  # 观察状态列表
        self.usrgrp_cntr = []  # 用户分组计数器
        action_space_size = 127  # 0 to 126, inclusive，动作空间大小，对应7个用户的所有可能组合
        self.action_space = gym.spaces.Discrete(action_space_size)  # 定义离散动作空间

        # 定义观察空间：每个用户有3个状态变量（频谱效率、历史数据、分组索引），共7个用户
        low = np.array([-np.inf, 0, 0] * 7)  # 状态变量的最小值
        high = np.array([np.inf, np.inf, 6] * 7)  # 状态变量的最大值
        self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float64)

        self.total_reward = None  # 总奖励
        self.history = None  # 历史记录

    # 重置环境到初始状态
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Reset the environment to the initial state.
        Returns:
            numpy.ndarray: Initial observation/state. 初始观察状态
            dict: Information about the environment. 环境信息字典
        """
        self.current_step = 0  # 重置当前步数
        self.total_reward = 0  # 重置总奖励
        self.history = {}  # 重置历史记录
        self.jfi = 0  # 重置Jain公平指数
        self.sys_se = 0  # 重置系统频谱效率
        # 根据当前信道状态对用户进行分组
        group_idx = usr_group(np.squeeze(self.H[self.current_step,:,:]))
        self.usrgrp_cntr.append(group_idx)  # 记录用户分组
        self.ue_history = np.zeros((7,))  # 重置用户历史数据
        # 构建初始状态：包含频谱效率、用户历史数据和分组索引
        # 假设有7个用户的状态示例
        # state_example = [
        #     # 频谱效率部分 (用户0-6的最大频谱效率)
        #     12.5, 8.3, 15.2, 6.7, 10.1, 9.8, 11.4,
        #
        #     # 用户历史部分 (用户0-6的累计频谱效率)
        #     45.2, 32.1, 58.7, 21.3, 39.8, 36.5, 42.9,
        #
        #     # 用户分组部分 (用户0-6的分组索引)
        #     0, 1, 0, 2, 1, 0, 2
        # ]
        initial_state = np.concatenate((np.reshape(self.se_max[self.current_step,:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        # initial_state = initial_state.flatten()
        self.obs_state.append(initial_state)  # 记录观察状态
        info = self.getinfo()  # 获取环境信息
        return initial_state, info

    # 在环境中执行一步动作
    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action: Action taken by the agent. 智能体选择的动作
        Returns:
            numpy.ndarray: Next observation/state. 下一个观察状态
            float: Reward received for the action. 动作获得的奖励
            bool: Whether the episode is finished. 是否结束
            dict: Information about the environment. 环境信息字典
        """
        # 将动作转换为选择的用户索引
        ue_select, idx = sel_ue(action)
        mod_select = np.ones((idx,)) * 4  # 设置调制方式为QPSK
        # 处理数据，计算频谱效率等指标
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[self.current_step,:,ue_select],(64,-1)),idx,mod_select)
        # 计算奖励并更新用户历史
        reward, self.ue_history, jfi, sys_se = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[self.current_step], self.se_max[self.current_step])
        self.jfi = jfi  # 更新Jain公平指数
        self.sys_se = sys_se  # 更新系统频谱效率
        self.total_reward += reward  # 累加总奖励
        self.current_step += 1  # 增加当前步数
        done_pm = self.total_steps - 1  # 计算结束步数
        done = self.current_step >= done_pm  # 判断是否结束
        truncated = False  # 截断标志（用于早期终止）
        # 更新用户分组
        group_idx = usr_group(np.squeeze(self.H[(self.current_step),:,:]))
        self.usrgrp_cntr.append(group_idx)
        # 构建下一个状态
        next_state = np.concatenate((np.reshape(self.se_max[(self.current_step),:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        # next_state = next_state.flatten()
        self.obs_state.append(next_state)  # 记录观察状态
        info = self.getinfo()  # 获取环境信息
        history = self.update_history(info)  # 更新历史记录
        return next_state, reward, done, truncated, info

    # 计算给定动作的奖励（不推进环境）
    def get_reward(self, action):
        """
        Calculate the reward for the given action without advancing the environment.
        Args:
            action: Action taken by the agent. 智能体选择的动作
        Returns:
            float: Reward for the action. 动作的奖励值
        """
        # 保存当前状态
        current_step = self.current_step
        ue_history = self.ue_history.copy()

        # 计算奖励
        ue_select, idx = sel_ue(action)
        mod_select = np.ones((idx,)) * 4
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[current_step, :, ue_select], (64, -1)), idx, mod_select)
        reward, _, _, _ = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[current_step], self.se_max[current_step], se_noise=True)

        # 恢复状态
        self.current_step = current_step
        self.ue_history = ue_history

        return reward

    # 设置环境到特定状态
    def set_state(self, state):
        """
        Set the environment to a specific state.
        Args:
            state (numpy.ndarray): State to set the environment to. 要设置的环境状态
        """
        # 从状态中提取频谱效率
        spectral_efficiencies = state[:self.num_ue]
        # 在se_max中查找匹配的行
        tolerance = 1  # 容差
        # 在se_max_ur中查找与next_obs2匹配的行索引（最多2位小数）
        row_index = -1
        for idx, row in enumerate(self.se_max):
            if np.all(np.isclose(row, spectral_efficiencies, atol=tolerance)):
                row_index = idx
                break
        if row_index == -1:
            raise ValueError("The provided state does not match any row in se_max.")

        self.current_step = row_index  # 设置当前步数
        # 根据提供的状态设置用户历史
        self.ue_history = state[self.num_ue:self.num_ue*2]
        # 更新分组索引历史
        group_idx = state[self.num_ue*2:]
        if len(self.usrgrp_cntr) > self.current_step:
            self.usrgrp_cntr[self.current_step] = group_idx
        else:
            self.usrgrp_cntr.append(group_idx)

        # 创建初始状态
        initial_state = np.concatenate(
            (np.reshape(self.se_max[self.current_step, :], (1, self.num_ue)),
             np.reshape(self.ue_history, (1, self.num_ue)),
             np.reshape(group_idx, (1, -1))),
            axis=1
        )
        self.obs_state.append(initial_state)  # 记录观察状态
        info = self.getinfo()  # 获取环境信息

    # 获取环境信息
    def getinfo(self):
        """
        Get information about the environment.
        Returns:
            dict: Information about the environment. 包含环境信息的字典
        """
        return dict(current_step = self.current_step, NSSE = self.sys_se, JFI = self.jfi)

    # 更新环境历史记录
    def update_history(self, info):
        """
        Update the history of the environment.
        Args:
            info (dict): Information about the environment. 环境信息字典
        """
        if not self.history:
            self.history = {key: [] for key in info.keys()}  # 初始化历史记录字典
        for key, value in info.items():
            self.history[key].append(value)  # 添加新的历史记录

    # 计算奖励值【注：会推进环境】
    def calculate_reward(self, ur_se_total, ur_min_snr, ur_se, ue_select, idx, usrgrp, semax, se_noise = False):
        """
        Calculate the reward based on the received spectral efficiency.
        Args:
            ur_se_total (float): Total spectral efficiency. 总频谱效率
            ur_min_snr (float): Minimum signal-to-noise ratio. 最小信噪比
            ur_se (numpy.ndarray): Spectral efficiency for each user. 每个用户的频谱效率
            ue_select (int): Selected user index. 选择的用户索引
            idx (int): Number of selected users. 选择的用户数量
            usrgrp (int): User group index. 用户分组索引
            semax (numpy.ndarray): Maximum achievable spectral efficiency. 最大可达频谱效率
        Returns:
            float: Calculated reward. 计算出的奖励值
            numpy.ndarray: Updated user history. 更新后的用户历史
        """
        beta = 0.5 # reward weight，奖励权重系数
        bin_act = transform_input_to_output(ue_select, 7) # Converting Action to Binary Encoding，将动作转换为二进制编码
        usrgrp2 = usrgrp + 1  # 用户分组索引加1（避免0值）
        sel = usrgrp2 * bin_act  # 计算选择的分组
        non_zero_elements = sel[sel != 0]  # 获取非零元素
        ue_select = np.array(ue_select)  # 确保ue_select是numpy数组
        sum_semax = np.sum(semax)  # 计算最大频谱效率总和
        Norm_Const = 1.15  # 归一化常数
        if se_noise:
            # 调整频谱效率考虑干扰
            ur_se, ur_se_total = adjust_se_interfernce(non_zero_elements, ur_se, ur_se_total, usrgrp, ue_select)
        # 奖励计算
        ur_se_total = ur_se_total / (sum_semax*Norm_Const) # Normalizing due to Randomization，归一化处理
        for i in range(0,idx):
            self.ue_history[ue_select[i]] += ur_se[i]  # 更新用户历史数据
        # 计算Jain公平指数【0~1,1：均相同，绝对公平】
        jfi = np.square((np.sum(self.ue_history))) / (7 * np.sum(np.square(self.ue_history)))
        # 计算最终奖励：频谱效率和公平性的加权组合
        reward  = round((beta*ur_se_total) + ((1-beta)*jfi), 3)
        return reward, self.ue_history, jfi, ur_se_total

    def render(self, mode = 'human'):
        pass  # 渲染环境（未实现）

    def __call__(self):
        # Implement the __call__ method to make the class callable，使类可调用
        return self

    def close(self):
        pass  # 关闭环境（未实现）


########################### IMPORTED FUNCTION BLOCKS ######################################################
'''
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023
'''

'''以下是Massive MIMO物理层仿真代码、还有一些工具函数，对于理解整个系统非常重要，但如果你主要关注SYMBXRL的DRL和可解释性部分，可以暂时不用深入理解所有细节'''

# 将输入动作序列转换为二进制编码的输出序列
def transform_input_to_output(input_sequence, total_variables):
    """
    Transform input action to binary coded output action.
    Args:
        input_sequence (list): Input sequence [1,3,4]. 输入的用户索引序列
        total_variables (int): Total number of users 7. 总用户数
    Returns:
        list: Output sequence  [0 1 0 1 1 0 0]. 二进制编码的输出序列
    """
    output_sequence = [0] * total_variables  # 初始化输出序列为全零
    for index in input_sequence:
        if index < total_variables:
            output_sequence[index] = 1  # 如果索引在输入序列中，设置对应位置为1
    return output_sequence

# 根据最大出现次数转换数组
# 在数组中找到出现次数最多的非零数字，然后创建一个新数组，只有这个数字的位置标记为1，其他位置都标记为0。
def transform_array(arr):
    """
    Transform the array based on maximum occurrences.
    Args:
        arr (list): Input array. 输入数组
    Returns:
        list: Transformed array. 转换后的数组
    """
    counts = {}  # 计数字典
    result = []  # 结果列表
    for num in arr:
        if num != 0:
            counts[num] = counts.get(num, 0) + 1  # 统计非零数字的出现次数
    max_count = max(counts.values()) if counts else 0  # 最大出现次数
    max_occurrence_numbers = [num for num, count in counts.items() if count == max_count]  # 出现次数最多的数字
    chosen_number = random.choice(max_occurrence_numbers) if max_occurrence_numbers else 0  # 随机选择一个最多出现的数字
    for num in arr:
        if num == chosen_number and num != 0:
            result.append(1)  # 如果是选择的数字且非零，添加1
        else:
            result.append(0)  # 否则添加0
    return result

# 获取选中索引及其值
def get_selected_indices_and_values(arr):
    """
    Get selected indices and their values from the array.
    Args:
        arr (list): Input array. 输入数组
    Returns:
        tuple: Number of selected indices and their values. 选中索引的数量和值
    """
    selected_indices = [i for i, num in enumerate(arr) if num != 0]  # 获取所有非零元素的索引
    return len(selected_indices), selected_indices

# 统计数组中变量的出现次数
def count_occurrences(arr):
    """
    This function counts the maximum occurrence of a variable in an array.
    Args:
        arr: A list of integers. 整数列表
    Returns:
        A tuple containing the variable with the maximum occurrence and its count. 包含最大出现次数的变量及其计数的元组
    """
    counts = Counter(arr)  # 使用Counter统计出现次数
    max_value = max(counts.values())  # 最大出现次数
    max_variable = [var for var, count in counts.items() if count == max_value]  # 出现次数最多的变量
    max_indexes = [i for i, x in enumerate(arr) if x in max_variable]  # 这些变量的索引
    return max_variable[0], max_value, max_indexes

# 根据干扰调整频谱效率【组间干扰、组内干扰】
def adjust_se_interfernce(non_zero_elements, ur_se, ur_se_total, usrgrp, ue_select):
    """
    Adjust the spectral efficiency based on the interference.
    Args:
        non_zero_elements (list): Non-zero elements. 非零元素列表
        ur_se (numpy.ndarray): Spectral efficiency for each user. 每个用户的频谱效率
        ur_se_total (float): Total spectral efficiency. 总频谱效率
        usrgrp (int): User group index. 用户分组索引
        ue_select (int): Selected user index. 选择的用户索引
    Returns:
        numpy.ndarray: Adjusted spectral efficiency. 调整后的频谱效率
        float: Adjusted total spectral efficiency. 调整后的总频谱效率
    """
    intf_penalty = 0.5  # 干扰惩罚系数
    bonus_reward = [1.1, 1.2, 1.25]  # 奖励加成系数
    # 如果选择的用户来自不同分组，应用干扰惩罚
    if np.any(non_zero_elements != non_zero_elements[0]):
        ur_se = ur_se * intf_penalty  # 对每个用户的频谱效率应用惩罚
        ur_se_total = ur_se_total * intf_penalty  # 对总频谱效率应用惩罚
    else:
        # 如果选择的用户来自同一分组，根据分组情况应用奖励
        _,case,max_ind = count_occurrences(usrgrp)  # 统计用户分组情况
        all_ind = np.arange(0,7)  # 所有用户索引
        min_ind = np.setdiff1d(all_ind,max_ind)  # 不在主要分组中的用户索引
        if case == 7:  # 所有用户在同一分组
            ur_se = ur_se  # 不调整
            ur_se_total = ur_se_total  # 不调整
        elif case == 6:  # 6个用户在同一分组
            if np.all(ue_select) in max_ind:  # 如果选择的用户都在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))]  # 不调整
            else:  # 如果选择的用户不在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]  # 应用最大奖励
            ur_se_total = np.sum(ur_se)  # 重新计算总频谱效率
        elif case == 5:  # 5个用户在同一分组
            if np.all(ue_select) in max_ind:  # 如果选择的用户都在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[0]  # 应用最小奖励
            else:  # 如果选择的用户不在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]  # 应用最大奖励
            ur_se_total = np.sum(ur_se)  # 重新计算总频谱效率
        elif case == 4:  # 4个用户在同一分组
            if np.all(ue_select) in max_ind:  # 如果选择的用户都在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[1]  # 应用中等奖励
            else:  # 如果选择的用户不在主要分组中
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[1]  # 应用中等奖励
            ur_se_total = np.sum(ur_se)  # 重新计算总频谱效率
        else:  # 其他情况
            ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]  # 应用最大奖励
            ur_se_total = np.sum(ur_se)  # 重新计算总频谱效率
    return ur_se, ur_se_total

# 基于信道相关性对用户进行分组【信道相似的用户分到不同组】
def usr_group(H):
    """
    This function groups users based on the correlation of their channel vectors(CSI).
    Parameters:
    H (numpy.ndarray): A matrix of channel vectors where each column corresponds to a user and each row corresponds to a base station antenna. 信道向量矩阵
    Returns:
    numpy.ndarray: An array where each element represents the group index of the corresponding user. 用户分组索引数组
    """
    N_UE = 7  # Number of user equipment (UE)，用户设备数量
    num_bs = 64  # Number of base station antennas，基站天线数量
    ur_group = [[] for i in range(N_UE)]  # 初始化用户分组列表
    group_idx = np.zeros(N_UE)  # 初始化分组索引数组
    ur_group[0].append(0)  # 将第一个用户分配到第一个分组
    N_group = 1  # 初始分组数量
    corr_h = 0.5  # Correlation threshold for grouping users，用户分组的相关性阈值
    meet_all = 0  # 标志：用户是否满足分组中所有用户的相关性标准
    assigned = 0  # 标志：用户是否已被分配到分组
    # 从第二个用户开始遍历所有用户
    for i in range(1, N_UE):
        # 遍历所有现有分组
        for j in range(N_group):
            # 遍历当前分组中的所有用户
            for k in ur_group[j]:
                # 计算当前用户信道向量与分组中第k个用户信道向量的相关性
                g_i = np.matrix(np.reshape(H[:, i], (num_bs, 1))).getH()  # 当前用户信道向量的共轭转置
                corr = abs(np.dot(g_i, np.reshape(H[:, k], (num_bs, 1)))) / (np.linalg.norm(np.reshape(H[:, i], (num_bs, 1))) * np.linalg.norm(np.reshape(H[:, k], (num_bs, 1))))
                if corr > corr_h:
                    # 如果相关性超过阈值，跳出并尝试下一个分组
                    break
                else:
                    if k == ur_group[j][-1]:
                        # 如果当前用户满足分组中所有用户的相关性标准，设置meet_all标志
                        meet_all = 1
                    continue
            if meet_all == 1:
                # 如果用户满足分组中所有用户的相关性标准，将用户添加到该分组
                ur_group[j].append(i)
                meet_all = 0  # 重置meet_all标志
                assigned = 1  # 设置assigned标志
                break
            else:
                continue
        if assigned == 0:
            # 如果用户未被分配到任何现有分组，为用户创建新分组
            ur_group[N_group].append(i)
            N_group += 1  # 增加分组数量
        else:
            assigned = 0  # 重置assigned标志
    # 为每个用户分配分组索引
    for i in range(N_group):
        for j in ur_group[i]:
            group_idx[j] = i

    return group_idx

# 处理信道数据，计算频谱效率和信噪比【Massive MIMO物理层的仿真】
def data_process (H, N_UE, MOD_ORDER):
    """
    This function converts channel vectors to Spectral efficiency per user and SINR per user.
    Parameters:
    H (numpy.ndarray): A matrix of channel vectors where each column corresponds to a user and each row corresponds to a base station antenna. 信道向量矩阵
    N_UE: No of Users，用户数量
    MOD_ORDER: Modulation Order，调制阶数
    Returns:
    System Spectral Efficiency, SINR (all users), spectral effecincy (all users) 系统频谱效率，所有用户的信噪比，所有用户的频谱效率
    """
    # Waveform params，波形参数
    N_OFDM_SYMS             = 24  # Number of OFDM symbols，OFDM符号数量
    # MOD_ORDER               = 4  # Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)，调制阶数
    TX_SCALE                = 1.0 # Scale for Tdata waveform ([0:1])，数据波形缩放因子

    # OFDM params，OFDM参数
    SC_IND_PILOTS           = np.array([7, 21, 43, 57])  # Pilot subcarrier indices，导频子载波索引
    SC_IND_DATA             = np.r_[1:7,8:21,22:27,38:43,44:57,58:64]  # Data subcarrier indices，数据子载波索引
    N_SC                    = 64           # Number of subcarriers，子载波数量
    # CP_LEN                  = 16          # Cyclic prefidata length，循环前缀长度
    N_DATA_SYMS             = N_OFDM_SYMS * len(SC_IND_DATA)     # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)，数据符号数量
    SAMP_FREQ               = 20e6  # 采样频率

    # Massive-MIMO params，大规模MIMO参数
    # N_UE                    = 7
    N_BS_ANT                = 64  # N_BS_ANT >> N_UE，基站天线数量
    # N_UPLINK_SYMBOLS        = N_OFDM_SYMS
    N_0                     = 1e-2  # 噪声功率
    H_var                   = 0.1  # 信道方差

    # LTS for CFO and channel estimation，用于载波频率偏移和信道估计的LTS
    lts_f = np.array([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
    pilot_in_mat = np.zeros((N_UE, N_SC, N_UE))  # 导频输入矩阵
    for i in range (0, N_UE):
        pilot_in_mat [i, :, i] = lts_f  # 为每个用户设置导频

    lts_f_mat = np.zeros((N_BS_ANT, N_SC, N_UE))  # LTS频率矩阵
    for i in range (0, N_UE):
        lts_f_mat[:, :, i] = numpy.matlib.repmat(lts_f, N_BS_ANT, 1)  # 复制LTS到所有天线

    ## Uplink，上行链路
    # 生成随机整数载荷
    tx_ul_data = np.zeros((N_UE, N_DATA_SYMS),dtype='int')
    for n_ue in range (0,N_UE):
        tx_ul_data[n_ue,:] = np.random.randint(low = 0, high = MOD_ORDER[n_ue], size=(1, N_DATA_SYMS))
    # 将数据值映射到复符号
    tx_ul_syms = np.zeros((N_UE, N_DATA_SYMS),dtype='complex')
    vec_mod = np.vectorize(modulation)  # 向量化调制函数
    for n_ue in range (0,N_UE):
        tx_ul_syms[n_ue,:] = vec_mod(MOD_ORDER[n_ue], tx_ul_data[n_ue,:])  # 对每个用户的数据进行调制
    # 将符号向量重塑为每OFDM符号一列的矩阵
    tx_ul_syms_mat = np.reshape(tx_ul_syms, (N_UE, len(SC_IND_DATA), N_OFDM_SYMS))
    # 定义导频音值作为BPSK符号
    pt_pilots = np.transpose(np.array([[1, 1, -1, 1]]))
    # 在所有OFDM符号上重复导频
    pt_pilots_mat = np.zeros((N_UE, 4, N_OFDM_SYMS),dtype= 'complex')
    for i in range (0,N_UE):
        pt_pilots_mat[i,:,:] = numpy.matlib.repmat(pt_pilots, 1, N_OFDM_SYMS)  # 复制导频到所有符号
    ## IFFT
    # 构建IFFT输入矩阵
    data_in_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    # 插入数据和导频值；其他子载波保持为0
    data_in_mat[:, SC_IND_DATA, :] = tx_ul_syms_mat
    data_in_mat[:, SC_IND_PILOTS, :] = pt_pilots_mat
    tx_mat_f = np.concatenate((pilot_in_mat, data_in_mat),axis=2)  # 连接导频和数据矩阵
    # 重塑为向量
    tx_payload_vec = np.reshape(tx_mat_f, (N_UE, -1))
    # 上行噪声矩阵
    Z_mat = np.sqrt(N_0/2) * ( np.random.random((N_BS_ANT,tx_payload_vec.shape[1])) + 1j*np.random.random((N_BS_ANT,tx_payload_vec.shape[1])))
    # H = np.sqrt(H_var/2) * ( np.random.random((N_BS_ANT, N_UE)) + 1j*np.random.random((N_BS_ANT, N_UE)))
    rx_payload_vec = np.matmul(H, tx_payload_vec) + Z_mat  # 接收信号：信道矩阵乘以发送信号加上噪声
    rx_mat_f = np.reshape(rx_payload_vec, (N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS))  # 重塑接收矩阵

    csi_mat = np.multiply(rx_mat_f[:, :, 0:N_UE], lts_f_mat)  # 信道状态信息矩阵
    fft_out_mat = rx_mat_f[:, :, N_UE:]  # FFT输出矩阵
    # precoding_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    demult_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')  # 解复用矩阵
    sc_csi_mat = np.zeros((N_BS_ANT, N_UE),dtype='complex')  # 子载波CSI矩阵

    for j in range (0,N_SC):
        sc_csi_mat = csi_mat[:, j, :]  # 获取当前子载波的CSI
        zf_mat = np.linalg.pinv(sc_csi_mat)   # ZF，计算迫零预编码矩阵
        demult_mat[:, j, :] = np.matmul(zf_mat, np.squeeze(fft_out_mat[:, j, :]))  # 解复用信号

    payload_syms_mat = demult_mat[:, SC_IND_DATA, :]  # 提取数据符号
    payload_syms_mat = np.reshape(payload_syms_mat, (N_UE, -1))  # 重塑为向量

    tx_ul_syms_vecs = np.reshape(tx_ul_syms_mat, (N_UE, -1))  # 发送符号向量
    ul_evm_mat = np.mean(np.square(np.abs(payload_syms_mat - tx_ul_syms_vecs)),1) / np.mean(np.square(np.abs(tx_ul_syms_vecs)),1)  # 计算误差向量幅度
    ul_sinrs = 1 / ul_evm_mat  # 计算信噪比

    ## Spectrual Efficiency，频谱效率
    ul_se = np.zeros(N_UE)  # 初始化频谱效率数组
    for n_ue in range (0,N_UE):
        ul_se[n_ue] = np.log2(1+ul_sinrs[n_ue])  # 根据香农公式计算每个用户的频谱效率
    ul_se_total = np.sum(ul_se)  # 计算总频谱效率

    return ul_se_total, ul_sinrs, ul_se

# 调制函数：根据调制阶数和数据返回复符号
def modulation (mod_order,data):
    '''
    Sub Functions of Previous Main Function - Data Process
    '''
    modvec_bpsk   =  (1/np.sqrt(2))  * np.array([-1, 1]) # and QPSK，BPSK和QPSK调制向量
    modvec_16qam  =  (1/np.sqrt(10)) * np.array([-3, -1, +3, +1])  # 16-QAM调制向量
    modvec_64qam  =  (1/np.sqrt(43)) * np.array([-7, -5, -1, -3, +7, +5, +1, +3])  # 64-QAM调制向量

    if (mod_order == 2): #BPSK
        return complex(modvec_bpsk[data],0) # data = 0/1，返回BPSK调制符号
    elif (mod_order == 4): #QPSK
        return complex(modvec_bpsk[data>>1],modvec_bpsk[np.mod(data,2)])  # 返回QPSK调制符号
    elif (mod_order == 16): #16-QAM
        return complex(modvec_16qam[data>>2],modvec_16qam[np.mod(data,4)])  # 返回16-QAM调制符号
    elif (mod_order == 64): #64-QAM
        return complex(modvec_64qam[data>>3],modvec_64qam[np.mod(data,8)])  # 返回64-QAM调制符号

# 解调函数：根据调制阶数和数据返回解调后的数据
def demodulation (mod_order, data):
    if (mod_order == 2): #BPSK
        return float(np.real(data)>0) # data = 0/1，BPSK解调
    elif (mod_order == 4): #QPSK
        return float(2*(np.real(data)>0) + 1*(np.imag(data)>0))  # QPSK解调
    elif (mod_order == 16): #16-QAM
        return float((8*(np.real(data)>0)) + (4*(abs(np.real(data))<0.6325)) + (2*(np.imag(data)>0)) + (1*(abs(np.imag(data))<0.6325)))  # 16-QAM解调
    elif (mod_order == 64): #64-QAM
        return float((32*(np.real(data)>0)) + (16*(abs(np.real(data))<0.6172)) + (8*((abs(np.real(data))<(0.9258))and((abs(np.real(data))>(0.3086))))) + (4*(np.imag(data)>0)) + (2*(abs(np.imag(data))<0.6172)) + (1*((abs(np.imag(data))<(0.9258))and((abs(np.imag(data))>(0.3086))))))  # 64-QAM解调

# 将动作转换为选择的用户索引
# 计算过程：
# i=1: sum_before = 7 (1用户组合数) → 15+1=16 > 7 → 继续
# i=2: sum_before = 7+21=28 → 16 ≤ 28 ✓ 找到！
#
# idx = 2 (选择2个用户)
# sum_before = 28-21 = 7
# position = 15-7 = 8
#
# # 2用户组合列表中的第8个：
# combinations([0,1,2,3,4,5,6], 2) =
# [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),...]
# # 索引8 → (1,4)
#
# 返回: ([1,4], 2)
def sel_ue(action):
    '''
    Converting Action into User Indexed Action
    '''
    user_set = [0,1,2,3,4,5,6]  # 用户集合
    sum_before = 0  # 之前的组合数总和
    # ue_select = []
    # idx = 0
    # 遍历所有可能的用户组合大小（从1到7）
    for i in range (1,8):
        sum_before += len(list(combinations(user_set, i)))  # 累加组合数
        if ((action+1)>sum_before):
            continue  # 如果动作不在当前组合大小范围内，继续
        else:
            idx = i  # 设置选择的用户数量
            sum_before -= len(list(combinations(user_set, i)))  # 减去当前组合数
            ue_select = list(combinations(user_set, i))[action-sum_before]  # 获取对应的用户组合
            break
    return ue_select,idx

# 将用户索引动作反向转换为系统动作
# ue_select = [1,4]
# idx = 2 (2个用户)
#
# # 累加1用户组合数
# action = len(combinations(7,1)) = 7
#
# # 在2用户组合中找[1,4]的位置
# comb_list = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
#              (1,2),(1,3),(1,4),(1,5),...]
# position = comb_list.index((1,4)) = 8
#
# action = 7 + 8 = 15
def reverse_sel_ue(ue_select):
    '''
    Reversing User Indexed action to system action
    '''
    user_set = [0,1,2,3,4,5,6]  # 用户集合
    idx = len(ue_select)  # 选择的用户数量
    action = 0  # 初始化动作值
    # 累加小于当前用户数量的所有组合数
    for i in range(1, idx):
        action += len(list(combinations(user_set, i)))
    comb_list = list(combinations(user_set, idx))  # 获取当前用户数量的所有组合
    position = comb_list.index(tuple(ue_select))  # 查找当前组合在列表中的位置
    action += position  # 加上位置得到最终动作值
    return action