'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

## Imports

from Action_Steering.p_square_quantile_approximator import PSquareQuantileApproximator  # P2分位数近似算法
import pandas as pd
import numpy as np
import ast  # 抽象语法树，用于安全地解析字符串

# 定义KPI变化阈值百分比（5%）
KPI_CHANGE_THRESHOLD_PERCENT = 5

# 符号化器：接收1或2个时间步的数据，返回符号表示
class QuantileManager:
    '''
    使用PSquareQuantileApproximator管理多个KPI的分位数近似。
    '''
    def __init__(self, kpi_list, p=50):
        # 为每个KPI创建分位数近似器
        self.quantile_approximators = {kpi: PSquareQuantileApproximator(p) for kpi in kpi_list}

    def fit(self):
        # 拟合所有近似器（当前为空实现）
        for approximator in self.quantile_approximators.values():
            approximator.fit([])

    def partial_fit(self, kpi_name, value):
        # 部分拟合：为指定KPI添加新数据点
        if kpi_name in self.quantile_approximators:
            self.quantile_approximators[kpi_name].partial_fit(value)

    def get_markers(self, kpi_name):
        # 获取指定KPI的分位数标记点
        if kpi_name in self.quantile_approximators:
            return self.quantile_approximators[kpi_name].get_markers()
        else:
            return []

    def reset(self):
        # 重置所有分位数近似器
        for kpi in self.quantile_approximators:
            self.quantile_approximators[kpi].reset() # TODO: 重置将标记点设为1,2,3,4,5，可能需要改进

    def represent_markers(self):
        """
        返回包含q0到q4分位数值的字典
        """
        markers_data = []
        for kpi in self.quantile_approximators:
            markers = self.get_markers(kpi)
            if len(markers) == 5:  # 确保有5个分位数点
                markers_data.append({
                    "kpi": kpi,      # KPI名称
                    "q0": markers[0],  # 最小值
                    "q1": markers[1],  # 25%分位数
                    "q2": markers[2],  # 50%分位数（中位数）
                    "q3": markers[3],  # 75%分位数
                    "q4": markers[4],  # 最大值
                })

        return pd.DataFrame(markers_data)  # 返回数据框格式的分位数信息

class Symbolizer:
    def __init__(self, quantile_manager: QuantileManager, kpi_list, users):
        self.quantile_manager = quantile_manager  # 分位数管理器
        self.kpis_name_list = kpi_list  # KPI名称列表
        self.users = users  # 用户列表

        # 存储前一个时间步的状态和决策数据
        self.prev_state_df = {}
        self.prev_decision_df = {}

        # 存储候选的状态和决策数据（用于下一步更新）
        self.prev_state_candid_df = {}
        self.prev_decision_candid_df = {}

    def create_symbolic_form(self, state_t_df, decision_t_df):
        """
        接收智能体在某个时间步的状态和决策，返回其符号表示
        """
        effects_symbolic_representation = []  # 存储符号表示结果

        ## 从状态数据中提取存在的分组和每个分组的成员
        groups = self._get_list_of_existing_groups_in_timestep(state_t_df)
        # 获取智能体决策的完整表示（调度和未调度用户）
        agent_complete_decision = self._get_actions_full_represetntation(decision_t_df['action'].iloc[0])

        # 遍历每个分组
        for group, group_members in groups.items():
            ## 对每个分组检查是否有前一个记录，如果存在则比较并创建符号形式，否则存储并等待下一个
            members_refined = None
            # 获取该分组所有成员的KPI列名
            kpi_columns = list(np.concatenate([self._get_list_of_kpi_column_for_users(kpi_name, group_members) for kpi_name in self.kpis_name_list]))

            # 如果该分组有前一个状态记录
            if group in self.prev_state_df:
                # 计算KPI的符号状态变化
                group_symbolic_effect = self._calculate_kpi_symbolic_state(state_t_df, self.prev_state_df[group], group_members)

                # 计算决策的符号状态变化
                group_symbolic_decision = self._calculate_decision_symbolic_state(decision_t_df, self.prev_decision_df[group], group, group_members)

                # 根据调度情况清理成员状态
                members_refined = self._clean_member_state_according_to_scheduling(group_members, decision_t_df['action'].iloc[0])

                # 构建符号表示条目
                effects_symbolic_representation.append({
                    "timestep": state_t_df['timestep'].iloc[0],  # 时间步
                    "group": group,  # 分组编号
                    "group_members": str(group_members),  # 分组成员
                    **group_symbolic_effect,  # KPI符号效果
                    "sched_members": str(members_refined),  # 调度成员
                    "sched_members_complete": str(agent_complete_decision),  # 完整调度决策
                    "decision": group_symbolic_decision  # 决策符号
                })
            else:
                # 如果没有前一个记录，只清理成员状态
                members_refined = self._clean_member_state_according_to_scheduling(group_members, decision_t_df['action'].iloc[0])

            # 准备要记忆的决策数据
            decision_to_be_rememeberd = decision_t_df.copy()
            decision_to_be_rememeberd.at[decision_to_be_rememeberd.index[0], 'action'] = members_refined[0]

            # 存储候选数据
            self.prev_state_candid_df[group] = state_t_df[kpi_columns]
            self.prev_decision_candid_df[group] = decision_to_be_rememeberd
            # 更新调度用户数量的分位数
            self.quantile_manager.partial_fit('scheduled_user', [len(members_refined[0])])

        # 将时间步的KPI数据添加到分位数近似器
        self._add_timestep_kpi_data_to_approximator(state_t_df)
        return pd.DataFrame(effects_symbolic_representation)  # 返回符号表示的数据框

    def step(self):
        """
        步进到下一个时间步，更新前一个状态和决策
        """
        self.prev_state_df = self.prev_state_candid_df.copy()
        self.prev_decision_df = self.prev_decision_candid_df.copy()

    def _clean_member_state_according_to_scheduling(self, members_list, decision):
        """
        接收成员列表，返回包含两个列表的列表：
        第一个元素包含调度的用户，第二个元素包含未调度的用户
        """
        # 解析决策（确保是集合格式）
        decision = set(ast.literal_eval(decision) if not isinstance(decision, tuple) else decision)
        # 筛选调度和未调度的成员
        scheduled_members = [member for member in members_list if member in decision]
        unscheduled_members = [member for member in members_list if member not in decision]
        return [scheduled_members, unscheduled_members]

    def _calculate_decision_symbolic_state(self, current_decision_df, previous_decision, group_num, group_users):
        """基于当前和之前的决策值计算决策的符号状态。"""
        # 格式：inc(G0, Quartile, Percent)

        # 清理当前决策
        current_decision = self._clean_member_state_according_to_scheduling(group_users, current_decision_df['action'].iloc[0])
        # 解析前一个决策
        previous_decision = ast.literal_eval(previous_decision['action'].iloc[0]) if not isinstance(previous_decision['action'].iloc[0], list) else previous_decision['action'].iloc[0]

        # 计算调度用户数量统计
        scheduled_users_count = len(current_decision[0])  # 当前调度用户数
        total_users_count = len(current_decision[0]) + len(current_decision[1])  # 总用户数

        ## 设置谓词
        predicate = "const"  # 默认：保持不变
        ## 比较当前调度用户数与之前的数量，设置方向
        if scheduled_users_count > len(previous_decision):
            predicate = "inc"  # 增加
        elif scheduled_users_count < len(previous_decision):
            predicate = 'dec'  # 减少

        ## 设置分组名称
        group_name = f"G{group_num}"

        ## 设置调度用户数的分位数
        quartile = self._get_kpi_quantile("scheduled_user", scheduled_users_count)

        ## 设置分组中调度用户的百分比（按25%的间隔取整）
        scheduled_percentage = round((scheduled_users_count / total_users_count) * 100 / 25) * 25

        return f"{predicate}({group_name}, {quartile}, {scheduled_percentage})"

    def _calculate_kpi_symbolic_state(self, curr_state_df:pd.DataFrame, prev_state_df:pd.DataFrame, members:list):
        """
        基于当前和之前的KPI值计算KPI切片的符号状态。
        """
        kpi_symbolic_representatino = {}  # 存储KPI符号表示

        # 遍历所有KPI分组
        for kpi_group in self.kpis_name_list:
            # 计算KPI的平均值的符号形式
            curr_mean = round(curr_state_df[self._get_list_of_kpi_column_for_users(kpi_group, members)].iloc[0].mean(), 4)
            prev_mean = round(prev_state_df.filter(regex=f"^{kpi_group}").iloc[0].mean(), 4)

            # 定义MSE或DTU的符号状态
            kpi_symbolic_representatino[f'{kpi_group}'] = self._define_MSE_or_DTU_symbolic_state(curr_mean, prev_mean, f'{kpi_group}', kpi_group)
        return kpi_symbolic_representatino

    def _define_MSE_or_DTU_symbolic_state(self, curr_value, prev_value, kpi_column, kpi_name):
        """
        计算并返回MSE列对不同用户的符号表示
        """
        change_percentage = self._find_change_percentage(curr_value, prev_value)  # 计算变化百分比
        predicate = self._get_predicate(change_percentage)  # 获取谓词
        return f'{predicate}({kpi_column}, {self._get_kpi_quantile(kpi_name, curr_value)})'  # 返回符号形式

    def _find_change_percentage(self, curr_value, prev_value):
        """ 计算给定参数的变化百分比 """
        if prev_value == 0:
            if curr_value == 0:
                return 0
            else:
                return 'inf'  # 无穷大（从0到非0）
        else:
            return int(100 * (curr_value - prev_value) / prev_value)  # 计算百分比变化

    def _get_predicate(self, change_percentage):
        """ 根据变化百分比返回正确的谓词 """
        if change_percentage == 'inf':
            return "inc"  # 增加（从0开始）
        elif change_percentage > KPI_CHANGE_THRESHOLD_PERCENT:
            return "inc"  # 增加（超过阈值）
        elif change_percentage < -KPI_CHANGE_THRESHOLD_PERCENT:
            return "dec"  # 减少（超过阈值）
        else:
            return "const"  # 保持不变

    def _get_kpi_quantile(self, kpi_name, kpi_value):
        """
        返回值的四分位数或是否为观测数据的最小/最大值
        """
        markers = self.quantile_manager.get_markers(kpi_name)  # 获取分位数标记

        if len(markers) < 5:
            return "NaN"  # 数据不足
        # 检查值是否在最小标记点以下或最大标记点以上
        if kpi_value <= markers[1]:  # Q1范围
            return "Q1"
        elif kpi_value <= markers[2]:  # Q2范围
            return "Q2"
        elif kpi_value <= markers[3]:  # Q3范围
            return "Q3"
        elif kpi_value <= 0.999*markers[4]:  # Q4范围（排除最大值）
            return "Q4"
        else:
            return "MAX"  # 最大值

    def _add_timestep_kpi_data_to_approximator(self, timestep_df):
        """将一个时间步的KPI数据添加到分位数近似器。"""
        for kpi_name in self.kpis_name_list:
            # 根据KPI名称创建df列列表
            kpi_columns = self._get_list_of_kpi_column_for_users(kpi_name, self.users)

            # 将新值列表发送给分位数管理器处理
            self.quantile_manager.partial_fit(kpi_name, timestep_df[kpi_columns].iloc[0].to_numpy())

    def _get_list_of_kpi_column_for_users(self, kpi_name, user_list):
        """
        组合KPI名称和给定的用户列表。输出用于从接收的状态数据框中获取列
        """
        return [f'{kpi_name}{user}' for user in user_list]

    def _get_list_of_existing_groups_in_timestep(self, data):
        """"
        接收一个时间步的数据，返回字典：键是分组编号，元素是该分组中的用户列表
        """
        # 查找分组编号
        groups = {}
        # 遍历每个'UGUr'列以确定每个用户的分组
        for i in self.users:  # 假设有7个UGUr列（0到6）
            group_number = int(data[f'UGUr{i}'].iloc[0])  # 获取用户i的分组编号

            # 将用户编号(i)添加到字典中对应的分组编号键中
            if group_number in groups:
                groups[group_number].append(i)
            else:
                groups[group_number] = [i]
        return groups

    def _get_actions_full_represetntation(self, agent_action_tuple):
        """
        接收动作并返回动作的完整表示
        """
        # 将输入元组转换为列表
        members = list(agent_action_tuple)

        # 创建从0到6的完整数字集合
        full_set = set(range(7))

        # 将输入元组转换为集合
        input_set = set(agent_action_tuple)

        # 通过从完整集合中减去输入集合来找到缺失的数字
        missing_numbers = list(full_set - input_set)

        # 返回包含两个列表的结果
        return [members, missing_numbers]