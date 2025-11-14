import numpy as np
import pandas as pd
import os
import ast  # 用于安全地解析字符串为Python数据结构
from script.experiments_constants import STORAGE_DIRECTORY, SYMBOLIC_DATA_FILE_SUFFIX, QUANTILE_DATA_FILE_SUFFIX, ENV_KPI_NAME_LIST
from script.utils import get_list_of_experiments_number_for_number_of_users, get_kpi_change_threshold_percent
from script.experiments_constants import PRB_CATEGORY_LIST # PRB资源分类定义
from script.p_square_quantile_approximator import PSquareQuantileApproximator  # 分位数估计算法

# 管理分位数
class KPIQuantileManager:
    """
       KPI分位数管理器 - 负责实时计算KPI指标的分位数
       论文应用：动态跟踪网络性能指标的变化范围，用于后续的符号化离散化
    """
    def __init__(self):
        # 为每个KPI指标创建一个分位数估计器（中位数p=50）
        self.quantile_approximators = {
            "tx_brate": PSquareQuantileApproximator(p=50),# 传输比特率分位数估计器
            "tx_pckts": PSquareQuantileApproximator(p=50),# 传输数据包数分位数估计器
            "dl_buffer": PSquareQuantileApproximator(p=50),# 下行缓冲区分位数估计器
        }

    def fit(self):
        """初始化所有分位数估计器"""
        for approximator in self.quantile_approximators.values():
            approximator.fit([]) # 用空数组初始化

    def partial_fit(self, kpi_name, value):
        """增量更新指定KPI的分位数估计"""
        if kpi_name in self.quantile_approximators:
            # 将新数据点加入分位数估计器
            self.quantile_approximators[kpi_name].partial_fit(value)

    def get_markers(self, kpi_name):
        """获取指定KPI的分位数标记点（q0-q4）"""
        if kpi_name in self.quantile_approximators:
            return self.quantile_approximators[kpi_name].get_markers()
        else:
            return []

    def reset(self):
        """重置所有分位数估计器"""
        for kpi in self.quantile_approximators:
            self.quantile_approximators[kpi].fit([])

# 符号化数据创建器
class SymbolicDataCreator:
    """
       符号化数据创建器 - 将连续的KPI值和决策转换为符号化表示
       论文应用：将复杂的网络状态和决策转化为离散符号，便于模式挖掘和规则提取
    """
    def __init__(self, quantile_manager, kpi_name_list):
        self.quantile_manager = quantile_manager # 分位数管理器
        self.kpi_name_list = kpi_name_list # KPI指标列表
        self.marker_data = [] # 存储分位数标记点数据

    # 将kpi状态和DRL决策符号化，会调用后续方法
    def create_symbolic_data(self, df_kpi, df_decision):
        """
        Create symbolic representation of data.
        """
        """
           创建数据的符号化表示
           核心思想：将"在状态S采取决策A，导致状态S'"转换为符号化规则
        """
        
        effects_symbolic_representation = [] # 存储符号化效果
        max_timestep = df_kpi['timestep'].max() # 最大时间步
        # 初始化：将第一个时间步的KPI数据加入分位数估计器
        self._add_timestep_kpi_data_to_approximator(df_kpi[df_kpi['timestep'] == 1], timestep=1)

        # 遍历每个时间步（从第2步到倒数第2步）
        for i in range(2, max_timestep - 1):
            # 获取结果KPI（下一时间步的状态）
            resulting_kpi = df_kpi[df_kpi['timestep'] == i + 1]
            # 获取有效决策（当前时间步的决策）
            effective_decision = df_decision[df_decision['timestep'] == i]['decision'].iloc[0]
            # 获取先前KPI（当前时间步的状态）
            previous_kpi = df_kpi[df_kpi['timestep'] == i]
            # 获取先前决策（上一时间步的决策）
            previous_decision = df_decision[df_decision['timestep'] == i - 1]['decision'].iloc[0]

            # 将当前KPI数据加入分位数估计器【先调整分位数，后分类】
            self._add_timestep_kpi_data_to_approximator(previous_kpi, timestep=i)

            # 获取所有网络切片ID
            slices = resulting_kpi['slice_id'].unique()

            # 计算决策的符号化状态【关键】
            effect_symbolic_decision = self.calculate_decision_symbolic_state(effective_decision, previous_decision, slices)

            # 对每个切片计算KPI的符号化状态【关键】
            for slice_id in slices:
                symbolic_effect_for_slice = self._calculate_kpi_symbolic_state(
                    resulting_kpi[resulting_kpi['slice_id'] == slice_id][['tx_brate', 'tx_pckts', 'dl_buffer']], 
                    previous_kpi[previous_kpi['slice_id'] == slice_id][['tx_brate', 'tx_pckts', 'dl_buffer']]
                )
                # 组装完整的符号化表示
                effects_symbolic_representation.append({
                    "timestep": i,
                    "slice_id": slice_id,
                    "prb_decision": effect_symbolic_decision.loc[effect_symbolic_decision['slice_id'] == slice_id, 'prb'].iloc[0],
                    "sched_decision": effect_symbolic_decision.loc[effect_symbolic_decision['slice_id'] == slice_id, 'sched'].iloc[0],
                    **symbolic_effect_for_slice  # 展开KPI符号化结果
                })
        
        return pd.DataFrame(effects_symbolic_representation), pd.DataFrame(self.marker_data)

    # 将单个时间步的KPI数据添加到分位数估计器中
    def _add_timestep_kpi_data_to_approximator(self, timestep_data, timestep):
        """Adds KPI data of one timestep to the quantile approximators."""
        """将单个时间步的KPI数据添加到分位数估计器中"""
        
        for kpi_name in self.kpi_name_list:
            # 将KPI数据加入对应的分位数估计器
            self.quantile_manager.partial_fit(kpi_name, timestep_data[kpi_name].to_numpy())
            # 获取当前的分位数标记点
            markers = self.quantile_manager.get_markers(kpi_name)
            # 如果已经有5个标记点（完整的分布信息），则记录下来
            if len(markers) == 5:
                self.marker_data.append({
                    "timestep": timestep,
                    "kpi": kpi_name,
                    "q0": markers[0],  # 最小值
                    "q1": markers[1],  # 25%分位数
                    "q2": markers[2],  # 中位数（50%）
                    "q3": markers[3],  # 75%分位数
                    "q4": markers[4],  # 最大值
                })

    # 决策=>符号化
    def calculate_decision_symbolic_state(self, current_decision, previous_decision, slices):
        """Calculate the symbolic state for a decision based on current and previous decision values."""
        """计算决策的符号化状态（基于当前和先前的决策值）"""
        
        symbolic_decision = []

        # 解析决策元组（确保是tuple类型）
        current_decision = ast.literal_eval(current_decision) if not isinstance(current_decision, tuple) else current_decision
        previous_decision = ast.literal_eval(previous_decision) if not isinstance(previous_decision, tuple) else previous_decision

        # 对每个切片生成符号化决策
        for slice_id in slices:
            symbolic_decision.append({
                "slice_id": slice_id,
                "prb": f"{self._define_prb_change_symbolic_representation(current_decision[slice_id], previous_decision[slice_id], 'prb')}",
                "sched": f"{self._define_scheduling_policy_change_symbol(current_decision[slice_id + 3], previous_decision[slice_id + 3])}(sched)"
            })

        return pd.DataFrame(symbolic_decision)

    def _define_prb_change_symbolic_representation(self, curr_value, prev_value, variable_name):
        """
        Convert the changes in the KPIs to symbolic representation.
        """
        curr_cat = self._get_prb_category(curr_value)
        prev_cat = self._get_prb_category(prev_value)
        
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        predicate = self._get_predicate(change_percentage)
        
        if curr_cat == prev_cat:
            return f"const({variable_name}, {curr_cat})"
        else:
            # return f"{predicate}({variable_name}, {prev_cat}, {curr_cat})"
            return f"{predicate}({variable_name}, {curr_cat})"
        
    # def _define_prb_change_symbolic_representation(self, curr_value, prev_value, variable_name):
    #     """
    #     Convert the changes in the KPIs to symbolic representation.
    #     """
    #     change_percentage = self._find_change_percentage(curr_value, prev_value)
    #     predicate = self._get_predicate(change_percentage)
        
    #     if predicate == "const":
    #         return f"{predicate}({variable_name})"
    #     else:
    #         return f"{predicate}({variable_name}, {abs(curr_value - prev_value)}, {curr_value})"

    # 将PRB分配数量映射为类别Ci
    def _get_prb_category(self, value, category_map=PRB_CATEGORY_LIST):
        """ 
        This function will return the category in which the prb value sits in which category
        Maps a numerical value to its corresponding category based on the provided category map.
        Args:
            value: The numerical value to be categorized.
            category_map: A dictionary where keys are category names and values are tuples representing the range (inclusive) for each category.
        Returns:
            The category name to which the value belongs, or None if no matching category is found.
            此函数将返回prb值所在的类别
        根据提供的类别映射，将数值映射到其对应的类别。
        参数：
            数值：待分类的数值。
            category_map：一个字典，其中键是类别名称，值是代表每个类别范围（包括端点）的元组。
        返回：
            该值所属的类别名称，如果未找到匹配的类别，则为None。
        """
        """ 
                将PRB数值映射到对应的类别
                论文应用：将连续的资源分配量离散化为有限的几个等级
        """
        for category, (lower_bound, upper_bound) in category_map.items():
            if lower_bound <= value <= upper_bound:
                return category # 返回对应的类别名（如"C1", "C2"等）
        return None

    # 调度策略映射为符号
    def _define_scheduling_policy_change_symbol(self, curr_sch_pl, prev_sch_pl):
        """Convert the changes in the scheduling policy to symbolic representation."""
        """将调度策略的变化转换为符号化表示"""
        scheduling_policy_string_helper = {0: "RR", 1: "WF", 2: "PF"}
        return "const" if curr_sch_pl == prev_sch_pl else f"to{scheduling_policy_string_helper[curr_sch_pl]}"
    # kpi=>符号化
    def _calculate_kpi_symbolic_state(self, curr_kpi, prev_kpi):
        """Calculate the symbolic state for a KPI slice based on current and previous KPI values."""
        """计算KPI切片的符号化状态（基于当前和先前的KPI值）"""
        return {
                f"{kpi}": f"{self._define_change_symbolic_representation(curr_kpi[kpi].iloc[0], prev_kpi[kpi].iloc[0], kpi)}" for kpi in curr_kpi.columns
            }

    # 定义 kpi=>符号化格式
    def _define_change_symbolic_representation(self, curr_value, prev_value, variable_name):
        """Define symbolic representation of changes for KPIs."""
        """定义KPI变化的符号化表示"""
        """Convert the changes in the KPIs to symbolic representation."""
        # 计算变化百分比
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        # 获取变化谓词
        predicate = self._get_predicate(change_percentage)
        
        # if predicate == "const":
        #     return f"{predicate}({variable_name})"
        # else:

        # 生成符号化表示：谓词(KPI名称, 当前分位区间)
        return f"{predicate}({variable_name}, {self._get_kpi_quantile(variable_name, curr_value)})"

    # 计算前后kpi变化百分比
    def _find_change_percentage(self, curr_value, prev_value):
        """ This function will calculate the change percentage of the given parameter """
        """计算变化百分比"""
        if prev_value == 0:
            if curr_value == 0:
                return 0 # 都是0，无变化
            else:
                return 'inf'  # 从0到非0，视为无限增长
        else:
            return int(100 * (curr_value - prev_value) / prev_value) # 正常计算百分比

    # 定义 inc、dec、const
    def _get_predicate(self, change_percentage):
        """ This function will return the correct predicate according to the change percentage """
        if change_percentage == 'inf':
            return "inc" # 无限增长视为增加
        elif change_percentage > get_kpi_change_threshold_percent():
            return "inc" # 增加
        elif change_percentage < -get_kpi_change_threshold_percent():
            return "dec" # 减少
        else:
            return "const" # 基本不变【注意：变化幅度小于阈值5%视为不变】

    # 定义Qi
    def _get_kpi_quantile(self, kpi_name, kpi_value):
        """
        This function will return the quarter or if the value is min/max of the observed data
        """
        """
                根据KPI值返回其所在的分位区间
                论文应用：将连续的KPI值离散化为4个分位区间，便于模式发现
        """
        markers = self.quantile_manager.get_markers(kpi_name)
        
        if len(markers) < 5:
            return "NaN" # 分位数数据不足

        # Check for values at or below the minimum marker or at or above the maximum marker
        # 根据值在分位数标记点中的位置确定区间
        if kpi_value <= markers[1]: # ≤25%分位数
            return "Q1"
        elif kpi_value <= markers[2]: # ≤50%分位数（中位数）
            return "Q2"
        elif kpi_value <= markers[3]:  # ≤75%分位数
            return "Q3"
        else: # >75%分位数
            return "Q4"

# 定义数据存放目录
def create_directory_path(agent_info, user_number, storage_directory):
    """
    Create the directory path for storing data based on agent information and user number.
    """
    """创建基于智能体信息和用户数量的目录路径来存储数据"""
    
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"
    
    return os.path.join(storage_directory, agent_info['name'], str_helper)

# 创建目录
def ensure_directory_exists(directory_path):
    """Ensure that the directory exists."""
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# 定义符号化文件路径
def get_symbolic_data_csv_path(directory_path, agent_info, user_number, symbolic_data_file_suffix):
    """
    Construct the full path to the symbolic data CSV file based on directory path, agent information, and user number.
    """
    """构建符号化数据CSV文件的完整路径"""
    
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"
    
    return os.path.join(directory_path, f"{agent_info['name']}_{str_helper}{symbolic_data_file_suffix}")


# 主函数，【SRG】，将清洗好的原始数据（状态+决策）转换为符号表示并保存和返回
def create_symbolic_state_decision_matrix(df_kpi, df_decision, agent_info, user_number, force_overwrite=False):
    """
    Main function to create or read symbolic state decision matrix.
    
    Args:
        df_kpi (DataFrame): DataFrame containing KPI data.
        df_decision (DataFrame): DataFrame containing decision data.
        agent_info (dict): Dictionary containing agent information.
        user_number (int): Number of users.
        force_overwrite (bool): If True, force overwrite the existing CSV file.
        主要功能是创建或读取符号状态决策矩阵。
    参数：
        df_kpi (DataFrame)：包含关键绩效指标（KPI）数据的DataFrame。
        df_decision (DataFrame)：包含决策数据的DataFrame。
        agent_info (dict)：包含智能体信息的字典。
        user_number (int): 用户数量。
        force_overwrite（bool）：如果为True，则强制覆盖现有的CSV文件。
    """
    """
        主函数：创建或读取符号化状态决策矩阵

        论文应用：将原始的网络状态和决策数据转换为符号化规则，用于后续的关联规则挖掘和策略分析
    """

    # 创建和确保目录存在
    directory_path = create_directory_path(agent_info, user_number, STORAGE_DIRECTORY)
    ensure_directory_exists(directory_path)

    # 构建文件路径
    symbolic_data_csv_path = get_symbolic_data_csv_path(directory_path, agent_info, user_number, SYMBOLIC_DATA_FILE_SUFFIX)
    qunatile_data_csv_path = get_symbolic_data_csv_path(directory_path, agent_info, user_number, QUANTILE_DATA_FILE_SUFFIX)


    # Check if the CSV exists and force_overwrite is False
    # 检查CSV是否存在且不强制覆盖
    if not force_overwrite and os.path.exists(symbolic_data_csv_path):
        # print("Reading existing symbolic data CSV: ", symbolic_data_csv_path, "and", qunatile_data_csv_path)
        # 直接读取现有的符号化数据
        return pd.read_csv(symbolic_data_csv_path), pd.read_csv(qunatile_data_csv_path)
    else:
        print("Creating new symbolic data CSV...")

        # 创建分位数管理器和符号化创建器
        quantile_manager = KPIQuantileManager()
        symbolic_creator = SymbolicDataCreator(quantile_manager, ENV_KPI_NAME_LIST)

        # 重置分位数估计器
        quantile_manager.reset()

        # 生成符号化数据
        df_symbolic_representation, df_marker_data = symbolic_creator.create_symbolic_data(df_kpi, df_decision)
        df_symbolic_representation.to_csv(symbolic_data_csv_path, index=False)

        # print(qunatile_data_csv_path)
        # 保存结果
        df_marker_data.to_csv(qunatile_data_csv_path, index=False)

        return pd.DataFrame(df_symbolic_representation), pd.DataFrame(df_marker_data)
