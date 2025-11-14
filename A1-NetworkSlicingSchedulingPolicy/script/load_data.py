from script.experiments_constants import EXPERIMENT_DATA_DIR_ADRESS  # 导入实验数据目录地址【注意：作者没有提供原生数据】
from script.experiments_constants import EXPERIMENT_DATA_LOG_FILE_NAME  # 导入实验日志文件名
from script.experiments_constants import STORAGE_DIRECTORY  # 导入数据存储目录【data下symbxrl】
from script.experiments_constants import CLEANED_EXPERIMENT_DATA_FILE_SUFFIX  # 导入清洗后实验数据文件后缀
from script.experiments_constants import CLEANED_SCHEDULING_POLICY_DATA_FILE_SUFFIX  # 导入清洗后调度策略文件后缀
from script.utils import get_list_of_experiments_number_for_number_of_users  # 导入根据用户数获取实验列表的工具函数
import pandas as pd  # 导入数据处理库
import re  # 导入正则表达式库
import os  # 导入操作系统接口库


# 数据缓存，加快后续数据加载，返回 kpi_data, decision_data
def handle_data(agent_info, user_number = 6):
    """
    This function helps us to reduce the execution time of the program by order of magnitude for after first run. 
    It checks if the program has been ran before, and csv files are stored, just retrieve data from csv files.
    """
    """
        核心数据管理函数 - 实现数据缓存机制
        首次运行：从原始日志文件读取并处理数据，保存为CSV
        后续运行：直接读取CSV文件，大幅提升执行速度
    """

    # 数据缓存位置及文件命名
    # create list of directories according to number of users
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"

    # 数据缓存文件命名【cleaned-data下的kpi和decision缓存文件】
    # Set address of the csv files
    agent_data_folder_address = f"{STORAGE_DIRECTORY}/{agent_info['name']}/{str_helper}/cleaned-data"
    agent_kpi_data_csv_address = f"{agent_data_folder_address}/{agent_info['name']}_{str_helper}{CLEANED_EXPERIMENT_DATA_FILE_SUFFIX}"
    agent_decision_data_csv_address = f"{agent_data_folder_address}/{agent_info['name']}_{str_helper}{CLEANED_SCHEDULING_POLICY_DATA_FILE_SUFFIX}"

    # 检查数据文件夹是否存在，如果不存在则递归创建
    if not os.path.exists(agent_data_folder_address):
        os.makedirs(agent_data_folder_address)

    # 检查缓存文件是否存在（同时检查KPI数据和决策数据两个文件）
    if os.path.exists(agent_kpi_data_csv_address) and os.path.exists(agent_decision_data_csv_address):
        # print(f"Cleaned data File for agent {agent_info['name']} exists, we load data from: {agent_kpi_data_csv_address}")
        # 如果缓存文件存在，直接从CSV文件加载数据，避免重复处理

        # 使用pandas读取KPI数据CSV文件
        kpi_data = pd.read_csv(agent_kpi_data_csv_address)
        decision_data = pd.read_csv(agent_decision_data_csv_address)
        # Todo: We may need to convert decision columns to tuple/list later
        # 待办事项：后续可能需要将决策列转换为元组或列表格式
        # 因为CSV存储时会丢失原始的数据结构，需要手动恢复
        
    else:
        # print(f"Cleaned data File for agent {agent_info['name']} does not exists, we load data from log file for {str_helper}")
        # 如果缓存文件不存在，需要从原始日志文件处理数据

        # 调用数据加载函数，从多个实验目录加载原始数据
        # 返回三个数据框：原始实验数据、原始PRB决策、原始调度策略
        raw_experiment_data, raw_prb_decision, raw_scheduling_policy = load_agent_data(directories, user_number)

        # 调用数据清洗函数，对原始数据进行清洗和处理
        # 返回两个同步的数据框：清洗后的KPI数据和决策数据
        kpi_data, decision_data = clean_process_experiment_data(raw_experiment_data, raw_prb_decision, raw_scheduling_policy)

        # 将清洗后的KPI数据保存到CSV文件，不保存行索引
        kpi_data.to_csv(agent_kpi_data_csv_address, index=False)
        # 将清洗后的决策数据保存到CSV文件，不保存行索引
        decision_data.to_csv(agent_decision_data_csv_address, index=False)

    return kpi_data, decision_data

################### ---------------------- Load Data From Files ----------------------- ############################
################### ---------------------- 从原生数据文件中加载原始数据 ----------------------- ####################
# load_agent_data会用到下面5个小函数，只需要明白这一坨代码是加载数据就行
def load_agent_data(agent_data_directories, user_number):
    agent_experiment_data = []
    agent_prb_decision_data = []
    agent_scheduling_policy_data = []

    timestep = 0

    for dir in agent_data_directories:
        
        # Load data of a log file
        received_data_df, prb_decision_df, sending_to_du_df = process_log_file(EXPERIMENT_DATA_DIR_ADRESS + f"exp{str(dir)}" + EXPERIMENT_DATA_LOG_FILE_NAME)
        
        # filter for having each a new data frame that at each time stamp the data 
        for i in range(0, len(received_data_df)):
            timestep += 1

            for data in received_data_df.iloc[i].data:

                extracted_info = data.split(",")

                if check_received_data_elemets_by_length(extracted_info):
                    agent_experiment_data.append({
                        "timestep": int(timestep),
                        "slice_id": int(extracted_info[0]),
                        "tx_brate": float(extracted_info[2]),
                        "tx_pckts": float(extracted_info[5]),
                        "dl_buffer": float(extracted_info[1]),
                        "slice_prb": int(extracted_info[4]),
                    })
            
            agent_prb_decision_data.append({
                "timestep": timestep,
                "prb_decision": prb_decision_df.iloc[i].data
            })
             
            agent_scheduling_policy_data.append({
                "timestep": timestep,
                "scheduling_policy": sending_to_du_df.iloc[i].data
            })
    
    
    return pd.DataFrame(agent_experiment_data), pd.DataFrame(agent_prb_decision_data), pd.DataFrame(agent_scheduling_policy_data)

def check_received_data_elemets_by_length(array, length = 6):
    return len(array) == length

def process_log_file(file_path):
    """
    Processes the log file and returns dataframes for received data and data sent to DU.

    Args:
    file_path (str): Path to the log file.

    Returns:
    tuple: Two pandas DataFrames, one for received data and another for data sent to DU.
    """
    log_received_data = []
    log_prb_decision_data = []
    log_send_to_du_data = []
    try:
        with open(file_path, 'r') as file:
            # valid_entry falg will determine whether the received data is updated or not, 
            # if the received data is not updated it means the state of the agent is not updated and
            # as a result the decision of the agent hasn't changed and there is no need to record the decision.
            valid_entry = False
            for line in file:
                if "Received data:" in line:
                    valid_entry = True
                    received_data = parse_received_data(line)
                    log_received_data.append({'data': received_data})
                elif "Using previous socket data" in line:
                    valid_entry = False
                elif "Action means slice_prb" in line and valid_entry:
                    prb_decision = parse_prb_data(line)
                    log_prb_decision_data.append({'data': prb_decision})
                elif "Sending to DU:" in line and valid_entry:
                    sending_data = parse_sending_data(line)
                    log_send_to_du_data.append({'data': sending_data})

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # Handle the exception or re-raise

    return pd.DataFrame(log_received_data), pd.DataFrame(log_prb_decision_data), pd.DataFrame(log_send_to_du_data)

def parse_prb_data(line):
    """
    Extract prb decisions for the next timestep from log file
    """
    match = re.search(r"slice_prb \[([0-9, ]+)\]", line)
    if match:
        numbers_str = match.group(1)
        numbers = numbers_str.split(", ")
        return numbers
    else:
        print("problem")

def parse_received_data(line):
    """
    Extracts and processes received data from a line.
    """

    return line.split("Received data:")[1][2:-2].split('\\n')

def parse_sending_data(line):
    """
    Extracts and processes sending data from a line.
    """
    return line.split("Sending to DU:")[1][2:7].strip().split(",")

################### ---------------------- Clean and Process Data ----------------------- ############################
################### ---------------------- 清洗和处理数据 ----------------------- ############################
def clean_process_experiment_data(extracted_data, prb_decisions, scheduling_policy):
    """
    Receives the raw read log data and perform cleaning and arranging on them.
    Output is two dataframe that the timestep is sync between them.
    At each time step the experiment data shows the KPI at that timestep and 
    the decision dataframe shows the decisions made according to that KPI which will be effective on the next timestep.
    """
    cleaned_experiment_data = []
    cleaned_decision = []
    
    timestep_counter = 0
    
    for i in range(1, extracted_data['timestep'].max()+1):
        data_at_timestep = extracted_data.loc[extracted_data.timestep == i]
        
        if is_complete_data(data_at_timestep):
            timestep_counter += 1
            
            timestep_data = group_and_average_data(data_at_timestep)
            cleaned_experiment_data.extend(
                build_experiment_data(timestep_counter, timestep_data)
            )

            policy_state = scheduling_policy.loc[scheduling_policy.timestep == i]
            prb_state = prb_decisions.loc[prb_decisions.timestep == i]
            
            decision = make_decision(prb_state, policy_state, timestep_counter)
            
            cleaned_decision.append(decision)

    
    return drop_non_frequent_slice_timesteps(pd.DataFrame(cleaned_experiment_data), pd.DataFrame(cleaned_decision))

def drop_non_frequent_slice_timesteps(kpi_data, decision_data):
    """
    Filters out timesteps from kpi_data and decision_data DataFrames that do not contain
    the most frequently occurring set of slice_ids. Rows with timesteps not containing the most
    frequent set are removed directly from both dataframes.
    """
    # Group by 'timestep' and collect unique 'slice_id' for each group in kpi_data
    slice_sets_per_timestep = kpi_data.groupby('timestep')['slice_id'].apply(lambda x: set(x))
    
    # Identify the most frequent set of 'slice_id'
    set_frequency = slice_sets_per_timestep.value_counts()
    most_frequent_set = set_frequency.idxmax()
    
    # Identify timesteps that do not contain the most frequent set
    timesteps_not_with_most_frequent_set = slice_sets_per_timestep[slice_sets_per_timestep != most_frequent_set].index
    
    # Filter out these timesteps from both kpi_data and decision_data
    filtered_kpi_data = kpi_data[~kpi_data['timestep'].isin(timesteps_not_with_most_frequent_set)]
    filtered_decision_data = decision_data[~decision_data['timestep'].isin(timesteps_not_with_most_frequent_set)]
    
    # Adjust the 'timestep' in filtered_kpi_data and filtered_decision_data to be continuous starting from 1
    filtered_kpi_data['timestep'], _ = pd.factorize(filtered_kpi_data['timestep'], sort=True)
    filtered_kpi_data['timestep'] += 1

    filtered_decision_data['timestep'], _ = pd.factorize(filtered_decision_data['timestep'], sort=True)
    filtered_decision_data['timestep'] += 1


    # Reset indexes
    filtered_kpi_data.reset_index(drop=True, inplace=True)
    filtered_decision_data.reset_index(drop=True, inplace=True)

    return filtered_kpi_data, filtered_decision_data

def is_complete_data(data):
    return len(data['slice_id'].unique()) >= 1

def group_and_average_data(data):
    return data.groupby(['slice_id']).mean().reset_index()

def build_experiment_data(timestep, grouped_data):
    return [
        {
            "timestep": timestep,
            "slice_id": int(item['slice_id']),
            "tx_brate": item['tx_brate'],
            "tx_pckts": item['tx_pckts'],
            "dl_buffer": item['dl_buffer']
        }
        for key, item in grouped_data.iterrows()
    ]

def make_decision(prb_data, policy_state, timestep):
    prb_state = define_prb_state(prb_data)
    scheduling_policy_state = define_scheduling_policy_state(policy_state)
    
    return {
        "timestep": timestep,
        "decision": tuple(prb_state + scheduling_policy_state),
        "prb_decision": prb_state,
        "policy_decision": scheduling_policy_state
    }

def define_prb_state(prb_data):    
    return [int(i) for i in prb_data['prb_decision'].iloc[0]]

def define_scheduling_policy_state(df_timestep_scheduling_policy_data):
    return [int(i) for i in df_timestep_scheduling_policy_data['scheduling_policy'].iloc[0]]