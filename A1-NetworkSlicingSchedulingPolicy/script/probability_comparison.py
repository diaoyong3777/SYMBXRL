# 导入必要的Python库
import os  # 操作系统接口，用于文件路径操作
import pandas as pd  # 数据处理库，用于处理表格数据
import numpy as np  # 数值计算库，用于数学运算
import plotly.express as px  # 数据可视化库，用于创建交互式图表
import plotly.graph_objects as go  # 更高级的绘图功能
from scipy.cluster.hierarchy import linkage, leaves_list  # 层次聚类算法，用于数据分组
from scipy.spatial.distance import squareform  # 距离矩阵格式转换工具



# 计算KL散度
def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length, and q must not contain zeros.
    计算两个离散概率分布之间的 Kullback-Leibler散度。
    p和q必须是长度相同的numpy数组，且q不能包含零。
    """
    """
        Kullback-Leibler散度（KL散度）
        论文应用：在强化学习中衡量智能体策略分布与目标分布的差异
        比如：比较不同网络切片策略的效果差异
    """

    # 将输入转换为numpy数组，确保数据类型为浮点数以提高计算精度
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    # 归一化处理：确保概率分布的和为1（概率论基本要求）
    p /= p.sum()
    q /= q.sum()
    
    # Replace zeros with a very small number to avoid division by zero
    # 处理零值问题：将概率为零的位置替换为极小数，避免计算log(0)出现数学错误
    q = np.where(q == 0, 1e-10, q) # np.where(条件, 满足时的值, 不满足时的值)
    p = np.where(p == 0, 1e-10, p)
    
    # Calculate KL divergence
    # 计算KL散度核心公式：Σ p(x) * log(p(x)/q(x))
    # np.where(p != 0, p * np.log(p / q), 0) 表示：当p不为零时计算，否则为0
    divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    # Ensure divergence is not negative due to numerical issues
    # 确保结果非负：由于数值计算误差可能产生极小负值，用max确保≥0
    return max(divergence, 0)


# 利用KL散度进一步计算JS散度
def js_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    """
    """
        Jensen-Shannon散度（JS散度）- KL散度的对称改进版本
        论文应用：比较不同实验条件下智能体策略的相似性，取值范围[0,1]更易解释
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    # 概率分布归一化
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the pointwise mean of p and q
    # 计算两个分布的中间分布M = (P+Q)/2
    m = 0.5 * (p + q)
    
    # Calculate the Jensen-Shannon divergence using the KL divergence
    # JS散度公式：0.5*KL(P||M) + 0.5*KL(Q||M)
    # 这样计算使得JS散度是对称的，且取值范围固定
    js_divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    return js_divergence


# 计算 Wasserstein距离【没用上】
def wasserstein_distance(p, q):
    """
    Calculate the Wasserstein distance (Earth Mover's distance) between two one-dimensional discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    计算两个一维离散概率分布之间的瓦瑟斯坦距离（即地球移动者距离）。
    p和q必须是长度相同的numpy数组。
    """
    """
        Wasserstein距离（地球移动距离）
        论文应用：衡量从一种资源分配策略转变为另一种策略的"成本"
        比如：网络切片中PRB资源重新分配的最小代价
    """
    # Normalize p and q to ensure they sum to 1
    # 输入处理和归一化
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the cumulative distribution functions (CDFs) of p and q
    # 计算累积分布函数(CDF)：每个点的累计概率值
    cdf_p = np.cumsum(p) # 例如：[0.1, 0.3, 0.6, 1.0]
    cdf_q = np.cumsum(q)
    
    # Calculate the Wasserstein distance as the L1 distance between the CDFs
    # Wasserstein距离公式：两个CDF曲线之间面积的绝对值之和
    distance = np.sum(np.abs(cdf_p - cdf_q))
    # 几何意义：将一个分布变成另一个分布需要移动的"土方量"
    
    return distance

# 计算 Hellinger距离
def hellinger_distance(p, q):
    """
    Calculate the Hellinger distance between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    """
    """
    Hellinger距离 - 基于概率平方根的距离度量
    论文应用：衡量网络切片策略分布的相似性，对异常值不敏感
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    # 归一化
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the Hellinger distance
    # Hellinger距离公式：√[0.5 * Σ(√p(x) - √q(x))²]
    # 先计算每个点的平方根差值，然后平方、求和、乘0.5、开方
    distance = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    # 取值范围[0,1]，0表示完全相同，1表示完全不同
    
    return distance

# 计算马氏距离【没用上】
def mahalanobis_distance(p, q):
    """
    Calculate the Mahalanobis distance between two discrete probability distributions.
    Both p and q must be numpy arrays of the same length.
    The covariance matrix is computed from the combined array of p and q.
    计算两个离散概率分布之间的马氏距离。
    p和q必须是长度相同的numpy数组。
    协方差矩阵是根据p和q的组合数组计算得出的。
    """
    """
       马氏距离 - 考虑数据协方差结构的统计距离
       论文应用：在考虑各KPI指标相关性的情况下比较策略效果
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize p and q to ensure they sum to 1
    p /= p.sum()
    q /= q.sum()
    
    # Calculate the difference vector
    # 计算两个分布的差值向量
    diff = p - q  # 每个维度上的差异
    
    # Combine p and q to compute the covariance matrix
    # 构建协方差矩阵：衡量p和q在各个维度上的变化关系
    combined = np.vstack([p, q])  # 将两个分布堆叠成2×n的矩阵
    cov = np.cov(combined, rowvar=False)  # Ensure that rows represent variables # 计算协方差矩阵
    
    # Invert the covariance matrix
    # 求协方差矩阵的逆矩阵：用于标准化不同维度的尺度
    inv_cov = np.linalg.inv(cov)  # 矩阵求逆
    
    # Calculate the Mahalanobis distance
    # 马氏距离公式：√[(p-q)ᵀ × Σ⁻¹ × (p-q)]
    # 考虑了各个维度之间的相关性，比欧氏距离更准确
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))  # diff.T是转置，np.dot是矩阵乘法
    
    return distance

# 计算 JS散度、Hellinger距离，并绘图
def plot_and_save_probability_dist_comparison_heatmaps_with_clustering(probs, unique_ids, name, path, methods=["hd"]):
    """
    This function will calculate and plot the probability distribution comparison for probability distribution a and b using 
    methods listed in the methods array amd save them in path with the name given
    This function can be only applied to symmetric methods. So fot this method that we want to apply clustering we will only use JS and Hellinger method
    此函数将计算并绘制概率分布a和b的概率分布对比图
    将方法数组中列出的方法以给定的名称保存在路径中
    此函数仅适用于对称方法。因此，对于我们想要应用聚类的方法，我们将仅使用JS和Hellinger方法
    """
    """
        主函数：生成概率分布比较热图并进行聚类分析
        论文应用：可视化不同实验条件下智能体策略的相似性模式

        参数说明：
        probs: DataFrame，包含'id'和'prob'列，每个id对应一个概率分布
        unique_ids: 数组，需要比较的ID列表
        name: 字符串，分析名称（如"embb-trf1_6users"）
        path: 字符串，结果保存路径
        methods: 列表，使用的距离计算方法，默认["hd"](Hellinger距离)
    """
    # 遍历每种距离计算方法
    for method in methods:
        # 初始化距离矩阵：n×n的零矩阵，n为unique_ids的数量
        div_matrix = np.zeros((len(unique_ids), len(unique_ids)))
        # Jensen-Shannon散度分析
        if method == "js":
            ## Calculate comparison value
            ## 计算所有分布对之间的距离
            for i, uid_i in enumerate(unique_ids): # i是行索引，uid_i是行的ID
                for j, uid_j in enumerate(unique_ids): # j是列索引，uid_j是列的ID
                    if i != j: # 非对角线元素（自己不与自己比较）
                        # 从DataFrame中提取两个分布的概率值
                        prob_i = probs[probs['id'] == uid_i]['prob'].values[0]
                        prob_j = probs[probs['id'] == uid_j]['prob'].values[0]
                        # 计算JS散度
                        divergence = js_divergence(prob_i, prob_j)
                        # 填充距离矩阵
                        div_matrix[i, j] = divergence
                        div_matrix[j, i] = divergence  # Ensure the matrix is symmetric # 确保矩阵对称
                    else:
                        div_matrix[i, j] = 0  # Diagonal should be zero # 对角线设为0（自己与自己的距离为0）
            # Perform hierarchical clustering
            # 层次聚类分析：将相似的分布分组
            # squareform将距离矩阵转换为压缩格式，linkage执行聚类
            linkage_matrix = linkage(squareform(div_matrix), method='average')
            # leaves_list获取聚类后的叶节点顺序（即重新排列的索引）
            sorted_indices = leaves_list(linkage_matrix)
            
            # Reorder the matrix according to the clustering
            # 根据聚类结果重新排列距离矩阵
            sorted_div_matrix = div_matrix[:, sorted_indices][sorted_indices, :]
            # [sorted_indices, :] 重新排列行，[:, sorted_indices] 重新排列列
            
            # Plot the comparison
            # 创建DataFrame用于绘图
            div_df = pd.DataFrame(sorted_div_matrix, index=unique_ids[sorted_indices], columns=unique_ids[sorted_indices])
            # 重新排序的行标签  # 重新排序的列标签

            # 使用Plotly创建交互式热图
            fig = px.imshow(
                div_df,  # 数据
                labels=dict(x="ID", y="ID", color=f"{method} divergence"),  # 坐标轴和颜色条标签
                title=f"{name} - {method} divergence", # 图表标题
                range_color=[0, 1] # 颜色映射范围固定为0到1
            )
            fig.update_layout(
                title=f'Heatmap of {method.capitalize()} Divergence between Distributions',
                xaxis_title='ID', # x轴标题
                yaxis_title='ID',  # y轴标题
                autosize=True,  # 自动调整大小
            )
            # Save the plot for each combination of agent and number of users
            # 保存为HTML文件（交互式图表）
            full_file_path = os.path.join(path, f"{name}_{method}_divergence.html")
            fig.write_html(full_file_path)  # 写入HTML文件

        # Hellinger距离分析（结构与JS散度相同）
        elif method == "hd":
            ## Calculate comparison value
            ## 计算Hellinger距离矩阵
            for i, uid_i in enumerate(unique_ids):
                for j, uid_j in enumerate(unique_ids):
                    if i != j:
                        prob_i = probs[probs['id'] == uid_i]['prob'].values[0]
                        prob_j = probs[probs['id'] == uid_j]['prob'].values[0]
                        # 计算Hellinger距离
                        divergence = hellinger_distance(prob_i, prob_j)
                        div_matrix[i, j] = divergence
                        div_matrix[j, i] = divergence  # Ensure the matrix is symmetric
                    else:
                        div_matrix[i, j] = 0  # Diagonal should be zero
            # Perform hierarchical clustering
            # 层次聚类
            linkage_matrix = linkage(squareform(div_matrix), method='average')
            sorted_indices = leaves_list(linkage_matrix)
            
            # Reorder the matrix according to the clustering
            # 重新排序矩阵
            sorted_div_matrix = div_matrix[:, sorted_indices][sorted_indices, :]
            
            # Plot the comparison
            # 创建热图
            div_df = pd.DataFrame(sorted_div_matrix, index=unique_ids[sorted_indices], columns=unique_ids[sorted_indices])
            fig = px.imshow(
                div_df,
                labels=dict(x="ID", y="ID", color=f"Hellinger Distance"),
                title=f"{name} - Hellinger Distance",
                range_color=[0, 1]
            )
            fig.update_layout(
                title=f'Heatmap of Hellinger Distance between Distributions',
                xaxis_title='ID',
                yaxis_title='ID',
                autosize=True,
            )
            # Save the plot for each combination of agent and number of users
            # 保存结果
            full_file_path = os.path.join(path, f"{name}_{method}_divergence.html")
            fig.write_html(full_file_path)

