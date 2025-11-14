'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import pandas as pd
import numpy as np
import networkx as nx  # 图论分析库
import pyvis.network as network  # 交互式网络可视化库

class DecisionGraph:
    def __init__(self, column_name) -> None:
        self.column_name = column_name  # 决策列名（如"decision"）
        self.decision_df = []  # 存储决策数据
        self.G = nx.DiGraph()  # 创建有向图对象
        self.net = None  # Pyvis网络对象（用于可视化）
        self.previous_decision = None  # 记录前一个决策
        return

    def update_graph(self, symbolic_form_df: pd.DataFrame):
        """
        接收新的决策并更新图对象中的决策和决策计数
        """
        current_decision = symbolic_form_df[self.column_name].iloc[0]  # 获取当前决策
        current_reward = symbolic_form_df['reward'].iloc[0]  # 获取当前奖励

        # 如果当前决策不存在，添加为新节点
        if current_decision not in self.G.nodes:
            self.G.add_node(current_decision,
                          title=current_decision,
                          occurrence=0,       # 出现次数
                          total_reward=0,     # 总奖励
                          mean_reward=0)      # 平均奖励

        # 更新当前决策节点的出现次数和总奖励
        self.G.nodes[current_decision]['occurrence'] += 1
        self.G.nodes[current_decision]['total_reward'] += current_reward
        self.G.nodes[current_decision]['mean_reward'] = self.G.nodes[current_decision]['total_reward'] / self.G.nodes[current_decision]['occurrence']

        # 更新从前一个决策到当前决策的转移
        if self.previous_decision is not None:  # 如果有前一个决策
            if self.G.has_edge(self.previous_decision, current_decision):
                # 如果边已存在，更新出现次数和总奖励
                self.G[self.previous_decision][current_decision]['occurrence'] += 1
                self.G[self.previous_decision][current_decision]['total_reward'] += current_reward
            else:
                # 如果边不存在，创建新边
                self.G.add_edge(self.previous_decision, current_decision,
                              occurrence=1,
                              total_reward=current_reward)

            # 计算边的平均奖励
            self.G[self.previous_decision][current_decision]['mean_reward'] = self.G[self.previous_decision][current_decision]['total_reward'] / self.G[self.previous_decision][current_decision]['occurrence']

        self.previous_decision = current_decision  # 更新前一个决策
        self.previous_reward = current_reward  # 更新前一个奖励

        # 更新节点和边的概率和大小
        self._update_probabilities_and_sizes()

        return

    def _update_probabilities_and_sizes(self):
        """
        更新图中节点和边的概率和大小
        """
        # 计算节点的总出现次数
        total_node_occurrence = sum(nx.get_node_attributes(self.G, 'occurrence').values())

        # 计算每个决策节点的概率
        node_probabilities = {}
        node_sizes = {}
        for node, data in self.G.nodes(data=True):
            node_probabilities[node] = data['occurrence'] / total_node_occurrence  # 节点概率
            # 应用指数缩放计算节点大小
            min_size = 10  # 最小大小
            scale_factor = 0.5  # 缩放因子，控制指数增长率
            node_sizes[node] = min_size + np.exp(scale_factor * np.log1p(data['occurrence'] - 1))

        # 在图中设置节点概率和大小属性
        nx.set_node_attributes(self.G, node_probabilities, 'probability')
        nx.set_node_attributes(self.G, node_sizes, 'size')

        # 计算每条边的概率
        edge_probabilities = {}
        for u, v, data in self.G.edges(data=True):
            # 计算从节点u出发的所有转移次数
            total_transitions_from_u = sum(self.G[u][nbr]['occurrence'] for nbr in self.G.successors(u))
            edge_probabilities[(u, v)] = data['occurrence'] / total_transitions_from_u if total_transitions_from_u > 0 else 0

        # 在图中设置边概率属性
        nx.set_edge_attributes(self.G, edge_probabilities, 'probability')

    def build_graph(self):
        """
        返回可用于绘制所创建图的pyvis对象
        """
        # 确保概率和大小是最新的
        self._update_probabilities_and_sizes()

        # 创建Pyvis网络对象
        self.net = network.Network(height="1500px",
                                 width="100%",
                                 bgcolor="#222222",
                                 font_color="white",
                                 directed=True,
                                 notebook=True,
                                 filter_menu=True,
                                 select_menu=True,
                                 cdn_resources="in_line")
        # 从NetworkX图创建Pyvis网络
        self.net.from_nx(self.G)

        # 遍历Pyvis网络中的每个节点，设置包含出现次数、平均奖励和概率的标题
        for node in self.net.nodes:
            occurrence = self.G.nodes[node['id']]['occurrence']
            probability = round(100 * self.G.nodes[node['id']]['probability'], 1)
            mean_reward = self.G.nodes[node['id']]['mean_reward']
            node['title'] = f"Node: {node['id']} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"

        # 遍历Pyvis网络中的每条边，设置包含出现次数、平均奖励和概率的标题
        for edge in self.net.edges:
            u, v = edge['from'], edge['to']
            occurrence = self.G[u][v]['occurrence']
            probability = round(100 * self.G[u][v]['probability'], 1)
            mean_reward = self.G[u][v]['mean_reward']
            edge['title'] = f"Edge from {u} to {v} \n Occurrence: {occurrence} \n Mean Reward: {mean_reward:.2f} \n Probability: {probability}%"

        # 计算图的大小信息
        num_nodes = self.G.number_of_nodes()  # 节点数量
        num_edges = self.G.number_of_edges()  # 边数量

        # 创建图大小信息的文本元素
        size_text = f"Number of Nodes: {num_nodes}<br>Number of Edges: {num_edges}"

        # 将大小信息作为HTML元素添加到Pyvis网络中
        self.net.add_node("size_info",
                         label=size_text,
                         shape="text",
                         x='-95%',
                         y=0,
                         physics=False)  # 禁用物理效果（固定位置）

        # 设置布局算法
        self.net.barnes_hut(overlap=1)
        # 显示控制按钮
        self.net.show_buttons(filter_=['physics'])
        return

    def get_graph(self, mode="all"):
        """
        返回图的networkX对象以执行"动作引导"分析
        """
        if mode == "all":
            return self.G, self.net  # 返回NetworkX图和Pyvis网络

        if mode == "networkX":
            return self.G  # 只返回NetworkX图

        if mode == "pyvis":
            return self.net  # 只返回Pyvis网络