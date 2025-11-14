# SYMBXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks

这是 SYMBXRL 项目的完整复现环境，包含两个独立的 DRL 应用案例：网络切片调度和大规模 MIMO 资源调度。

## 项目概述

SYMBXRL 是一个基于符号人工智能的可解释深度强化学习框架，通过一阶逻辑（FOL）为 DRL 智能体生成人类可理解的解释。本项目包含：

- **A1**: 网络切片和调度策略（RAN slicing and scheduling）
- **A2**: 大规模 MIMO 调度（Massive MIMO scheduling）

## 测试环境

- **操作系统**: Linux
- **显卡**: NVIDIA A10，21 GB
- **Python**: 3.10.14

## 环境配置

### 创建 Conda 环境
```bash
conda create -n symbxrl python=3.10.14
conda activate symbxrl
```

### 环境注意事项
- **依赖冲突**: 可以临时修改 `webcolors>=24.6.0`，安装完成后恢复至 1.13 版本
- **Graphviz**: A2 项目需要安装 Graphviz 用于决策图可视化
- 环境配置好后注意在constants.py中修改项目根路径(复制你本地的项目的绝对路径)

## 项目结构

```
SYMBXRL/
├── A1-NetworkSlicingSchedulingPolicy/    # 第一个项目 (RAN slicing and scheduling)
│   ├── data/                             # 数据目录
│   │   └── symbxrl/
│   │       └── embb-trf1/
│   │           └── 3-users_exp-4/
│   │               ├── cleaned-data/     # (1) 清理好的数据
│   │               │   └── embb-trf1_3-users_exp-4_cleaned_experiment_data.csv
│   │               └── embb-trf1_3-users_exp-4_quantile_data.csv  # (2) 符号化后的数据
│   ├── results/                          # 结果输出目录
│   │   ├── Probabilistic_Analysis/
│   │   │   ├── prob_dist/                # (3) 概率分布图
│   │   │   │   └── embb-trf1-users3/
│   │   │   │       └── Effect_Probability_Distribution_embb-trf1_user_3.pdf
│   │   │   └── heatmaps/                 # (4) 热力图
│   │   │       └── embb-trf1-users3/
│   │   │           └── Effect_Probabilities_Heatmaps_embb-trf1_user_3.pdf
│   │   ├── decision_graphs/              # (5) 知识图谱
│   │   │   └── embb-trf1-users3/
│   │   │       └── Decision_Graph_embb-trf1_user-3_slice-0.pdf
│   │   └── Plots_for_Paper/              # (x) 论文用图汇总
│   ├── script/                           # 脚本文件
│   │   ├── experiments_constants.py      # 实验参数常量配置
│   │   ├── load_data.py                  # 加载原生数据，清洗、缓存、返回
│   │   ├── clean_exps.py                 # load_data.py 的重复文件（实际未使用）
│   │   ├── p_square_quantile_approximator.py  # P² 分位数估计算法
│   │   ├── probability_comparison.py     # 离散概率分布比较（JS散度、Hellinger距离）
│   │   ├── symbolic_representation.py    # SRG - 数据符号化 (1) => (2)
│   │   └── utils.py                      # 工具函数
│   ├── 1_data_preprocess.ipynb           # SRG + EE - 概率分布（Jupyter可视化）
│   ├── 2_probabilistic_analysis.ipynb    # SRG + EE - 概率分布 + 热力图（保存PDF）(3)、(4)
│   ├── 3_graph_visualization.ipynb       # SRG + EE - KG分析 (5)
│   ├── 4_Plots_for_Paper.ipynb           # 汇总 - 状态图、概率分布、KG、热力图 (x)
│   └── README.md
└── A2-MassiveMIMOScheduler/              # 第二个项目 (Massive MIMO scheduling)
    ├── custom_mimo_env.py                # 环境类
    ├── action_steering_utils.py          # IAS 底层代码
    ├── decision_graph.py                 # 决策图辅助工具
    ├── 1_*.ipynb                         # 主要执行文件
    ├── 2_*.ipynb                         # 主要执行文件
    └── (其他相关文件)
```

## 复现指南

### A1 项目：网络切片和调度策略

#### 核心组件
- **SRG (Symbolic Representation Generator)**: 符号表示生成器
- **EE (Explanation Engine)**: 解释引擎

#### 复现步骤

1. **数据预处理**
   - 运行 `1_data_preprocess.ipynb` 进行数据预处理和符号化

2. **概率分析**
   - 运行 `2_probabilistic_analysis.ipynb` 生成：
     - 概率分布图 (3)
     - 热力图 (4)

3. **知识图谱分析**
   - 运行 `3_graph_visualization.ipynb` 生成决策图谱 (5)

4. **论文图表生成**
   - 运行 `4_Plots_for_Paper.ipynb` 汇总所有图表 (x)
   - **注意**: 执行时先运行导包和工具单元，然后跳转到 "Without Decimal Points by filtering out empty cells"（倒数第三个单元）
   - **已知问题**: 生成的 PDF 底色可能不是纯白色，可能与依赖包版本有关

#### 数据说明
- **原始数据**: 未提供，使用项目已处理的数据
- **处理数据**: 位于 `data/symbxrl/` 目录下
- **符号化数据**: 已缓存，可卸载重新生成

### A2 项目：大规模 MIMO 调度

#### 环境依赖
```bash
conda install graphviz -c conda-forge  # EE-决策图可视化工具
```

#### 复现步骤
1. 直接运行 `1_*.ipynb` 和 `2_*.ipynb` 即可完成复现
2. 核心文件说明：
   - `custom_mimo_env.py`: 环境类
   - `action_steering_utils.py`: IAS（基于意图的动作引导）底层实现
   - `decision_graph.py`: 决策图生成工具，IAS 的辅助组件

## 核心功能模块

### SRG (符号表示生成器)
- 将数值状态和动作转换为 FOL 术语
- 使用 P² 算法进行在线分位数计算
- 生成符号化数据存储

### EE (解释引擎)
- **概率分析**: 状态分布和动作相关性分析
- **KG分析**: 构建知识图谱揭示决策模式
- **可视化输出**: 概率分布图、热力图、决策图谱

### IAS (基于意图的动作引导)
- 奖励最大化
- 决策条件约束
- 加速学习



## 引用

如使用本项目，请引用原始论文：
```
S. Duttagupta, M. Jabbari, C. Fiandrino, M. Fiore, and J. Widmer, 
"SYMBXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks," 
IEEE INFOCOM 2025.
```

## 许可证

本项目基于原始 SYMBXRL 项目代码，遵循相应的许可证条款。
