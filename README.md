# IAIC-MIL: Interventional Attention-based Instance Clustering for Multiple Instance Learning

## 📖 项目简介

IAIC-MIL（Interventional Attention-based Instance Clustering for Multiple Instance Learning）是一个基于示例归因和因果推断的多实例学习框架，专门用于全切片病理图像（Whole Slide Image, WSI）分类任务。

### 主要特点

- **因果干预机制**：通过混杂因子建模和因果干预，减少虚假相关性对模型的影响
- **Sobol敏感性分析**：利用Sobol全局敏感性分析方法进行实例重要性评估，提高关键实例的选择精度
- **特征解耦**：通过Disentangler模块分离前景（因果）特征和背景（混杂）特征

## 🏗️ 项目结构

```
IAIC_MIL/
├── Main_Sobol.py              # 主训练脚本
├── utils.py                   # 工具函数（评估指标、特征解释等）
├── README.md                  # 项目说明文档
├── Models/                    # 模型定义
│   ├── IAIC/                  # IAIC核心模型
│   │   ├── network.py         # 维度降低网络
│   │   ├── Attention.py       # 注意力机制（带解耦）
│   │   ├── AttentionORI.py    # 原始注意力机制
│   │   └── loss.py            # 损失函数（MINE、SimMinLoss等）
│   ├── SimpleNet/             # 简单基线模型
│   │   ├── MaxNet.py          # 最大池化聚合
│   │   └── MeanNet.py         # 均值池化聚合
│   └── TransMIL/              # TransMIL模型
│       └── net.py             # Transformer-based MIL
└── sobolAnakysis/             # Sobol敏感性分析模块
    ├── sobol2WSI.py           # WSI的Sobol分析实现
    ├── estimator.py           # Sobol指数估计器
    ├── sampler.py             # Sobol序列采样器
    └── utils.py               # 分析工具函数
```

## 🔧 环境配置

### 依赖项

```bash
pip install torch torchvision
pip install numpy scipy scikit-learn
pip install tensorboard
pip install faiss-gpu  # 或 faiss-cpu
pip install opencv-python
pip install pandas
```

### 推荐环境

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (GPU训练)

## 🚀 快速开始

### 数据准备

1. 准备WSI特征文件（.pkl格式），包含以下结构：
```python
{
    'slide_name_1': [
        {'feature': np.array([...])},  # patch特征
        {'feature': np.array([...])},
        ...
    ],
    'slide_name_2': [...],
    ...
}
```

2. 切片命名规则：
   - 以 `lucs` 开头的切片标记为类别1
   - 以 `luad` 开头的切片标记为类别0

### 训练模型

```bash
python Main_Sobol.py \
    --mDATA_dir /path/to/features.pkl \
    --log_dir ./weight/experiment_name \
    --EPOCH 200 \
    --lr 1e-4 \
    --numGroup 50 \
    --instances_per_group 15 \
    --nb_design 8 \
    --seed 22
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mDATA_dir` | - | 特征文件路径 |
| `--log_dir` | - | 日志和模型保存路径 |
| `--EPOCH` | 200 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--numGroup` | 50 | 实例分组数量 |
| `--instances_per_group` | 15 | 每组选择的实例数 |
| `--nb_design` | 8 | Sobol采样设计点数（2的幂次） |
| `--feats_size` | 1024 | 输入特征维度 |
| `--num_cls` | 2 | 分类类别数 |
| `--distill_type` | MaxS | 蒸馏类型（MaxMinS/MaxS/MinS/AFS） |
| `--split_ratio` | 0.7 | 训练集比例 |
| `--seed` | 22 | 随机种子 |

## 📊 模型架构

### IAIC核心流程

1. **特征降维**：通过 `DimReduction` 将输入特征从高维映射到低维空间
2. **Sobol分析**：通过全局敏感性分析评估每个实例对预测的贡献
4. **实例选择**：根据Sobol指数选择最重要/最不重要的实例
5. **因果干预**：通过混杂因子建模进行因果推断
6. **特征解耦**：分离因果特征和混杂特征
7. **分类预测**：基于解耦后的因果特征进行最终分类

## 📈 评估指标

模型训练过程中会输出以下指标：

- **AUC**：ROC曲线下面积
- **Accuracy**：准确率
- **Precision**：精确率
- **Recall**：召回率
- **Specificity**：特异性
- **F1 Score**：F1分数

## 📝 引用

如果您使用了本项目的代码，请引用相关论文。

## 📄 许可证

本项目仅供学术研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 📧 联系方式

如有问题，请通过GitHub Issues联系我们。
