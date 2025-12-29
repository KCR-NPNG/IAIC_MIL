from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import qmc


class Sampler(ABC):
    """
    Base class for replicated design sampling.
    """

    def _build_replicated_design(self, a, b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        a: ndarray
          The masks values for the sampling matrix A.
        b: ndarray
          The masks values for the sampling matrix B.

        Returns
        -------
        c: ndarray
          The new replicated design matrix C generated from A & B.
        """
        c = np.array([a.copy() for _ in range(a.shape[-1])])
        for i in range(len(c)):
            c[i, :, i] = b[:, i]
        c = c.reshape(-1, a.shape[-1]) 
        return c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()

class ScipySobolSequence(Sampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of 
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __init__(self, scramble=False):
        self.scramble = scramble
        self.seed = 42

    def __call__(self, dimension, nb_design):

        sampler = qmc.Sobol(dimension*2, scramble=self.scramble, seed=self.seed)
        sampler.fast_forward(10)
        ab = sampler.random(nb_design)
        a, b = ab[:, :dimension], ab[:, dimension:]
        # a = (a > 0.5).astype(int)
        # b = (b > 0.5).astype(int)
        c = self._build_replicated_design(a, b)
        return np.concatenate([a, b, c], 0)


class AttentionAggregatorModel(nn.Module):
    def __init__(self, feature_dim=512, attn_hidden_dim=128):
        super(AttentionAggregatorModel, self).__init__()
        # 注意力权重子网络：输入每个样本的特征，输出一个标量分数
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)  # 输出注意力分数
        )
        # 输出预测网络：输入加权后的全局特征，输出最终预测值
        self.output_net = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        # x维度: (batch_size, 507, 512)
        # 1. 计算注意力分数并归一化为权重
        attn_scores = self.attention_net(x)            # (batch_size, 507, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, 507, 1), 对每组507个样本的得分做softmax
        # 2. 根据权重对特征加权求和，得到每组的全局特征表示
        weighted_feature = (attn_weights * x).sum(dim=1)   # (batch_size, 512)
        # 3. 将全局特征输入输出层，得到每组的预测结果
        group_output = self.output_net(weighted_feature)   # (batch_size, 1)
        return group_output.squeeze(-1)  # 返回维度 (batch_size,)
