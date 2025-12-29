import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_sim(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    # print(sim)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

class SafeCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(reduction='none', **kwargs)
        self.eps = eps

    def forward(self, input, target):        
        # 添加输入数值约束（关键点1）
        input = torch.clamp(input, min=-1e3, max=20.0)  # 防御极端logits值
        
        # 执行log(Softmax)并添加下界（关键点2）
        log_pred = F.log_softmax(input, dim=1)
        log_pred = log_pred.clamp(min=torch.finfo(log_pred.dtype).min + self.eps*10)  # 预防log(0)
        
        # 手动计算交叉熵（保持原有逻辑但防御计算）
        loss = - log_pred.gather(1, target.view(-1,1)).squeeze()
        return loss



class MINE(nn.Module):
    """ 互信息神经估计器: https://arxiv.org/abs/1801.04062 """
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.eps = 1e-8
    def forward(self, x, y):
        # x: [N, C], causal特征
        # y: [N, C], confounder特征
        batch_size = x.size(0)
        
        # 联合分布样本 (x_i, y_i)
        xy_joint = torch.cat([x, y], dim=1)        # [N, 2C]
        t_joint = self.net(xy_joint)               # [N, 1]
        
        # 边缘分布样本 (x_i, y_j), i≠j
        shuffled_idx = torch.randperm(batch_size)
        y_shuffled = y[shuffled_idx]               # 随机打乱y
        xy_marginal = torch.cat([x, y_shuffled], dim=1)  # [N, 2C]
        t_marginal = self.net(xy_marginal)         # [N, 1]
        t_joint = t_joint.clamp(min=-30.0, max=30.0)
        t_marginal = t_marginal.clamp(min=-30.0, max=30.0)
        # 计算互信息下界 (与论文公式对应)
        # mi_lb = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
        mi_lb = (t_joint.mean() - torch.logsumexp(t_marginal, dim=0)).clamp(min=self.eps)
        return mi_lb  # 最大化互信息下界 → 最小化此损失 (设为负数)

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

if __name__ == '__main__':

    fg_embedding = torch.randn((4, 12))
    bg_embedding = torch.randn((4, 12))
    # print(fg_embedding, bg_embedding)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
