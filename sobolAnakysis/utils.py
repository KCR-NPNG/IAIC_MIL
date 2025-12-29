import torch
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

class StabilitySobolAnalyzer:
    def __init__(self, classifier, num_design=128, batch_size=32, seed=42, device='cuda'):
        """
        参数说明:
            classifier: 分类器模型，接收聚合后的特征向量，输出logits
            num_design: Sobol采样设计点数（建议用2的幂次）
            batch_size: 每批次处理的掩码数量（根据GPU显存调整）
            seed: 随机种子（确保结果可复现）
            device: 计算设备
        """
        self.classifier = classifier.eval()
        self.num_design = num_design
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def _generate_sobol_masks(self, dim: int) -> tuple:
        """生成Sobol二进制掩码矩阵（A/B/Ci组）"""
        # 创建独立种子生成器避免全局种子干扰
        rng = np.random.default_rng(self.seed)
        
        # 生成A和B组的基础掩码
        sobol = qmc.Sobol(d=dim, scramble=True, seed=rng.integers(0, 2**16))
        A = sobol.random(self.num_design) > 0.5  # [N, D] bool矩阵
        B = sobol.random(self.num_design) > 0.5

        # 生成Ci组的交叉掩码
        Ci_masks = []
        for i in range(dim):
            # 将A的第i列替换为B的第i列
            Ci = A.copy()
            Ci[:, i] = B[:, i]
            Ci_masks.append(Ci)

        # 合并所有掩码并转换为tensor
        all_masks = np.vstack([A, B, np.vstack(Ci_masks)])
        return torch.from_numpy(all_masks).float().to(self.device)  # [N*(D+2), D]

    def _aggregate_features(self, feat_matrix: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """根据二值掩码聚合特征（均值池化）"""
        # feat_matrix: [B, D], masks: [N, B]
        masked_feats = feat_matrix.unsqueeze(0) * masks.unsqueeze(-1)  # [N, B, D]
        
        # 计算有效特征均值（避免除零）
        summed = masked_feats.sum(dim=1)  # [N, D]
        counts = masks.sum(dim=1, keepdim=True)  # [N, 1]
        counts = torch.maximum(counts, torch.ones_like(counts))  # 确保非空
        return summed / counts  # [N, D]

    @torch.no_grad()
    def analyze(self, tattFeats: torch.Tensor, tslideLabel: torch.Tensor) -> torch.Tensor:
        """
        执行Sobol敏感性分析主流程
        返回: 示例重要性得分 [batchsize]
        """
        # 数据准备
        tattFeats = tattFeats.to(self.device)  # [B, D]
        # label = tslideLabel.expand(self.num_design * 2).to(self.device)  # 拓展标签用于匹配AB组
        label_idx = tslideLabel.long().item()
        
        # Step 1: 生成所有Sobol采样掩码
        masks = self._generate_sobol_masks(dim=tattFeats.size(0))  # [N*(D+2), B]
        
        # Step 2: 分批次处理掩码
        num_masks = masks.size(0)
        all_logits = []
        for idx in range(0, num_masks, self.batch_size):
            batch_masks = masks[idx:idx+self.batch_size]
            
            # 特征聚合并推理
            agg_feats = self._aggregate_features(tattFeats, batch_masks)
            outputs = self.classifier(agg_feats) # [B, 1] -> [B]
            main_logits = outputs[0]                      # 提取主logits张量
            selected_logits = main_logits[:, label_idx]   # 正确: 在张量上索引
            # selected_logits = outputs[:, label_idx]
            all_logits.append(selected_logits.cpu())
        
        # 合并所有预测结果
        all_logits = torch.cat(all_logits)  # [N*(D+2)]
        
        # Step 3: 拆解A/B/Ci组的预测结果
        a_logits = all_logits[:self.num_design]                    # A组
        b_logits = all_logits[self.num_design:2*self.num_design]   # B组
        ci_logits = all_logits[2*self.num_design:]                 # Ci组 [D*N]
        
        # 通过交叉验证方式计算每个特征的重要性
        std_total = torch.var(torch.cat([a_logits, b_logits]))     # 总体方差
        importance = []
        for i in range(tattFeats.size(0)):
            # 提取该示例对应的Ci组结果
            start = i * self.num_design
            end = (i+1) * self.num_design
            ci = ci_logits[start:end]
            
            # Sobol总效应指数公式（Saltelli 2010）
            sti = (torch.mean((a_logits - ci)**2) / (2 * std_total)).item()
            importance.append(sti)
        
        return torch.tensor(importance)  # [B]

def sobolAnalysis(tattFeats, label, classifier, params):

    batch_size, feature_dim = tattFeats.shape
    analyzer = StabilitySobolAnalyzer(
    classifier=classifier,
    num_design=params.nb_design,
    batch_size=params.mask_batch,
    seed=42,
    device = params.device
    )
    importance = analyzer.analyze(tattFeats, label)
    return importance