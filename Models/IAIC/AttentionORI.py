import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            # A = F.softmax(A, dim=1)  # softmax over N
            A = A.sigmoid()
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
            # A = A.sigmoid()

        return A  ### K x N

class Classifier_1fc_ori(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc_ori, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, confounder_path=False):
        super(Classifier_1fc, self).__init__()
        self.confounder_path = confounder_path
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

        if confounder_path:
            conf_list = []
            for i in confounder_path:
                print(i)
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, n_channels).float())
            conf_tensor = torch.cat(conf_list, 0) 
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 256
            self.W_q = nn.Linear(n_channels, joint_space_dim)
            self.W_k = nn.Linear(n_channels, joint_space_dim)
            self.fc =  nn.Linear(n_channels*2, n_classes)
        else:
            self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)

        if self.confounder_path:
            M = x
            device = M.device
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            A = torch.mm(conf_k, bag_q.transpose(0, 1))
            A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            M = torch.cat((M, conf_feats),dim=1)
            pred = self.fc(M)
            Y_hat = torch.ge(pred, 0.5).float()
            return pred, M, A
        else:
            pred = self.fc(x)

            return pred, x, None

# class Attention_with_Classifier(nn.Module):
#     def __init__(self, args, L=512, D=128, K=1, num_cls=2, droprate=0, confounder_path=False):
#         super(Attention_with_Classifier, self).__init__()
        
        
#         if confounder_path:
#             self.attention = Attention_Gated(L, D, K)
#             self.confounder_path = confounder_path
#             conf_list = []
#             for i in confounder_path:
#                 conf_list.append(torch.from_numpy(np.load(i)).view(-1, L).float())
#             conf_tensor = torch.cat(conf_list, 0) 
#             self.register_buffer("confounder_feat",conf_tensor)
#             joint_space_dim = 256
#             dropout_v = 0.5
#             self.W_q = nn.Linear(L, joint_space_dim)
#             self.W_k = nn.Linear(L, joint_space_dim)
#             self.classifier =  nn.Linear(L*2, num_cls)
#             self.dropout = nn.Dropout(dropout_v)
#         else:
#             self.confounder_path = False
#             self.attention = Attention_Gated(L, D, K)
#             self.classifier = Classifier_1fc(L, num_cls, droprate)

#     def forward(self, x): ## x: N x L
#         AA = self.attention(x)  ## K x N
#         M = torch.mm(AA, x) ## K x L
#         # M = F.relu(M) + x
#         AA_neg = 1-AA
#         M_neg = torch.mm(AA_neg, x)  

#         if self.confounder_path:
#             # pred_ori, _, _ = self.classifier(M) ## K x num_cls
#             device = M.device
#             bag_q = self.W_q(M)
#             conf_k = self.W_k(self.confounder_feat)
#             A = torch.mm(conf_k, bag_q.transpose(0, 1))
#             A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
#             conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
#             M = torch.cat((M, conf_feats),dim=1)
#             pred = self.classifier(M)
#             Y_hat = torch.ge(pred, 0.5).float()
#             return pred, M, A
#         else:
#             pred, _, _ = self.classifier(M) ## K x num_cls
#             return pred, M, AA, M_neg, AA_neg


        # return Y_prob, M, A

class Attention_with_Classifier_ori(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier_ori, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc_ori(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class Disentangler(nn.Module): # 
    def __init__(self, L, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False) # 激活头
        self.bn_head = nn.BatchNorm2d(1)
        self.liner = nn.Linear(L, 1)
        # 可学习的动态阈值生成器
        self.threshold_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # 全局平均池化 [B,C,H,W] → [B,C,1,1]
            nn.Flatten(),               # [B,C,1,1] → [B,C]
            nn.Linear(L, 32),           # 输入特征维度L必须与Linear层输入匹配
            nn.ReLU(),
            nn.Linear(32, 1),           # 输出每个样本的阈值τ
            nn.Sigmoid()                # τ ∈ (0,1)
        )

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x):
        # print(x.size())
        assert x.dim() == 2, "输入应为二维张量 [batch, features]"
        x = x.view(1, 1, x.size(0), x.size(1))  # 明确调整为 [1, 1, N, L]
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(1)
        # print(x.size())
        N, C, H, W = x.size() # N 是多少张图片
        ccam = torch.sigmoid(self.bn_head(self.activation_head(x))) # [N, H, W] 归一化\
        # ccam = ccam.squeeze(dim=(0,1))
        ccam = ccam.squeeze(0)
        ccam = ccam.squeeze(0)
        ccam = self.liner(ccam)
        ccam = torch.transpose(ccam, 1, 0)  # KxN
        # 目前这种平滑方式可能会压缩特征区分
        ccam_adjusted = self.sigmoid(ccam.mean(dim=0, keepdim=True)) * ccam
       
        fg_feats = torch.matmul(ccam_adjusted, x) / H                # [N, 1, C] 前景和背景的特征表示
        bg_feats = torch.matmul(1 - ccam_adjusted, x) / H         # [N, 1, C]
        # print(fg_feats.size(),bg_feats.size(),ccam_adjusted.size())

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam_adjusted

class Disentangler1(nn.Module):
    def __init__(self, feature_dim=512):
        super(Disentangler1, self).__init__()
        self.feature_dim = feature_dim

        self.activation_head = nn.Linear(feature_dim, 1, bias=False)  # 输出每个样本的特征权重

        self.sliding_window = nn.Sequential(
            nn.Linear(3 * (feature_dim//3), 64),  # 假设划分特征为3段
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        # 动态阈值
        self.dynamic_tau = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出范围 [0,1]
        )

    def forward(self, x):
        """
        输入 x : [B, L] = [4000, 512]
        输出 : fg_feats, bg_feats, weights
        """
        B, L = x.shape
        
        # if hasattr(self, 'sliding_window'):
        #     segments = x.view(B, 3, -1)            # 划分特征为3段 → [B, 3, L/3] (确保可整除)
        #     interacted = self.sliding_window(segments.view(B, -1))  # 融合后的交互特征 → [B, L]
        #     x = x + interacted                     # 残差连接保留原始信息

        raw_weights = self.activation_head(x)      # [B, 1] 每个样本的权重标量
        tau = self.dynamic_tau(x)                  # [B, 1]
        adjusted_weights = torch.sigmoid(raw_weights - tau)  # 软阈值操作 → [B, 1]

        # fg_feats = torch.matmul(adjusted_weights, x) / B                # [N, 1, C] 前景和背景的特征表示
        # bg_feats = torch.matmul(1 - adjusted_weights, x) / B        # [N, 1, C]
        fg_feats = adjusted_weights * x            # 前景特征：加权激活部分 [B, L]
        bg_feats = (1 - adjusted_weights) * x      # 背景特征：剩余部分
        # print(fg_feats.size(), bg_feats.size(), adjusted_weights.size())
        return fg_feats.mean(dim=0, keepdim=True), bg_feats.mean(dim=0, keepdim=True), adjusted_weights


class Attention_with_Classifier(nn.Module):
    def __init__(self, args, L=512, D=128, K=1, num_cls=2, droprate=0, confounder_path=False, confounder=None):
        super(Attention_with_Classifier, self).__init__()
        
        joint_space_dim = 512
        dropout_v = 0.5
        self.confounder_path = confounder_path
        self.W_q = nn.Linear(L, joint_space_dim)
        self.W_k = nn.Linear(L, joint_space_dim)
        self.dropout = nn.Dropout(dropout_v)
        self.ac_head = Disentangler(L, K)
        # self.ac_head = Disentangler1(L, K, 512)
        self.ac_head1 = Disentangler(L//2, K)
        self.attention = Attention_Gated(L, D, K)
        self.attention1 = Attention_Gated(joint_space_dim, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        self.L = L

    def forward(self, x, confounder, testLabel): ## x: N x L
        # AA = self.attention(x)  ## K x N
        # M = torch.mm(AA, x) ## K x L
        # M = F.relu(M) + x
        # AA_neg = 1-AA
        # M_neg = torch.mm(AA_neg, x)  
        M, M_neg, AA = self.ac_head(x) # 现在更新激活方式为注意力激活，1-激活头=混杂特征的激活(1, 512)
        # print(M.size())
        # print('1111111111111111', confounder)
        if testLabel:
            # print(11111)
            conf_list = []
            confounder_files = glob.glob(os.path.join(self.confounder_path, '*.npy'))
            for i in confounder_files:
                # print(i)
                conf_data = np.load(i)
                if np.any(np.isnan(conf_data)) or np.any(np.isinf(conf_data)):
                    print(f"NaN or Inf detected in confounder data file: {i}")
                    continue  # 跳过当前文件或采取其他补救措施
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, self.L).float())
            if len(conf_list) > 0 and all(t.numel() > 0 for t in conf_list):
                conf_tensor = torch.cat(conf_list, dim=0).to('cuda')
            else:
                conf_tensor = torch.empty(0).to('cuda')  # 或者 raise Warning/Skip
            self.register_buffer("confounder_feat",conf_tensor)
            bag_q = self.W_q(M)
            if self.confounder_feat.numel() > 0:  # 检查是否非空
                conf_k = self.W_k(self.confounder_feat)
                A = F.relu(bag_q + conf_k)  # 只有 confounder_feat 存在时才参与计算
            else:
                # 如果没有 confounder_feat，则 A 只有 bag_q
                A = F.relu(bag_q) 
            M, M_neg, AA = self.ac_head(A) 
            pred, _, _ = self.classifier(M)
            return pred, M, AA, M_neg
        else:
            if confounder != None:
                # print(2222)
                # print('yyyyyy')
                # pred_ori, _, _ = self.classifier(M) ## K x num_cls
                # conf_list = torch.tensor(confounder)
                if len(confounder) > 0:
                    conf_list = []
                    for i in confounder:
                        if np.any(np.isnan(i)) or np.any(np.isinf(i)):
                            print(f"NaN or Inf detected in confounder data: {i}")
                            continue  # 跳过当前 i 或进行其他处理
                        conf_list.append(torch.from_numpy(i).view(-1, self.L).float())
                    if len(conf_list) > 0 and all(t.numel() > 0 for t in conf_list):
                        conf_tensor = torch.cat(conf_list, dim=0).to('cuda')
                    else:
                        conf_tensor = torch.empty(0).to('cuda')  # 或者 raise Warning/Skip
                    # conf_list = torch.stack(confounder)
                    # conf_tensor = torch.cat(conf_list, 0).to('cuda')
                    
                    self.register_buffer("confounder_feat",conf_tensor)
                    device = M.device
                    bag_q = self.W_q(M)
                    if self.confounder_feat.numel() > 0:  # 检查是否非空
                        conf_k = self.W_k(self.confounder_feat)
                        A = F.relu(bag_q + conf_k)  # 只有 confounder_feat 存在时才参与计算
                    else:
                        # 如果没有 confounder_feat，则 A 只有 bag_q
                        A = F.relu(bag_q) 
                    # conf_k = self.W_k(self.confounder_feat)
                    # A = torch.cat((bag_q, conf_k), dim=1)
                    # A=F.relu(bag_q+conf_k) # 30*256
                    # weights = self.attention1(A) #torch.Size([1, 30]) torch.Size([1, 256]) torch.Size([30, 256])
                    M, M_neg, AA = self.ac_head(A) # torch.Size([1, 1, 1, 512]) torch.Size([1, 1, 1, 512]) torch.Size([1, 4000])
                    # print(M.size(), M_neg.size())
                    # M = bag_q * weights + conf_k * weights
                    # M = torch.mm(weights, A)  # 1*256
                    # M_neg = torch.mm(1-weights, A)
                    pred, _, _ = self.classifier(M) ## K x num_cls
                else:
                    print("[⚠️] Warning: empty confounder passed!")
                    self.confounder_feat = None

                # A = torch.mm(conf_k, bag_q.transpose(0, 1))
                
                # A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
                # conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
                # M = torch.cat((M, conf_feats),dim=1)
                # pred = self.classifier1(M)
                # Y_hat = torch.ge(pred, 0.5).float()
                return pred, M, AA, M_neg
            else:
                pred, _, _ = self.classifier(M) ## K x num_cls
                return pred, M, AA, M_neg

class Attention_with_Classifier_nodistangle(nn.Module):
    def __init__(self, args, L=512, D=128, K=1, num_cls=2, droprate=0, confounder_path=False, confounder=None):
        super(Attention_with_Classifier_nodistangle, self).__init__()
        
        joint_space_dim = 512
        dropout_v = 0.5
        self.confounder_path = confounder_path
        self.W_q = nn.Linear(L, joint_space_dim)
        self.W_k = nn.Linear(L, joint_space_dim)
        self.dropout = nn.Dropout(dropout_v)
        self.ac_head = Disentangler(L, K)
        # self.ac_head = Disentangler1(L, K, 512)
        self.ac_head1 = Disentangler(L//2, K)
        self.attention = Attention_Gated(L, D, K)
        self.attention1 = Attention_Gated(joint_space_dim, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        self.L = L

    def forward(self, x, confounder, testLabel): ## x: N x L
        # AA = self.attention(x)  ## K x N
        # M = torch.mm(AA, x) ## K x L
        # M = F.relu(M) + x
        # AA_neg = 1-AA
        # M_neg = torch.mm(AA_neg, x)  
        M, M_neg, AA = self.ac_head(x) # 现在更新激活方式为注意力激活，1-激活头=混杂特征的激活(1, 512)
        # print(M.size())
        # print('1111111111111111', confounder)
        if testLabel:
            # print(11111)
            conf_list = []
            confounder_files = glob.glob(os.path.join(self.confounder_path, '*.npy'))
            for i in confounder_files:
                # print(i)
                conf_data = np.load(i)
                if np.any(np.isnan(conf_data)) or np.any(np.isinf(conf_data)):
                    print(f"NaN or Inf detected in confounder data file: {i}")
                    continue  # 跳过当前文件或采取其他补救措施
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, self.L).float())
            if len(conf_list) > 0 and all(t.numel() > 0 for t in conf_list):
                conf_tensor = torch.cat(conf_list, dim=0).to('cuda')
            else:
                conf_tensor = torch.empty(0).to('cuda')  # 或者 raise Warning/Skip
            self.register_buffer("confounder_feat",conf_tensor)
            bag_q = self.W_q(M)
            if self.confounder_feat.numel() > 0:  # 检查是否非空
                conf_k = self.W_k(self.confounder_feat)
                A = F.relu(bag_q + conf_k)  # 只有 confounder_feat 存在时才参与计算
            else:
                # 如果没有 confounder_feat，则 A 只有 bag_q
                A = F.relu(bag_q) 
            M, M_neg, AA = self.ac_head(A) 
            pred, _, _ = self.classifier(M)
            return pred, M, AA, M_neg
        else:
            if confounder != None:
                # print(2222)
                # print('yyyyyy')
                # pred_ori, _, _ = self.classifier(M) ## K x num_cls
                # conf_list = torch.tensor(confounder)
                if len(confounder) > 0:
                    conf_list = []
                    for i in confounder:
                        if np.any(np.isnan(i)) or np.any(np.isinf(i)):
                            print(f"NaN or Inf detected in confounder data: {i}")
                            continue  # 跳过当前 i 或进行其他处理
                        conf_list.append(torch.from_numpy(i).view(-1, self.L).float())
                    if len(conf_list) > 0 and all(t.numel() > 0 for t in conf_list):
                        conf_tensor = torch.cat(conf_list, dim=0).to('cuda')
                    else:
                        conf_tensor = torch.empty(0).to('cuda')  # 或者 raise Warning/Skip
                    # conf_list = torch.stack(confounder)
                    # conf_tensor = torch.cat(conf_list, 0).to('cuda')
                    
                    self.register_buffer("confounder_feat",conf_tensor)
                    device = M.device
                    bag_q = self.W_q(M)
                    if self.confounder_feat.numel() > 0:  # 检查是否非空
                        conf_k = self.W_k(self.confounder_feat)
                        A = F.relu(bag_q + conf_k)  # 只有 confounder_feat 存在时才参与计算
                    else:
                        # 如果没有 confounder_feat，则 A 只有 bag_q
                        A = F.relu(bag_q) 
                    # conf_k = self.W_k(self.confounder_feat)
                    # A = torch.cat((bag_q, conf_k), dim=1)
                    # A=F.relu(bag_q+conf_k) # 30*256
                    # weights = self.attention1(A) #torch.Size([1, 30]) torch.Size([1, 256]) torch.Size([30, 256])
                    M, M_neg, AA = self.ac_head(A) # torch.Size([1, 1, 1, 512]) torch.Size([1, 1, 1, 512]) torch.Size([1, 4000])
                    # print(M.size(), M_neg.size())
                    # M = bag_q * weights + conf_k * weights
                    # M = torch.mm(weights, A)  # 1*256
                    # M_neg = torch.mm(1-weights, A)
                    pred, _, _ = self.classifier(M) ## K x num_cls
                else:
                    print("[⚠️] Warning: empty confounder passed!")
                    self.confounder_feat = None

                # A = torch.mm(conf_k, bag_q.transpose(0, 1))
                
                # A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
                # conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
                # M = torch.cat((M, conf_feats),dim=1)
                # pred = self.classifier1(M)
                # Y_hat = torch.ge(pred, 0.5).float()
                return pred, M, AA, M_neg
            else:
                pred, _, _ = self.classifier(M) ## K x num_cls
                return pred, M, AA, M_neg