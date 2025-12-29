from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def divide_list_equally(input_list, num_sublists):

    sublist_length = len(input_list) // num_sublists
    remainder = len(input_list) % num_sublists
    sublists = [input_list[i * sublist_length: (i + 1) * sublist_length] for i in range(num_sublists - 1)]
    sublists.append(input_list[(num_sublists - 1) * sublist_length:])
    return sublists

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

# def explain_patches(tattFeats , model):#tattfeats:n*512
#     # Apply temperature scaling
#     temp = 40
#     pred = model(tattFeats)

#     num_patches = tattFeats.shape[0]
#     # loss = nn.CrossEntropyLoss(pred / temp,lable)
#     mse_loss = nn.MSELoss()

#     node_mask = torch.zeros(num_patches)
#     batch_size = 10
#     pred_labels = torch.argmax(pred, dim=1)

#     for b in range(0, num_patches, batch_size):
#         b = b // batch_size
#         start = b * batch_size
#         end = min((b+1) * batch_size, num_patches)

#         before_part = tattFeats[:start]  #######保留前 start 个样本
#         after_part = tattFeats[end:]     ####### 保留 end 之后的样本
#         res_patches = torch.cat((before_part, after_part), dim=0)  # 在第一个维度上拼接，得到 (n - (end - start)) x 512 的特征
#         with torch.no_grad():
#             pred_res = model(res_patches)
#         before_part = pred_res[:start]
#         middle_part = pred[start:end]
#         after_part = pred_res[start:]  
#         new_pred =  torch.cat((before_part,middle_part))
#         new_pred = torch.cat((new_pred,after_part))
   
#         # lf = torch.nn.CrossEntropyLoss()
#         # lb = torch.ones((end - start,), dtype=torch.int)* label
#         # delta = lf(new_pred,pred_labels.long()) 
#         # M = ot.dist(pred,new_pred,metric='euclidean')#距离矩阵
#         # distance = ot.emd2(pred*temp,new_pred*temp,M)#temp增大差异
#         loss = mse_loss(pred*temp, new_pred*temp)
#         node_mask[start:end] = loss

#     node_mask = node_mask.detach().numpy()
#     node_mask = (node_mask - np.min(node_mask)) / (np.max(node_mask) - np.min(node_mask))#归一化

#     return node_mask

def explain_patches(tattFeats , label, model,same_mask_num):
    pred = model(tattFeats)
    num_patches = tattFeats.shape[0]
    patches_label = label.clone().detach().to('cuda').long().expand(num_patches)
    loss_fcn = nn.CrossEntropyLoss().to('cuda')
    loss1 = loss_fcn(pred[0], patches_label)
    # get_weight = Focal_rev_weight().to('cuda')
    # weight = get_weight(pred, patches_label)
    node_mask = torch.zeros(num_patches, device='cuda')
    allocated_memory = torch.cuda.memory_allocated()
    for b in range(0, num_patches,same_mask_num):
        b = b // same_mask_num
        start = b * same_mask_num
        end = min((b+1) * same_mask_num, num_patches)
        if end <= num_patches:
            before_part = tattFeats[:start]  #######保留前 start 个样本
            be=patches_label[:start]
            after_part = tattFeats[end:]     ####### 保留 end 之后的样本
            af=patches_label[end:]
            res_patches = torch.cat((before_part, after_part), dim=0)  # 在第一个维度上拼接，得到 (n - (end - start)) x 512 的特征
            res_label=torch.cat((be, af), dim=0)
            end1 = end
            with torch.no_grad():
                pred_res = model(res_patches) #8*1024->8*2
            loss2=loss_fcn(pred_res[0],res_label)
            node_mask[start:end] = loss2-loss1
        else: 
            break
    # node_mask *= weight
    return node_mask

# def explain_patches(tattFeats , label, model):

#     pred = model(tattFeats)
#     num_patches = tattFeats.shape[0]
#     patches_label = label.clone().detach().to('cuda').long().expand(num_patches)
#     loss_fcn = nn.CrossEntropyLoss().to('cuda')
#     get_weight = Focal_rev_weight().to('cuda')
#     weight = get_weight(pred, patches_label)
#     node_mask = torch.zeros(num_patches, device='cuda')
#     batch_size = 10
#     for b in range(0, num_patches, batch_size):
#         b = b // batch_size
#         start = b * batch_size
#         end = min((b+1) * batch_size, num_patches)

#         before_part = tattFeats[:start]  #######保留前 start 个样本
#         after_part = tattFeats[end:]     ####### 保留 end 之后的样本
#         res_patches = torch.cat((before_part, after_part), dim=0)  # 在第一个维度上拼接，得到 (n - (end - start)) x 512 的特征
#         with torch.no_grad():
#             pred_res = model(res_patches)
#         before_part = pred_res[:start]
#         middle_part = pred[start:end]
#         after_part = pred_res[start:]  
#         new_pred =  torch.cat((before_part,middle_part))
#         new_pred = torch.cat((new_pred,after_part))
#         delta = loss_fcn ((pred - new_pred), patches_label)
#         node_mask[start:end] = delta

#     return node_mask

class Focal_rev_weight(nn.Module):
    def __init__(self, alpha=1, gamma=5):
        super(Focal_rev_weight, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        weight =  1/(1 - pt) ** self.gamma 
        return weight

def roc_threshold(label, prediction):
    # print(label)
    # print(prediction)
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    try:
        c_auc = roc_auc_score(label, prediction)
        if c_auc == float('inf'):
            c_auc = 0.0
    except ValueError as e:
        if "Only one class present in y_true" in str(e):
            # 如果标签全一致，设置AUC分数为默认值或者跳过计算
            c_auc = 0.0
        else:
            # 如果是其他异常，继续引发异常
            raise e
    if threshold_optimal == float('inf') or threshold_optimal == float('-inf'):
        threshold_optimal = 0.5  # 设置为默认值
    # c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):
    
    
    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc
