import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import pandas
import random
import torch.nn as nn
from utils import get_cam_1d
import torch.nn.functional as F
import numpy as np
from utils import eval_metric,explain_patches
import subprocess
import time
import faiss
from sobolAnakysis.sobol2WSI import sobol_analysis
from sobolAnakysis.utils import sobolAnalysis
from Models.IAIC.network import DimReduction
from Models.IAIC.AttentionORI import Attention_Gated as Attention
from Models.IAIC.AttentionORI import Attention_with_Classifier, Classifier_1fc
from Models.IAIC.loss import SimMinLoss, MINE




parser = argparse.ArgumentParser(description='abc')
testMask_dir = '/home/Data/kongchaoran/camelyon16/baseline/testLabel/reference.csv' ## Point to the Camelyon test set mask location

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[200]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./weight/final/tcga/main_tcga_nsclc_scribble0.6_resnet_fold3_group50_select15', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=50, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.0002, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA_dir', default='/home/disk/kongchaoran/Data/datasets/features/TCGA_NSCLC_SCRIBBLE_0.6_RESNET.pkl', type=str)
# parser.add_argument('--mDATA0_dir_train0', default='/home/disk/kongchaoran/Data/tcga/IAIC/feature/fold_1/mDATA_folder/train.pkl', type=str)  ## Train Set
# parser.add_argument('--mDATA0_dir_val0', default='/home/disk/kongchaoran/Data/tcga/IAIC/feature/fold_1/mDATA_folder/val.pkl', type=str)      ## Validation Set
# parser.add_argument('--mDATA_dir_test0', default='/home/disk/kongchaoran/Data/tcga/IAIC/feature/fold_1/mDATA_folder/test.pkl', type=str)     ## Test Set
# parser.add_argument('--mDATA0_dir_train0', default='/home/disk/kongchaoran/Data/camelyon16/IAIC-MIL/feature/fold_1/mDATA_folder/train.pkl', type=str)  ## Train Set
# parser.add_argument('--mDATA0_dir_val0', default='/home/disk/kongchaoran/Data/camelyon16/IAIC-MIL/feature/fold_1/mDATA_folder/val.pkl', type=str)      ## Validation Set
# parser.add_argument('--mDATA_dir_test0', default='/home/disk/kongchaoran/Data/camelyon16/IAIC-MIL/feature/fold_1/mDATA_folder/test.pkl', type=str)     ## Test Set
parser.add_argument('--numGroup', default=50, type=int)
parser.add_argument('--total_instance', default=4, type=int)
# parser.add_argument('--numGroup_test', default=50, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--feats_size', default=1024, type=int)
parser.add_argument('--nb_design', default=8, type=int)
parser.add_argument('--mask_batch', default=256, type=int)
parser.add_argument('--select_num', default=10, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', default='False')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='MaxS', type=str)   ## MaxMinS, MaxS, MinS, AFS
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--instances_per_group', default=15, type=int)
parser.add_argument('--same_mask_num', default=2, type=int)
parser.add_argument('--seed', type=int, default=22, metavar='S',
                    help='random seed (default: 1)') # fold1-5:12\22\32\42\52
parser.add_argument('--split_ratio', type=int, default=0.7, metavar='S',
                    help='train set ratio')
parser.add_argument('--c_path', nargs='+', default='/home/disk/kongchaoran/code/IAIC_NEW/IAIC-MIL/datasets_deconf/main_tcga_nsclc_scribble0.6_resnet_fold3_group50_select15/', type=str,help='directory to confounders')

                    

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)

def main():
    start = time.time()
    params = parser.parse_args()
    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))

    in_chn = 1024
    mDim = params.feats_size//2

    classifier = Classifier_1fc(mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(mDim).to(params.device)
    dimReduction = DimReduction(params.feats_size, mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(params, L=mDim, num_cls=params.num_cls, droprate=params.droprate_2, confounder_path=params.c_path).to(params.device)

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)

    # ce_cri = [torch.nn.CrossEntropyLoss(reduction='none').to(params.device),SimMinLoss(metric='cos').to(params.device)]
    ce_cri = [torch.nn.CrossEntropyLoss(reduction='none').to(params.device), MINE(feat_dim=512).to(params.device)]

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log2_maxminx.txt')
    save_dir = os.path.join(params.log_dir, 'best_model.pth')
    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')

    print("00000")
    # with open(params.mDATA0_dir_train0, 'rb') as f:
    #     mDATA_train = pickle.load(f)
    # with open(params.mDATA0_dir_val0, 'rb') as f:
    #     mDATA_val = pickle.load(f)
    # with open(params.mDATA_dir_test0, 'rb') as f:
    #     mDATA_test = pickle.load(f)
    (train_names, train_feats, train_labels), (test_names,  test_feats,  test_labels) = readPkl(params.mDATA_dir, split_ratio=params.split_ratio, seed=params.seed)
    print_log(f'training slides: {len(train_names)}, test slides: {len(test_names)}', log_file)
    # SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train)
    # SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val)
    # SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA(mDATA_test)
    # SlideNames_combined= SlideNames_val + SlideNames_test
    # FeatList_combined = FeatList_val + FeatList_test  # List[Tensor] 拼接（外层 list 拼接即可）
    # Label_combined = Label_val + Label_test
    # print_log(f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}, test slides: {len(SlideNames_test)}', log_file)

    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr,  weight_decay=params.weight_decay)

    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    best_auc = 0
    best_epoch = -1
    test_auc = 0
    Confound = None

    for ii in range(params.EPOCH):

        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f' current learn rate {curLR}', log_file )

        Confound = train_attention_preFeature_IAIC(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=attCls, mDATA_list=(train_names, train_feats, train_labels), ce_cri=ce_cri,
                                                   optimizer0=optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii, params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, total_instance=params.total_instance, distill=params.distill_type,same_mask_num=params.same_mask_num,instance_per_group=params.instances_per_group, Confound=Confound)
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)

        if ii >= 150:
            auc_val = test_attention_IAIC_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,UClassifier=attCls, mDATA_list=(test_names,  test_feats,  test_labels), criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, total_instance=params.total_instance_test, distill=params.distill_type,same_mask_num=params.same_mask_num,instance_per_group=params.instances_per_group, Confound=Confound)
            print_log(' ', log_file)
        # print_log(f'>>>>>>>>>>> Test Epoch: {ii}', log_file)
        # tauc = test_attention_IAIC_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
        #                                                 UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test, total_instance=params.total_instance_test, distill=params.distill_type,same_mask_num=params.same_mask_num,instance_per_group=params.instances_per_group_test, Confound=Confound)
        # print_log(' ', log_file)



        if ii > int(params.EPOCH*0.8):
            if auc_val > best_auc:
                best_auc = auc_val
                best_epoch = ii
                # test_auc = tauc
                if params.isSaveModel:
                    tsave_dict = {
                        'classifier': classifier.state_dict(),
                        'dim_reduction': dimReduction.state_dict(),
                        'attention': attention.state_dict(),
                        'att_classifier': attCls.state_dict()
                    }
                    torch.save(tsave_dict, save_dir)

            print_log(f' test auc: {best_auc}, from epoch {best_epoch}', log_file)

        scheduler0.step()
        scheduler1.step()
    end = time.time()
    print('程序运行时间为: %s Seconds'%(end-start))

def test_attention_IAIC_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, criterion=None,  params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS',same_mask_num=4,instance_per_group=1, Confound=None):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list
    # instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst].to(params.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))

                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    converted_list = [sst.tolist() for sst in index_chunk_list]
                    index_chunk_list = [lst for lst in converted_list if lst]
                    # index_chunk_list = [feat_index]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)
                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs

                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict[0])
                        '''
                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                        '''
                        # patch_mask=explain_patches(tattFeats,tslideLabel,classifier,same_mask_num)
                        patch_mask = sobol_analysis(tattFeats, tslideLabel, classifier, params)
                        # patch_mask = sobolAnalysis(tattFeats, tslideLabel, classifier, params)
                        patch_mask = torch.tensor(patch_mask).to(params.device)
                        # patch_mask=patch_mask.to(params.device)
                        _,sort_idx=torch.sort(patch_mask,descending=True)
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_maxs = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx_mins = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_maxs)
                        min_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_mins)
                        af_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)

                        if distill == 'MaxMinS':
                            slide_d_feat.append(MaxMin_inst_feat)
                        elif distill == 'MaxS':
                            slide_d_feat.append(max_inst_feat)
                        elif distill == 'MinS':
                            slide_d_feat.append(min_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeats)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion[0](slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    # gSlidePred = UClassifier(slide_d_feat,Confound)
                    gSlidePred, bag_feat, DAtt, bag_feat_neg = UClassifier(slide_d_feat, Confound, True)
                    # print('gSlidePred', gSlidePred)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))
                    # print('allSlide_pred_softmax', allSlide_pred_softmax)

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                # print('allSlide_pred_softmax', allSlide_pred_softmax)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                # print('allSlide_pred_softmax', allSlide_pred_softmax)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                # print('gPred_1', gPred_1)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]
    # print('gPred_1', gPred_1)
    # print('gt_1',gt_1)
    nan_indices = np.isnan(gPred_1.detach().cpu().numpy()).any(axis=None)
    if nan_indices:
        print("NumPy检查结果：", nan_indices)
        print(gPred_1)
        return 0
    else:
        macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

        print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
        print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

        writer.add_scalar(f'auc_0 ', auc_0, epoch)
        writer.add_scalar(f'auc_1 ', auc_1, epoch)
        return auc_1


def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip().split('\n')
        gpu_memory_used = [int(x) for x in output]
        return gpu_memory_used
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def train_attention_preFeature_IAIC(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, epoch, ce_cri=None, params=None,
                                          f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS',same_mask_num=4,instance_per_group=1,Confound=None):

    SlideNames_list, mFeat_list, Label_dict = mDATA_list
    print(len(mFeat_list))

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    # instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)
    feats_list = []
    feats_list_neg = []
    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]

        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):

            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            converted_list = [sst.tolist() for sst in index_chunk_list]
            index_chunk_list = [lst for lst in converted_list if lst]
            feat_index = list(range(tfeat_tensor.shape[0]))

            # 👉 改成一个大包（仅一个 group，包含所有 patch）
            # index_chunk_list = [feat_index]

            for tindex in index_chunk_list:
                # print(tindex)
                idx_tensor = torch.LongTensor(tindex).to(params.device)
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor,0, idx_tensor)
                # print('hahahahaha', subFeat_tensor.size())
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict[0])
                '''
                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                '''
                # patch_mask=explain_patches(tattFeats,tslideLabel,classifier,same_mask_num)
                # patch_mask = patch_mask.to(params.device)
                patch_mask = sobol_analysis(tattFeats, tslideLabel, classifier, params)
                # patch_mask = sobolAnalysis(tattFeats, tslideLabel, classifier, params)

                patch_mask = torch.tensor(patch_mask).to(params.device)
                _,sort_idx=torch.sort(patch_mask,descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_maxs = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx_mins = sort_idx[-instance_per_group*40:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_maxs)
                min_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_mins)
                af_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'MinS':
                    slide_pseudo_feat.append(min_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(tattFeats)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = ce_cri[0](slide_sub_preds, slide_sub_labels).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            optimizer0.step()

            ## optimization for the second tier
            # gSlidePred = UClassifier(slide_pseudo_feat)
            gSlidePred, bag_feat, DAtt, bag_feat_neg = UClassifier(slide_pseudo_feat,Confound,False)

            loss1 = ce_cri[0](gSlidePred, tslideLabel).mean()
            loss2 = ce_cri[1](bag_feat, bag_feat_neg)
            loss = 0.7*loss1+0.3*loss2
            # loss = loss1
            optimizer1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        feats_list.append(bag_feat)
        feats_list_neg.append(bag_feat_neg)
        # print(feats_list_neg.size())

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} '
            print_log(tstr, f_log)
    bag_tensor = torch.cat(feats_list_neg,dim=0)

    # bag_tensor=torch.load(f'datasets/{args.dataset}/abmil/ft_feats.pth')
    bag_tensor_ag = bag_tensor.view(-1,params.feats_size//4)
    # bag_tensor_ag_neg = bag_tensor_neg.view(-1,args.feats_size//2)
    conf_list = []
    for i in [2,4,6,8]:
        prototypes = reduce(params, bag_tensor_ag, i)
        # conf_list.append(torch.from_numpy(prototypes).view(-1, params.feats_size//2).float())
        conf_list.append(prototypes)
    Confound = conf_list
    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)
    
    return Confound


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)

def readPkl(pklPath, split_ratio=0.8, seed=42):
    slideNames = []
    FeatList = []
    Label = []
    with open(pklPath, 'rb') as f:
        mDATA = pickle.load(f)
    for slideName in mDATA.keys():
        slideNames.append(slideName)
        # if slideName.startswith('tumor'):
        if slideName.startswith('lucs'):
            label = 1
        # elif slideName.startswith('normal'):
        elif slideName.startswith('luad'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slideName]
        featGroup = []
        np_feats = [tpatch['feature'] for tpatch in patch_data_list]
        np_feats = np.stack(np_feats, axis=0)
        featGroup = torch.from_numpy(np_feats)
        FeatList.append(featGroup)
        # -------- split 8:2 --------
    N = len(slideNames)
    idx = np.arange(N)

    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    train_size = int(split_ratio * N)
    train_idx = idx[:train_size]
    test_idx  = idx[train_size:]

    # -------- create subsets --------
    train_slides = [slideNames[i] for i in train_idx]
    test_slides  = [slideNames[i] for i in test_idx]

    train_feats  = [FeatList[i] for i in train_idx]
    test_feats   = [FeatList[i] for i in test_idx]

    train_labels = [Label[i] for i in train_idx]
    test_labels  = [Label[i] for i in test_idx]

    return (train_slides, train_feats, train_labels), \
           (test_slides,  test_feats,  test_labels)


def reOrganize_mDATA(mDATA):

    SlideNames = []
    FeatList = []
    Label = []
    print("111111111")
    for slide_name in mDATA.keys():
        # print(slide_name)
        SlideNames.append(slide_name)
        # if slide_name.startswith('tumor'):
        if slide_name.startswith('lucs'):
            label = 1
        # elif slide_name.startswith('normal'):
        elif slide_name.startswith('luad'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label

def reduce(args, feats, k):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    prototypes_neg = []
    semantic_shifts = []
    feats = feats.detach().cpu().numpy()
    # feats_neg = feats
    # print(feats_neg)
    # minn, maxx, feats_norm = minmaxscaler(feats_neg)
    # feats_neg = (1-feats_norm) * (maxx-minn) + minn
    # print('hhhhhh',feats_neg)
    kmeans = Kmeans(k=k, pca_dim=-1)
    # kmeans_neg = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    # kmeans_neg.cluster(feats_neg, seed=67)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # assignments_neg = kmeans_neg.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])
    print(2222222)                   
    os.makedirs(args.c_path, exist_ok=True)
    prototypes.append(centroids)

    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, args.feats_size//2)
    np.save(args.c_path + f'/train_bag_cls_agnostic_feats_proto_{k}.npy', prototypes)

    return prototypes



def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]
def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata
class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))



if __name__ == "__main__":
    main()
