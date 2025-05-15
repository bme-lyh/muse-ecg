import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import remove_small_holes, remove_small_objects
from data_augmentation import *


# 模型预测相关函数
# (1) 后处理方法
def  Multi_lead_Correction(refs, lead, weight=[0.4,1/12,0.4], threshold=[30,30,30]):
    def multi_lead_weighted_average(labels, lead, weight):
        weights = np.ones((12,1,1))
        weights *= (1-weight)/11
        weights[lead] = weight
        label = np.sum(labels*weights, axis=0)
        return label
    
    def list2mask(pqrstlist):
        mask = np.zeros((5000,4))
        for i in range(3):
            onsets = pqrstlist[2*i]
            offsets = pqrstlist[2*i+1]
            for j in range(len(onsets)):
                mask[onsets[j]:offsets[j]+1,i+1] = 1
        mask_nowave = np.ones(5000)
        is_wave = (mask[:,1]==1)+(mask[:,2]==1)+(mask[:,3]==1)
        mask_nowave[is_wave] = 0
        mask[:,0] = mask_nowave
        return mask
    
    def mask2list(mask):
        mask_padded = np.zeros(mask.shape[0]+2)
        mask_padded[1:-1] = mask
        index = np.linspace(0,mask.shape[0],mask.shape[0]+1).astype('int')
        diff = np.diff(mask_padded)
        onsets = index[diff==1]
        offsets = index[diff==-1]-1
        assert len(onsets)==len(offsets)
        return onsets, offsets
    
    ave = np.mean(refs, axis=0)
    raw = refs[lead,:,:]
    pqrstlist = []
    for wavetype in range(3):
        wt = weight[wavetype]
        th = threshold[wavetype]
        ref = multi_lead_weighted_average(refs, lead, weight=wt)
        raw_s = raw[:,wavetype+1]>=0.5
        ref_s = ref[:,wavetype+1]>=0.5
        ref_s = remove_small_holes(ref_s,40)
        ref_s = remove_small_objects(ref_s,10)
        onset_ref, offset_ref = mask2list(ref_s)
        onset_raw, offset_raw = mask2list(raw_s)
        onset_result, offset_result = [], []
        for wave in range(len(onset_ref)):
            onsets = [i for i in onset_raw if i>=onset_ref[wave]-th and i<=onset_ref[wave]+th]
            if len(onsets)==1:
                onset_result.append(onsets[0])
            elif len(onsets)>1:
                onset_result.append(np.min(onsets))
            else:
                onset_result.append(onset_ref[wave])
            offsets = [i for i in offset_raw if i>=offset_ref[wave]-th and i<=offset_ref[wave]+th]
            if len(offsets)==1:
                offset_result.append(offsets[0])
            elif len(offsets)>1:
                offset_result.append(np.max(offsets))
            else:
                offset_result.append(offset_ref[wave])
        pqrstlist.append(onset_result)
        pqrstlist.append(offset_result)
    result = list2mask(pqrstlist)
    return result


# (2) 模型预测
def model_predict(model: nn.Module, model_save_path: str, test_dataset: Dataset, device: torch.device = torch.device('cuda'),
                  model_output_length: int = 2500, stride: int = 2500, 
                  multi_lead_correction: bool = False, batch_size: int = 32, logits: bool = False) :
    def predict_batch(model, test_loader, start, end):
        y_pred = []
        for x, _ in test_loader:
            x = x.to(device)
            with torch.no_grad():
                if logits:
                    output = torch.softmax(model(x[:,:,:,start:end]), dim=1)
                else:
                    output = model(x[:,:,:,start:end])
                if isinstance(output, tuple):
                    output = output[0]
                y_pred.append(output.cpu().squeeze(2).permute(0, 2, 1).numpy())
        return np.concatenate(y_pred, axis=0)
    
    # 1. Load Model Weights
    model.load_state_dict(torch.load(model_save_path))  # PyTorch method for loading weights
    model.eval()  # Set the model to evaluation mode

    # 2. Data Loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # no shuffle here
    
    # 3. Prepare Variables
    num_samples = len(test_dataset)
    num_segments = model_output_length // stride + 1
    pred_segmented = np.zeros((num_segments, num_samples, 5000, 4))
    pred_segmented[:] = np.nan
    
    # 4. Perform Segmented Predictions
    for s in range(num_segments):
        start = stride*s
        end = start + 2500
        pred_segmented[s,:,start:end,:] = predict_batch(model, test_loader, start, end)
    y_pred = np.nanmean(pred_segmented, axis=0)

    # 5. Average Segmented Predictions
    y_pred = np.nanmean(pred_segmented, axis=0)

    # 6. Multi-Lead Correction (if enabled)
    if multi_lead_correction:
      y_pred_copy = copy.deepcopy(y_pred)
      for i in range(len(y_pred)):
          refs = y_pred_copy[i//12*12:i//12*12+12,:,:]
          lead = i%12
          y_pred[i,:,:] = Multi_lead_Correction(refs, lead, weight=[0.4,1/12,0.4], threshold=[30,30,30])

    return y_pred


# 结果统计相关函数
# (1) 将单个通道转换为波形起始点和终止点的列表
def label2list(label):
    label[label>=0.5] = 1
    label[label<=0.5] = 0
    label_padded = np.zeros(label.shape[0]+2)
    label_padded[1:-1] = label[:]
    index = np.linspace(0,label.shape[0],label.shape[0]+1).astype('int')
    diff = np.diff(label_padded)
    onsets = index[diff==1]
    ends = index[diff==-1]-1
    assert len(onsets)==len(ends)
    return onsets, ends


# (2) 计算单个通道的交集、并集和并集中的元素个数, 用于计算Dice系数(2*I/S)，或者计算IoU系数(交并比, I/U)
def area_eval(Ypred, Ytrue, start=0, end=4999):
    ypred = Ypred[start:end+1]
    ytrue = Ytrue[start:end+1]
    Iarea = np.sum((ypred>=0.5)*(ytrue>=0.5))
    Uarea = np.sum(((ypred>=0.5)+(ytrue>=0.5))>0)
    Sarea = np.sum(ypred>=0.5)+np.sum(ytrue>=0.5)
    return Iarea, Uarea, Sarea
    

# (3) 计算统计波形起止点的TP, FP, FN以及误差
def error_eval(onsets, offsets, onsets_test, offsets_test, threshold=150, fs=500, start=0, end=4999):
    assert len(onsets)==len(offsets)
    assert len(onsets_test)==len(offsets_test)
    error = np.zeros((len(onsets),2)) 
    error[:] = np.nan
    tp = np.zeros(2)
    fp = np.zeros(2)
    fn = np.zeros(2)
    th = threshold*fs/1000
    valid = (offsets_test>=start)*(onsets_test<=end)
    ons = onsets_test[valid]
    offs = offsets_test[valid]
    assert len(ons)==len(offs)
    total = np.array([len(ons),len(ons)]).reshape(2)
    
    for i in range(len(onsets)):
        ons_in_range = (ons>=onsets[i]-th)*(ons<=onsets[i]+th)
        offs_in_range = (offs>=offsets[i]-th)*(offs<=offsets[i]+th)
        if np.sum(ons_in_range)>0:
            tp[0] += 1
            imin = np.argmin(np.abs(ons[ons_in_range]-onsets[i]))
            error[i,0] = (np.abs(ons[ons_in_range]-onsets[i]))[imin]*np.sign((ons[ons_in_range]-onsets[i])[imin])
        else:
            fn[0] += 1
        if np.sum(offs_in_range)>0:
            tp[1] += 1
            imin = np.argmin(np.abs(offs[offs_in_range]-offsets[i]))
            error[i,1] = (np.abs(offs[offs_in_range]-offsets[i]))[imin]*np.sign((offs[offs_in_range]-offsets[i])[imin])
        else:
            fn[1] += 1
            
    fp = total - tp
    return error, tp, fp, fn


# (4) 结果统计
def dataset_eval(dataset, pred, th_delineation=150, verbose=0):
    data, ann, rhythm, flag = dataset
    flag = np.zeros((data.shape[0],)) if flag is None else flag
    assert data.shape[0] == ann.shape[0] == rhythm.shape[0] == flag.shape[0]
    assert ann.shape == pred.shape

    num_sample = data.shape[0]
    seg_counts = np.zeros((num_sample,8))
    deli_counts = np.zeros((num_sample,18))
    # 统计每个样本各个波形的 交集、并集和并集的采样点数量 以及 TP, FP, FN 和 误差
    points_true = 0
    points_all = 0
    for i in range(num_sample):
        # 每个样本有效的统计区间
        index = np.linspace(0,4999,5000).astype('int')
        temp = index[np.argmax(ann[i,:,:], axis=-1)>0]
        start, end = temp[0], temp[-1]
        cur_points_true = np.sum((np.argmax(ann[i,:,:], axis=-1) == np.argmax(pred[i,:,:], axis=-1))[start:end+1])
        cur_points_all = end - start + 1
        seg_counts[i,6], seg_counts[i,7] = cur_points_true, cur_points_all
        points_true += cur_points_true
        points_all += cur_points_all
        for k in range(3):
            iarea, uarea, sarea = area_eval(ann[i,:,k+1], pred[i,:,k+1], start, end)
            seg_counts[i,k], seg_counts[i,k+3] = iarea, uarea
            onsets, offsets = label2list(ann[i,:,k+1])
            onsets_test, offsets_test = label2list(pred[i,:,k+1])
            
            err, tp, fp, fn = error_eval(onsets, offsets, onsets_test, offsets_test, threshold=th_delineation, fs=500, start=start, end=end)
            deli_counts[i,2*k:2*k+2], deli_counts[i,2*k+6:2*k+8], deli_counts[i,2*k+12:2*k+14] = tp[:], fp[:], fn[:]

    ACC = points_true / points_all
    if verbose:
        print('ACC: %.2f' % (100*ACC))

    list_rhythm = np.unique(rhythm)
    num_rhythm = len(list_rhythm)
    seg_metrics = np.zeros((num_rhythm+1,4))
    deli_metrics = np.zeros((num_rhythm+1,36))
    # 计算按心律类型分类的指标
    for r in range(num_rhythm+1):
        if r < num_rhythm:
            select = ((rhythm == list_rhythm[r]) * (flag == 0)) > 0
            type_name = list_rhythm[r]
        else:
            select = flag == 0
            type_name = 'Total'

        IoU = np.sum(seg_counts[select,0:3], axis=0) / np.sum(seg_counts[select,3:6], axis=0)
        IoU[np.sum(seg_counts[select,3:6], axis=0) == 0] = -1
        Acc = np.sum(seg_counts[select,6], axis=0) / np.sum(seg_counts[select,7], axis=0)
        seg_metrics[r, 0:3] = IoU[:]
        seg_metrics[r, 3] = Acc
            
        epsilon = 1e-6
        TP, FP, FN = np.sum(deli_counts[select,0:6], axis=0), np.sum(deli_counts[select,6:12], axis=0), np.sum(deli_counts[select,12:18], axis=0)
        Se = (TP + epsilon) / (TP + FN + epsilon)
        PPV = (TP + epsilon) / (TP + FP + epsilon)
        F1 = (2*Se*PPV + epsilon) / (Se + PPV + epsilon)

        if verbose:
            print('Type: %s  Ave. F1:\t%.2f\tF1:\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (type_name, 100*np.mean(F1), 100*F1[0], 100*F1[1], 100*F1[2], 100*F1[3], 100*F1[4], 100*F1[5]))

        deli_metrics[r, [0,3,6,9,12,15]] = Se[:]
        deli_metrics[r, [1,4,7,10,13,16]] = PPV[:]
        deli_metrics[r, [2,5,8,11,14,17]] = F1[:]

        deli_metrics[r, [18,21,24,27,30,33]] = TP[:]
        deli_metrics[r, [19,22,25,28,31,34]] = FP[:]
        deli_metrics[r, [20,23,26,29,32,35]] = FN[:]
        

    header_seg_counts = ['iarea_p', 'iarea_qrs', 'iarea_t', 'uarea_p', 'uarea_qrs', 'uarea_t', 'points_true', 'points_all']
    seg_counts = pd.DataFrame(seg_counts, columns=header_seg_counts)

    header_deli_counts = ['tp_p_on','tp_p_end', 'tp_qrs_on', 'tp_qrs_end', 'tp_t_on', 'tp_t_end',
                          'fp_p_on','fp_p_end', 'fp_qrs_on', 'fp_qrs_end', 'fp_t_on', 'fp_t_end',
                          'fn_p_on','fn_p_end', 'fn_qrs_on', 'fn_qrs_end', 'fn_t_on', 'fn_t_end',
                          ]
    deli_counts = pd.DataFrame(deli_counts, columns=header_deli_counts)

    row_name = sorted(list_rhythm.tolist()) + ['Total']
    header_seg_metrics = ['type', 'iou_p', 'iou_qrs', 'iou_t', 'miou', 'acc']
    seg_metrics = pd.DataFrame(seg_metrics)
    seg_metrics.insert(0, 'type', row_name)
    seg_metrics.insert(4, 'miou', np.mean(seg_metrics.iloc[:,1:4], axis=1))
    seg_metrics.columns = header_seg_metrics

    header_deli_metrics = ['type', 'ave_f1',
                           'se_p_on', 'ppv_p_on', 'f1_p_on', 'se_p_end', 'ppv_p_end', 'f1_p_end',
                           'se_qrs_on', 'ppv_qrs_on', 'f1_qrs_on', 'se_qrs_end', 'ppv_qrs_end', 'f1_qrs_end',
                           'se_t_on', 'ppv_t_on', 'f1_t_on', 'se_t_end', 'ppv_t_end', 'f1_t_end',
                           'tp_p_on', 'fp_p_on', 'fn_p_on', 'tp_p_end', 'fp_p_end', 'fn_p_end',
                           'tp_qrs_on', 'fp_qrs_on', 'fn_qrs_on', 'tp_qrs_end', 'fp_qrs_end', 'fn_qrs_end',
                           'tp_t_on', 'fp_t_on', 'fn_t_on', 'tp_t_end', 'fp_t_end', 'fn_t_end',
                           ]
    deli_metrics = pd.DataFrame(deli_metrics)
    deli_metrics.insert(0, 'type', row_name)
    deli_metrics.insert(1, 'ave_f1', np.mean(deli_metrics.iloc[:,[3,6,9,12,15,18]], axis=1))
    deli_metrics.columns = header_deli_metrics

    return seg_counts, deli_counts, seg_metrics, deli_metrics
