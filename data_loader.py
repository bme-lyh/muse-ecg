# data_loader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import einops
from data_augmentation import *


class ECGDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        if self.transform:
            for transform_func in self.transform:
                x_sample, y_sample = transform_func(x_sample, y_sample)

        x_sample = torch.from_numpy(x_sample.astype(np.float32)).permute(1, 0).unsqueeze(1) # (channel=1, 1, length)
        y_sample = torch.from_numpy(y_sample.astype(np.float32)).permute(1, 0).unsqueeze(1) # (channel=4, 1, length)
        return x_sample, y_sample


# def raw_data_load_ludb(num_test, num_labeled, fold, crop=[1250, 3750], unlabel_contain_labeled=False):
#     # load data (LUDB)
#     x_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/data.npy')
#     y_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/label.npy')
#     flag_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/flag.npy')
#     # index_shuffled_5fold = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/ludb_index_shuffled_5fold_250113.npy')
#     index_shuffled_5fold = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/ludb_index_shuffled_5fold_250113.npy')

#     x_ludb = x_ludb[:,crop[0]:crop[1],:]
#     y_ludb = y_ludb[:,crop[0]:crop[1],:]
    
#     index_shuffled = index_shuffled_5fold[:,fold]
#     index_shuffled_lead = []
#     for i in np.array(index_shuffled):
#         index_shuffled_lead.extend([k for k in range(12*i,12*i+12,1)])
#     x_test = x_ludb[index_shuffled_lead[0:num_test*12]]
#     y_test = y_ludb[index_shuffled_lead[0:num_test*12]]

#     if num_labeled + num_test == x_ludb.shape[0]/12:
#         x_train = x_ludb[index_shuffled_lead[num_test*12:]]
#         y_train = y_ludb[index_shuffled_lead[num_test*12:]]
#         return x_train, y_train, None, x_test, y_test
    
#     else: 
#         flag_labeled = flag_ludb[index_shuffled_lead[num_test*12:num_test*12+num_labeled*12]]
#         index_labeled = np.array(index_shuffled_lead[num_test*12:num_test*12+num_labeled*12])[flag_labeled==0]
#         x_labeled = x_ludb[index_labeled,:,:]
#         y_labeled = y_ludb[index_labeled,:,:]

#         if unlabel_contain_labeled:
#             flag_unlabeled = flag_ludb[index_shuffled_lead[num_test*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12:])[flag_unlabeled==0]
#             x_unlabeled = x_ludb[index_unlabeled,:,:]
#         else:
#             flag_unlabeled = flag_ludb[index_shuffled_lead[num_test*12+num_labeled*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12+num_labeled*12:])[flag_unlabeled==0]
#             x_unlabeled = x_ludb[index_unlabeled,:,:]

#         return x_labeled, y_labeled, x_unlabeled, x_test, y_test
    

# def raw_data_load_rdb(num_test, num_labeled, fold, crop=[1250, 3750], unlabel_contain_labeled=False):
#     # load data (rdb)
#     x_rdb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/rdb/data.npy')
#     y_rdb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/rdb/label.npy')
#     flag_rdb = np.sum(x_rdb, axis=(1,2)) == 0
#     # print(f'Invaild data: {np.sum(flag_rdb)}/{len(flag_rdb)}')
#     index_shuffled_5fold = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/rdb/rdb_index_shuffled_5fold_250113.npy')

#     x_rdb = x_rdb[:,crop[0]:crop[1],:]
#     y_rdb = y_rdb[:,crop[0]:crop[1],:]
    
#     index_shuffled = index_shuffled_5fold[:,fold]
#     index_shuffled_lead = []
#     for i in np.array(index_shuffled):
#         index_shuffled_lead.extend([k for k in range(12*i,12*i+12,1)])
#     flag_test = flag_rdb[index_shuffled_lead[0:num_test*12]]
#     index_test = np.array(index_shuffled_lead[0:num_test*12])[flag_test==0]
#     x_test = x_rdb[index_test,:,:]
#     y_test = y_rdb[index_test,:,:]
#     # x_test = x_rdb[index_shuffled_lead[0:num_test*12]]
#     # y_test = y_rdb[index_shuffled_lead[0:num_test*12]]

#     if num_labeled + num_test == x_rdb.shape[0]/12:
#         flag_labeled = flag_rdb[index_shuffled_lead[num_test*12:num_test*12+num_labeled*12]]
#         index_labeled = np.array(index_shuffled_lead[num_test*12:num_test*12+num_labeled*12])[flag_labeled==0]
#         x_labeled = x_rdb[index_labeled,:,:]
#         y_labeled = y_rdb[index_labeled,:,:]

#         if unlabel_contain_labeled:
#             flag_unlabeled = flag_rdb[index_shuffled_lead[num_test*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12:])[flag_unlabeled==0]
#             x_unlabeled = None
#         else:
#             flag_unlabeled = flag_rdb[index_shuffled_lead[num_test*12+num_labeled*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12+num_labeled*12:])[flag_unlabeled==0]
#             x_unlabeled = None

#         return x_labeled, y_labeled, x_unlabeled, x_test, y_test
    
#     else: 
#         flag_labeled = flag_rdb[index_shuffled_lead[num_test*12:num_test*12+num_labeled*12]]
#         index_labeled = np.array(index_shuffled_lead[num_test*12:num_test*12+num_labeled*12])[flag_labeled==0]
#         x_labeled = x_rdb[index_labeled,:,:]
#         y_labeled = y_rdb[index_labeled,:,:]

#         if unlabel_contain_labeled:
#             flag_unlabeled = flag_rdb[index_shuffled_lead[num_test*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12:])[flag_unlabeled==0]
#             x_unlabeled = x_rdb[index_unlabeled,:,:]
#         else:
#             flag_unlabeled = flag_rdb[index_shuffled_lead[num_test*12+num_labeled*12:]]
#             index_unlabeled = np.array(index_shuffled_lead[num_test*12+num_labeled*12:])[flag_unlabeled==0]
#             x_unlabeled = x_rdb[index_unlabeled,:,:]

#         return x_labeled, y_labeled, x_unlabeled, x_test, y_test


def reshape_stacked_leads_einops(data: np.ndarray, leads_per_subject: int = 12) -> np.ndarray:
    """
    Reshapes stacked single-lead ECG data using NumPy and einops.

    Args:
        data (np.ndarray): Input array with shape (B L 1) or (B L),
                           where B = num_subjects * leads_per_subject.
        leads_per_subject (int): Number of leads per subject (typically 12).

    Returns:
        np.ndarray: Output array with shape (B L leads_per_subject).
    """
    if data.ndim == 3 and data.shape[2] == 1:
        pattern_in = '(s l) len 1 -> s len l'
    elif data.ndim == 2:
        pattern_in = '(s l) len -> s len l'
    else:
         raise ValueError(f"Input tensor must have shape (B, L, 1) or (B, L), got {data.shape}")

    num_total_rows = data.shape[0]
    if num_total_rows % leads_per_subject != 0:
        raise ValueError(f"First dimension ({num_total_rows}) is not divisible by "
                         f"leads_per_subject ({leads_per_subject})")

    num_subjects = num_total_rows // leads_per_subject

    # 1. Rearrange to group leads into channel dimension: (S*L, len, 1) -> (S, len, L)
    grouped_data = einops.rearrange(data, pattern_in, l=leads_per_subject)
    # grouped_data has shape (num_subjects, length, leads_per_subject)

    # 2. Repeat each subject's data 'leads_per_subject' times along the first dim
    output_data = einops.repeat(grouped_data, 's len l -> (s r) len l', r=leads_per_subject)
    # output_data has shape (num_subjects * leads_per_subject, length, leads_per_subject)
    return output_data


def split_train_val_indices(index_array: np.ndarray,
                           val_ratio: float,
                           seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits an array of indices into training and validation sets randomly.

    Args:
        index_array (np.ndarray): The input array of indices to be split.
        val_ratio (float): The proportion of indices to allocate to the
                           validation set (must be between 0.0 and 1.0).
        seed (int): The random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - train_idx: NumPy array of indices for the training set.
            - val_idx: NumPy array of indices for the validation set.

    Raises:
        ValueError: If val_ratio is not between 0.0 and 1.0.
    """
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0.0 and 1.0")

    # --- Ensure reproducibility ---
    rng = np.random.default_rng(seed)

    # --- Create a shuffled copy of the indices ---
    # It's important to shuffle a copy so the original array remains unchanged.
    shuffled_indices = index_array.copy()
    rng.shuffle(shuffled_indices) # Shuffles the array in-place

    # --- Calculate the split point ---
    total_size = len(shuffled_indices)
    val_size = int(total_size * val_ratio) # Number of validation samples

    # --- Split the shuffled indices ---
    val_idx = shuffled_indices[:val_size]
    train_idx = shuffled_indices[val_size:]

    return train_idx, val_idx


# Convert the record index value to the index value of 12 lead data 将record索引值转化为12导联数据的索引值
def index_convert(index_record, start=0):
    index12 = []
    for i in np.array(index_record)-start:
        index12.extend([k for k in range(12*i,12*i+12,1)])
    return index12


def raw_data_load_ludb(num_test, num_labeled, fold, crop=[1250, 3750], apply_flag=True):
    # load data (LUDB)
    x_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/data.npy')
    y_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/label.npy')
    flag_ludb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/ludb/flag.npy')
    info_ludb = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/ludb_info.csv')
    rhythm_ludb = np.repeat(info_ludb['Rhythm'].values, 12)
    index_5fold_train = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/ludb_5fold_train_index.csv', header=None).values
    index_5fold_test = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/ludb_5fold_test_index.csv', header=None).values

    x_ludb = x_ludb[:,crop[0]:crop[1],:]
    # x_ludb_12leads = reshape_stacked_leads_einops(x_ludb, leads_per_subject=12)
    # x_ludb = np.concatenate([x_ludb, x_ludb_12leads], axis=-1) # (B, L, 1+12)
    y_ludb = y_ludb[:,crop[0]:crop[1],:]
    
    index_train = index_convert((index_5fold_train[:,fold]).tolist(), start=1)
    unlabeled_ratio = (200- num_test - num_labeled) / (200- num_test)
    index_labeled, index_unlabeled = split_train_val_indices(index_train, val_ratio=unlabeled_ratio, seed=42+fold)
    index_test = index_convert((index_5fold_test[:,fold]).tolist(), start=1)

    flag_labeled = flag_ludb[index_labeled]
    flag_unlabeled = flag_ludb[index_unlabeled]
    flag_test = flag_ludb[index_test]

    rhythm_labeled = rhythm_ludb[index_labeled]
    rhythm_unlabeled = rhythm_ludb[index_unlabeled]
    rhythm_test = rhythm_ludb[index_test]

    if apply_flag:
        x_labeled = x_ludb[index_labeled,:,:][flag_labeled==0]
        y_labeled = y_ludb[index_labeled,:,:][flag_labeled==0]
        x_unlabeled = x_ludb[index_unlabeled,:,:][flag_unlabeled==0]
        y_unlabeled = y_ludb[index_unlabeled,:,:][flag_unlabeled==0]
        x_test = x_ludb[index_test,:,:][flag_test==0]
        y_test = y_ludb[index_test,:,:][flag_test==0]
        return x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test
    else:
        x_labeled = x_ludb[index_labeled,:,:]
        y_labeled = y_ludb[index_labeled,:,:]
        x_unlabeled = x_ludb[index_unlabeled,:,:]
        y_unlabeled = y_ludb[index_unlabeled,:,:]
        x_test = x_ludb[index_test,:,:]
        y_test = y_ludb[index_test,:,:]
        return x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test, flag_labeled, flag_unlabeled, flag_test, rhythm_labeled, rhythm_unlabeled, rhythm_test

    

def raw_data_load_rdb(num_test, num_labeled, fold, crop=[1250, 3750], apply_flag=True):
    # load data (rdb)
    x_rdb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/rdb/data.npy')
    y_rdb = np.load('/data/liuyuhang/code/ecg_seg_torch/semi_supervised/dataset/rdb/label.npy')
    flag_rdb = np.sum(x_rdb, axis=(1,2)) == 0

    info_rdb = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/rdb_info.csv')
    rhythm_rdb = np.repeat(info_rdb['data_name'].values, 12)
    rhythm_rdb = np.array([name[0:-4] for name in rhythm_rdb])

    index_5fold_train = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/rdb_5fold_train_index.csv', header=None).values
    index_5fold_test = pd.read_csv('/data/liuyuhang/code/ecg_seg_torch/supervised/dataset/rdb_5fold_test_index.csv', header=None).values

    x_rdb = x_rdb[:,crop[0]:crop[1],:]
    # x_rdb_12leads = reshape_stacked_leads_einops(x_rdb, leads_per_subject=12)
    # x_rdb = np.concatenate([x_rdb, x_rdb_12leads], axis=-1) # (B, L, 1+12)
    y_rdb = y_rdb[:,crop[0]:crop[1],:]

    if num_test == 0:
        return x_rdb, y_rdb
    
    fold_indices = index_5fold_train[:, fold]
    fold_indices = fold_indices[fold_indices != -1].tolist()
    index_train = index_convert(fold_indices)
    unlabeled_ratio = (2399 - num_test - num_labeled) / (2399 - num_test)
    index_labeled, index_unlabeled = split_train_val_indices(index_train, val_ratio=unlabeled_ratio, seed=42+fold)
    fold_indices = index_5fold_test[:, fold]
    fold_indices = fold_indices[fold_indices != -1].tolist()
    index_test = index_convert(fold_indices)

    flag_labeled = flag_rdb[index_labeled]
    flag_unlabeled = flag_rdb[index_unlabeled]
    flag_test = flag_rdb[index_test]

    rhythm_labeled = rhythm_rdb[index_labeled]
    rhythm_unlabeled = rhythm_rdb[index_unlabeled]
    rhythm_test = rhythm_rdb[index_test]

    if apply_flag:
        x_labeled = x_rdb[index_labeled,:,:][flag_labeled==0]
        y_labeled = y_rdb[index_labeled,:,:][flag_labeled==0]
        x_unlabeled = x_rdb[index_unlabeled,:,:][flag_unlabeled==0]
        y_unlabeled = y_rdb[index_unlabeled,:,:][flag_unlabeled==0]
        x_test = x_rdb[index_test,:,:][flag_test==0]
        y_test = y_rdb[index_test,:,:][flag_test==0]
        return x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test
    else:
        x_labeled = x_rdb[index_labeled,:,:]
        y_labeled = y_rdb[index_labeled,:,:]
        x_unlabeled = x_rdb[index_unlabeled,:,:]
        y_unlabeled = y_rdb[index_unlabeled,:,:]
        x_test = x_rdb[index_test,:,:]
        y_test = y_rdb[index_test,:,:]
        return x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test, flag_labeled, flag_unlabeled, flag_test, rhythm_labeled, rhythm_unlabeled, rhythm_test
    

# Example usage with a list of transformations:
def get_train_transforms():
    transforms = [
        lambda x, y: (random_resize(x, y, scale_range=(0.5, 2))),
        lambda x, y: (x + additive_white_gaussian_noise(x[:,0], snr=10)[:,np.newaxis] if np.random.random() < 0.5 else x, y),
        lambda x, y: (x + baseline_wander_noise(x[:,0], fs=500, snr=-10, freq=0.15)[:,np.newaxis]  if np.random.random() < 0.5 else x, y),
        lambda x, y: (zscore_normalize(x, axis=0), y)
    ]
    return transforms


def base_transforms():
    transforms = [
        lambda x, y: (zscore_normalize(x, axis=0), y),
    ]
    return transforms