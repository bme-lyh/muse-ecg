import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from network.model_unet_a_2d import *
from data_loader import *
from data_augmentation import *
from loss_utils import *
from test_utils import model_predict, dataset_eval
from torchinfo import summary

import random
import torch.backends.cudnn as cudnn


def _evaluate_epoch(model, data_loader, device, loss_func, is_training=False, optimizer=None, weights=None, deep_supervision=True):
    """Helper function to evaluate or train the model on a given dataset."""
    model.train() if is_training else model.eval()
    total_loss = 0.0
    total_acc = [0.0] * 4  # 假设有 4 个输出
    num_outputs = 4 # 假设有4个输出

    # 设置默认权重
    if weights is None:
        # weights = [1.0] * num_outputs
        weights = [1, 1/2, 1/4, 1/8]
    # 权重归一化， 总和为1
    # weights = [w / sum(weights) for w in weights]

    with torch.set_grad_enabled(is_training):
        for batch, (ecg, label) in enumerate(data_loader):
            ecg = ecg.to(device)
            label = label.to(device)

            if is_training:
                optimizer.zero_grad()

            # 假设 model 返回一个包含 4 个输出的列表或元组
            preds = model(ecg, full_output=True)
            if deep_supervision:
                losses = [loss_func(preds[i], label) for i in range(num_outputs)]
                # 使用权重计算加权损失
                loss = sum(w * l for w, l in zip(weights, losses))
            else:
                loss = loss_func(preds[0], label)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            for i, pred in enumerate(preds[:num_outputs]):
                total_acc[i] += (pred.argmax(dim=1) == label.argmax(dim=1)).type(torch.float).mean().item()

    avg_loss = total_loss / len(data_loader)
    avg_accs = [acc / len(data_loader) for acc in total_acc]
    return avg_loss, avg_accs


def supervised_train(
    model,
    model_save_path,
    train_loader,
    val_loader,
    test_loader,
    scheduler,
    optimizer,
    deep_supervision=True,
    epochs=40,
    save_epochs=[],
    loss_func=nn.CrossEntropyLoss(),
    early_stop_patience=5,  # Early stopping patience
    early_stop_ignore_epochs=10,  # Number of epochs to ignore before starting early stopping
    tqdm_header="",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    show_val = val_loader is not None
    show_test = test_loader is not None

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stop_patience, verbose=0, model_save_path= model_save_path)

    # Combined progress bar
    with tqdm(total=epochs, desc=f'{tqdm_header}Training Progress', unit='epoch') as pbar:
        for epoch in range(epochs):
            # Training
            train_loss_epoch, train_acc_epoch = _evaluate_epoch(model, train_loader, device, loss_func, is_training=True, optimizer=optimizer, deep_supervision=deep_supervision)
            train_losses.append(train_loss_epoch)
            train_accs.append(train_acc_epoch)

            postfix_str = f"train_loss={train_losses[-1]:.4f}, acc="
            postfix_str += ", ".join([f"{acc:.4f}" for i, acc in enumerate(train_accs[-1])])

            # Validation
            if show_val:
                val_loss, val_acc = _evaluate_epoch(model, val_loader, device, loss_func, is_training=False, deep_supervision=deep_supervision)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                cur_val_loss = val_loss

                postfix_str += f", val_loss={val_losses[-1]:.4f}, acc="
                postfix_str += ", ".join([f"{acc:.4f}" for i, acc in enumerate(val_accs[-1])])

                # Call early stopping check
                if epoch >= early_stop_ignore_epochs:
                    early_stopping(val_loss=val_loss, model=model, epoch=epoch)
                    if early_stopping.early_stop:
                        break

            # Test
            if show_test:
                test_loss, test_acc = _evaluate_epoch(model, test_loader, device, loss_func)
                test_losses.append(test_loss)
                test_accs.append(test_acc)

            # Update the progress bar with metrics for the epoch
            pbar.set_postfix_str(postfix_str)
            pbar.update(1)

            # Learning rate schedule
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(cur_val_loss)
            elif isinstance(scheduler, LambdaLR):
                scheduler.step()

            # Save model
            if (epoch + 1) in save_epochs:
                torch.save(model.state_dict(), f"{model_save_path}.epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), f"{model_save_path}.final.pth")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-th', type=int, default=150)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-aug', type=int, default=2)
    parser.add_argument('-ds', type=int, default=0)
    args = parser.parse_args()
    th_delineation = args.th
    gpu = args.gpu
    aug = args.aug
    deep_supervision = args.ds
    torch.cuda.set_device(gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameters
    max_epochs = 100
    # max_epochs = 40
    save_epochs = [5,20]
    val_ratio = 0.2
    batch_size = 32 
    lr_ini = 1e-3
    reduce_lr_patience = 5
    factor = 0.1
    min_lr = 1e-6
    cooldown = 0
    loss_func = nn.CrossEntropyLoss()
    # loss_func=DiceCELoss()
    model_save_path = f'./checkpoints/unet_a_ds{int(deep_supervision)}_202505_cce'
    metrics_save_path = f'./metrics/unet_a_ds{int(deep_supervision)}_202505_cce'
    early_stop_patience = 10 
    early_stop_ignore_epochs = 10
    # records_list = [5,10,20,50,160]
    records_list = [160, 50, 20, 10, 5]
    df_list = []

    for num_labeled in records_list:
        ## Train
        print(f"Number of labeled data: {num_labeled}")
        print("Training Stage")
        for fold in range(5):
            if os.path.exists(f"{model_save_path}.num_labeled_{num_labeled}_aug_{aug}.fold_{fold}.final.pth"):
                print(f"Fold {fold+1}/{5}: model already exists. Skipping...")
                continue
            # Set random seed for reproducibility
            seed = 42
            cudnn.benchmark = False
            cudnn.deterministic = True
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            x_train, y_train, _, _, x_test, y_test = raw_data_load_ludb(40, num_labeled, fold, crop=[1250, 3750])
            # num_val =  np.ceil(x_train.shape[0]//12 * val_ratio).astype(int) * 12
            num_val =  np.round(x_train.shape[0] * val_ratio).astype(int)
            x_val, y_val = x_train[:num_val], y_train[:num_val]
            x_train, y_train = x_train[num_val:], y_train[num_val:]
            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=42)
            print(f"Fold {fold+1}/{5}: Train: {x_train.shape[0]} Val: {x_val.shape[0]} Test: {x_test.shape[0]}")

            if aug == 0:
                train_dataset = ECGDataset(x_train, y_train, transform=base_transforms())
                val_dataset = ECGDataset(x_val, y_val, transform=base_transforms())
            elif aug == 1:
                train_dataset = ECGDataset(x_train, y_train, transform=get_train_transforms())
                val_dataset = ECGDataset(x_val, y_val, transform=get_train_transforms())
            elif aug == 2:
                train_dataset = ECGDataset(x_train, y_train, transform=base_transforms())
                train_dataset_aug = ECGDataset(x_train, y_train, transform=get_train_transforms())
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_aug])
                val_dataset = ECGDataset(x_val, y_val, transform=base_transforms())
                val_dataset_aug = ECGDataset(x_val, y_val, transform=get_train_transforms())
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, val_dataset_aug])
            else:
                raise ValueError("Invalid aug value. Choose from 0, 1, 2")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = None

            model = UNet1D_A(length=2500, base_channels=16, kernel_size=9, dropout='channels', droprate=.2, num_classes=2)
            # input_tensor = torch.randn(32, 1, 1 ,2500) 
            # summary(model, input_data=input_tensor)
            # output = model(input_tensor)

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_ini)

            # LR Scheduler (ReduceLROnPlateau)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=reduce_lr_patience, cooldown=cooldown, min_lr=min_lr, verbose=0)
            model = supervised_train(
                model,
                f"{model_save_path}.num_labeled_{num_labeled}_aug_{aug}.fold_{fold}",
                train_loader,
                val_loader,
                test_loader,
                scheduler,
                optimizer,
                deep_supervision=deep_supervision,
                epochs=max_epochs,
                save_epochs=save_epochs,
                loss_func=loss_func,
                early_stop_patience=early_stop_patience,
                early_stop_ignore_epochs=early_stop_ignore_epochs,
                tqdm_header=f"Fold {fold+1}/{5}: ",
            )

        ## Test
        print("Test Stage")
        data = []
        label = []
        preds = []
        seg_metrics_macro = []
        deli_metrics_macro = []
        
        for fold in range(5):
            model = UNet1D_A(length=2500, base_channels=16, kernel_size=9, dropout='channels', droprate=.2, num_classes=2).to(device)
            # model_load_path = f"{model_save_path}.num_labeled_{num_labeled}_aug_{aug}.fold_{fold}.best.pth"
            model_load_path = f"{model_save_path}.num_labeled_{num_labeled}_aug_{aug}.fold_{fold}.final.pth"
            model.load_state_dict(torch.load(model_load_path))

            _, _, _, _, x_test, y_test, _, _, f_test, _, _, r_test = raw_data_load_ludb(40, 160, fold, crop=[0, 5000], apply_flag=False)
            test_dataset = ECGDataset(x_test, y_test, transform=base_transforms())

            pred = model_predict(model, model_load_path, test_dataset, device, multi_lead_correction=False)
            dataset = (x_test, y_test, r_test, f_test)
            _, _, seg_metrics, deli_metrics = dataset_eval(dataset, pred, th_delineation=th_delineation, verbose=0)

            data.append(x_test)
            label.append(y_test)
            preds.append(pred)
            seg_metrics_macro.append(seg_metrics)
            deli_metrics_macro.append(deli_metrics)

        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        preds = np.concatenate(preds, axis=0)

        def summarize_total_rows(dfs):
            """
            Extracts 'Total' rows from DataFrames, calculates summary statistics,
            and returns a new DataFrame.

            Args:
            dfs: A list of pandas DataFrames with identical structure.

            Returns:
                A pandas DataFrame containing the mean, std, max, and min
                of each column of the 'Total' rows from all input DataFrames.
            """
            total_rows = [df[df['type'] == 'Total'].iloc[0] for df in dfs]
            total_df = pd.DataFrame(total_rows)

            # Get original headers, and remove 'type'
            original_headers = total_df.columns.tolist()
            original_headers.remove('type')


            # Calculate summary stats for each column (excluding type)
            summary_data = {
                'mean': total_df[original_headers].mean().to_list(),
                'std': total_df[original_headers].std().to_list(),
                'max': total_df[original_headers].max().to_list(),
                'min': total_df[original_headers].min().to_list()
            }
            # Create the summary DataFrame
            summary_df = pd.DataFrame(summary_data, index = original_headers)
            return summary_df
        
        # Macro average metrics of 5 folds
        seg_metrics_macro = summarize_total_rows(seg_metrics_macro)
        deli_metrics_macro = summarize_total_rows(deli_metrics_macro) 
        filtered_df = deli_metrics_macro[deli_metrics_macro.index.str.contains('f1')]
        merged_df_macro = pd.concat([seg_metrics_macro, filtered_df], axis = 0)                

        # Micro average metrics of 5 folds
        dataset = (data, label, np.zeros((data.shape[0],)), np.zeros((data.shape[0],)))
        _, _, seg_metrics_micro, deli_metrics_micro = dataset_eval(dataset, preds, th_delineation=th_delineation, verbose=0)
        # Filter df2 to include rows where column name contain 'f1'
        filtered_df = deli_metrics_micro[['type'] + [col for col in deli_metrics_micro.columns if 'f1' in col]]
        merged_df_micro = pd.merge(seg_metrics_micro, filtered_df, on='type', how='outer')
        micro_row = merged_df_micro[merged_df_micro['type'] == 'Total'].iloc[0]
        # Remove 'type' and convert to series
        micro_row_values = micro_row.drop('type')
        merged_df = copy.deepcopy(merged_df_macro)
        merged_df['micro'] = micro_row_values
        
        df_list.append(merged_df)
        # Final results
        print(merged_df)

    # Save results
    # 拼接数据，将总标题作为列上方的“标题行”
    concat_frames = []
    for i, df in enumerate(df_list):
        # 插入标题行
        df_with_title = df.copy()
        df_with_title.columns = pd.MultiIndex.from_tuples([(str(records_list[i]), col) for col in df.columns])
        concat_frames.append(df_with_title)

    # 按列拼接，并保留行标题
    result = pd.concat(concat_frames, axis=1)

    # 写入 Excel
    result.to_excel(f"{metrics_save_path}.aug_{aug}.xlsx")

