import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_coeff(input_tensor, target_tensor, pixel_weight=None, epsilon=1e-6):
    """
    Compute the Dice coefficient between two tensors.

    Args:
        input_tensor (torch.Tensor): the predicted tensor, with shape (B, C, H, W), after the activation function (sigmoid or softmax).
        target_tensor (torch.Tensor): the target tensor, with shape (B, C, H, W) , one-hot encoding.
        reduce_batch_first (bool, optional): If True, compute Dice coeff batch first, otherwise average the dice coeff across batch. Defaults to False.
        epsilon (float, optional): small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Dice coefficient.
    """
    intersection = torch.sum(input_tensor * target_tensor, dim=(2, 3), keepdim=True)
    cardinality = torch.sum(input_tensor, dim=(2, 3), keepdim=True) + torch.sum(target_tensor, dim=(2, 3), keepdim=True)
    
    dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)

    if pixel_weight is not None:
        dice = dice * pixel_weight
    
    return dice


class DiceLoss(nn.Module):
    """
    Dice loss function for semantic segmentation tasks.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Defaults to 'mean'.
        weight (torch.Tensor, optional): Weight for each class. Should have a shape of (C,), where C is the number of classes. Defaults to None.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Dice loss.

    """
    def __init__(self, reduction = 'mean', weight = None, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, input_tensor, target_tensor, pixel_weight=None):
      
        dice = dice_coeff(input_tensor, target_tensor, pixel_weight, epsilon=self.epsilon)

        if self.weight is not None:
            dice = dice * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (B, C, H, W)

        loss = 1 - dice

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass

        return loss
    

def weighted_cce_loss(input_tensor, target_tensor, pixel_weight=None, is_logit=False, epsilon=1e-6):
    """
    Compute the weighted Cross-Entropy loss between two tensors with support for pixel-level weights.

    Args:
        input_tensor (torch.Tensor): the predicted tensor, with shape (B, C, H, W), after the softmax activation.
        target_tensor (torch.Tensor): the target tensor, with shape (B, C, H, W) , one-hot encoding.
        pixel_weight (torch.Tensor, optional): Per pixel weight tensor, with shape (B, 1, H, W). Defaults to None.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Weighted Cross-Entropy loss.
    """
    # Flatten input and target to (B * H * W, C)
    channels = input_tensor.size(1)
    input_tensor = input_tensor.permute(0, 2, 3, 1).reshape(-1, channels)
    target_tensor = target_tensor.permute(0, 2, 3, 1).reshape(-1, channels)

    # Calculate the cross-entropy loss
    if is_logit:
        log_input = F.log_softmax(input_tensor, dim=1)
    else: 
        log_input = torch.log(input_tensor + epsilon)
    loss = -torch.sum(target_tensor * log_input, dim=1)  # (B * H * W,)

    if pixel_weight is not None:
        # If pixel_weight is provided, use it
        pixel_weight = pixel_weight.view(-1)  # Flatten pixel_weight to (B * H * W,)
        loss = loss * pixel_weight

    # Return the loss, optionally apply reduction
    return torch.mean(loss)


class WeightedCCELoss(nn.Module):
    """
    Weighted Cross-Entropy loss function for segmentation tasks with per-sample pixel weights.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Defaults to 'mean'.
        weight (torch.Tensor, optional): Weight for each class. Should have a shape of (C,). Defaults to None.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Weighted Cross-Entropy loss.
    """
    def __init__(self, reduction='mean', weight=None, is_logit=False):
        super(WeightedCCELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.is_logit = is_logit

    def forward(self, input_tensor, target_tensor, pixel_weight=None):
        loss = weighted_cce_loss(input_tensor, target_tensor, pixel_weight, is_logit=self.is_logit)

        # Apply class weights if provided
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass

        return loss
    

class DiceCELoss(nn.Module):
    """
    A combined loss function with Dice loss and cross entropy loss.

    Args:
        dice_weight (float, optional): Weight for Dice loss. Defaults to 0.5.
        ce_weight (float, optional): Weight for cross entropy loss. Defaults to 0.5.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Defaults to 'mean'.
        weight (torch.Tensor, optional): Weight for each class. Should have a shape of (C,), where C is the number of classes. Defaults to None.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-6.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, reduction='mean', epsilon=1e-6):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.reduction = reduction
        self.dice_loss = DiceLoss(reduction=reduction, epsilon=epsilon)
        self.cross_entropy = WeightedCCELoss(reduction=reduction)

    def forward(self, input_tensor, target_tensor, pixel_weight=None):
       
        dice_loss = self.dice_loss(input_tensor, target_tensor, pixel_weight)
        ce_loss = self.cross_entropy(input_tensor, target_tensor, pixel_weight)
        
        # Combine the two losses using the specified weights
        loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return loss
    

def morphological_op(input_tensor, kernel_size, operation='open'):
    """
    对输入的 tensor 进行形态学开运算或闭运算

    Args:
        input_tensor (torch.Tensor): 形状为 (batchsize, channels, H, W) 的输入张量
        kernel_size (int): 形态学结构元素的大小
        operation (str): 'open' 或 'close'，指定是开运算还是闭运算

    Returns:
        torch.Tensor: 处理后的张量
    """
    # 如果 kernel_size 是 float，则转换为整数
    if isinstance(kernel_size, float):
        kernel_size = int(round(kernel_size))
    elif isinstance(kernel_size, tuple):
        kernel_size = (int(round(kernel_size[0])), int(round(kernel_size[1])))
    else:
        kernel_size = (kernel_size, kernel_size)
    
    # 定义形态学结构元素，使用卷积核来模拟
    kernel = torch.ones((1, 1, kernel_size[0], kernel_size[1]), device=input_tensor.device)
    
    if operation == 'open':
        # 开运算：先腐蚀，再膨胀
        eroded = F.conv2d(input_tensor, kernel, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        opened = F.conv2d(eroded, kernel, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        return opened
    
    elif operation == 'close':
        # 闭运算：先膨胀，再腐蚀
        dilated = F.conv2d(input_tensor, kernel, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        closed = F.conv2d(dilated, kernel, stride=1, padding=(kernel_size[0]//2, kernel_size[1]//2))
        return closed
    else:
        raise ValueError("Invalid operation. Choose either 'open' or 'close'.")
    

def pseudo_label_refine(pred, kernel_size=(1,5)):
    """
    对预测的伪标签进行后处理，包括形态学开运算和闭运算

    Args:
        pred (torch.Tensor): 预测的伪标签，形状为 (batchsize, channels, H, W)
        kernel_size (tuple): 形态学结构元素的大小

    Returns:
        torch.Tensor: 处理后的伪标签
    """
    pred = pred.squeeze(2).permute(0, 2, 1)
    pred_binary = F.one_hot(torch.argmax(pred, dim=-1), num_classes=4).float().permute(0, 2, 1).unsqueeze(2)

    # for ch in range(pred_binary.size(1)):
    for ch in [1, 2, 3]:
        pred_binary[:, ch:ch+1, :, :] = morphological_op(pred_binary[:, ch:ch+1, :, :], kernel_size, operation='close')
        pred_binary[:, ch:ch+1, :, :] = morphological_op(pred_binary[:, ch:ch+1, :, :], kernel_size, operation='open')
    pred_binary[:, 0:1, :, :] = (torch.sum(pred_binary[:, 1:, :, :], dim=1, keepdim=True) < 0.5).float()
    
    refined = pred_binary.squeeze(2).permute(0, 2, 1)
    refined = F.one_hot(torch.argmax(refined, dim=-1), num_classes=4).float().permute(0, 2, 1).unsqueeze(2)
    return refined


def pyramid_consistency(pred_s, pred_ave):
    pred_s = torch.clamp(pred_s, min=1e-8, max=1.0)
    pred_ave = torch.clamp(pred_ave, min=1e-8, max=1.0)
    pc = torch.norm(pred_s - pred_ave, p=2, dim=1)
    return pc


def uncertainty_estimation(pred_s, pred_ave):
    pred_s = torch.clamp(pred_s, min=1e-8, max=1.0)
    pred_ave = torch.clamp(pred_ave, min=1e-8, max=1.0)
    uncertainty = torch.sum(pred_s * torch.log(pred_s / (pred_ave)), dim=1, keepdim=True)
    return uncertainty


def pc_loss(pred_s, pred_ave):
    loss = torch.mean(pyramid_consistency(pred_s, pred_ave))
    return loss


def weighted_pc_loss(pred_s, pred_ave, sample_weights):
    loss = torch.mean(pyramid_consistency(pred_s, pred_ave) * sample_weights)
    return loss


def piecewise_function(x, power):
    condition = x < 0.5
    return torch.where(condition, torch.zeros_like(x), (2 * x - 1) ** power)


def weighted_mse_loss(label, pred, sample_weights):
    # Assuming label is already in one-hot format
    loss = torch.mean((label - pred)**2 * sample_weights)
    return loss


def pred2plabel(pred):
    pred = pred.squeeze(2).permute(0, 2, 1)
    pred_sharpen = F.one_hot(torch.argmax(pred, dim=-1), num_classes=4).float()
    return pred_sharpen.permute(0, 2, 1).unsqueeze(2)


class EarlyStopping:
    """Early stopping utility class to stop training when validation loss does not improve."""
    def __init__(self, patience=5, verbose=False, delta=0, model_save_path=None):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            verbose (bool): Whether to print messages when stopping early.
            delta (float): Minimum change to qualify as an improvement.
            model_save_path (str): Path to save the best model and final model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.model_save_path = model_save_path

    def __call__(self, val_loss, model, epoch):
        """Call the EarlyStopping object during training to check for improvement."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model
            if self.model_save_path is not None:
                torch.save(model.state_dict(), f"{self.model_save_path}.best.pth")
                if self.verbose>1:
                    print(f"\tValidation loss improved. Saving best model to {self.model_save_path}.best.pth")
        else:
            self.counter += 1
            if self.verbose>1:
                print(f"\tEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\tEarly stopping triggered at epoch {epoch + 1}")
                # Save the final model
                if self.model_save_path is not None:
                    torch.save(model.state_dict(), f"{self.model_save_path}.final.pth")
                    if self.verbose:
                        print(f"\tSaving final model to {self.model_save_path}.final.pth")

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False