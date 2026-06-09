# -*- coding: utf-8 -*-
"""训练/验证指标：pixel-level soft IoU 等。"""
import torch
import torch.nn.functional as F


def calculate_pixel_soft_iou(logits, targets, eps=1e-6):
    """
    Pixel-level soft IOU：全体像素上先求 soft 交、并，再算一个 IOU 标量。
    Stage 1 训练中用于 loss 调权、epoch 统计、scheduler.step。

    logits:  (B, 1, H, W)  raw model output
    targets: (B, 1, H, W)  binary {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten()
    targets = targets.flatten()
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)

def calculate_hard_iou(logits, targets, eps=1e-6):
    """
    Hard IOU（二值化 pred = sigmoid>0.5 后算 IOU）。仅用于评估/打印，不参与训练反向。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def calculate_hard_dice(logits, targets, eps=1e-6):
    """
    Hard Dice（二值化 pred = sigmoid>0.5 后算 Dice）。仅用于评估/打印，不参与训练反向。
    返回标量 [0,1]，越高越好。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1) + eps
    gt_sum = targets.sum(dim=1) + eps
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum)
    return dice.mean()
