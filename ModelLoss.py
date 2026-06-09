import torch
import torch.nn.functional as F


def dice_loss_from_prob(prob, targets, smooth=1.0):
    """Dice loss 从概率图计算，与 DiceLoss 公式一致但输入为 prob 而非 logits。"""
    prob = prob.view(prob.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (prob * targets).sum(dim=1)
    union = prob.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def topk_bce_with_logits(logits, targets, k=0.15):
    """
    logits: [B,1,H,W]
    targets: [B,1,H,W] float(0/1)
    只对最难的 top k 像素计算 BCE，解决 hard negative（暗植被/阴影/黄土误检）。
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B,1,H,W]
    bce = bce.view(bce.size(0), -1)  # [B, HW]
    k = max(1, int(bce.size(1) * k))
    topk_loss, _ = torch.topk(bce, k, dim=1, largest=True, sorted=False)
    return topk_loss.mean()


def compute_edge_mask(mask, dilation=2):
    """二值 mask 的边缘（膨胀版）。mask: [B,1,H,W] 或 [1,H,W]."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = (mask > 0).float()
    # 用 max_pool 近似膨胀，然后取边界环
    k = 1 + 2 * max(0, int(dilation))
    pad = k // 2
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
    edge = (dil - m) > 0
    return edge.float()


def weighted_bce_with_logits(logits, targets, weight_map):
    """加权 BCEWithLogitsLoss，weight_map 同尺寸。"""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * weight_map).mean()


def weighted_topk_bce_with_logits(logits, targets, weight_map, k=0.15):
    """带权重的 Top-k BCE。"""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = loss * weight_map
    loss = loss.view(loss.size(0), -1)
    k_num = max(1, int(loss.size(1) * k))
    topk_vals, _ = torch.topk(loss, k_num, dim=1, largest=True, sorted=False)
    return topk_vals.mean()


