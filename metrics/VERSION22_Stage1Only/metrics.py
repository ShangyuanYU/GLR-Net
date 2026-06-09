import math

import numpy as np
import torch

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    HAS_SCIPY = True
except Exception:
    binary_erosion = None
    distance_transform_edt = None
    HAS_SCIPY = False

try:
    import cv2

    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


def calculate_pixel_soft_iou(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten()
    targets = targets.flatten()
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def calculate_hard_iou(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return calculate_binary_iou(preds, targets, eps=eps)


def calculate_hard_dice(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return calculate_binary_dice(preds, targets, eps=eps)


def calculate_binary_iou(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def calculate_binary_dice(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1) + eps
    gt_sum = targets.sum(dim=1) + eps
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum)
    return dice.mean()


def calculate_lake_iou(preds, targets, eps=1e-6):
    return calculate_binary_iou(preds, targets, eps=eps)


def logits_to_binary_mask(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).float()


def _to_numpy_bool_mask(mask):
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={arr.shape}")
    return arr > 0.5


def _extract_boundary(mask_bool):
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected 2D boolean mask, got shape={mask_bool.shape}")
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)

    if HAS_SCIPY:
        eroded = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return np.logical_and(mask_bool, np.logical_not(eroded))

    if HAS_CV2:
        mask_u8 = mask_bool.astype(np.uint8)
        eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
        return np.logical_and(mask_bool, eroded == 0)

    raise RuntimeError("Boundary metrics require scipy or cv2.")


def _distance_to_boundary(boundary_bool):
    if boundary_bool.ndim != 2:
        raise ValueError(f"Expected 2D boundary mask, got shape={boundary_bool.shape}")
    if not boundary_bool.any():
        return None

    inverted = (~boundary_bool).astype(np.uint8)
    if HAS_SCIPY:
        return distance_transform_edt(inverted)

    if HAS_CV2:
        return cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

    raise RuntimeError("Boundary metrics require scipy or cv2.")


def calculate_boundary_metrics(pred_mask, target_mask, tolerance=3.0):
    pred_bool = _to_numpy_bool_mask(pred_mask)
    target_bool = _to_numpy_bool_mask(target_mask)
    pred_boundary = _extract_boundary(pred_bool)
    target_boundary = _extract_boundary(target_bool)

    if not pred_boundary.any() and not target_boundary.any():
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "boundary_mean_error": 0.0,
            "boundary_median_error": 0.0,
            "boundary_p90_error": 0.0,
        }

    fallback_dist = float(math.hypot(pred_bool.shape[0], pred_bool.shape[1]))

    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())

    if target_count > 0 and pred_count > 0:
        dist_to_target = _distance_to_boundary(target_boundary)
        dist_to_pred = _distance_to_boundary(pred_boundary)
        pred_match = dist_to_target[pred_boundary] <= tolerance
        target_match = dist_to_pred[target_boundary] <= tolerance
        precision = float(pred_match.mean()) if pred_match.size > 0 else 0.0
        recall = float(target_match.mean()) if target_match.size > 0 else 0.0
        dist_g_to_p = dist_to_pred[target_boundary]
        dist_p_to_g = dist_to_target[pred_boundary]
    elif pred_count == 0 and target_count > 0:
        precision = 0.0
        recall = 0.0
        dist_g_to_p = np.full(target_count, fallback_dist, dtype=np.float32)
        dist_p_to_g = np.zeros((0,), dtype=np.float32)
    elif pred_count > 0 and target_count == 0:
        precision = 0.0
        recall = 0.0
        dist_g_to_p = np.zeros((0,), dtype=np.float32)
        dist_p_to_g = np.full(pred_count, fallback_dist, dtype=np.float32)
    else:
        precision = 1.0
        recall = 1.0
        dist_g_to_p = np.zeros((0,), dtype=np.float32)
        dist_p_to_g = np.zeros((0,), dtype=np.float32)

    if precision + recall > 0:
        boundary_f1 = 2.0 * precision * recall / (precision + recall)
    else:
        boundary_f1 = 0.0

    all_dist = np.concatenate([dist_g_to_p, dist_p_to_g], axis=0)
    if all_dist.size == 0:
        all_dist = np.array([0.0], dtype=np.float32)

    return {
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "boundary_mean_error": float(np.mean(all_dist)),
        "boundary_median_error": float(np.median(all_dist)),
        "boundary_p90_error": float(np.percentile(all_dist, 90)),
    }
