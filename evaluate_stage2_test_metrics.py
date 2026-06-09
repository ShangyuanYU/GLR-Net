# -*- coding: utf-8 -*-
"""
Stage2 测试集汇总评估（checkpoints/stage2.ckpt）。

在 Test 集上逐湖 patch 推理、拼湖后计算 hard IoU、Dice 等，写入 logs/stage2_test_eval_metrics.txt。

用法: python evaluate_stage2_test_metrics.py
数据路径见 paths_config.py
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ModelStructure_CUDA import ModelStructure
from MyDataLoader_numpy0127 import WaterDataset, custom_collate_fn
from utils.train_helpers import extract_lake_ids_from_folder, get_device
from train import (
    TEST_SPLIT_NAME,
    IMAGE_DIRNAME,
    MASK_DIRNAME,
    MODEL_IN_CHANNELS,
    MODEL_BACKBONE_DEPTH,
    MODEL_BACKBONE_STRIDES,
    MODEL_BACKBONE_DILATIONS,
    MODEL_BACKBONE_OUT_INDICES,
    MODEL_FPN_IN_CHANNELS,
    MODEL_FPN_OUT_CHANNELS,
    MODEL_FPN_NUM_OUTS,
    PIN_MEMORY,
    TEST_DROP_LAST,
    PERSISTENT_WORKERS,
    NUM_WORKERS,
    NUM_WORKERS_MIN,
    NUM_WORKERS_MAX,
    NUM_WORKERS_DIV,
    USE_AMP,
)

warnings.simplefilter("ignore")

from paths_config import DATASET_ROOT as DATASET_FOLDER, TRAIN_BAND_MEAN_STD_PATH as DEFAULT_MEAN_STD_PATH

CKPT_PATH = os.path.join(current_dir, "checkpoints", "stage2.ckpt")
OUT_TXT = os.path.join(current_dir, "logs", "stage2_test_eval_metrics.txt")
THRESHOLD = 0.5
BOUNDARY_DILATION = 2


def get_optimal_workers():
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    if NUM_WORKERS == "auto":
        return min(NUM_WORKERS_MAX, max(NUM_WORKERS_MIN, cpu_count // NUM_WORKERS_DIV))
    return int(NUM_WORKERS)


def assemble_patches_to_lake(patch_prob, patch_coords, orig_hw):
    """
    patch_prob: [P,1,H,W] in [0,1]
    return: [1,1,H_orig,W_orig]
    """
    h_orig, w_orig = int(orig_hw[0]), int(orig_hw[1])
    p, _, ph, pw = patch_prob.shape
    if p != len(patch_coords):
        raise ValueError(f"patch 数与坐标数不一致: {p} vs {len(patch_coords)}")

    accum = torch.zeros((1, h_orig, w_orig), dtype=patch_prob.dtype, device=patch_prob.device)
    count = torch.zeros((1, h_orig, w_orig), dtype=patch_prob.dtype, device=patch_prob.device)
    for i, (x, y) in enumerate(patch_coords):
        x, y = int(x), int(y)
        h_end = min(y + ph, h_orig)
        w_end = min(x + pw, w_orig)
        crop_h = h_end - y
        crop_w = w_end - x
        if crop_h <= 0 or crop_w <= 0:
            continue
        accum[:, y:h_end, x:w_end] += patch_prob[i, :, :crop_h, :crop_w]
        count[:, y:h_end, x:w_end] += 1.0
    return (accum / count.clamp_min(1e-6)).unsqueeze(0)


def mask_to_boundary(mask_bin):
    inv = 1.0 - mask_bin
    eroded_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - eroded_inv
    bnd = (mask_bin - eroded).clamp(0.0, 1.0)
    return bnd


def boundary_f1(pred_bin, gt_bin, dilation=2, eps=1e-6):
    pred_b = mask_to_boundary(pred_bin)
    gt_b = mask_to_boundary(gt_bin)
    pred_cnt = pred_b.sum().item()
    gt_cnt = gt_b.sum().item()
    if pred_cnt < eps and gt_cnt < eps:
        return 1.0
    if pred_cnt < eps or gt_cnt < eps:
        return 0.0

    k = 2 * int(dilation) + 1
    gt_dilate = F.max_pool2d(gt_b, kernel_size=k, stride=1, padding=dilation)
    pred_dilate = F.max_pool2d(pred_b, kernel_size=k, stride=1, padding=dilation)

    pred_match = (pred_b * (gt_dilate > 0).float()).sum()
    gt_match = (gt_b * (pred_dilate > 0).float()).sum()
    precision = pred_match / (pred_b.sum() + eps)
    recall = gt_match / (gt_b.sum() + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return float(f1.item())


def compute_iou_dice(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    union = pred_sum + gt_sum - inter
    iou = (inter + eps) / (union + eps)
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    return float(iou.item()), float(dice.item()), float(inter.item()), float(pred_sum.item()), float(gt_sum.item())


def build_model(device):
    backbone_cfg = dict(
        type="ResNet",
        depth=MODEL_BACKBONE_DEPTH,
        in_channels=MODEL_IN_CHANNELS,
        num_stages=4,
        out_indices=MODEL_BACKBONE_OUT_INDICES,
        strides=MODEL_BACKBONE_STRIDES,
        dilations=MODEL_BACKBONE_DILATIONS,
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
    )
    neck_cfg = dict(
        type="FPN",
        in_channels=MODEL_FPN_IN_CHANNELS,
        out_channels=MODEL_FPN_OUT_CHANNELS,
        num_outs=MODEL_FPN_NUM_OUTS,
    )
    model = ModelStructure(backbone_cfg, neck_cfg).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"未找到 checkpoint: {CKPT_PATH}")
    if not os.path.isfile(DEFAULT_MEAN_STD_PATH):
        raise FileNotFoundError(f"mean/std 文件不存在: {DEFAULT_MEAN_STD_PATH}")

    test_image_folder = os.path.normpath(os.path.join(DATASET_FOLDER, TEST_SPLIT_NAME, IMAGE_DIRNAME))
    test_mask_folder = os.path.normpath(os.path.join(DATASET_FOLDER, TEST_SPLIT_NAME, MASK_DIRNAME))
    test_ids = extract_lake_ids_from_folder(test_image_folder)
    if len(test_ids) == 0:
        raise RuntimeError(f"测试集为空: {test_image_folder}")

    dataset = WaterDataset(
        test_image_folder,
        test_mask_folder,
        is_train=False,
        global_image_folder=test_image_folder,
        global_mask_folder=test_mask_folder,
        mean_std_path=DEFAULT_MEAN_STD_PATH,
    )
    workers = get_optimal_workers()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        drop_last=TEST_DROP_LAST,
        collate_fn=custom_collate_fn,
        persistent_workers=PERSISTENT_WORKERS if workers > 0 else False,
    )

    device, device_name = get_device()
    print(f"设备: {device} ({device_name})")
    model = build_model(device)

    lake_rows = []
    lake_iou_list = []
    bf1_list = []
    total_inter = 0.0
    total_pred = 0.0
    total_gt = 0.0

    with torch.no_grad():
        for data in loader:
            lake_id = int(data["lake_ids"][0].item())
            global_image = data["global_images"][0:1].to(device, non_blocking=True)
            patches = data["patches_list"][0].to(device, non_blocking=True)
            patch_coords = data["patch_coords_list"][0]
            orig_hw = data["orig_hw_list"][0]

            with torch.cuda.amp.autocast(enabled=(device_name == "CUDA" and USE_AMP)):
                _, patch_logits, _ = model(
                    global_image,
                    patches=patches,
                    patch_coords=patch_coords,
                    orig_hw=orig_hw,
                    global_no_grad=True,
                )

            patch_prob = torch.sigmoid(patch_logits)
            pred_full = assemble_patches_to_lake(patch_prob, patch_coords, orig_hw)
            pred_bin = (pred_full > THRESHOLD).float()

            gt_path = os.path.join(test_mask_folder, f"{lake_id}.npy")
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"缺少测试 mask: {gt_path}")
            gt_np = np.load(gt_path)
            if gt_np.ndim == 3:
                gt_np = gt_np.squeeze(0)
            gt_t = torch.from_numpy((gt_np > 0.5).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            if gt_t.shape[-2:] != pred_bin.shape[-2:]:
                gt_t = F.interpolate(gt_t, size=pred_bin.shape[-2:], mode="nearest")

            iou, dice, inter, pred_sum, gt_sum = compute_iou_dice(pred_bin, gt_t)
            bf1 = boundary_f1(pred_bin, gt_t, dilation=BOUNDARY_DILATION)

            total_inter += inter
            total_pred += pred_sum
            total_gt += gt_sum
            lake_iou_list.append(iou)
            bf1_list.append(bf1)
            lake_rows.append((lake_id, iou, dice, bf1))

            print(f"lake {lake_id}: hard_iou={iou:.4f}, dice={dice:.4f}, boundary_f1={bf1:.4f}")

    hard_iou = (total_inter + 1e-6) / (total_pred + total_gt - total_inter + 1e-6)
    hard_dice = (2.0 * total_inter + 1e-6) / (total_pred + total_gt + 1e-6)
    lake_iou = float(np.mean(lake_iou_list)) if lake_iou_list else 0.0
    boundary_f1_mean = float(np.mean(bf1_list)) if bf1_list else 0.0

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("LakeID\thard_iou\tdice\tboundary_f1\n")
        for lake_id, iou, dice, bf1 in sorted(lake_rows, key=lambda x: x[0]):
            f.write(f"{lake_id}\t{iou:.6f}\t{dice:.6f}\t{bf1:.6f}\n")
        f.write("\nsummary_metric\tvalue\n")
        f.write(f"hard_iou\t{hard_iou:.6f}\n")
        f.write(f"dice\t{hard_dice:.6f}\n")
        f.write(f"lake_iou\t{lake_iou:.6f}\n")
        f.write(f"boundary_f1\t{boundary_f1_mean:.6f}\n")
        f.write(f"count\t{len(lake_rows)}\n")

    print("\n测试集汇总:")
    print(f"hard IOU      : {hard_iou:.4f}")
    print(f"Dice          : {hard_dice:.4f}")
    print(f"Lake IOU(mean): {lake_iou:.4f}")
    print(f"Boundary F1   : {boundary_f1_mean:.4f}")
    print(f"结果已保存: {OUT_TXT}")


if __name__ == "__main__":
    main()

