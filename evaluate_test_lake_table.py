# -*- coding: utf-8 -*-
"""
Stage2 测试集逐湖指标表（checkpoints/stage2.ckpt）。

输出列（一湖一行）：
  Lake ID | Lake-level IoU | Boundary F1 | Mean BDE (px) | BDE Samples

BF1@3px、BDE 实现见 metrics/VERSION22_Stage1Only/metrics.py（3×3 腐蚀边界）。
用法: python evaluate_test_lake_table.py
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
v22_dir = os.path.join(current_dir, "metrics", "VERSION22_Stage1Only")
sys.path.insert(0, current_dir)
if v22_dir not in sys.path:
    sys.path.append(v22_dir)

from ModelStructure_CUDA import ModelStructure
from MyDataLoader_numpy0127 import WaterDataset, custom_collate_fn
from utils.train_helpers import extract_lake_ids_from_folder, get_device
from metrics import (
    _extract_boundary,
    _to_numpy_bool_mask,
    calculate_binary_iou,
    calculate_boundary_metrics,
    logits_to_binary_mask,
)
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
OUT_CSV = os.path.join(current_dir, "logs", "stage2_test_lake_table.csv")
MAX_PATCHES_PER_FORWARD = 8
BOUNDARY_TOLERANCE = 3.0


def get_optimal_workers():
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    if NUM_WORKERS == "auto":
        return min(NUM_WORKERS_MAX, max(NUM_WORKERS_MIN, cpu_count // NUM_WORKERS_DIV))
    return int(NUM_WORKERS)


def assemble_patches_to_lake_logits(patch_logits, patch_coords, orig_hw):
    h_orig, w_orig = int(orig_hw[0]), int(orig_hw[1])
    p, _, ph, pw = patch_logits.shape
    if p != len(patch_coords):
        raise ValueError(f"patch 数与坐标数不一致: {p} vs {len(patch_coords)}")

    accum = torch.zeros((1, h_orig, w_orig), dtype=patch_logits.dtype)
    count = torch.zeros((1, h_orig, w_orig), dtype=patch_logits.dtype)
    for i, (x, y) in enumerate(patch_coords):
        x, y = int(x), int(y)
        h_end = min(y + ph, h_orig)
        w_end = min(x + pw, w_orig)
        crop_h = h_end - y
        crop_w = w_end - x
        if crop_h <= 0 or crop_w <= 0:
            continue
        accum[:, y:h_end, x:w_end] += patch_logits[i, 0, :crop_h, :crop_w]
        count[:, y:h_end, x:w_end] += 1.0
    return (accum / count.clamp_min(1e-6)).unsqueeze(0)


def count_bde_samples(pred_mask, gt_mask) -> int:
    pred_bool = _to_numpy_bool_mask(pred_mask)
    gt_bool = _to_numpy_bool_mask(gt_mask)
    pred_n = int(_extract_boundary(pred_bool).sum())
    gt_n = int(_extract_boundary(gt_bool).sum())
    return pred_n + gt_n


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


@torch.no_grad()
def predict_full_logits_for_one_lake(
    model,
    global_image: torch.Tensor,
    patches: torch.Tensor,
    patch_coords: List[Tuple[int, int]],
    orig_hw,
    device,
    device_name: str,
) -> torch.Tensor:
    model.eval()
    p_total = int(patches.shape[0])
    parts: List[torch.Tensor] = []
    use_amp = device_name == "CUDA" and USE_AMP

    for s in range(0, p_total, MAX_PATCHES_PER_FORWARD):
        e = min(p_total, s + MAX_PATCHES_PER_FORWARD)
        chunk = patches[s:e].to(device, non_blocking=True)
        chunk_coords = patch_coords[s:e]
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, patch_logits, _ = model(
                global_image,
                patches=chunk,
                patch_coords=chunk_coords,
                orig_hw=orig_hw,
                global_no_grad=True,
            )
        parts.append(patch_logits.float().cpu())

    patch_logits_all = torch.cat(parts, dim=0)
    full_logits = assemble_patches_to_lake_logits(patch_logits_all, patch_coords, orig_hw)
    return full_logits.to(device)


def _load_full_mask(lake_id, test_mask_folder, device):
    gt_path = os.path.join(test_mask_folder, f"{lake_id}.npy")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"缺少测试 mask: {gt_path}")
    gt_np = np.load(gt_path)
    if gt_np.ndim == 3:
        gt_np = gt_np.squeeze(0)
    return torch.from_numpy((gt_np > 0.5).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)


def main():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"未找到 checkpoint: {CKPT_PATH}")
    if not os.path.isfile(DEFAULT_MEAN_STD_PATH):
        raise FileNotFoundError(f"mean/std 文件不存在: {DEFAULT_MEAN_STD_PATH}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    test_image_folder = os.path.normpath(os.path.join(DATASET_FOLDER, TEST_SPLIT_NAME, IMAGE_DIRNAME))
    test_mask_folder = os.path.normpath(os.path.join(DATASET_FOLDER, TEST_SPLIT_NAME, MASK_DIRNAME))
    if len(extract_lake_ids_from_folder(test_image_folder)) == 0:
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
    print(f"ckpt: {CKPT_PATH}")
    model = build_model(device)

    rows = []

    for idx, data in enumerate(loader):
        lake_id = int(data["lake_ids"][0].item())
        global_image = data["global_images"][0:1].to(device, non_blocking=True)
        patches = data["patches_list"][0]
        patch_coords = data["patch_coords_list"][0]
        orig_hw = data["orig_hw_list"][0]

        full_logits = predict_full_logits_for_one_lake(
            model, global_image, patches, patch_coords, orig_hw, device, device_name
        )
        gt = _load_full_mask(lake_id, test_mask_folder, device)
        if full_logits.shape[-2:] != gt.shape[-2:]:
            full_logits = F.interpolate(
                full_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )

        pred = logits_to_binary_mask(full_logits)
        lake_iou = float(calculate_binary_iou(pred, gt).item())
        b = calculate_boundary_metrics(pred, gt, tolerance=BOUNDARY_TOLERANCE)
        bde_samples = count_bde_samples(pred, gt)
        mean_bde = float(b["boundary_mean_error"])

        rows.append((lake_id, lake_iou, float(b["boundary_f1"]), mean_bde, bde_samples))
        print(
            f"lake {lake_id}: lake_iou={lake_iou:.4f} bf1={b['boundary_f1']:.4f} "
            f"mean_bde={mean_bde:.4f} bde_samples={bde_samples}",
            flush=True,
        )

        if device_name == "CUDA" and (idx + 1) % 4 == 0:
            torch.cuda.empty_cache()

    rows.sort(key=lambda x: x[0])

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("Lake ID,Lake-level IoU,Boundary F1,Mean BDE (px),BDE Samples\n")
        for lake_id, lake_iou, bf1, mean_bde, bde_samples in rows:
            f.write(f"{lake_id},{lake_iou:.6f},{bf1:.6f},{mean_bde:.6f},{bde_samples}\n")

    print(f"\n共 {len(rows)} 个湖，结果已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
