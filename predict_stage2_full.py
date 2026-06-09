# -*- coding: utf-8 -*-
"""
使用 checkpoints/stage2.ckpt 对 Train/Val/Test 做整湖预测。

- Global + Patch 推理，patch 概率拼回原始分辨率
- 输出 JPG（叠加可视化）与 GeoTIFF（沿用 newimage_merged 空间参考）
- 输出目录：predict_stage2_output/{train,val,test}/tif/{lake_id}.tif

用法: python predict_stage2_full.py
路径见 paths_config.py
"""
import os
import sys
import math
import warnings

warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds

from MyDataLoader_numpy0127 import WaterDataset, downsample_whole_image
from ModelStructure_CUDA import ModelStructure
from utils.train_helpers import extract_lake_ids_from_folder
from paths_config import (
    DATASET_ROOT as DATASET_FOLDER,
    ORIG_TIFF_ROOT as ORIG_TIFF_FOLDER,
    TRAIN_BAND_MEAN_STD_PATH as DEFAULT_MEAN_STD_PATH,
)

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("请安装 Pillow: pip install Pillow")

# ========================
# 配置参数
# ========================
CKPT_PATH = os.path.join(current_dir, "checkpoints", "stage2.ckpt")
OUTPUT_ROOT = os.path.join(current_dir, "predict_stage2_output")

PROB_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45
OVERLAY_COLOR = (0, 255, 0)  # 绿色
PATCH_SIZE = 512

# 模型配置（与 train.py 一致）
MODEL_IN_CHANNELS = 4
MODEL_BACKBONE_DEPTH = 50
MODEL_BACKBONE_STRIDES = (1, 2, 2, 1)
MODEL_BACKBONE_DILATIONS = (1, 1, 1, 2)
MODEL_BACKBONE_OUT_INDICES = (0, 1, 2, 3)
MODEL_FPN_IN_CHANNELS = [256, 512, 1024, 2048]
MODEL_FPN_OUT_CHANNELS = 256
MODEL_FPN_NUM_OUTS = 4
MAX_PATCHES_PER_LAKE = 10

USE_AMP = True  # 混合精度加速


# ========================
# 工具函数
# ========================

def _load_band_mean_std(path):
    """从文件加载波段均值和标准差"""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("band"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                band, mean, std = int(parts[0]), float(parts[1]), float(parts[2])
                entries.append((band, mean, std))
            except ValueError:
                continue
    entries.sort(key=lambda x: x[0])
    means = torch.tensor([m for _, m, _ in entries], dtype=torch.float32).view(-1, 1, 1)
    stds = torch.tensor([s for _, _, s in entries], dtype=torch.float32).view(-1, 1, 1)
    return means, stds


def normalize_image(img_t, band_mean, band_std):
    """对图像张量做波段归一化"""
    mean = band_mean.to(img_t.device)
    std = band_std.to(img_t.device)
    return (img_t - mean) / (std + 1e-6)


def compute_patch_starts(length, patch_size=512):
    """计算 patch 的起始坐标（与 WaterDataset 一致）"""
    if length <= patch_size:
        return [0]
    num = int(math.ceil(length / patch_size))
    if num <= 1:
        return [0]
    stride = (length - patch_size) / (num - 1)
    starts = [int(round(i * stride)) for i in range(num)]
    starts[-1] = length - patch_size
    starts = sorted(set(starts))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def crop_patches_from_image(image_t, patch_size=512):
    """
    从原始分辨率图像裁剪 patches（不含 mask）。
    image_t: [C, H, W]
    返回: patches [P, C, ps, ps], coords [(x, y), ...]
    """
    C, H, W = image_t.shape
    hs = compute_patch_starts(H, patch_size)
    ws = compute_patch_starts(W, patch_size)
    patches = []
    coords = []
    for y in hs:
        for x in ws:
            patch = image_t[:, y:y + patch_size, x:x + patch_size]
            if patch.shape[-2:] != (patch_size, patch_size):
                patch = F.interpolate(
                    patch.unsqueeze(0), size=(patch_size, patch_size),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            patches.append(patch)
            coords.append((x, y))
    return torch.stack(patches, dim=0), coords


def assemble_patches(patch_probs, patch_coords, orig_h, orig_w, patch_size=512):
    """
    将 patch 的概率图拼接回原始分辨率。重叠区域取平均。
    patch_probs: [P, 1, ps, ps] numpy or tensor
    patch_coords: [(x, y), ...]
    返回: [H, W] numpy 概率图
    """
    if isinstance(patch_probs, torch.Tensor):
        patch_probs = patch_probs.cpu().numpy()
    accum = np.zeros((orig_h, orig_w), dtype=np.float64)
    count = np.zeros((orig_h, orig_w), dtype=np.float64)
    for i, (x, y) in enumerate(patch_coords):
        prob = patch_probs[i, 0]  # [ps, ps]
        ph, pw = prob.shape
        # 对于边界处 patch，确保不超出范围
        h_end = min(y + ph, orig_h)
        w_end = min(x + pw, orig_w)
        crop_h = h_end - y
        crop_w = w_end - x
        accum[y:h_end, x:w_end] += prob[:crop_h, :crop_w]
        count[y:h_end, x:w_end] += 1.0
    count = np.maximum(count, 1e-8)
    return accum / count


def tensor_to_rgb_uint8(x):
    """Tensor [C,H,W] -> numpy [H,W,3] uint8。4 波段取前 3 通道做可视化"""
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float64)
    if a.ndim == 2:
        a = a[np.newaxis, :, :]
    a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] > 3:
        a = a[..., :3]
    mn, mx = a.min(), a.max()
    if mx - mn > 1e-6:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def overlay_mask_on_image(image_rgb, binary_mask, alpha=OVERLAY_ALPHA, color=OVERLAY_COLOR):
    """将二值 mask 透明叠加到原图上"""
    h, w = image_rgb.shape[:2]
    m = np.asarray(binary_mask, dtype=np.float64)
    if m.ndim != 2:
        m = m.squeeze()
    if m.shape[0] != h or m.shape[1] != w:
        t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode="nearest")
        m = t.squeeze().numpy()
    m = np.clip(m, 0.0, 1.0)
    img = image_rgb.astype(np.float64)
    r, g, b = color
    blend = img * (1.0 - alpha * m)[:, :, np.newaxis]
    blend[:, :, 0] += alpha * m * r
    blend[:, :, 1] += alpha * m * g
    blend[:, :, 2] += alpha * m * b
    return np.clip(blend, 0, 255).astype(np.uint8)


def compute_hard_iou(pred_binary, gt_binary, eps=1e-6):
    """计算 Hard IoU"""
    pred = pred_binary.astype(np.float64).ravel()
    gt = gt_binary.astype(np.float64).ravel()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + eps) / (union + eps)


def read_geotiff_metadata(tif_path):
    """
    使用 rasterio 从原始 GeoTIFF 读取空间参考元数据。
    返回: dict with keys 'crs', 'transform', 'nodata'
    """
    metadata = {}
    try:
        with rasterio.open(tif_path) as ds:
            metadata["crs"] = ds.crs
            metadata["transform"] = ds.transform
            metadata["nodata"] = ds.nodata
    except Exception as e:
        print(f"  读取 GeoTIFF 元数据失败: {tif_path}: {e}")
    return metadata


def save_geotiff_with_metadata(out_path, data, geo_metadata, nodata=255):
    """
    使用 rasterio 保存单波段 GeoTIFF，带空间参考信息和 LZW 压缩。
    data: [H, W] numpy uint8
    geo_metadata: 从 read_geotiff_metadata 返回的字典
    """
    H, W = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": W,
        "height": H,
        "count": 1,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
    }
    if geo_metadata.get("crs") is not None:
        profile["crs"] = geo_metadata["crs"]
    if geo_metadata.get("transform") is not None:
        profile["transform"] = geo_metadata["transform"]

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


# ========================
# 主预测函数
# ========================

def predict_lake_stage2(
    model, device, lake_id, image_folder, mask_folder,
    band_mean, band_std, orig_tiff_folder,
    out_dir_jpg, out_dir_tif, use_amp=True,
):
    """
    使用 stage2 完整模型（global + patch 两个分支）对单个湖泊进行预测。
    
    流程：
      1. 加载原始 npy（全分辨率）→ 归一化
      2. Downsample 到 max_side=1024 作为 global_image
      3. 从全分辨率图像裁剪 512x512 patches
      4. 分批送入模型（global + patch），获得 patch_logits
      5. 将 patch 概率图拼接回原始分辨率
      6. 叠加到原图输出 JPG
      7. 读取原始 TIFF 的空间元数据，输出 GeoTIFF
    
    返回: (binary_mask_fullres, hard_iou_or_None)
    """
    # 1. 加载原始 npy
    npy_path = os.path.normpath(os.path.join(image_folder, f"{lake_id}.npy"))
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"未找到 npy: {npy_path}")
    raw_image = np.load(npy_path)  # [C, H, W]
    orig_h, orig_w = raw_image.shape[1], raw_image.shape[2]
    raw_t = torch.from_numpy(raw_image).float()  # 未归一化

    # 2. 归一化后 downsample 作为 global_image
    normed_t = normalize_image(raw_t, band_mean, band_std)
    global_image = downsample_whole_image(normed_t, max_side=1024)  # [C, H', W']
    global_image = global_image.unsqueeze(0).to(device)  # [1, C, H', W']

    # 3. 从全分辨率裁剪 patches 并归一化
    patches, patch_coords = crop_patches_from_image(normed_t, patch_size=PATCH_SIZE)
    # patches: [P, C, 512, 512]
    n_total = patches.shape[0]

    # 4. 分批推理（每次最多 MAX_PATCHES_PER_LAKE 个 patch，节省显存）
    all_patch_probs = []
    batch_size = MAX_PATCHES_PER_LAKE

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_patches = patches[start:end].to(device)
            batch_coords = patch_coords[start:end]
            orig_hw = (orig_h, orig_w)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                global_logits, patch_logits, patch_bnd_logits = model(
                    global_image,
                    patches=batch_patches,
                    patch_coords=batch_coords,
                    orig_hw=orig_hw,
                    global_no_grad=True,
                )

            if patch_logits is not None:
                # 确保 patch_logits 尺寸与 patch 一致
                if patch_logits.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
                    patch_logits = F.interpolate(
                        patch_logits, size=(PATCH_SIZE, PATCH_SIZE),
                        mode="bilinear", align_corners=False
                    )
                patch_prob = torch.sigmoid(patch_logits).cpu().numpy()
                all_patch_probs.append(patch_prob)

            del batch_patches, global_logits, patch_logits, patch_bnd_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # 5. 拼接 patch 预测到原始分辨率
    if len(all_patch_probs) > 0:
        all_patch_probs_np = np.concatenate(all_patch_probs, axis=0)  # [P, 1, 512, 512]
        prob_fullres = assemble_patches(
            all_patch_probs_np, patch_coords, orig_h, orig_w, patch_size=PATCH_SIZE
        )
    else:
        # fallback：使用 global 预测
        print(f"  警告: lake {lake_id} 没有 patch 预测结果，使用 global 预测")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                global_logits, _, _ = model(
                    global_image, patches=None, patch_coords=None, orig_hw=None
                )
        prob_global = torch.sigmoid(global_logits[0, 0]).cpu().numpy()
        prob_fullres_t = torch.from_numpy(prob_global).float().unsqueeze(0).unsqueeze(0)
        prob_fullres_t = F.interpolate(
            prob_fullres_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        prob_fullres = prob_fullres_t.squeeze().numpy()

    binary_fullres = (prob_fullres >= PROB_THRESHOLD).astype(np.uint8)

    # 6. 输出 JPG：叠加到原图
    os.makedirs(out_dir_jpg, exist_ok=True)
    img_rgb = tensor_to_rgb_uint8(raw_t)  # [H, W, 3] uint8，使用原始像素值（非归一化）
    overlay = overlay_mask_on_image(img_rgb, binary_fullres)
    jpg_path = os.path.join(out_dir_jpg, f"{lake_id}.jpg")
    PILImage.fromarray(overlay).save(jpg_path, "JPEG", quality=95)

    # 7. 输出 GeoTIFF
    os.makedirs(out_dir_tif, exist_ok=True)
    orig_tif_path = os.path.normpath(os.path.join(orig_tiff_folder, f"{lake_id}.tif"))
    tif_path = os.path.join(out_dir_tif, f"{lake_id}.tif")
    if os.path.isfile(orig_tif_path):
        geo_meta = read_geotiff_metadata(orig_tif_path)
        # 保存为 uint8: 0=非水体, 1=水体, nodata=255
        save_geotiff_with_metadata(tif_path, binary_fullres, geo_meta, nodata=255)
    else:
        # 没有原始 TIFF，直接保存不含空间信息的 GeoTIFF
        print(f"  警告: 未找到原始 TIFF {orig_tif_path}，保存不含空间信息的 TIFF")
        save_geotiff_with_metadata(tif_path, binary_fullres, {}, nodata=255)

    # 8. 计算 Hard IoU（若有 mask）
    hard_iou = None
    mask_npy = os.path.normpath(os.path.join(mask_folder, f"{lake_id}.npy"))
    if os.path.isfile(mask_npy):
        try:
            gt = np.load(mask_npy)
            if gt.ndim == 3:
                gt = gt.squeeze(0)
            if gt.shape != binary_fullres.shape:
                gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
                gt_t = F.interpolate(gt_t, size=binary_fullres.shape, mode="nearest")
                gt = gt_t.squeeze().numpy()
            gt_bin = (gt > 0.5).astype(np.float64)
            hard_iou = compute_hard_iou(binary_fullres, gt_bin)
        except Exception as e:
            print(f"  计算 IoU 失败 lake {lake_id}: {e}")

    return binary_fullres, hard_iou


def run_predict_split(model, device, split_name, image_folder, mask_folder,
                      band_mean, band_std, orig_tiff_folder, out_root, lake_ids, use_amp=True):
    """
    对一个数据集分割做预测。
    返回: (mean_iou, lake_iou_details)
        lake_iou_details: [(lake_id, hard_iou), ...] 每个湖泊的 IoU（无 mask 的为 None）
    """
    out_jpg = os.path.join(out_root, split_name, "jpg")
    out_tif = os.path.join(out_root, split_name, "tif")
    os.makedirs(out_jpg, exist_ok=True)
    os.makedirs(out_tif, exist_ok=True)

    ious = []
    lake_iou_details = []  # [(lake_id, hard_iou or None), ...]
    n_ok, n_err = 0, 0

    for idx, lake_id in enumerate(sorted(lake_ids)):
        try:
            _, hard_iou = predict_lake_stage2(
                model, device, lake_id, image_folder, mask_folder,
                band_mean, band_std, orig_tiff_folder,
                out_jpg, out_tif, use_amp=use_amp,
            )
            lake_iou_details.append((lake_id, hard_iou))
            if hard_iou is not None:
                ious.append(hard_iou)
                print(f"  [{split_name}] ({idx+1}/{len(lake_ids)}) LakeID={lake_id}: "
                      f"Hard IoU = {hard_iou:.4f}")
            else:
                print(f"  [{split_name}] ({idx+1}/{len(lake_ids)}) LakeID={lake_id}: "
                      f"无 mask，跳过 IoU")
            n_ok += 1
        except Exception as e:
            n_err += 1
            lake_iou_details.append((lake_id, None))
            print(f"  [{split_name}] ({idx+1}/{len(lake_ids)}) LakeID={lake_id}: 错误 - {e}")
            import traceback
            traceback.print_exc()

    mean_iou = np.mean(ious) if ious else 0.0
    print(f"\n  [{split_name}] 完成: 成功 {n_ok}, 失败 {n_err}")
    print(f"  [{split_name}] 平均 Hard IoU = {mean_iou:.4f} (共 {len(ious)} 个有效样本)")
    return mean_iou, lake_iou_details


# ========================
# 主程序
# ========================

def main():
    print("=" * 70)
    print("Stage2 全模型预测（global + patch 双分支）")
    print("=" * 70)

    # 检查 checkpoint
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"未找到 checkpoint: {CKPT_PATH}")
    print(f"Checkpoint: {CKPT_PATH}")

    # 加载波段均值/标准差
    band_mean, band_std = _load_band_mean_std(DEFAULT_MEAN_STD_PATH)
    print(f"Mean/Std: {DEFAULT_MEAN_STD_PATH}")

    # 数据集路径
    Folder = os.path.normpath(DATASET_FOLDER)
    train_img = os.path.join(Folder, "Train", "image")
    train_mask = os.path.join(Folder, "Train", "mask")
    val_img = os.path.join(Folder, "Val", "image")
    val_mask = os.path.join(Folder, "Val", "mask")
    test_img = os.path.join(Folder, "Test", "image")
    test_mask = os.path.join(Folder, "Test", "mask")

    # 提取 LakeID
    train_ids = extract_lake_ids_from_folder(train_img)
    val_ids = extract_lake_ids_from_folder(val_img)
    test_ids = extract_lake_ids_from_folder(test_img)
    # 确保训练集不包含验证/测试集的 lake
    train_ids = train_ids - val_ids - test_ids
    print(f"训练集: {len(train_ids)} 个, 验证集: {len(val_ids)} 个, 测试集: {len(test_ids)} 个")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if device.type == "cuda" else "CPU"
    print(f"使用设备: {device} ({device_name})")
    if device_name == "CUDA":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 构建模型
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

    # 加载 checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"模型加载成功: {CKPT_PATH}")
    print(f"预测阈值: {PROB_THRESHOLD}, 叠加透明度: {OVERLAY_ALPHA}")
    print(f"原始 TIFF 目录: {ORIG_TIFF_FOLDER}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 70)

    # 对三个数据集分别预测，收集每个湖泊的 IoU
    results = {}
    all_details = {}  # {split_name: [(lake_id, iou), ...]}
    orig_tiff = os.path.normpath(ORIG_TIFF_FOLDER)

    print("\n>>> 训练集预测 ...")
    train_iou, train_details = run_predict_split(
        model, device, "train", train_img, train_mask,
        band_mean, band_std, orig_tiff, OUTPUT_ROOT, train_ids, use_amp=USE_AMP,
    )
    results["train"] = train_iou
    all_details["train"] = train_details

    print("\n>>> 验证集预测 ...")
    val_iou, val_details = run_predict_split(
        model, device, "val", val_img, val_mask,
        band_mean, band_std, orig_tiff, OUTPUT_ROOT, val_ids, use_amp=USE_AMP,
    )
    results["val"] = val_iou
    all_details["val"] = val_details

    print("\n>>> 测试集预测 ...")
    test_iou, test_details = run_predict_split(
        model, device, "test", test_img, test_mask,
        band_mean, band_std, orig_tiff, OUTPUT_ROOT, test_ids, use_amp=USE_AMP,
    )
    results["test"] = test_iou
    all_details["test"] = test_details

    # 汇总
    print("\n" + "=" * 70)
    print("汇总 Hard IoU（阈值=%.2f）：" % PROB_THRESHOLD)
    print(f"  训练集: {results['train']:.4f}")
    print(f"  验证集: {results['val']:.4f}")
    print(f"  测试集: {results['test']:.4f}")
    print("=" * 70)

    # 保存详细 IoU 结果到文本（每个湖泊一行 + 每个数据集的平均值）
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    iou_txt = os.path.join(OUTPUT_ROOT, "iou_details.txt")
    with open(iou_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Stage2 全模型预测 IoU 详细结果（global + patch 双分支）\n")
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"阈值: {PROB_THRESHOLD}\n")
        f.write("=" * 70 + "\n\n")

        for split_name, display_name in [("train", "训练集"), ("val", "验证集"), ("test", "测试集")]:
            details = all_details[split_name]
            valid_ious = [iou for _, iou in details if iou is not None]
            mean_val = np.mean(valid_ious) if valid_ious else 0.0

            f.write("-" * 50 + "\n")
            f.write(f"{display_name} ({split_name})  共 {len(details)} 个湖泊\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'LakeID':<12}{'Hard IoU':<12}\n")
            for lake_id, iou in details:
                if iou is not None:
                    f.write(f"{lake_id:<12}{iou:.4f}\n")
                else:
                    f.write(f"{lake_id:<12}{'N/A'}\n")
            f.write(f"\n{display_name} 平均 Hard IoU: {mean_val:.4f} "
                    f"(有效样本: {len(valid_ious)}/{len(details)})\n\n")

        f.write("=" * 70 + "\n")
        f.write("汇总\n")
        f.write("=" * 70 + "\n")
        f.write(f"  训练集 平均 Hard IoU: {results['train']:.4f}\n")
        f.write(f"  验证集 平均 Hard IoU: {results['val']:.4f}\n")
        f.write(f"  测试集 平均 Hard IoU: {results['test']:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"IoU 详细结果已保存至: {iou_txt}")
    print("完成！")


if __name__ == "__main__":
    main()
