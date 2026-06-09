# -*- coding: utf-8 -*-
"""
从 SHARE 包 TIFF 构建训练用 NPY 数据集。

输入（默认 paths_config 中 SHARE 路径）：
  Lake_sentinel2_GEE/{lake_id}.tif   4 波段 Sentinel-2，uint16 DN
  Lake_Mask/{lake_id}.tif            单波段 mask，0/1

输出（默认 SHARE/DataSet6）：
  {Train,Val,Test}/{image,mask}/{lake_id}.npy
  Train/train_band_mean_std.txt
  dataset_split_manifest.json

影像：DN × REFLECTANCE_COEF → float32 反射率，形状 (4, H, W)
Mask：二值 uint8，形状 (H, W)，取值 {0, 1}

用法:
  python preprocess/build_dataset_from_tiff.py
  python preprocess/build_dataset_from_tiff.py --output-dir D:/my/DataSet6 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from paths_config import (  # noqa: E402
    SHARE_DATASET_ROOT,
    SHARE_IMAGE_TIFF_DIR,
    SHARE_MASK_TIFF_DIR,
)

REFLECTANCE_COEF = 0.0001
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_BANDS = 4


def _lake_id_from_name(filename: str) -> int | None:
    stem, ext = os.path.splitext(filename)
    if ext.lower() not in (".tif", ".tiff"):
        return None
    try:
        return int(stem)
    except ValueError:
        return None


def find_paired_lake_ids(image_tiff_dir: str, mask_tiff_dir: str) -> List[int]:
    image_ids = set()
    for fn in os.listdir(image_tiff_dir):
        lid = _lake_id_from_name(fn)
        if lid is not None:
            image_ids.add(lid)

    mask_ids = set()
    for fn in os.listdir(mask_tiff_dir):
        lid = _lake_id_from_name(fn)
        if lid is not None:
            mask_ids.add(lid)

    paired = sorted(image_ids & mask_ids)
    only_img = sorted(image_ids - mask_ids)
    only_mask = sorted(mask_ids - image_ids)
    if only_img:
        print(f"警告: {len(only_img)} 个影像无对应 mask，已跳过（示例: {only_img[:5]}）")
    if only_mask:
        print(f"警告: {len(only_mask)} 个 mask 无对应影像，已跳过（示例: {only_mask[:5]}）")
    if not paired:
        raise RuntimeError("未找到成对的 lake_id（需同名 {id}.tif）")
    return paired


def split_lake_ids(
    lake_ids: Sequence[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    if abs(train_ratio + val_ratio + (1.0 - train_ratio - val_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio 须小于等于 1")
    ids = list(lake_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else max(0, min(1, n_total - n_train))
    n_test = n_total - n_train - n_val
    if n_test <= 0 and n_total >= 2:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def image_tiff_to_reflectance_npy(
    tiff_path: str,
    npy_path: str,
    coef: float = REFLECTANCE_COEF,
) -> np.ndarray:
    with rasterio.open(tiff_path) as ds:
        arr = ds.read().astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"影像须为 (C,H,W): {tiff_path}, got {arr.shape}")
    if arr.shape[0] < NUM_BANDS:
        raise ValueError(f"影像波段数不足 {NUM_BANDS}: {tiff_path}, shape={arr.shape}")
    arr = arr[:NUM_BANDS] * coef
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, arr)
    return arr


def mask_tiff_to_binary_npy(tiff_path: str, npy_path: str) -> np.ndarray:
    with rasterio.open(tiff_path) as ds:
        arr = ds.read(1)
    binary = (arr > 0).astype(np.uint8)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, binary)
    return binary


def convert_one_lake(
    lake_id: int,
    image_tiff_dir: str,
    mask_tiff_dir: str,
    out_image_path: str,
    out_mask_path: str,
    skip_existing: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    img_tif = os.path.join(image_tiff_dir, f"{lake_id}.tif")
    mask_tif = os.path.join(mask_tiff_dir, f"{lake_id}.tif")
    if skip_existing and os.path.isfile(out_image_path) and os.path.isfile(out_mask_path):
        img = np.load(out_image_path)
        mask = np.load(out_mask_path)
        return img, mask

    img = image_tiff_to_reflectance_npy(img_tif, out_image_path)
    mask = mask_tiff_to_binary_npy(mask_tif, out_mask_path)
    if img.shape[1:] != mask.shape:
        raise ValueError(
            f"lake {lake_id} 尺寸不一致: image {img.shape[1:]} vs mask {mask.shape}"
        )
    return img, mask


def compute_train_band_mean_std(train_image_dir: str, out_txt: str, num_bands: int = NUM_BANDS) -> Dict[str, List[float]]:
    npy_list = sorted(
        f for f in os.listdir(train_image_dir) if f.endswith(".npy") and not f.startswith(".")
    )
    if not npy_list:
        raise RuntimeError(f"训练集 image 目录为空: {train_image_dir}")

    band_sums = np.zeros(num_bands, dtype=np.float64)
    band_sq_sums = np.zeros(num_bands, dtype=np.float64)
    band_counts = np.zeros(num_bands, dtype=np.float64)

    for fn in npy_list:
        arr = np.load(os.path.join(train_image_dir, fn))
        if arr.ndim != 3 or arr.shape[0] < num_bands:
            raise ValueError(f"无效训练影像 {fn}: shape={arr.shape}")
        for b in range(num_bands):
            band = arr[b].astype(np.float64).ravel()
            band_sums[b] += band.sum()
            band_sq_sums[b] += (band ** 2).sum()
            band_counts[b] += band.size

    means = band_sums / band_counts
    stds = np.sqrt(np.maximum(band_sq_sums / band_counts - means ** 2, 0.0))

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fp:
        fp.write("# 训练集 4 波段 mean/std（Train/image 全像素统计）\n")
        fp.write("band\tmean\tstd\n")
        for b in range(num_bands):
            fp.write(f"{b + 1}\t{means[b]:.8f}\t{stds[b]:.8f}\n")

    return {"mean": means.tolist(), "std": stds.tolist()}


def build_dataset(
    image_tiff_dir: str,
    mask_tiff_dir: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    skip_existing: bool = False,
) -> None:
    lake_ids = find_paired_lake_ids(image_tiff_dir, mask_tiff_dir)
    train_ids, val_ids, test_ids = split_lake_ids(lake_ids, seed, train_ratio, val_ratio)
    print(f"成对湖泊: {len(lake_ids)}")
    print(f"划分 (seed={seed}): Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    split_map = {
        "Train": train_ids,
        "Val": val_ids,
        "Test": test_ids,
    }

    for split_name, ids in split_map.items():
        for sub in ("image", "mask"):
            os.makedirs(os.path.join(output_dir, split_name, sub), exist_ok=True)

    for split_name, ids in split_map.items():
        for i, lake_id in enumerate(ids, start=1):
            out_img = os.path.join(output_dir, split_name, "image", f"{lake_id}.npy")
            out_msk = os.path.join(output_dir, split_name, "mask", f"{lake_id}.npy")
            convert_one_lake(
                lake_id,
                image_tiff_dir,
                mask_tiff_dir,
                out_img,
                out_msk,
                skip_existing=skip_existing,
            )
            if i % 20 == 0 or i == len(ids):
                print(f"  [{split_name}] {i}/{len(ids)} lake {lake_id}")

    mean_std_path = os.path.join(output_dir, "Train", "train_band_mean_std.txt")
    stats = compute_train_band_mean_std(
        os.path.join(output_dir, "Train", "image"),
        mean_std_path,
    )
    print("训练集波段统计:")
    for b in range(NUM_BANDS):
        print(f"  band {b + 1}: mean={stats['mean'][b]:.8f}, std={stats['std'][b]:.8f}")

    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "reflectance_coef": REFLECTANCE_COEF,
        "image_tiff_dir": os.path.abspath(image_tiff_dir),
        "mask_tiff_dir": os.path.abspath(mask_tiff_dir),
        "output_dir": os.path.abspath(output_dir),
        "n_total": len(lake_ids),
        "Train": train_ids,
        "Val": val_ids,
        "Test": test_ids,
        "train_band_mean_std": stats,
    }
    manifest_path = os.path.join(output_dir, "dataset_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"已写入: {mean_std_path}")
    print(f"已写入: {manifest_path}")
    print("完成后请将 paths_config.py 中 DATASET_ROOT 指向上述 output_dir。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHARE TIFF → NPY 数据集 + 随机划分 + mean/std")
    parser.add_argument(
        "--image-tiff-dir",
        default=SHARE_IMAGE_TIFF_DIR,
        help="Sentinel-2 GeoTIFF 目录（Lake_sentinel2_GEE）",
    )
    parser.add_argument(
        "--mask-tiff-dir",
        default=SHARE_MASK_TIFF_DIR,
        help="Mask GeoTIFF 目录（Lake_Mask）",
    )
    parser.add_argument(
        "--output-dir",
        default=SHARE_DATASET_ROOT,
        help="输出 DataSet 根目录（Train/Val/Test）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 NPY 已存在则跳过转换（仍重算 mean/std 与 manifest）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio 须小于 1")
    build_dataset(
        image_tiff_dir=args.image_tiff_dir,
        mask_tiff_dir=args.mask_tiff_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
