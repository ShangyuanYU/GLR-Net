# -*- coding: utf-8 -*-
"""
边界距离误差（BDE）核心函数与可选分布图。

boundary_error_pixels(gt, pred)：GT/预测 mask 经 3×3 腐蚀得边界，双向距离变换后拼接边界像元误差。
供 metrics/recalc_test_metrics_unified.py 调用。

独立运行（需先执行 predict_stage2_full.py）：
  读取 predict_stage2_output/test/tif/ 与 Test/mask，绘制 histogram 与 CDF。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, distance_transform_edt

_METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_METRICS_DIR)

if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from paths_config import TEST_MASK_DIR  # noqa: E402

DATASET_MASK_DIR = TEST_MASK_DIR
PRED_TIF_DIR = os.path.join(_CODE_ROOT, "predict_stage2_output", "test", "tif")
OUT_DIR = os.path.join(_METRICS_DIR, "output")
OUT_FIG_HIST = os.path.join(OUT_DIR, "boundary_error_histogram_test.png")
OUT_FIG_CDF = os.path.join(OUT_DIR, "boundary_error_cdf_test.png")
OUT_NPY = os.path.join(OUT_DIR, "boundary_error_values_test.npz")
X_MAX_FOCUS = 20.0
MODEL_LABEL = "Ours"


def load_gt_mask(mask_path: str) -> np.ndarray:
    arr = np.load(mask_path)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return (arr > 0).astype(np.uint8)


def load_pred_mask(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
    return (arr == 1).astype(np.uint8)


def mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(
        mask.astype(bool), structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return mask.astype(bool) & (~eroded)


def boundary_error_pixels(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt_b = mask_to_boundary(gt_mask)
    pr_b = mask_to_boundary(pred_mask)

    if (not gt_b.any()) and (not pr_b.any()):
        return np.empty((0,), dtype=np.float32)
    if not gt_b.any() or not pr_b.any():
        return np.empty((0,), dtype=np.float32)

    dist_to_pred = distance_transform_edt(~pr_b)
    d1 = dist_to_pred[gt_b]
    dist_to_gt = distance_transform_edt(~gt_b)
    d2 = dist_to_gt[pr_b]
    return np.concatenate([d1, d2]).astype(np.float32)


def collect_lake_ids_from_mask_dir(mask_dir: str) -> List[int]:
    ids = []
    for fn in os.listdir(mask_dir):
        if not fn.lower().endswith(".npy"):
            continue
        stem = os.path.splitext(fn)[0]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(ids)


def compute_errors_for_pred_dir(tif_dir: str, lake_ids: List[int]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for idx, lake_id in enumerate(lake_ids, start=1):
        gt_path = os.path.join(DATASET_MASK_DIR, f"{lake_id}.npy")
        tif_path = os.path.join(tif_dir, f"{lake_id}.tif")
        if not os.path.isfile(gt_path) or not os.path.isfile(tif_path):
            continue
        gt = load_gt_mask(gt_path)
        pred = load_pred_mask(tif_path)
        if pred.shape != gt.shape:
            continue
        errs = boundary_error_pixels(gt, pred)
        if errs.size > 0:
            chunks.append(errs)
        if idx % 10 == 0 or idx == len(lake_ids):
            print(f"[{idx}/{len(lake_ids)}] lake {lake_id}")
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)


def plot_hist(vals: np.ndarray, out_png: str, label: str = MODEL_LABEL):
    if vals.size == 0:
        raise RuntimeError("没有可用误差数据，无法绘图。")
    bins = np.linspace(0.0, X_MAX_FOCUS, 100)
    plt.figure(figsize=(10, 6), dpi=220)
    plt.hist(
        vals,
        bins=bins,
        histtype="step",
        linewidth=2.0,
        alpha=0.6,
        density=True,
        color="#90ee90",
        label=label,
    )
    plt.xlabel("Boundary distance error (pixels)", fontsize=13)
    plt.ylabel("Probability density", fontsize=13)
    plt.xlim(0.0, X_MAX_FOCUS)
    plt.title("Distribution of boundary distance errors (Test)", fontsize=14)
    plt.legend(frameon=False, fontsize=12)
    plt.grid(alpha=0.2, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_cdf(vals: np.ndarray, out_png: str, label: str = MODEL_LABEL):
    if vals.size == 0:
        raise RuntimeError("没有可用误差数据，无法绘图。")
    tick_labelsize = 13
    axis_labelsize = 15
    legend_fontsize = 14
    plt.figure(figsize=(10, 6), dpi=220)
    ax = plt.gca()
    sorted_vals = np.sort(vals)
    cdf_y = np.arange(1, sorted_vals.size + 1, dtype=np.float64) / sorted_vals.size
    ax.plot(sorted_vals, cdf_y, linewidth=2.0, color="#90ee90", label=label)
    ax.set_xlabel("Boundary distance error (pixels)", fontsize=axis_labelsize)
    ax.set_ylabel("Cumulative proportion", fontsize=axis_labelsize)
    ax.set_xlim(0.0, X_MAX_FOCUS)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.legend(frameon=False, fontsize=legend_fontsize)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(PRED_TIF_DIR):
        raise FileNotFoundError(
            f"预测 TIF 目录不存在: {PRED_TIF_DIR}\n请先运行 predict_stage2_full.py"
        )
    lake_ids = collect_lake_ids_from_mask_dir(DATASET_MASK_DIR)
    print(f"Test 湖泊数: {len(lake_ids)}")
    vals = compute_errors_for_pred_dir(PRED_TIF_DIR, lake_ids)
    if vals.size == 0:
        raise RuntimeError("未收集到 BDE 数据，请检查预测 TIF 与 GT mask 是否对齐。")
    print(
        f"{MODEL_LABEL}: n={vals.size}, mean={vals.mean():.4f}, "
        f"median={np.median(vals):.4f}, p95={np.percentile(vals, 95):.4f}"
    )
    np.savez_compressed(OUT_NPY, **{MODEL_LABEL: vals})
    plot_hist(vals, OUT_FIG_HIST)
    plot_cdf(vals, OUT_FIG_CDF)
    print(f"直方图: {OUT_FIG_HIST}")
    print(f"CDF: {OUT_FIG_CDF}")
    print(f"数据: {OUT_NPY}")


if __name__ == "__main__":
    main()
