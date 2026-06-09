# -*- coding: utf-8 -*-
"""
发布包路径配置。使用前请按本地环境修改下列变量。

默认假定 DataSet6 目录结构：
  {DATASET_ROOT}/{Train,Val,Test}/{image,mask}
  {DATASET_ROOT}/Train/train_band_mean_std.txt

全幅原始 GeoTIFF（预测输出带坐标 GeoTIFF 时使用）：
  {ORIG_TIFF_ROOT}/{lake_id}.tif
"""
from __future__ import annotations

import os

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
SHARE_ROOT = os.path.dirname(CODE_ROOT)

# SHARE 发布包原始 TIFF（preprocess/build_dataset_from_tiff.py 输入）
SHARE_IMAGE_TIFF_DIR = os.path.join(SHARE_ROOT, "Lake_sentinel2_GEE")
SHARE_MASK_TIFF_DIR = os.path.join(SHARE_ROOT, "Lake_Mask")
# 由 TIFF 构建的 NPY 数据集默认输出位置
SHARE_DATASET_ROOT = os.path.join(SHARE_ROOT, "DataSet6")

# 训练/评估使用的 NPY 数据集（运行 build_dataset_from_tiff.py 后即为 SHARE_DATASET_ROOT）
DATASET_ROOT = SHARE_DATASET_ROOT
ORIG_TIFF_ROOT = SHARE_IMAGE_TIFF_DIR

TRAIN_BAND_MEAN_STD_PATH = os.path.normpath(
    os.path.join(DATASET_ROOT, "Train", "train_band_mean_std.txt")
)
TEST_MASK_DIR = os.path.join(DATASET_ROOT, "Test", "mask")
