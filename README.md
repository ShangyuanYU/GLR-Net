================================================================================
SHARE/CODE — 湖泊水体分割模型（Global + Patch，Stage1/Stage2）
================================================================================

目录结构
--------
  train.py                      Stage1/Stage2 训练
  predict_stage2_full.py        Stage2 全数据集预测（JPG + GeoTIFF）
  evaluate_stage2_test_metrics.py   Test 集汇总指标（hard IoU / Dice 等）
  evaluate_test_lake_table.py   Test 集逐湖指标表（IoU、BF1@3px、BDE）
  tiff_to_vector.py             预测 GeoTIFF 转 Shapefile
  count_stage2_params_flops.py  Stage2 参数量与 FLOPs
  paths_config.py               数据路径（发布前请修改）
  preprocess/
    build_dataset_from_tiff.py  TIFF→NPY、随机划分 Train/Val/Test、计算 mean/std
  ModelStructure_CUDA.py        模型结构（依赖 mmsegmentation）
  MyDataLoader_numpy0127.py     数据加载与 patch 采样
  ModelLoss.py / elvate.py      损失与 soft IoU 指标
  utils/                        训练辅助与增强
  checkpoints/                  stage1.ckpt, stage2.ckpt
  metrics/
    recalc_test_metrics_unified.py   论文口径 Test 指标（patch logits 拼湖）
    recalc_global_hard_dice_test.py  global hard Dice
    boundary_error_histogram.py      BDE 计算与可选 CDF 图
    VERSION22_Stage1Only/metrics.py  Boundary F1 / BDE 实现
    test_metrics_unified_METHOD.txt  指标定义说明

环境依赖
--------
  Python 3.8+
  PyTorch, torchvision, numpy, rasterio, Pillow, scipy
  mmsegmentation（pip 安装，或将源码置于 CODE 上级目录 mmsegmentation-main/）
  评估矢量脚本另需：fiona, shapely

路径配置
--------
  编辑 paths_config.py 中的 DATASET_ROOT、ORIG_TIFF_ROOT。
  从 SHARE TIFF 构建数据集后，将 DATASET_ROOT 设为 ../DataSet6（即 SHARE_DATASET_ROOT）。

数据预处理（TIFF → NPY）
-------------------------
  输入:
    ../Lake_sentinel2_GEE/{lake_id}.tif   4 波段 Sentinel-2（DN）
    ../Lake_Mask/{lake_id}.tif            水体 mask（0/1）

  运行:
    python preprocess/build_dataset_from_tiff.py

  输出 ../DataSet6/:
    Train|Val|Test/image|mask/{lake_id}.npy
    Train/train_band_mean_std.txt         训练归一化 mean/std
    dataset_split_manifest.json           划分清单（默认 seed=42, 7:1.5:1.5）

  可选: --output-dir --seed --train-ratio --val-ratio --skip-existing

典型流程
--------
  0) 构建 NPY 数据集（首次使用 SHARE 数据）:
       python preprocess/build_dataset_from_tiff.py

  1) 训练（可选，已附带权重）：
       python train.py

  2) Test 集预测：
       python predict_stage2_full.py
     输出：predict_stage2_output/{train,val,test}/tif/{lake_id}.tif

  3) Test 汇总评估：
       python evaluate_stage2_test_metrics.py
     输出：logs/stage2_test_eval_metrics.txt

  4) 论文口径统一指标（patch logits 拼湖 + Test/mask GT）：
       python metrics/recalc_test_metrics_unified.py
     输出：metrics/output/test_metrics_unified_v17.csv

  5) 逐湖指标表：
       python evaluate_test_lake_table.py
     输出：logs/stage2_test_lake_table.csv

指标口径（详见 metrics/test_metrics_unified_METHOD.txt）
--------
  - hard IoU：全 Test 像素 micro 池化
  - Lake-level IoU：逐湖 hard IoU 算术平均
  - Boundary F1@3px：逐湖计算后平均（3×3 腐蚀边界，tolerance=3px）
  - BDE：GT/预测边界双向距离变换，全湖边界像元拼接后统计 mean/median/P90

关联数据（SHARE 包内）
--------
  Lake_sentinel2_GEE/   179 湖 Sentinel-2 影像 GeoTIFF
  Lake_Mask/            179 湖 mask GeoTIFF
  DataSet6/             运行预处理后生成的 NPY 训练集（可选）

引用
--------
如果本项目对您的研究有帮助，请引用本仓库。关联论文已经接收，正式论文引用和 DOI
将在出版后补充。当前引用元数据见 CITATION.cff。

许可证
--------
本仓库中的源代码采用 MIT License，详见 LICENSE。

除非另有明确说明，MIT License 不自动适用于相关数据集、湖泊标注、预训练模型权重
或第三方依赖。这些材料应遵循各自的许可和使用条款。
