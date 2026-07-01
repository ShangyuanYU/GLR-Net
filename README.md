# GLR-Net

> [!IMPORTANT]
> **仓库状态：** 完整代码、复现说明和数据集已经公开。本研究未使用预训练模型参数。
>
> **Repository status:** The complete code, reproduction instructions, and dataset
> are now available. This study did not use pretrained model parameters.

湖泊水体分割模型，采用 Global + Patch 两阶段训练与预测流程。

论文：[A Global–Local Residual Refinement Framework for Accurate Lake Boundary Delineation in Remote Sensing Imagery](https://doi.org/10.3390/rs18121919)

发表于 *Remote Sensing*, 2026, 18(12), 1919。

本模型的详细说明和设计请参考论文。如有疑问，可在 GitHub Issues 中留言或通过邮件咨询；
欢迎围绕湖泊边界提取、遥感语义分割和相关方向开展学术合作。

For detailed model descriptions and design choices, please refer to the paper.
Questions are welcome via GitHub Issues or email, and academic collaboration is encouraged.

## 仓库结构

以下是 GitHub 仓库中实际包含的主要文件：

```text
GLR-Net/
├── train.py                         # Stage1/Stage2 训练
├── predict_stage2_full.py           # Stage2 整湖预测（JPG + GeoTIFF）
├── evaluate_stage2_test_metrics.py  # Test 集汇总指标
├── evaluate_test_lake_table.py      # Test 集逐湖指标表
├── count_stage2_params_flops.py     # Stage2 参数量与 FLOPs
├── tiff_to_vector.py                # 预测 GeoTIFF 转 Shapefile
├── paths_config.py                  # 数据路径配置
├── ModelStructure_CUDA.py           # 模型结构
├── MyDataLoader_numpy0127.py        # 数据加载与 patch 采样
├── ModelLoss.py                     # 损失函数
├── elvate.py                        # soft IoU 指标
├── preprocess/
│   └── build_dataset_from_tiff.py   # TIFF 转 NPY 并划分数据集
├── metrics/
│   ├── recalc_test_metrics_unified.py
│   ├── recalc_global_hard_dice_test.py
│   ├── boundary_error_histogram.py
│   └── VERSION22_Stage1Only/
│       └── metrics.py
├── utils/
├── requirements.txt
└── requirements-lock-windows.txt
```

## 数据集与训练检查点

数据集已通过 Zenodo 公开发布。本研究未使用预训练模型参数，模型从随机初始化开始训练。

- 训练过程中生成的 `checkpoints/stage1.ckpt` 和 `checkpoints/stage2.ckpt` 未上传至 GitHub。
- `Lake_sentinel2_GEE/`、`Lake_Mask/` 和生成的 `DataSet6/` 不属于本 GitHub 仓库。

公开数据集：

- **Title:** Tibetan Plateau Lake Boundary Delineation Dataset
- **Version:** v1.0
- **DOI:** [10.5281/zenodo.20662094](https://doi.org/10.5281/zenodo.20662094)
- **Zenodo record:** <https://zenodo.org/records/20662094>
- **License:** CC BY 4.0
- **Files:**
  - `Lake_Mask.zip`，MD5: `640cd7df4d54b1ef98827e04f57f8e75`
  - `Lake_sentinel2_GEE.zip`，MD5: `b07ce069c4b537985e60997b9b4cd607`

GitHub 仓库中不包含 `SHARE/` 目录。下载 Zenodo 数据后，推荐在本地整理为如下目录布局：

```text
SHARE/
├── GLR-Net/               # 本 GitHub 仓库
├── Lake_sentinel2_GEE/    # Sentinel-2 GeoTIFF
├── Lake_Mask/             # 水体 mask GeoTIFF
└── DataSet6/              # 预处理脚本生成的 NPY 数据集
```

`SHARE/` 只是本地工作目录示例，不是 GitHub 仓库的一部分。如使用其他目录布局，
请修改 `paths_config.py`。

## 环境依赖

论文实验使用并验证的环境：

- Windows
- Python 3.12.7
- CUDA 12.1
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- mmcv 2.1.0
- mmengine 0.10.7
- mmsegmentation 1.2.2

建议先创建虚拟环境，再单独安装 CUDA 12.1 对应的 PyTorch：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` 记录项目直接依赖及验证版本。完整实验环境快照见
`requirements-lock-windows.txt`；该文件包含 Jupyter 和间接依赖，主要用于精确复现，
不建议作为常规安装入口。

## 数据预处理

输入文件：

```text
Lake_sentinel2_GEE/{lake_id}.tif  # 4 波段 Sentinel-2 DN
Lake_Mask/{lake_id}.tif           # 水体 mask（0/1）
```

运行：

```bash
python preprocess/build_dataset_from_tiff.py
```

默认输出至 `DataSet6/`：

```text
DataSet6/
├── Train/
├── Val/
├── Test/
└── dataset_split_manifest.json
```

其中 `Train/train_band_mean_std.txt` 保存训练集归一化统计量。默认随机种子为 `42`，
Train/Val/Test 比例为 `7:1.5:1.5`。

## 使用流程

1. 构建 NPY 数据集：

   ```bash
   python preprocess/build_dataset_from_tiff.py
   ```

2. 训练 Stage1 和 Stage2：

   ```bash
   python train.py
   ```

3. 执行 Test 集预测：

   ```bash
   python predict_stage2_full.py
   ```

4. 计算 Test 集汇总指标：

   ```bash
   python evaluate_stage2_test_metrics.py
   ```

5. 计算论文口径统一指标：

   ```bash
   python metrics/recalc_test_metrics_unified.py
   ```

6. 生成逐湖指标表：

   ```bash
   python evaluate_test_lake_table.py
   ```

## 指标定义

- **Hard IoU**：合并全部 Test 集前景像素后计算 micro IoU。
- **Lake-level IoU**：逐湖 hard IoU 的算术平均。
- **Boundary F1@3px**：逐湖计算后平均，边界容差为 3 像素。
- **BDE**：双向边界距离误差，报告 mean、median 和 P90。

详细定义见 `metrics/test_metrics_unified_METHOD.txt`。

## 引用

如果本项目对您的研究有帮助，请引用关联论文。

MDPI 推荐引用格式：

> Yu, S.; Tu, J.; Guo, Z.; He, P. A Global–Local Residual Refinement Framework
> for Accurate Lake Boundary Delineation in Remote Sensing Imagery.
> *Remote Sens.* **2026**, *18*, 1919.
> https://doi.org/10.3390/rs18121919

BibTeX：

```bibtex
@article{yu2026global,
  author  = {Yu, Shangyuan and Tu, Jienan and Guo, Zhaocheng and He, Peng},
  title   = {A Global--Local Residual Refinement Framework for Accurate Lake Boundary Delineation in Remote Sensing Imagery},
  journal = {Remote Sensing},
  year    = {2026},
  volume  = {18},
  number  = {12},
  pages   = {1919},
  doi     = {10.3390/rs18121919},
  url     = {https://www.mdpi.com/2072-4292/18/12/1919}
}
```

论文链接：[MDPI](https://www.mdpi.com/2072-4292/18/12/1919) |
[DOI](https://doi.org/10.3390/rs18121919)

软件引用元数据见 `CITATION.cff`。GitHub 的 **Cite this repository** 功能可基于该文件
生成 APA、Chicago 等其他引用格式。

数据集引用：

> YU, Shangyuan. Tibetan Plateau Lake Boundary Delineation Dataset. Zenodo,
> 2026. https://doi.org/10.5281/zenodo.20662094

## 许可证

本仓库中的源代码采用 [MIT License](LICENSE)。

Zenodo 数据集采用 CC BY 4.0。除非另有明确说明，MIT License 不自动适用于数据集、
湖泊标注、训练检查点或第三方依赖。这些材料应遵循各自的许可和使用条款。
