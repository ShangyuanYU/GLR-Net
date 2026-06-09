"""
湖泊水体分割 Stage1/Stage2 训练（Global + Patch 双分支，ResNet50-FPN）。

用法: python train.py
数据路径见 paths_config.py；权重输出 checkpoints/stage1.ckpt、stage2.ckpt。
"""
import warnings
import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from paths_config import DATASET_ROOT as DATASET_FOLDER, TRAIN_BAND_MEAN_STD_PATH

# 设置环境变量抑制警告（在导入其他模块之前）
# 这会影响所有子进程（包括 DataLoader workers）
os.environ['PYTHONWARNINGS'] = 'ignore'

# 抑制所有警告信息（在导入其他模块之前设置）
# 这样可以避免 DataLoader worker 进程重复输出警告
warnings.simplefilter('ignore')  # 全局忽略所有警告

# 配置 logging 来捕获警告（可选，用于调试）
logging.captureWarnings(True)

# =========================
# 可修改参数（集中放这里）
# =========================
TRAIN_SPLIT_NAME = "Train"
VAL_SPLIT_NAME = "Val"
TEST_SPLIT_NAME = "Test"
IMAGE_DIRNAME = "image"
MASK_DIRNAME = "mask"
# 若有完整湖泊 mask（{lake_id}.npy），可指定文件夹；为 None 则回退到 Train/Val/Test 的 mask 子目录
TRAIN_GLOBAL_MASK_FOLDER = None
VAL_GLOBAL_MASK_FOLDER = None
TEST_GLOBAL_MASK_FOLDER = None
MAX_PATCHES_PER_LAKE_STAGE1 = 10
MAX_PATCHES_PER_LAKE_STAGE2 = 24
STAGE2_PATCH_SAMPLING = "mixed"  # "mixed" 或 "random"
PATCH_SAMPLE_BOUNDARY_RATIO = 0.4
PATCH_SAMPLE_MISMATCH_RATIO = 0.3
PATCH_SAMPLE_RANDOM_RATIO = 0.3
# 可选：stage1 概率图目录（{lake_id}.npy, 值域[0,1]），用于 mixed 中“不一致 patch”采样
STAGE1_PROB_FOLDER = None
STAGE1_PROB_THRESHOLD = 0.5
DEFAULT_MEAN_STD_PATH = TRAIN_BAND_MEAN_STD_PATH

# DataLoader
BATCH_SIZE = 1  # lake 数量，每个 batch 包含 1 个 lake 的所有 patches
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TEST_SHUFFLE = False
PIN_MEMORY = True
TRAIN_DROP_LAST = True
VAL_DROP_LAST = False
TEST_DROP_LAST = False
NUM_WORKERS = "auto"  # "auto" 或整数（0 表示单进程）
NUM_WORKERS_MIN = 4
NUM_WORKERS_MAX = 8
NUM_WORKERS_DIV = 4
PERSISTENT_WORKERS = True

# 模型结构
MODEL_IN_CHANNELS = 4
MODEL_BACKBONE_DEPTH = 50
MODEL_BACKBONE_STRIDES = (1, 2, 2, 1)
MODEL_BACKBONE_DILATIONS = (1, 1, 1, 2)
MODEL_BACKBONE_OUT_INDICES = (0, 1, 2, 3)
MODEL_FPN_IN_CHANNELS = [256, 512, 1024, 2048]
MODEL_FPN_OUT_CHANNELS = 256
MODEL_FPN_NUM_OUTS = 4

# 模型信息输出
PRINT_MODEL_SUMMARY = True
SUMMARY_GLOBAL_SIZE = 512
SUMMARY_PATCHES = 2
SUMMARY_PATCH_SIZE = 512

# 训练阶段与增强
STAGE_MODE = "stage2"  # "stage1"=只训global, "stage2"=只训patch(冻结global)
STAGE1_AUGMENT = False
STAGE1_ICE_AUGMENT = False
STAGE1_JITTER = 0.1
STAGE1_EVAL_INTERVAL = 5
STAGE2_EVAL_INTERVAL = 10
TOPK_RATIO = 0.15
PATCH_BND_ALPHA = 4.0
PATCH_SEG_W = 1.0
PATCH_BND_W = 0.3
PATCH_BND_POS_WEIGHT = 10.0
CONSIST_W = 0.3
CONSIST_HIGH_CONF_TH = 0.8
RESIDUAL_REG_W = 0.1
CLEAR_CACHE_EVERY = 10
USE_AMP = True  # 混合精度以降低显存占用
STAGE1_CKPT_NAME = "stage1.ckpt"

# 优化器与调度器
WEIGHT_DECAY = 5e-4
BASE_LR_STAGE1 = 5e-5
BASE_LR_STAGE2 = 5e-5
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-5

# 训练轮数与早停
MAX_EPOCH = 500
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001

# 输出目录
LOG_DIRNAME = "logs"
CKPT_DIRNAME = "checkpoints"

from ModelStructure_CUDA import ModelStructure
import torch
import torch.nn.functional as F
import numpy as np
from MyDataLoader_numpy0127 import WaterDataset, custom_collate_fn
from torch.utils.data import DataLoader
from ModelLoss import (
    dice_loss_from_prob,
    topk_bce_with_logits,
    compute_edge_mask,
    weighted_bce_with_logits,
)
from elvate import (
    calculate_pixel_soft_iou,
    calculate_hard_iou,
    calculate_hard_dice,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing
import platform
from utils.augment import augment_stage1, HAS_TV
from utils.train_helpers import (
    diagnose_no_global_mask,
    extract_lake_ids_from_folder,
    get_device,
    clear_cache,
)
try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except Exception:
    torchinfo_summary = None
    HAS_TORCHINFO = False

def _freeze_bn_running_stats(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def _freeze_frozen_parts_bn(model, stage_mode):
    if stage_mode == "stage1":
        model.apply(_freeze_bn_running_stats)
        return
    if stage_mode == "stage2":
        for name in ("backbone", "neck", "global_context", "decoder"):
            if hasattr(model, name):
                getattr(model, name).apply(_freeze_bn_running_stats)


def assemble_patches_to_lake_tensor(patch_tensor, patch_coords, orig_hw):
    """
    将 patch tensor 按 patch_coords 拼回整湖，重叠区域取平均。
    patch_tensor: [P, 1, ph, pw]
    返回: [1, 1, H_orig, W_orig]
    """
    if patch_tensor is None or patch_tensor.numel() == 0 or patch_coords is None or len(patch_coords) == 0:
        return None
    if patch_tensor.dim() != 4 or patch_tensor.shape[1] != 1:
        raise ValueError(f"patch_tensor 期望 [P,1,H,W]，实际为 {tuple(patch_tensor.shape)}")

    p, _, ph, pw = patch_tensor.shape
    if p != len(patch_coords):
        raise ValueError(f"patch 数与 patch_coords 数不一致: P={p}, coords={len(patch_coords)}")

    h_orig, w_orig = int(orig_hw[0]), int(orig_hw[1])
    accum = torch.zeros((1, h_orig, w_orig), dtype=patch_tensor.dtype, device=patch_tensor.device)
    count = torch.zeros((1, h_orig, w_orig), dtype=patch_tensor.dtype, device=patch_tensor.device)

    for i, (x, y) in enumerate(patch_coords):
        x = int(x)
        y = int(y)
        h_end = min(y + ph, h_orig)
        w_end = min(x + pw, w_orig)
        crop_h = h_end - y
        crop_w = w_end - x
        if crop_h <= 0 or crop_w <= 0:
            continue
        accum[:, y:h_end, x:w_end] += patch_tensor[i, :, :crop_h, :crop_w]
        count[:, y:h_end, x:w_end] += 1.0

    full = accum / count.clamp_min(1e-6)
    return full.unsqueeze(0)


def evaluate_split(model, data_loader, split_name, stage_mode, device, device_name, use_amp):
    model.eval()
    lake_iou_list = []

    if stage_mode == "stage1":
        # 按LakeID存储global_pred和global_mask（lake-level评估）
        lake_global_logits = {}  # {LakeID: tensor}，存储global_pred的logits
        lake_global_masks = {}   # {LakeID: tensor}，存储global_mask

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                try:
                    lake_id = int(data['lake_ids'][0].item())
                    global_image = data['global_images'][0:1].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(device_name == 'CUDA' and use_amp)):
                        global_pred, _, _ = model(
                            global_image, patches=None, patch_coords=None, orig_hw=None
                        )
                    if 'global_masks' not in data or data['global_masks'] is None:
                        del global_pred, global_image
                        continue
                    global_mask = data['global_masks'][0:1].to(device, non_blocking=True)
                    if global_mask.dim() == 3:
                        global_mask = global_mask.unsqueeze(1)
                    lake_global_logits[lake_id] = global_pred[0].cpu()
                    lake_global_masks[lake_id] = global_mask[0].cpu()
                    del global_pred, global_image, global_mask

                except Exception as e:
                    print(f"{split_name} lake {i} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 使用 lake-level 评估：soft IoU
        print(f"\n  开始计算{split_name} soft IoU（lake-level 评估）...")
        print(f"  共找到 {len(lake_global_logits)} 个不同的LakeID")

        for lake_id in sorted(lake_global_logits.keys()):
            global_pred_logits = lake_global_logits[lake_id]  # [1, H, W]（与 mask 对齐后计算）
            global_mask_tensor = lake_global_masks[lake_id]  # [1, H, W] full-res

            if not isinstance(global_pred_logits, torch.Tensor):
                global_pred_logits = torch.tensor(global_pred_logits, dtype=torch.float32)
            if not isinstance(global_mask_tensor, torch.Tensor):
                global_mask_tensor = torch.tensor(global_mask_tensor, dtype=torch.float32)

            global_pred_logits = global_pred_logits.unsqueeze(0)  # [1, 1, H, W]
            global_mask_tensor = global_mask_tensor.unsqueeze(0)  # [1, 1, H, W]
            if global_pred_logits.shape[-2:] != global_mask_tensor.shape[-2:]:
                global_pred_logits = F.interpolate(
                    global_pred_logits,
                    size=global_mask_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

            lake_soft_iou = calculate_pixel_soft_iou(global_pred_logits, global_mask_tensor).item()

            lake_iou_list.append(lake_soft_iou)
            print(f"  LakeID {lake_id}: soft IoU = {lake_soft_iou:.4f}")
    else:
        # stage2: lake-level soft IoU（先拼接 patch 到整湖）
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                try:
                    lake_id = int(data['lake_ids'][0].item())
                    global_image = data['global_images'][0:1].to(device)
                    patches = data['patches_list'][0].to(device)
                    patch_masks = data['patch_masks_list'][0].to(device)
                    patch_coords = data['patch_coords_list'][0]
                    patch_meta = None
                    if 'patch_meta_list' in data and data['patch_meta_list'] is not None:
                        patch_meta = data['patch_meta_list'][0].to(device)
                    orig_hw = data['orig_hw_list'][0]

                    _, patch_logits, patch_bnd_logits = model(
                        global_image,
                        patches=patches,
                        patch_coords=patch_coords,
                        orig_hw=orig_hw,
                        patch_meta=patch_meta,
                        global_no_grad=True,
                    )

                    if patch_logits.shape[-2:] != patch_masks.shape[-2:]:
                        patch_logits = F.interpolate(
                            patch_logits,
                            size=patch_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )

                    patch_logits_cpu = patch_logits.detach().cpu()
                    patch_masks_cpu = patch_masks.detach().cpu()
                    lake_logits = assemble_patches_to_lake_tensor(
                        patch_logits_cpu, patch_coords, orig_hw
                    )  # [1,1,H_orig,W_orig]
                    lake_mask = assemble_patches_to_lake_tensor(
                        patch_masks_cpu, patch_coords, orig_hw
                    )  # [1,1,H_orig,W_orig]

                    if lake_logits is None or lake_mask is None:
                        lake_patch_soft_iou = 0.0
                    else:
                        lake_mask = (lake_mask > 0.5).float()
                        lake_patch_soft_iou = calculate_pixel_soft_iou(lake_logits, lake_mask).item()

                    lake_iou_list.append(lake_patch_soft_iou)
                    print(f"  LakeID {lake_id}: lake-level patch soft IoU = {lake_patch_soft_iou:.4f}")

                    del patch_logits, patch_bnd_logits, global_image, patches, patch_masks, patch_meta
                    del patch_logits_cpu, patch_masks_cpu, lake_logits, lake_mask
                except Exception as e:
                    print(f"{split_name} lake {i} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    avg_iou = np.mean(lake_iou_list) if len(lake_iou_list) > 0 else 0.0
    if stage_mode == "stage1":
        print(f"  平均 soft IoU: {avg_iou:.4f} (共 {len(lake_iou_list)} 个LakeID)")
    else:
        print(f"  平均 lake-level patch soft IoU: {avg_iou:.4f} (共 {len(lake_iou_list)} 个LakeID)")

    return avg_iou




# Windows 多进程兼容性设置（必须在 if __name__ 之前）
if platform.system() == 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误

# 所有执行代码必须放在 if __name__ == '__main__': 保护下
if __name__ == '__main__':
    print("所有模块导入成功！")
    Folder = DATASET_FOLDER
    try:
        import MyDataLoader_numpy0127
        imported_path = os.path.abspath(MyDataLoader_numpy0127.__file__)
        expected_path = os.path.join(current_dir, 'MyDataLoader_numpy0127.py')
        if os.path.abspath(imported_path) == os.path.abspath(expected_path):
            print(f"✓ MyDataLoader_numpy0127: {imported_path}")
        else:
            print(f"⚠ MyDataLoader_numpy0127 非本地路径: {imported_path}")
            print(f"  期望: {expected_path}")
    except Exception as e:
        print(f"无法验证导入路径: {e}")
    # 数据路径配置（规范化路径，确保路径分隔符一致）
    trainImageFoler = os.path.normpath(os.path.join(Folder, TRAIN_SPLIT_NAME, IMAGE_DIRNAME))
    trainMaskFoler = os.path.normpath(os.path.join(Folder, TRAIN_SPLIT_NAME, MASK_DIRNAME))
    valImageFoler = os.path.normpath(os.path.join(Folder, VAL_SPLIT_NAME, IMAGE_DIRNAME))
    valMaskFoler = os.path.normpath(os.path.join(Folder, VAL_SPLIT_NAME, MASK_DIRNAME))
    testImageFoler = os.path.normpath(os.path.join(Folder, TEST_SPLIT_NAME, IMAGE_DIRNAME))
    testMaskFoler = os.path.normpath(os.path.join(Folder, TEST_SPLIT_NAME, MASK_DIRNAME))

    # 检查 mean/std 文件是否存在
    if not os.path.isfile(DEFAULT_MEAN_STD_PATH):
        raise FileNotFoundError(f"DEFAULT_MEAN_STD_PATH not found: {DEFAULT_MEAN_STD_PATH}")
    
    # 从训练/验证/测试集文件夹中提取 LakeID
    trainID = extract_lake_ids_from_folder(trainImageFoler)
    valID = extract_lake_ids_from_folder(valImageFoler)
    testID = extract_lake_ids_from_folder(testImageFoler)

    # 检查是否有重叠，并从训练集中移除验证/测试集的 LakeID
    overlap_train_val = trainID & valID
    overlap_train_test = trainID & testID
    if overlap_train_val:
        print(f"警告: 训练集和验证集有重叠的 LakeID: {sorted(overlap_train_val)}")
        print("  从训练集中移除验证集的 LakeID，确保训练集和验证集不重叠")
        trainID = trainID - valID
    if overlap_train_test:
        print(f"警告: 训练集和测试集有重叠的 LakeID: {sorted(overlap_train_test)}")
        print("  从训练集中移除测试集的 LakeID，确保训练集和测试集不重叠")
        trainID = trainID - testID

    overlap_val_test = valID & testID
    if overlap_val_test:
        print(f"警告: 验证集和测试集有重叠的 LakeID: {sorted(overlap_val_test)}")

    print(f"从训练集文件夹提取到 {len(trainID)} 个 LakeID: {sorted(trainID)}")
    print(f"从验证集文件夹提取到 {len(valID)} 个 LakeID: {sorted(valID)}")
    print(f"从测试集文件夹提取到 {len(testID)} 个 LakeID: {sorted(testID)}")

    # 完整湖泊 mask：优先从 MASK_numpy 读取 {lake_id}.npy；
    train_global_mask_folder = os.path.normpath(TRAIN_GLOBAL_MASK_FOLDER) if TRAIN_GLOBAL_MASK_FOLDER else trainMaskFoler
    val_global_mask_folder = os.path.normpath(VAL_GLOBAL_MASK_FOLDER) if VAL_GLOBAL_MASK_FOLDER else valMaskFoler
    test_global_mask_folder = os.path.normpath(TEST_GLOBAL_MASK_FOLDER) if TEST_GLOBAL_MASK_FOLDER else testMaskFoler
    train_max_patches = MAX_PATCHES_PER_LAKE_STAGE2 if STAGE_MODE == "stage2" else MAX_PATCHES_PER_LAKE_STAGE1
    train_patch_sampling = STAGE2_PATCH_SAMPLING if STAGE_MODE == "stage2" else "random"
    stage1_prob_folder = os.path.normpath(STAGE1_PROB_FOLDER) if (STAGE_MODE == "stage2" and STAGE1_PROB_FOLDER) else None

    # 创建数据集（每个 batch 单位是一个 lake；global_image/global_mask 始终完整湖泊；
    # stage2 可用 mixed 采样：边界/不一致/随机）
    try:
        train_dataset = WaterDataset(
            trainImageFoler,
            trainMaskFoler,
            is_train=True,
            trainID=trainID,
            global_image_folder=trainImageFoler,
            global_mask_folder=train_global_mask_folder,
            max_patches_per_lake=train_max_patches,
            sampling_strategy=train_patch_sampling,
            boundary_sample_ratio=PATCH_SAMPLE_BOUNDARY_RATIO,
            mismatch_sample_ratio=PATCH_SAMPLE_MISMATCH_RATIO,
            random_sample_ratio=PATCH_SAMPLE_RANDOM_RATIO,
            stage1_prob_folder=stage1_prob_folder,
            stage1_prob_threshold=STAGE1_PROB_THRESHOLD,
            mean_std_path=DEFAULT_MEAN_STD_PATH,
        )
        val_dataset = WaterDataset(
            valImageFoler,
            valMaskFoler,
            is_train=False,
            global_image_folder=valImageFoler,
            global_mask_folder=val_global_mask_folder,
            mean_std_path=DEFAULT_MEAN_STD_PATH,
        )
        test_dataset = WaterDataset(
            testImageFoler,
            testMaskFoler,
            is_train=False,
            global_image_folder=testImageFoler,
            global_mask_folder=test_global_mask_folder,
            mean_std_path=DEFAULT_MEAN_STD_PATH,
        )
        print(f"训练数据集创建成功，共 {len(train_dataset)} 个样本")
        print(f"验证数据集创建成功，共 {len(val_dataset)} 个样本")
        print(f"测试数据集创建成功，共 {len(test_dataset)} 个样本")
        print(
            f"训练 patch 采样: strategy={train_patch_sampling}, max_patches={train_max_patches}, "
            f"ratios(boundary/mismatch/random)="
            f"{PATCH_SAMPLE_BOUNDARY_RATIO:.2f}/{PATCH_SAMPLE_MISMATCH_RATIO:.2f}/{PATCH_SAMPLE_RANDOM_RATIO:.2f}"
        )
        if STAGE_MODE == "stage2" and train_patch_sampling == "mixed" and stage1_prob_folder is None:
            print("提示: 未设置 STAGE1_PROB_FOLDER，mixed 采样将自动退化为 边界+随机。")
    except Exception as e:
        print(f"数据集创建失败: {e}")
        raise

    # 注意：custom_collate_fn 已从 MyDataLoader_numpy 模块导入
    # 这样可以确保在 Windows 多进程环境中正确序列化

    # 自动检测最优的 num_workers
    # 通常设置为 CPU 核心数的一半到全部，但4-8个通常就足够了
    # 注意：在 Windows 上，如果遇到多进程问题，可以尝试减少 num_workers 或设为 0
    cpu_count = multiprocessing.cpu_count()
    if NUM_WORKERS == "auto":
        optimal_workers = min(NUM_WORKERS_MAX, max(NUM_WORKERS_MIN, cpu_count // NUM_WORKERS_DIV))
    else:
        optimal_workers = int(NUM_WORKERS)

    # 如果遇到多进程错误，可以尝试以下值：
    # optimal_workers = 4  # 或 2, 1, 0
    # 0 表示不使用多进程（单进程，但更稳定）

    print(f"CPU 核心数: {cpu_count}")
    if NUM_WORKERS == "auto":
        print(f"推荐的 num_workers: {optimal_workers}")
        print("提示：如果遇到多进程错误，可以尝试减少 num_workers 或设为 0")
    else:
        print(f"使用固定 num_workers: {optimal_workers}")

    # 创建训练、验证、测试 DataLoader
    # 注意：batch_size 表示 lake 数量（不是 patch 数量）
    # 每个 batch 包含一个 lake 的完整数据（包括该 lake 的所有 patches）
    # 通常设为 1，即每个 batch 处理一个 lake
    trainDataLoader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # lake 数量，每个 batch 包含 1 个 lake 的所有 patches
        shuffle=TRAIN_SHUFFLE,  # 打乱 lake 的顺序（不影响每个 lake 的 patch 完整性）
        num_workers=optimal_workers,  # 使用多进程加速数据加载
        pin_memory=PIN_MEMORY,  # CUDA可以使用pin_memory加速
        drop_last=TRAIN_DROP_LAST,
        collate_fn=custom_collate_fn,  # 使用自定义 collate 函数
        persistent_workers=PERSISTENT_WORKERS if optimal_workers > 0 else False  # num_workers > 0 时启用持久化worker
    )

    valDataLoader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,  # lake 数量，每个 batch 包含 1 个 lake 的所有 patches
        shuffle=VAL_SHUFFLE,  # 验证集不需要shuffle
        num_workers=optimal_workers,  # 使用多进程加速数据加载
        pin_memory=PIN_MEMORY,
        drop_last=VAL_DROP_LAST,  # 验证集保留所有样本
        collate_fn=custom_collate_fn,  # 使用自定义 collate 函数
        persistent_workers=PERSISTENT_WORKERS if optimal_workers > 0 else False  # num_workers > 0 时启用持久化worker
    )

    testDataLoader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,  # lake 数量，每个 batch 包含 1 个 lake 的所有 patches
        shuffle=TEST_SHUFFLE,  # 测试集不需要shuffle
        num_workers=optimal_workers,  # 使用多进程加速数据加载
        pin_memory=PIN_MEMORY,
        drop_last=TEST_DROP_LAST,  # 测试集保留所有样本
        collate_fn=custom_collate_fn,  # 使用自定义 collate 函数
        persistent_workers=PERSISTENT_WORKERS if optimal_workers > 0 else False  # num_workers > 0 时启用持久化worker
    )

    print(f"训练 DataLoader 创建成功，batch_size={trainDataLoader.batch_size} (lake数量), 批次数={len(trainDataLoader)} (lake数量)")
    print(f"验证 DataLoader 创建成功，batch_size={valDataLoader.batch_size} (lake数量), 批次数={len(valDataLoader)} (lake数量)")
    print(f"测试 DataLoader 创建成功，batch_size={testDataLoader.batch_size} (lake数量), 批次数={len(testDataLoader)} (lake数量)")

    # 设置设备（支持 CUDA 和 CPU）
    device, device_name = get_device()
    print(f'使用设备: {device} ({device_name})')

    # 如果使用 GPU，显示设备信息
    if device_name == 'CUDA':
        print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
        print(f'GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # 初始化模型
    # layer4: stride=1, dilation=2, padding=2 → 1/16 分辨率，感受野扩大；其余 stage 默认
    backbone_cfg = dict(
        type='ResNet',
        depth=MODEL_BACKBONE_DEPTH,
        in_channels=MODEL_IN_CHANNELS,
        num_stages=4,
        out_indices=MODEL_BACKBONE_OUT_INDICES,
        strides=MODEL_BACKBONE_STRIDES,
        dilations=MODEL_BACKBONE_DILATIONS,
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
    )

    # ResNet50 的输出通道数: [256, 512, 1024, 2048]
    neck_cfg = dict(
        type='FPN',
        in_channels=MODEL_FPN_IN_CHANNELS,
        out_channels=MODEL_FPN_OUT_CHANNELS,
        num_outs=MODEL_FPN_NUM_OUTS
    )

    model = ModelStructure(backbone_cfg, neck_cfg)
    model = model.to(device)
    print("模型已创建并移动到设备:", device)
    if PRINT_MODEL_SUMMARY:
        if HAS_TORCHINFO:
            try:
                dummy_global = torch.zeros(1, MODEL_IN_CHANNELS, SUMMARY_GLOBAL_SIZE, SUMMARY_GLOBAL_SIZE, device=device)
                dummy_patches = torch.zeros(SUMMARY_PATCHES, MODEL_IN_CHANNELS, SUMMARY_PATCH_SIZE, SUMMARY_PATCH_SIZE, device=device)
                dummy_coords = [(0, 0)] * SUMMARY_PATCHES
                dummy_orig_hw = (SUMMARY_GLOBAL_SIZE, SUMMARY_GLOBAL_SIZE)
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    print(torchinfo_summary(
                        model,
                        input_data=(dummy_global, dummy_patches, dummy_coords, dummy_orig_hw),
                        depth=4,
                        verbose=1,
                    ))
                model.train(was_training)
            except Exception as e:
                print(f"torchinfo.summary 失败: {e}")
        else:
            print("torchinfo 未安装，跳过 summary")

    if STAGE_MODE == "stage2":
        stage1_ckpt_path = os.path.join(current_dir, CKPT_DIRNAME, STAGE1_CKPT_NAME)
        print(f"Stage2 训练将加载 Stage1 参数: {stage1_ckpt_path}")
        if not os.path.isfile(stage1_ckpt_path):
            print("未找到 Stage1 权重文件，退出训练。")
            sys.exit(1)
        try:
            ckpt = torch.load(stage1_ckpt_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"加载 Stage1 权重失败: {e}")
            sys.exit(1)
    if STAGE_MODE == "stage2":
        for p in model.parameters():
            p.requires_grad = False
        stage2_trainable_prefixes = (
            "patch_backbone.",
            "patch_neck.",
            "patch_decoder.",
            "patch_meta_mlp.",
            "patch_global_proj.",
            "patch_fuse.",
            # patch_global_film 属于 stage2 新增融合模块，也需要训练
            "patch_global_film.",
        )
        for name, p in model.named_parameters():
            if name.startswith(stage2_trainable_prefixes):
                p.requires_grad = True
    elif STAGE_MODE == "stage1":
        for p in model.parameters():
            p.requires_grad = True
        for name, p in model.named_parameters():
            if name.startswith((
                "patch_backbone.",
                "patch_neck.",
                "patch_decoder.",
                "patch_meta_mlp.",
                "patch_global_proj.",
                "patch_fuse.",
                "patch_global_film.",
            )):
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable param groups:")
    for n in trainable[:50]:
        print("  ", n)
    print("Total trainable:", len(trainable))
    wd = WEIGHT_DECAY
    base_lr = BASE_LR_STAGE2 if STAGE_MODE == "stage2" else BASE_LR_STAGE1
    optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_name == 'CUDA' and USE_AMP))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR
    )

    max_epoch = MAX_EPOCH
    patience = EARLY_STOP_PATIENCE
    min_delta = EARLY_STOP_MIN_DELTA
    ckpt_dir = os.path.join(current_dir, CKPT_DIRNAME)
    os.makedirs(ckpt_dir, exist_ok=True)
    stage_prefix = "stage2" if STAGE_MODE == "stage2" else "stage1"
    best_model_path = os.path.join(ckpt_dir, f"{stage_prefix}.ckpt")

    best_val_iou = 0.0
    patience_counter = 0
    best_model_state = None
    last_val_iou = 0.0

    print("损失函数和优化器已设置")
    print(f"Early Stopping: patience={patience}, min_delta={min_delta}, max_epoch={max_epoch}")
    print(f"Backbone+FPN+轻量Decoder+SegHead；loss = Dice + 0.3*Top-k BCE (topk={TOPK_RATIO:.2f})")
    if STAGE_MODE == "stage2":
        print(
            f"Patch分支：seg边界加权 α={PATCH_BND_ALPHA}, "
            f"loss权重 seg={PATCH_SEG_W}, bnd={PATCH_BND_W}, consistency={CONSIST_W}, residual={RESIDUAL_REG_W}, "
            f"pos_weight={PATCH_BND_POS_WEIGHT}, fg阈值={CONSIST_HIGH_CONF_TH}"
        )
    else:
        print("Patch分支：stage1 不训练（冻结）")
    print("loss 一律在 full-res 上计算（logits 上采样到 mask 尺寸）")
    if HAS_TV and STAGE_MODE == "stage1" and STAGE1_AUGMENT:
        ice_str = "；结冰模拟（ice_prob=0.25）" if STAGE1_ICE_AUGMENT else "；结冰模拟：关"
        print(f"轻量增强：ColorJitter(0.1) + 随机水平翻转{ice_str}")
    else:
        print("轻量增强：未启用（需 torchvision 且 STAGE1_AUGMENT=True）")
    print(f"scheduler 用验证集 mean soft IoU；lr={base_lr}, weight_decay={wd}")
    print(f"保存路径: {best_model_path}")
    log_dir = os.path.join(current_dir, LOG_DIRNAME)
    os.makedirs(log_dir, exist_ok=True)
    loss_txt = os.path.join(log_dir, f"{stage_prefix}_train_loss.txt")
    train_iou_txt = os.path.join(log_dir, f"{stage_prefix}_train_iou.txt")
    val_iou_txt = os.path.join(log_dir, f"{stage_prefix}_val_iou.txt")
    test_iou_txt = os.path.join(log_dir, f"{stage_prefix}_test_iou.txt")
    for p, header in [
        (loss_txt, "epoch\tloss\n"),
        (train_iou_txt, "epoch\ttrain_iou\n"),
        (val_iou_txt, "epoch\tval_iou\n"),
    ]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(header)
    print(f"日志将保存至: {loss_txt}, {train_iou_txt}, {val_iou_txt}, {test_iou_txt}")

    # 训练循环（带 Early Stopping）
    model.train()
    _freeze_frozen_parts_bn(model, STAGE_MODE)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(max_epoch):
        model.train()
        _freeze_frozen_parts_bn(model, STAGE_MODE)
        epoch_loss = 0.0
        num_lakes = 0
        lake_losses = []
        lake_ious = []
        epoch_soft_iou_list = []
        epoch_bce_list = []
        epoch_dice_list = []
        epoch_patch_bce_list = []
        epoch_patch_dice_list = []
        epoch_patch_bnd_list = []
        epoch_patch_consist_list = []
        epoch_patch_residual_list = []
        epoch_hard_iou_list = []
        epoch_hard_dice_list = []
        epoch_inter_sum = 0.0
        epoch_probs_sum = 0.0
        epoch_gt_sum = 0.0
        print(f"\nEpoch {epoch+1}/{max_epoch} (lake_id / loss / hard_iou / hard_dice):")

        for i, data in enumerate(trainDataLoader):
            try:
                optimizer.zero_grad(set_to_none=True)
                global_image = data['global_images'][0:1].to(device, non_blocking=True)
                has_global_mask = 'global_masks' in data and data['global_masks'] is not None
                if has_global_mask:
                    global_mask = data['global_masks'][0:1].to(device, non_blocking=True)
                    if global_mask.dim() == 3:
                        global_mask = global_mask.unsqueeze(1)
                lake_id = int(data['lake_ids'][0].item())

                if not has_global_mask:
                    print(f"  跳过 lake {lake_id}: 无 global_mask")
                    diagnose_no_global_mask(lake_id, train_global_mask_folder)
                    continue

                if STAGE_MODE == "stage1" and STAGE1_AUGMENT:
                    ice_p = 0.25 if STAGE1_ICE_AUGMENT else 0.0
                    j = STAGE1_JITTER
                    global_image, global_mask = augment_stage1(
                        global_image, global_mask,
                        brightness=j, contrast=j, saturation=j,
                        ice_prob=ice_p
                    )
                with torch.cuda.amp.autocast(enabled=(device_name == 'CUDA' and USE_AMP)):
                    if STAGE_MODE == "stage1":
                        global_logits, _, _ = model(
                            global_image, patches=None, patch_coords=None, orig_hw=None
                        )
                        logits_full = global_logits
                        if logits_full.shape[-2:] != global_mask.shape[-2:]:
                            logits_full = F.interpolate(
                                logits_full,
                                size=global_mask.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )
                        gt_full = global_mask.float()
                        prob_full = torch.sigmoid(logits_full)
                        topk_bce = topk_bce_with_logits(logits_full, gt_full, k=TOPK_RATIO)
                        dice_loss = dice_loss_from_prob(prob_full, gt_full)
                    else:
                        patches = data['patches_list'][0].to(device, non_blocking=True)  # [P, C, 512, 512]
                        patch_masks = data['patch_masks_list'][0].to(device, non_blocking=True)  # [P, 1, 512, 512]
                        patch_coords = data['patch_coords_list'][0]
                        patch_meta = None
                        if 'patch_meta_list' in data and data['patch_meta_list'] is not None:
                            patch_meta = data['patch_meta_list'][0].to(device, non_blocking=True)
                        orig_hw = data['orig_hw_list'][0]
                        global_logits, patch_logits, patch_bnd_logits, patch_delta_logits, patch_coarse_logits = model(
                            global_image,
                            patches=patches,
                            patch_coords=patch_coords,
                            orig_hw=orig_hw,
                            patch_meta=patch_meta,
                            return_patch_aux=True,
                            global_no_grad=True,
                        )
                # ----- stage1: global loss -----
                if STAGE_MODE == "stage1":
                    total_loss = dice_loss + 0.3 * topk_bce

                # ----- stage2: patch loss only -----
                else:
                    # 不让 global loss 参与 stage2（避免污染）
                    if patch_logits is None or patch_bnd_logits is None:
                        raise RuntimeError("stage2 期望 patch_logits/patch_bnd_logits，但拿到 None")

                    if patch_logits.shape[-2:] != patch_masks.shape[-2:]:
                        patch_logits = F.interpolate(
                            patch_logits,
                            size=patch_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    if patch_coarse_logits is not None and patch_coarse_logits.shape[-2:] != patch_masks.shape[-2:]:
                        patch_coarse_logits = F.interpolate(
                            patch_coarse_logits,
                            size=patch_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    if patch_delta_logits is not None and patch_delta_logits.shape[-2:] != patch_masks.shape[-2:]:
                        patch_delta_logits = F.interpolate(
                            patch_delta_logits,
                            size=patch_masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    with torch.no_grad():
                        per_patch = []
                        for p in range(patch_logits.shape[0]):
                            per_patch.append(calculate_hard_iou(
                                patch_logits[p:p+1], patch_masks[p:p+1]
                            ).item())
                        patch_hard_iou = float(np.mean(per_patch)) if len(per_patch) else 0.0

                    patch_prob = torch.sigmoid(patch_logits)
                    patch_dice = dice_loss_from_prob(patch_prob, patch_masks.float())

                    with torch.no_grad():
                        coarse_patch_logits = patch_coarse_logits
                        coarse_patch_prob = None
                        high_conf_fg = None
                        if coarse_patch_logits is not None:
                            coarse_patch_prob = torch.sigmoid(coarse_patch_logits)
                            high_conf_fg = (coarse_patch_prob > CONSIST_HIGH_CONF_TH).float()

                    if coarse_patch_prob is None:
                        consistency_loss = patch_logits.new_zeros(())
                    else:
                        consistency_map = F.relu(coarse_patch_prob - patch_prob)
                        consistency_loss = (consistency_map * high_conf_fg).sum() / (high_conf_fg.sum() + 1e-6)

                    if coarse_patch_logits is None:
                        residual_reg_loss = patch_logits.new_zeros(())
                    else:
                        delta_logits = patch_delta_logits if patch_delta_logits is not None else (patch_logits - coarse_patch_logits)
                        residual_reg_loss = delta_logits.abs().mean()

                    edge_mask = compute_edge_mask(patch_masks)
                    weight_map = 1.0 + PATCH_BND_ALPHA * edge_mask
                    patch_bce = weighted_bce_with_logits(patch_logits, patch_masks.float(), weight_map)
                    patch_seg_loss = patch_dice + 0.3 * patch_bce

                    pos_w = torch.tensor([PATCH_BND_POS_WEIGHT], device=patch_bnd_logits.device)
                    bnd_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
                    patch_bnd_loss = bnd_loss_fn(patch_bnd_logits, edge_mask)

                    total_loss = (
                        PATCH_SEG_W * patch_seg_loss
                        + PATCH_BND_W * patch_bnd_loss
                        + CONSIST_W * consistency_loss
                        + RESIDUAL_REG_W * residual_reg_loss
                    )
                if STAGE_MODE == "stage1":
                    with torch.no_grad():
                        all_zero = torch.all(global_mask == 0).item() and torch.all(logits_full == 0).item()
                        if not all_zero:
                            _iou = calculate_pixel_soft_iou(logits_full, global_mask).item()
                            epoch_inter_sum += (prob_full * global_mask).sum().item()
                            epoch_probs_sum += prob_full.sum().item()
                            epoch_gt_sum += global_mask.sum().item()
                            hard_iou = calculate_hard_iou(logits_full, global_mask).item()
                            hard_dice = calculate_hard_dice(logits_full, global_mask).item()
                        else:
                            _iou = 0.0
                            hard_iou = 0.0
                            hard_dice = 0.0
                    epoch_hard_iou_list.append(hard_iou)
                    epoch_hard_dice_list.append(hard_dice)
                if STAGE_MODE == "stage1":
                    epoch_bce_list.append(topk_bce.item())
                    epoch_dice_list.append(dice_loss.item())
                else:
                    epoch_patch_bce_list.append(patch_bce.item())
                    epoch_patch_dice_list.append(patch_dice.item())
                    epoch_patch_bnd_list.append(patch_bnd_loss.item())
                    epoch_patch_consist_list.append(consistency_loss.item())
                    epoch_patch_residual_list.append(residual_reg_loss.item())
                if STAGE_MODE == "stage1":
                    _soft_for_scheduler = _iou

                tl_val = total_loss.item()
                low_iou_tag = ""
                if STAGE_MODE == "stage2":
                    print(f"  lake {lake_id} | loss={tl_val:.4f} patch_hard_iou={patch_hard_iou:.4f}")
                else:
                    print(f"  lake {lake_id} | loss={tl_val:.4f} hard_iou={hard_iou:.4f} hard_dice={hard_dice:.4f}{low_iou_tag}")

                if device_name == 'CUDA' and USE_AMP:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    total_loss.backward()
                with torch.no_grad():
                    lake_losses.append(tl_val)
                    if STAGE_MODE == "stage2":
                        lake_ious.append(patch_hard_iou)
                    else:
                        lake_ious.append(hard_iou)
                    if STAGE_MODE == "stage1":
                        epoch_soft_iou_list.append(_soft_for_scheduler)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                if device_name == 'CUDA' and USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                epoch_loss += tl_val
                num_lakes += 1

                del global_logits, total_loss, global_image
                if has_global_mask:
                    del global_mask
                if STAGE_MODE != "stage1":
                    del patches, patch_masks, patch_coords, orig_hw, patch_logits, patch_bnd_logits, patch_prob, patch_delta_logits, patch_coarse_logits
                    if patch_meta is not None:
                        del patch_meta
                if num_lakes % CLEAR_CACHE_EVERY == 0:
                    clear_cache(device_name)
                
            except Exception as e:
                print(f"训练 lake {i} 出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果出错，需要清零梯度，避免影响后续训练
                optimizer.zero_grad(set_to_none=True)
                clear_cache(device_name)
                continue
        
        # 计算训练集平均指标（以 lake 为单位）
        if num_lakes > 0:
            avg_loss = epoch_loss / num_lakes  # 平均每个 lake 的 loss
            avg_train_iou = np.mean(lake_ious) if len(lake_ious) > 0 else 0.0  # lake-level mean IOU
            epoch_mean_soft_iou = np.mean(epoch_soft_iou_list) if len(epoch_soft_iou_list) > 0 else 0.0
            union_epoch = epoch_probs_sum + epoch_gt_sum - epoch_inter_sum
            if union_epoch > 1e-9:
                epoch_level_pixel_soft_iou = (epoch_inter_sum + 1e-6) / (union_epoch + 1e-6)
            else:
                epoch_level_pixel_soft_iou = 0.0
        else:
            avg_loss = 0.0
            avg_train_iou = 0.0
            epoch_mean_soft_iou = 0.0
            epoch_level_pixel_soft_iou = 0.0
            print(f"警告: Epoch {epoch+1} 没有成功处理的 lake")
        
        eval_interval = STAGE1_EVAL_INTERVAL if STAGE_MODE == "stage1" else STAGE2_EVAL_INTERVAL
        do_eval = ((epoch + 1) % eval_interval == 0)
        if do_eval:
            # ========== 验证集评估阶段 ==========
            avg_val_iou = evaluate_split(
                model,
                valDataLoader,
                split_name="验证集",
                stage_mode=STAGE_MODE,
                device=device,
                device_name=device_name,
                use_amp=USE_AMP,
            )
            metric_for_stop = avg_val_iou
            last_val_iou = avg_val_iou

            # ========== 更新学习率 ==========
            scheduler.step(metric_for_stop)
            clear_cache(device_name)

            # ========== Early Stopping 逻辑 ==========
            # 检查是否有改进
            improved = False
            if metric_for_stop > best_val_iou + min_delta:
                improved = True
                iou_improvement = metric_for_stop - best_val_iou
                best_val_iou = metric_for_stop
                patience_counter = 0
                # 保存最佳模型
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict().copy(),
                    'val_iou': metric_for_stop,
                    'train_loss': avg_loss,
                    'train_iou': avg_train_iou,
                }
                with open(best_model_path, "wb") as f:
                    torch.save(best_model_state, f)
                print(f"✓ 发现更好的模型！验证 soft IoU: {metric_for_stop:.4f} (提升 {iou_improvement:.4f})")
            else:
                patience_counter += 1
        else:
            avg_val_iou = last_val_iou
            metric_for_stop = None
            improved = False
# 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{max_epoch}]')
        print(f'  训练 - 损失: {avg_loss:.6f} (处理了 {num_lakes} 个 lake)')
        if STAGE_MODE == "stage1":
            print(f'  mean pixel soft IOU (lake-level): {epoch_mean_soft_iou:.4f}')
            print(f'  epoch-level pixel soft IOU: {epoch_level_pixel_soft_iou:.4f}')
        if STAGE_MODE == "stage1" and len(epoch_bce_list) > 0:
            print(f'  loss 分项: topk_bce={np.mean(epoch_bce_list):.4f} dice={np.mean(epoch_dice_list):.4f}')
        if STAGE_MODE == "stage2" and len(epoch_patch_bce_list) > 0:
            print(
                f'  loss 分项: patch_bce={np.mean(epoch_patch_bce_list):.4f} '
                f'patch_dice={np.mean(epoch_patch_dice_list):.4f} '
                f'patch_bnd={np.mean(epoch_patch_bnd_list):.4f} '
                f'consistency={np.mean(epoch_patch_consist_list):.4f} '
                f'residual={np.mean(epoch_patch_residual_list):.4f}'
            )
        if STAGE_MODE == "stage1" and len(epoch_hard_iou_list) > 0:
            print(f'  评估 (hard): mean hard IoU={np.mean(epoch_hard_iou_list):.4f} mean hard Dice={np.mean(epoch_hard_dice_list):.4f}')
        if do_eval:
            if STAGE_MODE == "stage1":
                print(f'  验证 - mean soft IoU: {avg_val_iou:.4f} (最佳 soft IoU: {best_val_iou:.4f})')
            else:
                print(f'  验证 - mean patch soft IoU: {avg_val_iou:.4f} (最佳 soft IoU: {best_val_iou:.4f})')
            print(f'  学习率: {current_lr:.6f}, 处理 lake 数: {num_lakes}')
            print(f'  Early Stopping: {patience_counter}/{patience} ({"改进!" if improved else "无改进"})')
        else:
            print(f'  验证 - 跳过（每{eval_interval}个epoch评估一次）')
            print(f'  学习率: {current_lr:.6f}, 处理 lake 数: {num_lakes}')
            print(f'  Early Stopping: {patience_counter}/{patience} (未评估)')
        print('-' * 60)

        # 追加到 TXT 日志：损失、训练集平均 IoU、验证集平均 IoU
        with open(loss_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{avg_loss:.6f}\n")
        with open(train_iou_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{avg_train_iou:.6f}\n")
        with open(val_iou_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{avg_val_iou:.6f}\n")

        # 检查是否应该提前停止
        if do_eval and patience_counter >= patience:
            print(f"\nEarly Stopping 触发！")
            if best_model_state is not None:
                print(f"最佳验证 soft IoU: {best_val_iou:.4f} (Epoch {best_model_state['epoch']})")
            else:
                print("最佳验证 soft IoU: 0.0000 (未保存最佳模型)")
            print(f"已训练 {epoch+1} 个 epoch，但 {patience} 个 epoch 内没有改进")
            break
        
        # 恢复训练模式
        model.train()
        _freeze_frozen_parts_bn(model, STAGE_MODE)

    print(f"\n训练完成！")
    print(f"模型权重已保存: {best_model_path}")
    print(f"最佳验证 soft IoU: {best_val_iou:.4f}")
    print("=" * 60)

    # ========== 测试集评估 ==========
    if os.path.isfile(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
            print("已加载最佳模型用于测试集评估")
        except Exception as e:
            print(f"加载最佳模型失败，使用当前模型进行测试评估: {e}")
    else:
        print("未找到最佳模型文件，使用当前模型进行测试评估")

    avg_test_iou = evaluate_split(
        model,
        testDataLoader,
        split_name="测试集",
        stage_mode=STAGE_MODE,
        device=device,
        device_name=device_name,
        use_amp=USE_AMP,
    )
    with open(test_iou_txt, "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"test_iou\t{avg_test_iou:.6f}\n")
    if STAGE_MODE == "stage1":
        print(f"测试集 mean soft IoU: {avg_test_iou:.4f}")
    else:
        print(f"测试集 mean patch soft IoU: {avg_test_iou:.4f}")
    print("=" * 60)
