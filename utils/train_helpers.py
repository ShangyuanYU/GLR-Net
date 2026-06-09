import os
import numpy as np
import torch

from MyDataLoader_numpy0127 import downsample_whole_image


def diagnose_no_global_mask(lake_id, global_mask_folder):
    """检查某 lake 没有 global_mask 的具体原因（与 Dataset __getitem__ 逻辑一致）。"""
    print(f"  [诊断] lake_id={lake_id}, global_mask_folder={repr(global_mask_folder)}")
    if global_mask_folder is None or global_mask_folder == "":
        print("  [诊断] 原因: global_mask_folder 为空，Dataset 不加载 global mask")
        return
    if not os.path.isdir(global_mask_folder):
        print("  [诊断] 原因: global_mask_folder 不是目录或不存在")
        return
    path = os.path.normpath(os.path.join(global_mask_folder, str(lake_id) + ".npy"))
    print(f"  [诊断] 预期文件: {path}")
    if not os.path.exists(path):
        print(f"  [诊断] 原因: 文件不存在（MASK_numpy 下缺少 {lake_id}.npy）")
        return
    try:
        gm = np.load(path)
        downsample_whole_image(gm, max_side=1024)
        print(f"  [诊断] 加载与 downsample 成功，原始 shape={gm.shape}")
    except Exception as e:
        print(f"  [诊断] 原因: 加载或 downsample 失败: {e}")
        import traceback
        traceback.print_exc()


def extract_lake_ids_from_folder(folder_path):
    """
    从文件夹中读取文件，提取所有的 LakeID
    """
    if not os.path.exists(folder_path):
        print(f"警告: 文件夹不存在: {folder_path}")
        return set()

    lake_ids = set()
    all_files = os.listdir(folder_path)

    for filename in all_files:
        if filename.startswith('.') or not (filename.endswith('.npy') or filename.endswith('.tif')):
            continue
        if '_' in filename:
            lake_id_str = filename.split('_')[0]
        else:
            lake_id_str = os.path.splitext(filename)[0]
        try:
            lake_id = int(lake_id_str)
            lake_ids.add(lake_id)
        except ValueError:
            continue

    return lake_ids


def get_device():
    """自动选择可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'CUDA'
    return torch.device('cpu'), 'CPU'


def clear_cache(device_type):
    """清理设备缓存"""
    try:
        if device_type == 'CUDA':
            torch.cuda.empty_cache()
    except Exception:
        pass
