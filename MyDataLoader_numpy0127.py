import torch
import numpy as np
import os
import random
import math
from torch.utils.data import Dataset
import torch.nn.functional as F


def _load_band_mean_std(path):
  if path is None:
    raise ValueError("mean/std path is None")
  path = os.path.normpath(path)
  if not os.path.isfile(path):
    fallback = os.path.normpath(os.path.join(os.path.dirname(__file__), "test_band_mean_std.txt"))
    if os.path.isfile(fallback):
      path = fallback
    else:
      raise FileNotFoundError(f"mean/std file not found: {path}")
  entries = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      if line.lower().startswith("band"):
        continue
      parts = line.split()
      if len(parts) < 3:
        continue
      try:
        band = int(parts[0])
        mean = float(parts[1])
        std = float(parts[2])
      except ValueError:
        continue
      entries.append((band, mean, std))
  if not entries:
    raise ValueError(f"No valid mean/std entries found in: {path}")
  entries.sort(key=lambda x: x[0])
  means = [m for _, m, _ in entries]
  stds = [s for _, _, s in entries]
  mean_t = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
  std_t = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)
  return mean_t, std_t


class WaterDataset(Dataset):
  def __init__(self, ImageFoler, MaskFoler, is_train=False, trainID=None, 
               global_image_folder=None, global_mask_folder=None, max_patches_per_lake=10,
               mean_std_path=None, sampling_strategy="random",
               boundary_sample_ratio=0.4, mismatch_sample_ratio=0.3, random_sample_ratio=0.3,
               stage1_prob_folder=None, stage1_prob_threshold=0.5):
    """
    Args:
        ImageFoler: 当前使用的图像文件夹（可能是已复制的文件夹，用于存储全局图像）
        MaskFoler: 当前使用的掩码文件夹（可能是已复制的文件夹）
        is_train: 是否为训练集
        trainID: 训练集ID列表（用于指定训练集的lake）
        global_image_folder: 全局图像文件夹路径（完整湖泊图像，如 {lake_id}.npy）
        global_mask_folder: 全局掩码文件夹路径（完整湖泊掩码，如 {lake_id}.npy）。为空则不加载全局掩码。
        max_patches_per_lake: 训练时每个 lake 最多使用的 patch 数 N（默认 10）。
        sampling_strategy: patch 采样策略，"random" 或 "mixed"。
        boundary_sample_ratio/mismatch_sample_ratio/random_sample_ratio:
            mixed 采样中边界/不一致/随机三类占比。
        stage1_prob_folder: stage1 概率图目录（{lake_id}.npy），用于不一致 patch 采样。
        stage1_prob_threshold: stage1 概率二值化阈值（用于计算与 GT 不一致率）。
    
    约定：每个 batch 的数据单位是一个 lake。global_image / global_mask 始终是完整湖泊。
    若 patch 数 > N，按 sampling_strategy 采样，最大数量为 N。
    """
    self.ImageFoler = ImageFoler
    self.MaskFoler = MaskFoler
    self.is_train = is_train
    self.trainID = trainID if trainID is not None else set()
    self.max_patches_per_lake = max_patches_per_lake if is_train else None  # 仅训练模式使用
    self.sampling_strategy = sampling_strategy if is_train else "all"
    self.boundary_sample_ratio = float(boundary_sample_ratio)
    self.mismatch_sample_ratio = float(mismatch_sample_ratio)
    self.random_sample_ratio = float(random_sample_ratio)
    total_ratio = self.boundary_sample_ratio + self.mismatch_sample_ratio + self.random_sample_ratio
    if total_ratio <= 0:
      self.boundary_sample_ratio, self.mismatch_sample_ratio, self.random_sample_ratio = 0.4, 0.3, 0.3
      total_ratio = 1.0
    self.boundary_sample_ratio /= total_ratio
    self.mismatch_sample_ratio /= total_ratio
    self.random_sample_ratio /= total_ratio
    self.stage1_prob_folder = os.path.normpath(stage1_prob_folder) if stage1_prob_folder else None
    self.stage1_prob_threshold = float(stage1_prob_threshold)
    self._warned_missing_stage1_file = False
    self._warned_invalid_stage1_prob = False
    self.global_mask_folder = os.path.normpath(global_mask_folder) if global_mask_folder else None  # 完整湖泊 mask
    self.band_mean, self.band_std = _load_band_mean_std(mean_std_path)
    
    # 确定全局图像文件夹路径（规范化路径）
    if global_image_folder:
      self.global_image_folder = os.path.normpath(global_image_folder)
    else:
      parent_dir = os.path.dirname(ImageFoler)
      self.global_image_folder = os.path.normpath(os.path.join(parent_dir, 'image'))
    
    # 初始化 lake 列表（每个 lake 包含**全部** patch 文件名，不在此处采样）
    if is_train:
      self.lake_list = self._generate_lake_list()
    else:
      self.lake_list = self._generate_lake_list_from_folder()
    print(f"Found {len(self.lake_list)} lakes")

  def _normalize_image(self, img_t):
    if img_t.ndim != 3:
      raise ValueError(f"Expected 3D image tensor [C,H,W], got {img_t.ndim}D")
    if self.band_mean is None or self.band_std is None:
      return img_t
    if img_t.shape[0] != self.band_mean.shape[0]:
      raise ValueError(f"Channel mismatch: image has {img_t.shape[0]} channels, "
                       f"but mean/std has {self.band_mean.shape[0]}")
    mean = self.band_mean
    std = self.band_std
    if img_t.device != mean.device:
      mean = mean.to(img_t.device)
      std = std.to(img_t.device)
    img_t = (img_t - mean) / (std + 1e-6)
    return img_t

  def _generate_lake_list(self):
    """
    根据 trainID 生成 lake 列表。patch 从 global image 裁剪生成。
    """
    lake_list = []
    for lake_id in self.trainID:
      lake_list.append({'lake_id': lake_id})
    return lake_list
  
  def _generate_lake_list_from_folder(self):
    """
    从全局图像文件夹读取所有 lake（用于测试集）
    """
    lake_ids = []
    if os.path.isdir(self.global_image_folder):
      for name in os.listdir(self.global_image_folder):
        if name.startswith('.') or not name.endswith('.npy'):
          continue
        base = os.path.splitext(name)[0]
        if '_' in base:
          continue
        try:
          lake_ids.append(int(base))
        except ValueError:
          continue
    lake_ids = sorted(set(lake_ids))
    return [{'lake_id': lake_id} for lake_id in lake_ids]
  
  def __getitem__(self, index):
    """
    返回一个 lake 的数据。每个 batch 单位是一个 lake。
    global_image / global_mask 始终是完整湖泊；patches 若数量 > N 则随机采样最多 N 个。
    
    Returns:
        lake_id: int
        global_image: Tensor[C, H, W] - 完整湖泊
        global_mask: Tensor[1, H', W'] - 完整湖泊（若提供 global_mask_folder，downsample 后）
        patches: Tensor[B, C, 512, 512] - 该 lake 的 patches（可能为采样子集）
        patch_masks: Tensor[B, 1, 512, 512] - 对应的 masks
    """
    lake_info = self.lake_list[index]
    lake_id = lake_info['lake_id']
    
    # 加载全局图像（始终完整湖泊）
    ImageGlobelFile = os.path.normpath(os.path.join(self.global_image_folder, str(lake_id) + ".npy"))
    try:
      global_patch = np.load(ImageGlobelFile)  # [C, H, W]
      orig_h, orig_w = global_patch.shape[1], global_patch.shape[2]
      global_patch_t = torch.from_numpy(global_patch).float()
      global_patch_t = self._normalize_image(global_patch_t)
      global_image = downsample_whole_image(global_patch_t, max_side=1024)  # [C, H', W']
    except Exception as e:
      print(f"加载全局图像失败 (LakeID {lake_id}): {e}")
      raise e
    
    # 加载全局掩码（始终完整湖泊）
    global_mask = None
    global_mask_np = None
    if self.global_mask_folder:
      MaskGlobalFile = os.path.normpath(os.path.join(self.global_mask_folder, str(lake_id) + ".npy"))
      if os.path.exists(MaskGlobalFile):
        try:
          gm = np.load(MaskGlobalFile)
          global_mask_np = gm
          global_mask = downsample_whole_image(gm, max_side=1024)

        except Exception as e:
          print(f"加载全局掩码失败 (LakeID {lake_id}): {e}")
    
    # 从原始 global 图像/掩码裁剪生成 patches（基于原图坐标）
    if global_mask_np is None:
      raise ValueError(f"LakeID {lake_id} 缺少 global_mask，无法生成 patch mask")
    global_image_np = global_patch_t
    if global_mask_np.ndim == 2:
      global_mask_np = global_mask_np[np.newaxis, :, :]
    global_mask_t = torch.from_numpy(global_mask_np).float()
    patches_list, patch_masks_list, patch_coords = self._crop_patches_from_global(global_image_np, global_mask_t)
    n_patches_total = len(patches_list)

    # 若 patch 数 > N，按策略采样（random 或 mixed），最大数量为 N
    N = self.max_patches_per_lake
    if self.is_train and N is not None and len(patches_list) > N:
      idx = self._sample_patch_indices(
        lake_id=lake_id,
        patch_masks_list=patch_masks_list,
        patch_coords=patch_coords,
        orig_hw=(orig_h, orig_w),
        patch_size=patches_list[0].shape[-1],
        num_to_sample=N,
      )
      patches_list = [patches_list[i] for i in idx]
      patch_masks_list = [patch_masks_list[i] for i in idx]
      patch_coords = [patch_coords[i] for i in idx]
    
    patches = torch.stack(patches_list, dim=0)  # [B, C, 512, 512]
    patch_masks = torch.stack(patch_masks_list, dim=0)  # [B, 1, 512, 512]
    patch_meta = self._build_patch_meta(
      patch_coords=patch_coords,
      orig_hw=(orig_h, orig_w),
      patch_size=patches.shape[-1],
    )  # [B, 8]
    n_patches_used = len(patches_list)  # 实际参与 loss 的 patch 数（加载成功的数量）
    
    out = {
      'lake_id': lake_id,
      'global_image': global_image,  # [C, H', W'] 完整湖泊（downsample 后）
      'patches': patches,
      'patch_masks': patch_masks,
      'patch_coords': patch_coords,
      'patch_meta': patch_meta,
      'orig_hw': (orig_h, orig_w),
      'n_patches_total': n_patches_total,
      'n_patches_used': n_patches_used,
    }
    if global_mask is not None:
      out['global_mask'] = global_mask  # [1, H', W'] 完整湖泊（downsample 后）
    return out

  def _compute_patch_starts(self, length, patch_size=512):
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

  def _crop_patches_from_global(self, global_image, global_mask, patch_size=512):
    # global_image: [C,H,W], global_mask: [1,H,W]
    C, H, W = global_image.shape
    _, Hm, Wm = global_mask.shape
    if Hm != H or Wm != W:
      global_mask = global_mask.unsqueeze(0)
      global_mask = F.interpolate(
        global_mask,
        size=(H, W),
        mode='nearest',
        align_corners=None
      ).squeeze(0)
    hs = self._compute_patch_starts(H, patch_size)
    ws = self._compute_patch_starts(W, patch_size)
    patches_list = []
    patch_masks_list = []
    patch_coords = []
    for y in hs:
      for x in ws:
        patch = global_image[:, y:y + patch_size, x:x + patch_size]
        mask = global_mask[:, y:y + patch_size, x:x + patch_size]
        if patch.shape[-2:] != (patch_size, patch_size):
          patch = F.interpolate(
            patch.unsqueeze(0),
            size=(patch_size, patch_size),
            mode='bilinear',
            align_corners=False
          ).squeeze(0)
        if mask.shape[-2:] != (patch_size, patch_size):
          mask = F.interpolate(
            mask.unsqueeze(0),
            size=(patch_size, patch_size),
            mode='nearest',
            align_corners=None
          ).squeeze(0)
        patches_list.append(patch)
        patch_masks_list.append(mask)
        patch_coords.append((x, y))
    return patches_list, patch_masks_list, patch_coords

  def _sample_patch_indices(self, lake_id, patch_masks_list, patch_coords, orig_hw, patch_size, num_to_sample):
    total = len(patch_coords)
    if total <= num_to_sample:
      return list(range(total))
    if self.sampling_strategy != "mixed":
      idx = random.sample(range(total), num_to_sample)
      idx.sort()
      return idx
    return self._sample_patch_indices_mixed(
      lake_id=lake_id,
      patch_masks_list=patch_masks_list,
      patch_coords=patch_coords,
      orig_hw=orig_hw,
      patch_size=patch_size,
      num_to_sample=num_to_sample,
    )

  def _sample_patch_indices_mixed(self, lake_id, patch_masks_list, patch_coords, orig_hw, patch_size, num_to_sample):
    total = len(patch_coords)
    all_indices = list(range(total))
    if total <= num_to_sample:
      return all_indices

    n_boundary = int(round(num_to_sample * self.boundary_sample_ratio))
    n_mismatch = int(round(num_to_sample * self.mismatch_sample_ratio))
    n_random = num_to_sample - n_boundary - n_mismatch
    if n_random < 0:
      n_random = 0
      used = n_boundary + n_mismatch
      if used > num_to_sample:
        shrink = used - num_to_sample
        if n_mismatch >= shrink:
          n_mismatch -= shrink
        else:
          shrink -= n_mismatch
          n_mismatch = 0
          n_boundary = max(0, n_boundary - shrink)

    selected = []
    remaining = set(all_indices)

    boundary_scores = self._compute_boundary_scores(patch_masks_list)
    take_boundary = min(n_boundary, len(remaining))
    if take_boundary > 0:
      boundary_idx = sorted(remaining, key=lambda i: boundary_scores[i], reverse=True)[:take_boundary]
      selected.extend(boundary_idx)
      remaining.difference_update(boundary_idx)

    stage1_prob_map = self._load_stage1_prob_map(lake_id=lake_id, orig_hw=orig_hw)
    mismatch_scores = self._compute_mismatch_scores(
      patch_masks_list=patch_masks_list,
      patch_coords=patch_coords,
      stage1_prob_map=stage1_prob_map,
      patch_size=patch_size,
    )
    if mismatch_scores is None:
      n_random += n_mismatch
      n_mismatch = 0

    take_mismatch = min(n_mismatch, len(remaining))
    if take_mismatch > 0:
      mismatch_idx = sorted(remaining, key=lambda i: mismatch_scores[i], reverse=True)[:take_mismatch]
      selected.extend(mismatch_idx)
      remaining.difference_update(mismatch_idx)

    take_random = min(n_random, len(remaining))
    if take_random > 0:
      random_idx = random.sample(list(remaining), take_random)
      selected.extend(random_idx)
      remaining.difference_update(random_idx)

    if len(selected) < num_to_sample and len(remaining) > 0:
      fill_k = min(num_to_sample - len(selected), len(remaining))
      fill_idx = random.sample(list(remaining), fill_k)
      selected.extend(fill_idx)

    selected = sorted(set(selected))
    if len(selected) > num_to_sample:
      selected = selected[:num_to_sample]
    return selected

  def _compute_boundary_scores(self, patch_masks_list):
    scores = []
    for patch_mask in patch_masks_list:
      m = patch_mask[0] if patch_mask.dim() == 3 else patch_mask
      m = m > 0.5
      edge = torch.zeros_like(m, dtype=torch.bool)
      edge[:, 1:] |= (m[:, 1:] != m[:, :-1])
      edge[1:, :] |= (m[1:, :] != m[:-1, :])
      scores.append(float(edge.float().mean().item()))
    return scores

  def _load_stage1_prob_map(self, lake_id, orig_hw):
    if self.stage1_prob_folder is None:
      return None
    npy_path = os.path.normpath(os.path.join(self.stage1_prob_folder, f"{lake_id}.npy"))
    if not os.path.isfile(npy_path):
      if not self._warned_missing_stage1_file:
        print(f"警告: stage1 概率图缺失，mixed 采样将退化为边界+随机。示例文件不存在: {npy_path}")
        self._warned_missing_stage1_file = True
      return None
    try:
      arr = np.load(npy_path)
    except Exception as e:
      if not self._warned_invalid_stage1_prob:
        print(f"警告: 读取 stage1 概率图失败，mixed 采样将退化为边界+随机: {e}")
        self._warned_invalid_stage1_prob = True
      return None
    if arr.ndim == 3:
      if arr.shape[0] == 1:
        arr = arr[0]
      elif arr.shape[-1] == 1:
        arr = arr[..., 0]
      else:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
      if not self._warned_invalid_stage1_prob:
        print(f"警告: stage1 概率图维度异常 ({arr.shape})，mixed 采样将退化为边界+随机。")
        self._warned_invalid_stage1_prob = True
      return None
    prob = torch.from_numpy(arr).float()
    h_orig, w_orig = int(orig_hw[0]), int(orig_hw[1])
    if prob.shape[-2:] != (h_orig, w_orig):
      prob = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(h_orig, w_orig),
        mode='bilinear',
        align_corners=False,
      ).squeeze(0).squeeze(0)
    return prob.clamp(0.0, 1.0)

  def _compute_mismatch_scores(self, patch_masks_list, patch_coords, stage1_prob_map, patch_size):
    if stage1_prob_map is None:
      return None
    patch_size = int(patch_size)
    scores = []
    for i, (x, y) in enumerate(patch_coords):
      x = int(x)
      y = int(y)
      pred_patch = stage1_prob_map[y:y + patch_size, x:x + patch_size]
      if pred_patch.shape[-2:] != (patch_size, patch_size):
        pred_patch = F.interpolate(
          pred_patch.unsqueeze(0).unsqueeze(0),
          size=(patch_size, patch_size),
          mode='bilinear',
          align_corners=False,
        ).squeeze(0).squeeze(0)
      gt_patch = patch_masks_list[i][0] if patch_masks_list[i].dim() == 3 else patch_masks_list[i]
      pred_bin = (pred_patch > self.stage1_prob_threshold).float()
      gt_bin = (gt_patch > 0.5).float()
      mismatch = (pred_bin != gt_bin).float().mean()
      scores.append(float(mismatch.item()))
    return scores

  def _build_patch_meta(self, patch_coords, orig_hw, patch_size=512):
    """
    为每个 patch 构造显式位置/尺度向量:
    [x_norm, y_norm, x2_norm, y2_norm, cx_norm, cy_norm, pw_norm, ph_norm]
    """
    if patch_coords is None or len(patch_coords) == 0:
      return torch.empty((0, 8), dtype=torch.float32)
    h_orig, w_orig = orig_hw
    h_orig = float(h_orig)
    w_orig = float(w_orig)
    pw = float(patch_size)
    ph = float(patch_size)
    w_safe = max(w_orig, 1.0)
    h_safe = max(h_orig, 1.0)
    meta = []
    for (x, y) in patch_coords:
      x = float(x)
      y = float(y)
      x2 = x + pw
      y2 = y + ph
      cx = x + 0.5 * pw
      cy = y + 0.5 * ph
      meta.append([
        x / w_safe,
        y / h_safe,
        x2 / w_safe,
        y2 / h_safe,
        cx / w_safe,
        cy / h_safe,
        pw / w_safe,
        ph / h_safe,
      ])
    return torch.tensor(meta, dtype=torch.float32)
  
  def __len__(self):
    return len(self.lake_list)

def downsample_whole_image(img, max_side=2048):
  """
  img: (C, H, W) or (H, W), numpy or torch
  """
  if isinstance(img, torch.Tensor):
    img_t = img.float()
  else:
    img_t = torch.from_numpy(img).float()
  if img_t.ndim == 3:
    _, H, W = img_t.shape
    scale = min(max_side / H, max_side / W, 1.0)
    if scale == 1.0:
      return img_t
    new_h = int(H * scale)
    new_w = int(W * scale)
    img_t = img_t.unsqueeze(0)  # (1,C,H,W)
    img_t = F.interpolate(
      img_t,
      size=(new_h, new_w),
      mode="bilinear",
      align_corners=False
    )
    return img_t.squeeze(0)
  elif img_t.ndim == 2:
    H, W = img_t.shape
    scale = min(max_side / H, max_side / W, 1.0)
    if scale == 1.0:
      return img_t
    new_h = int(H * scale)
    new_w = int(W * scale)
    img_t = img_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    img_t = F.interpolate(
      img_t,
      size=(new_h, new_w),
      mode="nearest",
      align_corners=None
    )
    return img_t.squeeze(0)  # (C,H',W')
  else:
    raise ValueError(f"Unsupported image dimension: {img_t.ndim}")

def custom_collate_fn(batch):
    """
    自定义 collate：每个 batch 的数据单位是一个 lake。
    输入: [{'lake_id', 'global_image', 'global_mask'?, 'patches', 'patch_masks', 'patch_coords', 'patch_meta'}, ...]
    输出: 'lake_ids', 'global_images', 'global_masks'?(可选), 'patches_list', 'patch_masks_list', 'patch_coords_list', 'patch_meta_list'
    global_image / global_mask 始终是完整湖泊；patches 可能为采样子集。
    """
    lake_ids = []
    global_images = []
    global_masks = []  # 仅当 Dataset 提供 global_mask 时存在
    patches_list = []
    patch_masks_list = []
    patch_coords_list = []
    patch_meta_list = []
    orig_hw_list = []
    n_patches_total_list = []
    n_patches_used_list = []
    
    for item in batch:
        lake_ids.append(item['lake_id'])
        global_images.append(item['global_image'])
        if 'global_mask' in item and item['global_mask'] is not None:
            global_masks.append(item['global_mask'])
        patches_list.append(item['patches'])
        patch_masks_list.append(item['patch_masks'])
        patch_coords_list.append(item['patch_coords'])
        patch_meta_list.append(item['patch_meta'])
        orig_hw_list.append(item['orig_hw'])
        n_patches_total_list.append(item['n_patches_total'])
        n_patches_used_list.append(item['n_patches_used'])
    
    # 将 global_images 填充到相同尺寸后堆叠
    if len(global_images) > 0:
        global_shapes = [g.shape for g in global_images]
        max_shape = list(global_shapes[0])
        for shape in global_shapes[1:]:
            for i in range(len(shape)):
                if shape[i] > max_shape[i]:
                    max_shape[i] = shape[i]
        padded_global_images = []
        for g in global_images:
            if list(g.shape) == max_shape:
                padded_global_images.append(g)
            else:
                pad_sizes = []
                for i in range(len(g.shape) - 1, -1, -1):
                    pad_size = max_shape[i] - g.shape[i]
                    pad_sizes.extend([0, pad_size])
                pad_sizes = tuple(pad_sizes)
                padded_g = F.pad(g, pad_sizes, mode='constant', value=0)
                padded_global_images.append(padded_g)
        global_images_batch = torch.stack(padded_global_images, dim=0)  # [batch_size, C, H, W]
    else:
        global_images_batch = None
    
    # 直接 stack（batch_size=1 时通常尺寸一致）
    if len(global_masks) == len(batch):
        global_masks_batch = torch.stack(global_masks, dim=0)  # [batch_size, 1, H', W']
    else:
        global_masks_batch = None
    
    lake_ids_tensor = torch.tensor(lake_ids, dtype=torch.long)
    
    out = {
        'lake_ids': lake_ids_tensor,
        'global_images': global_images_batch,
        'patches_list': patches_list,
        'patch_masks_list': patch_masks_list,
        'patch_coords_list': patch_coords_list,
        'patch_meta_list': patch_meta_list,
        'orig_hw_list': orig_hw_list,
        'n_patches_total': n_patches_total_list,   # List[int]，每个 lake 的 patch 总数
        'n_patches_used': n_patches_used_list,     # List[int]，实际参与 loss 的 patch 数
    }
    if global_masks_batch is not None:
        out['global_masks'] = global_masks_batch
    return out
