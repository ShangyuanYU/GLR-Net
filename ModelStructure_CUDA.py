import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# mmseg：优先 pip 安装；否则尝试 CODE 上级目录下的 mmsegmentation-main/
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
for _root in (
    os.environ.get("MMSEG_PATH", ""),
    os.path.dirname(_CODE_DIR),
    os.path.dirname(os.path.dirname(_CODE_DIR)),
):
    if not _root:
        continue
    _mmseg = os.path.join(_root, "mmsegmentation-main")
    if os.path.isdir(_mmseg) and _mmseg not in sys.path:
        sys.path.insert(0, _mmseg)
        break

from mmseg.models import build_backbone
from mmseg.models.builder import build_neck


class Decoder(nn.Module):
    """L3 -> L2 -> L1 逐级上采样融合，输出 full-res logits。"""
    def __init__(self, channels=256, num_classes=1):
        super().__init__()
        self.fuse_l2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_l1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, l3, l2, l1, out_size=None, return_feat=False):
        x = F.interpolate(l3, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_l2(torch.cat([x, l2], dim=1))
        x = F.interpolate(x, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_l1(torch.cat([x, l1], dim=1))
        feat = x
        x = self.head(x)
        if out_size is not None and x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return (x, feat) if return_feat else x


class GlobalContextBlock(nn.Module):
    """1×1 → BN → ReLU → 3×3 → BN → ReLU. 256→256，FPN 之后、decoder 之前；3×3 无 dilation。"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv3x3_dilated = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False)
        self.conv3x3_normal = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, use_dilation=True):
        x = self.conv1(x)
        x = self.conv3x3_dilated(x) if use_dilation else self.conv3x3_normal(x)
        x = self.relu(self.bn(x))
        return x


class PatchDecoder(nn.Module):
    """FPN-UNet 混合解码器：L4->L3->L2->L1 逐级上采样 + concat."""
    def __init__(self, channels=256, num_classes=1):
        super().__init__()
        self.fuse_l3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_l2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_l1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, l4, l3, l2, l1, out_size=None):
        x = F.interpolate(l4, size=l3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_l3(torch.cat([x, l3], dim=1))
        x = F.interpolate(x, size=l2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_l2(torch.cat([x, l2], dim=1))
        x = F.interpolate(x, size=l1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_l1(torch.cat([x, l1], dim=1))
        seg_logits = self.seg_head(x)
        bnd_logits = self.boundary_head(x)
        if out_size is not None and seg_logits.shape[-2:] != out_size:
            seg_logits = F.interpolate(seg_logits, size=out_size, mode='bilinear', align_corners=False)
            bnd_logits = F.interpolate(bnd_logits, size=out_size, mode='bilinear', align_corners=False)
        return seg_logits, bnd_logits


class ModelStructure(nn.Module):
    def __init__(self, backbone_cfg, neck_cfg):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.neck = build_neck(neck_cfg)

        # FPN 输出通道
        fpn_out_channels = neck_cfg['out_channels']  # 256
        self.global_context = GlobalContextBlock(fpn_out_channels)
        self.decoder = Decoder(channels=fpn_out_channels, num_classes=1)
        self.patch_global_proj = nn.Sequential(
            nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.patch_fuse = nn.Sequential(
            nn.Conv2d(fpn_out_channels * 2, fpn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.patch_global_film = nn.Conv2d(
            fpn_out_channels, fpn_out_channels * 2, kernel_size=1
        )
        self.patch_meta_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, fpn_out_channels * 2),
        )

        # Patch branch (独立 backbone / neck)
        self.patch_backbone = build_backbone(backbone_cfg)
        self.patch_neck = build_neck(neck_cfg)
        self.patch_decoder = PatchDecoder(channels=fpn_out_channels, num_classes=1)

    
    def forward(self, global_image, patches=None, patch_coords=None, orig_hw=None, patch_meta=None, return_features=False, return_patch_aux=False, global_only=False, global_no_grad=None):
        """
        Global 分支：Backbone → FPN → Decoder → full-res logits。
        Patch 分支：Backbone → FPN → PatchDecoder(delta)；最终输出为 coarse(global crop) + delta。
        可用 global fuse_l1 特征做对齐融合。
        patch_meta 可选传入 [P,8] 显式位置/尺度编码；不传则由 patch_coords + orig_hw 现场构造。
        return_patch_aux=True 时，额外返回 (patch_delta_logits, patch_coarse_logits)。
        global_only 当前未使用；global_no_grad=True 用于冻结 global 前向。
        """
        if return_features and return_patch_aux:
            raise ValueError("return_features 与 return_patch_aux 不能同时为 True")
        # 如果走 patch 分支，默认冻结 global 前向（除非显式传 global_no_grad）
        if global_no_grad is None:
            global_no_grad = patches is not None

        with torch.set_grad_enabled(not global_no_grad):
            feats = self.backbone(global_image)
            fpn_feats = self.neck(feats)
            L1_global = fpn_feats[0]
            L2_global = fpn_feats[1]
            L3_global = fpn_feats[2]
            L3_global = self.global_context(L3_global, use_dilation=False)
            global_logits, fuse_l1_feat = self.decoder(
                L3_global, L2_global, L1_global, out_size=global_image.shape[-2:], return_feat=True
            )

        patch_logits = None
        patch_bnd_logits = None
        patch_delta_logits = None
        patch_coarse_logits = None
        if patches is not None:
            patch_feats = self.patch_backbone(patches)
            patch_fpn = self.patch_neck(patch_feats)
            P1, P2, P3, P4 = patch_fpn[0], patch_fpn[1], patch_fpn[2], patch_fpn[3]
            patch_size = patches.shape[-1]

            meta_gamma = None
            meta_beta = None
            meta_tensor = None
            if patch_meta is not None:
                meta_tensor = patch_meta.to(device=patches.device, dtype=P1.dtype)
            elif patch_coords is not None and orig_hw is not None:
                meta_tensor = self._build_patch_meta(
                    patch_coords=patch_coords,
                    orig_hw=orig_hw,
                    patch_size=patch_size,
                    device=patches.device,
                    dtype=P1.dtype,
                )  # [P, 8]
            if meta_tensor is not None and meta_tensor.numel() > 0:
                if meta_tensor.shape[0] != P1.shape[0]:
                    raise ValueError(
                        f"patch_meta 数量与 patch 数不一致: meta={meta_tensor.shape[0]}, patch={P1.shape[0]}"
                    )
                meta_gamma_beta = self.patch_meta_mlp(meta_tensor)  # [P, 2C]
                meta_gamma, meta_beta = torch.chunk(meta_gamma_beta, 2, dim=1)
                meta_gamma = meta_gamma[:, :, None, None]
                meta_beta = meta_beta[:, :, None, None]

            gamma_term = 0.0
            beta_term = 0.0
            has_mod = False

            if patch_coords is not None and orig_hw is not None:
                patch_feats_from_global = self._crop_fuse_l1_by_coords(
                    fuse_l1_feat,
                    patch_coords,
                    orig_hw,
                    global_image.shape[-2:],
                    patch_size=patch_size,
                    out_size=P1.shape[-2:],
                )
                if patch_feats_from_global is not None:
                    global_local = self.patch_global_proj(patch_feats_from_global)
                    P1 = self.patch_fuse(torch.cat([P1, global_local], dim=1))
                    gamma_beta = self.patch_global_film(patch_feats_from_global)
                    gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
                    gamma_term = gamma_term + 0.5 * torch.tanh(gamma)
                    beta_term = beta_term + 0.5 * torch.tanh(beta)
                    has_mod = True

            if meta_gamma is not None and meta_beta is not None:
                gamma_term = gamma_term + 0.25 * torch.tanh(meta_gamma)
                beta_term = beta_term + 0.25 * torch.tanh(meta_beta)
                has_mod = True

            if has_mod:
                P1 = P1 * (1 + gamma_term) + beta_term
            patch_delta_logits, patch_bnd_logits = self.patch_decoder(
                P4, P3, P2, P1, out_size=patches.shape[-2:]
            )
            if patch_coords is not None and orig_hw is not None:
                patch_coarse_logits = self._crop_global_logits_by_coords(
                    global_logits=global_logits,
                    patch_coords=patch_coords,
                    orig_hw=orig_hw,
                    global_hw=global_image.shape[-2:],
                    patch_size=patch_size,
                    out_size=patches.shape[-2:],
                )
            patch_logits = patch_delta_logits if patch_coarse_logits is None else (patch_coarse_logits + patch_delta_logits)

        if return_features:
            return (
                global_logits,
                patch_logits,
                patch_bnd_logits,
                L3_global,
                L2_global,
                L1_global,
            )
        if return_patch_aux:
            return global_logits, patch_logits, patch_bnd_logits, patch_delta_logits, patch_coarse_logits
        return global_logits, patch_logits, patch_bnd_logits

    def _build_patch_meta(self, patch_coords, orig_hw, patch_size=512, device=None, dtype=torch.float32):
        """
        为每个 patch 构造显式位置/尺度向量:
        [x_norm, y_norm, x2_norm, y2_norm, cx_norm, cy_norm, pw_norm, ph_norm]
        """
        if patch_coords is None or len(patch_coords) == 0:
            return torch.empty((0, 8), device=device, dtype=dtype)

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

        return torch.tensor(meta, device=device, dtype=dtype)

    def _crop_global_logits_by_coords(self, global_logits, patch_coords, orig_hw, global_hw, patch_size=512, out_size=None):
        """
        从 global logits 中按 patch 原图坐标裁剪 coarse patch logits。
        global_logits: [1, 1, H_glb, W_glb]
        patch_coords: list[(x, y)]，patch 左上角在 orig 图坐标系
        orig_hw: (H_orig, W_orig)
        global_hw: (H_glb, W_glb) = global_image.shape[-2:]
        """
        assert global_logits.dim() == 4 and global_logits.size(0) == 1, "global_logits 应该是 [1,1,H,W]"
        _, _, h_glb, w_glb = global_logits.shape
        h_orig, w_orig = orig_hw
        h_expect, w_expect = global_hw
        if h_glb != int(h_expect) or w_glb != int(w_expect):
            # 保守处理：若输入尺寸不一致，以 global_logits 实际尺寸为准
            h_expect, w_expect = h_glb, w_glb

        sx = float(w_expect) / float(w_orig)
        sy = float(h_expect) / float(h_orig)
        rois = []
        for (x, y) in patch_coords:
            x1 = x * sx
            y1 = y * sy
            x2 = (x + patch_size) * sx
            y2 = (y + patch_size) * sy

            x1 = max(0.0, min(x1, w_glb - 1e-3))
            y1 = max(0.0, min(y1, h_glb - 1e-3))
            x2 = max(0.0, min(x2, w_glb - 1e-3))
            y2 = max(0.0, min(y2, h_glb - 1e-3))
            if x2 <= x1 + 1e-3:
                x2 = min(w_glb - 1e-3, x1 + 1.0)
            if y2 <= y1 + 1e-3:
                y2 = min(h_glb - 1e-3, y1 + 1.0)
            rois.append([0.0, x1, y1, x2, y2])

        if len(rois) == 0:
            return None
        if out_size is None:
            out_h = max(1, int(round(patch_size * sy)))
            out_w = max(1, int(round(patch_size * sx)))
            out_size = (out_h, out_w)

        rois = torch.tensor(rois, device=global_logits.device, dtype=torch.float32)
        try:
            crops = roi_align(
                global_logits,
                rois,
                output_size=out_size,
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )
        except TypeError:
            crops = roi_align(
                global_logits,
                rois,
                output_size=out_size,
                spatial_scale=1.0,
                sampling_ratio=-1,
            )
        return crops

    def _crop_fuse_l1_by_coords(self, fuse_l1, patch_coords, orig_hw, global_hw, patch_size=512, out_size=None):
        """
        用 ROIAlign 从 fuse_l1 里按 patch 坐标裁出对应区域，并直接输出到 out_size。
        
        fuse_l1: [1, C, FH, FW]  (global decoder 的 fuse_l1_feat)
        patch_coords: list[(x, y)]，patch 左上角在 orig 图坐标系
        orig_hw: (H_orig, W_orig)
        global_hw: (H_glb, W_glb) = global_image.shape[-2:]
        patch_size: patch 边长（在 orig 坐标系下）
        out_size: (out_h, out_w)，通常设为 P1.shape[-2:]
        """
        assert fuse_l1.dim() == 4 and fuse_l1.size(0) == 1, "fuse_l1 应该是 [1,C,H,W]"
        _, _, FH, FW = fuse_l1.shape
        H_orig, W_orig = orig_hw
        H_glb, W_glb = global_hw

        # 关键：不要写死 /4，用真实尺寸推映射
        # orig -> global 的缩放
        sx_g = W_glb / float(W_orig)
        sy_g = H_glb / float(H_orig)

        # global -> fuse_l1 的缩放（用 fuse_l1 实际大小）
        sx_f = FW / float(W_glb)
        sy_f = FH / float(H_glb)

        # ROIAlign 的 boxes 用的是 (x1,y1,x2,y2)，坐标系和输入 feature map 对应
        rois = []
        for (x, y) in patch_coords:
            # patch 在 orig 坐标系的框：[(x,y),(x+patch_size,y+patch_size)]
            # 映射到 fuse_l1 坐标系
            x1 = x * sx_g * sx_f
            y1 = y * sy_g * sy_f
            x2 = (x + patch_size) * sx_g * sx_f
            y2 = (y + patch_size) * sy_g * sy_f

            # clamp 到 feature map 范围内（ROIAlign 允许浮点，但范围要合理）
            x1 = max(0.0, min(x1, FW - 1e-3))
            y1 = max(0.0, min(y1, FH - 1e-3))
            x2 = max(0.0, min(x2, FW - 1e-3))
            y2 = max(0.0, min(y2, FH - 1e-3))

            # 避免退化框（x2<=x1 或 y2<=y1）
            if x2 <= x1 + 1e-3:
                x2 = min(FW - 1e-3, x1 + 1.0)
            if y2 <= y1 + 1e-3:
                y2 = min(FH - 1e-3, y1 + 1.0)

            # batch_idx=0，因为 fuse_l1 batch 是 1
            rois.append([0.0, x1, y1, x2, y2])

        if len(rois) == 0:
            return None

        rois = torch.tensor(rois, device=fuse_l1.device, dtype=torch.float32)

        if out_size is None:
            # 如果你不传 out_size，就默认输出到“裁剪框大小的整数近似”
            # 但强烈建议传 out_size=P1.shape[-2:]
            out_h = max(1, int(round(patch_size * sy_g * sy_f)))
            out_w = max(1, int(round(patch_size * sx_g * sx_f)))
            out_size = (out_h, out_w)

        # ✅ debug: 只在第一次打印一下，确认框在合理范围
        if not hasattr(self, "_roi_debug_printed"):
            self._roi_debug_printed = True
            print("[ROIAlign debug] fuse_l1:", fuse_l1.shape, "global_hw:", global_hw, "orig_hw:", orig_hw)
            print("[ROIAlign debug] first roi:", rois[0].tolist(), "out_size:", out_size)

        # aligned=True 通常更准（torchvision>=0.7）
        # spatial_scale=1.0，因为我们 rois 已经是在 fuse_l1 坐标系下了
        try:
            crops = roi_align(
                fuse_l1, rois,
                output_size=out_size,
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True
            )  # [P, C, out_h, out_w]
        except TypeError:
            crops = roi_align(
                fuse_l1, rois,
                output_size=out_size,
                spatial_scale=1.0,
                sampling_ratio=-1
            )  # [P, C, out_h, out_w]

        return crops
