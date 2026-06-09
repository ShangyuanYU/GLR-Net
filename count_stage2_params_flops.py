# -*- coding: utf-8 -*-
"""
统计本发布包 Stage2 可训练参数量、参数体积与 FLOPs。

与 train.py 一致：
- STAGE_MODE=stage2：冻结 global 分支，仅训练 patch_* 模块
- 结构：ResNet50 + FPN，输入 4 通道

FLOPs 说明（默认输入尺度，与 train.py SUMMARY_* 一致）：
- Global 前向（冻结但仍执行）：1 × 4 × 512 × 512
- Patch 前向（可训练）：P × 4 × 512 × 512，默认 P=24（MAX_PATCHES_PER_LAKE_STAGE2）
- 融合小模块（patch_fuse / film / meta_mlp 等）单独估算或忽略（相对 ResNet 很小）
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ModelStructure_CUDA import ModelStructure  # noqa: E402
from mmcv.cnn import get_model_complexity_info  # noqa: E402

# 与 train.py 对齐
MODEL_IN_CHANNELS = 4
MODEL_BACKBONE_DEPTH = 50
MODEL_BACKBONE_STRIDES = (1, 2, 2, 1)
MODEL_BACKBONE_DILATIONS = (1, 1, 1, 2)
MODEL_BACKBONE_OUT_INDICES = (0, 1, 2, 3)
MODEL_FPN_IN_CHANNELS = [256, 512, 1024, 2048]
MODEL_FPN_OUT_CHANNELS = 256
MODEL_FPN_NUM_OUTS = 4
SUMMARY_GLOBAL_SIZE = 512
MAX_PATCHES_PER_LAKE_STAGE2 = 24

STAGE2_TRAINABLE_PREFIXES = (
    "patch_backbone.",
    "patch_neck.",
    "patch_decoder.",
    "patch_meta_mlp.",
    "patch_global_proj.",
    "patch_fuse.",
    "patch_global_film.",
)


def build_model() -> ModelStructure:
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
    return ModelStructure(backbone_cfg, neck_cfg)


def apply_stage2_freeze(model: ModelStructure) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith(STAGE2_TRAINABLE_PREFIXES):
            p.requires_grad = True


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f} G"
    if n >= 1e6:
        return f"{n / 1e6:.3f} M"
    if n >= 1e3:
        return f"{n / 1e3:.3f} K"
    return str(n)


def flops_of_module(
    module: nn.Module,
    input_shape: tuple,
    input_constructor=None,
) -> tuple:
    """input_shape: (C,H,W)；mmcv 会自动加 batch 维 -> (1,C,H,W)。"""
    kwargs = dict(
        print_per_layer_stat=False,
        as_strings=False,
    )
    if input_constructor is not None:
        kwargs["input_constructor"] = input_constructor
    flops_val, params_val = get_model_complexity_info(module, input_shape, **kwargs)
    flops_str, _ = get_model_complexity_info(
        module,
        input_shape,
        print_per_layer_stat=False,
        as_strings=True,
        input_constructor=input_constructor,
    )
    return flops_str, int(flops_val), int(params_val)


class GlobalBranchCore(nn.Module):
    def __init__(self, model: ModelStructure):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.global_context = model.global_context
        self.decoder = model.decoder

    def forward(self, x):
        feats = self.backbone(x)
        fpn_feats = self.neck(feats)
        l1, l2, l3 = fpn_feats[0], fpn_feats[1], fpn_feats[2]
        l3 = self.global_context(l3, use_dilation=False)
        logits, _ = self.decoder(
            l3, l2, l1, out_size=x.shape[-2:], return_feat=True
        )
        return logits


class PatchBranchCore(nn.Module):
    def __init__(self, model: ModelStructure):
        super().__init__()
        self.patch_backbone = model.patch_backbone
        self.patch_neck = model.patch_neck
        self.patch_decoder = model.patch_decoder

    def forward(self, x):
        feats = self.patch_backbone(x)
        fpn = self.patch_neck(feats)
        p1, p2, p3, p4 = fpn[0], fpn[1], fpn[2], fpn[3]
        seg, bnd = self.patch_decoder(p4, p3, p2, p1, out_size=x.shape[-2:])
        return seg, bnd


def main():
    model = build_model()
    apply_stage2_freeze(model)
    model.eval()

    total, trainable = count_parameters(model)
    frozen = total - trainable

    print("=" * 60)
    print("Stage2 模型统计（与 train.py 配置一致）")
    print("=" * 60)
    print(f"总参数量:           {total:,}  ({fmt_params(total)})")
    print(f"可训练参数量:       {trainable:,}  ({fmt_params(trainable)})")
    print(f"冻结参数量:         {frozen:,}  ({fmt_params(frozen)})")
    print(f"可训练参数占比:     {100.0 * trainable / total:.2f}%")
    print(f"参数体积 (fp32):    可训练 {trainable * 4 / 1024**2:.2f} MB | 全部 {total * 4 / 1024**2:.2f} MB")
    print(f"参数体积 (fp16):    可训练 {trainable * 2 / 1024**2:.2f} MB | 全部 {total * 2 / 1024**2:.2f} MB")
    print()

    print("按顶层模块（参数量 / 可训练）:")
    groups = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups.setdefault(top, [0, 0])
        groups[top][0] += p.numel()
        if p.requires_grad:
            groups[top][1] += p.numel()
    for top in sorted(groups):
        t, tr = groups[top]
        print(f"  {top:20s}  total={t:>12,}  trainable={tr:>12,}")

    print()
    print("FLOPs（mmcv.get_model_complexity_info，MACs×2 记为 FLOPs）")
    chw = (MODEL_IN_CHANNELS, SUMMARY_GLOBAL_SIZE, SUMMARY_GLOBAL_SIZE)
    p_batch = MAX_PATCHES_PER_LAKE_STAGE2

    global_mod = GlobalBranchCore(model).eval()
    patch_mod = PatchBranchCore(model).eval()

    g_flops_s, g_flops, _ = flops_of_module(global_mod, chw)

    def patch_input_ctor(_input_shape):
        return {
            "x": torch.zeros(
                (p_batch, *_input_shape),
                dtype=next(patch_mod.parameters()).dtype,
                device=next(patch_mod.parameters()).device,
            )
        }

    p_flops_s, p_flops, _ = flops_of_module(patch_mod, chw, input_constructor=patch_input_ctor)

    # 小型融合头：粗略用一次 256 通道特征图上的 conv/linear 估算（数量级补充）
    fuse_extra = 0
    try:
        dummy_p1 = torch.zeros(1, 256, 128, 128)
        fuse_mod = nn.Sequential(model.patch_global_proj, model.patch_fuse).eval()
        _, fuse_extra, _ = flops_of_module(fuse_mod, (1, 256, 128, 128))
        # meta_mlp: 24 x 8
        meta_mod = model.patch_meta_mlp.eval()
        _, meta_flops, _ = flops_of_module(meta_mod, (MAX_PATCHES_PER_LAKE_STAGE2, 8))
        film_mod = model.patch_global_film.eval()
        _, film_flops, _ = flops_of_module(film_mod, (1, 256, 128, 128))
        fuse_extra = int(fuse_extra) + int(meta_flops) + int(film_flops)
    except Exception as e:
        print(f"  (融合模块 FLOPs 估算跳过: {e})")

    per_lake_flops = int(g_flops) + int(p_flops) + int(fuse_extra) * MAX_PATCHES_PER_LAKE_STAGE2

    def gflops(x):
        return x / 1e9

    print(f"  Global 分支 @ {SUMMARY_GLOBAL_SIZE}×{SUMMARY_GLOBAL_SIZE}:     {g_flops_s}  ({gflops(g_flops):.3f} GFLOPs)")
    print(f"  Patch 分支 @ P={p_batch}, 512×512:  {p_flops_s}  ({gflops(p_flops):.3f} GFLOPs)")
    print(f"  融合小模块估算 (×P):              ~{gflops(fuse_extra * p_batch):.4f} GFLOPs")
    print(f"  单次 Stage2 前向合计 (约):        ~{gflops(per_lake_flops):.3f} GFLOPs / lake")
    print()
    print("说明:")
    print("  - 可训练参数仅 patch_backbone/neck/decoder 及 patch_global_* / patch_meta_mlp")
    print("  - Stage2 训练时 global_no_grad=True，global 仍前向但不反传")
    print("  - roi_align 裁剪 coarse logits 未计入 mmcv FLOPs（通常远小于 ResNet）")
    print("  - 实际每湖 P 与 global 分辨率随样本变化，上表为 train.py 典型配置")

    out_txt = os.path.join(current_dir, "logs", "stage2_params_flops.txt")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"total_params\t{total}\n")
        f.write(f"trainable_params\t{trainable}\n")
        f.write(f"frozen_params\t{frozen}\n")
        f.write(f"trainable_mb_fp32\t{trainable * 4 / 1024**2:.4f}\n")
        f.write(f"global_flops\t{g_flops}\n")
        f.write(f"patch_flops_P{MAX_PATCHES_PER_LAKE_STAGE2}\t{p_flops}\n")
        f.write(f"per_lake_forward_flops_approx\t{per_lake_flops}\n")
    print(f"\n已保存: {out_txt}")


if __name__ == "__main__":
    main()
