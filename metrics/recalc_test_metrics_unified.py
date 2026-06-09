# -*- coding: utf-8 -*-
"""
V17 (Ours) Test 集统一重算指标：

- hard IOU：全局像素池（micro）
- Lake-level IOU：逐湖 hard IoU 算术平均（macro）
- Boundary F1：逐湖 @3px（metrics/VERSION22_Stage1Only/metrics.py）再平均
- Mean/Median/P90 BDE：全测试集边界像元误差拼接（boundary_error_histogram.py）

预测：patch logits 拼回整湖后 sigmoid>0.5；GT=Test/mask/{lake_id}.npy
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(METRICS_DIR)
V22_DIR = os.path.join(METRICS_DIR, "VERSION22_Stage1Only")

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)
if METRICS_DIR not in sys.path:
    sys.path.insert(0, METRICS_DIR)

from boundary_error_histogram import boundary_error_pixels  # noqa: E402

from paths_config import DATASET_ROOT, TEST_MASK_DIR  # noqa: E402

DATASET_FOLDER = DATASET_ROOT
OUT_DIR = os.path.join(METRICS_DIR, "output")
OUT_CSV = os.path.join(OUT_DIR, "test_metrics_unified_v17.csv")
OUT_TXT = os.path.join(OUT_DIR, "test_metrics_unified_v17.txt")

BOUNDARY_TOLERANCE = 3.0
THRESHOLD = 0.5
EPS = 1e-6

_INFER_CACHE: Dict[str, object] = {}


def load_v22_metrics():
    path = os.path.join(V22_DIR, "metrics.py")
    spec = importlib.util.spec_from_file_location("v22_metrics_unified", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V22M = load_v22_metrics()


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def list_lake_ids() -> List[int]:
    ids = []
    for fn in os.listdir(TEST_MASK_DIR):
        if fn.endswith(".npy"):
            ids.append(int(os.path.splitext(fn)[0]))
    return sorted(ids)


def load_gt_mask(lake_id: int) -> np.ndarray:
    path = os.path.join(TEST_MASK_DIR, f"{lake_id}.npy")
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return (arr > 0.5).astype(np.uint8)


def align_pred_gt(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pred.shape == gt.shape:
        return pred, gt
    from scipy.ndimage import zoom

    zh = gt.shape[0] / pred.shape[0]
    zw = gt.shape[1] / pred.shape[1]
    pred_r = zoom(pred.astype(np.float32), (zh, zw), order=0) > 0.5
    return pred_r.astype(np.uint8), gt


def compute_iou_stats(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    inter = float((p * g).sum())
    pred_sum = float(p.sum())
    gt_sum = float(g.sum())
    iou = (inter + EPS) / (pred_sum + gt_sum - inter + EPS)
    return iou, inter, pred_sum, gt_sum


def assemble_patch_logits_to_full(
    patch_logits: torch.Tensor,
    patch_coords,
    orig_h: int,
    orig_w: int,
    patch_size: int = 512,
) -> np.ndarray:
    accum = torch.zeros(1, orig_h, orig_w, dtype=torch.float64)
    count = torch.zeros(1, orig_h, orig_w, dtype=torch.float64)
    for i, (x, y) in enumerate(patch_coords):
        x, y = int(x), int(y)
        he, we = min(y + patch_size, orig_h), min(x + patch_size, orig_w)
        ch, cw = he - y, we - x
        if ch <= 0 or cw <= 0:
            continue
        accum[0, y:he, x:we] += patch_logits[i, 0, :ch, :cw].double()
        count[0, y:he, x:we] += 1.0
    full_logits = (accum / count.clamp(min=1e-8)).float()
    mask = V22M.logits_to_binary_mask(full_logits, threshold=THRESHOLD)
    return mask.squeeze().cpu().numpy().astype(np.uint8)


def compute_unified_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred, gt = align_pred_gt(pred, gt)
    iou, inter, ps, gs = compute_iou_stats(pred, gt)

    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    b = V22M.calculate_boundary_metrics(pred_t, gt_t, tolerance=BOUNDARY_TOLERANCE)
    errs = boundary_error_pixels(gt, pred)

    return {
        "lake_iou": iou,
        "inter": inter,
        "pred_sum": ps,
        "gt_sum": gs,
        "boundary_f1": float(b["boundary_f1"]),
        "bde_pixels": errs,
    }


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    return torch.device("cpu"), "CPU"


def predict_v17(lake_id: int) -> np.ndarray:
    if "v17" not in _INFER_CACHE:
        if CODE_ROOT not in sys.path:
            sys.path.insert(0, CODE_ROOT)
        loader_mod = _load_module(
            "v17_dataloader", os.path.join(CODE_ROOT, "MyDataLoader_numpy0127.py")
        )
        ms_mod = _load_module(
            "v17_model", os.path.join(CODE_ROOT, "ModelStructure_CUDA.py")
        )
        from torch.utils.data import DataLoader

        mean_std = os.path.join(DATASET_FOLDER, "Train", "train_band_mean_std.txt")
        test_img = os.path.join(DATASET_FOLDER, "Test", "image")
        test_msk = os.path.join(DATASET_FOLDER, "Test", "mask")
        ds = loader_mod.WaterDataset(
            test_img,
            test_msk,
            is_train=False,
            global_image_folder=test_img,
            global_mask_folder=test_msk,
            mean_std_path=mean_std,
        )
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, collate_fn=loader_mod.custom_collate_fn
        )
        cfg_b = dict(
            type="ResNet",
            depth=50,
            in_channels=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=-1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=False,
            style="pytorch",
        )
        cfg_n = dict(
            type="FPN",
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )
        device, dname = _get_device()
        model = ms_mod.ModelStructure(cfg_b, cfg_n).to(device)
        ckpt = torch.load(
            os.path.join(CODE_ROOT, "checkpoints", "stage2.ckpt"), map_location=device
        )
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        model.eval()
        _INFER_CACHE["v17"] = (model, device, dname, loader)

    model, device, dname, loader = _INFER_CACHE["v17"]
    for batch in loader:
        if int(batch["lake_ids"][0]) != lake_id:
            continue
        global_image = batch["global_images"][0:1].to(device)
        patches = batch["patches_list"][0]
        patch_coords = batch["patch_coords_list"][0]
        orig_hw = batch["orig_hw_list"][0]
        oh, ow = int(orig_hw[0]), int(orig_hw[1])
        parts = []
        p_total = patches.shape[0]
        use_amp = dname == "CUDA"
        for s in range(0, p_total, 8):
            e = min(p_total, s + 8)
            chunk = patches[s:e].to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, pl, _ = model(
                        global_image,
                        patches=chunk,
                        patch_coords=patch_coords[s:e],
                        orig_hw=orig_hw,
                        global_no_grad=True,
                    )
            parts.append(pl.float().cpu())
        return assemble_patch_logits_to_full(
            torch.cat(parts, 0), patch_coords, oh, ow, 512
        )
    raise RuntimeError(f"lake {lake_id} not found in Test loader")


def evaluate_v17() -> Dict:
    lake_ids = list_lake_ids()
    total_inter = total_pred = total_gt = 0.0
    lake_ious: List[float] = []
    lake_bf1s: List[float] = []
    bde_chunks: List[np.ndarray] = []
    per_lake_rows: List[Dict] = []

    print("\n[V17] Ours", flush=True)
    for lake_id in lake_ids:
        gt = load_gt_mask(lake_id)
        pred = predict_v17(lake_id)
        m = compute_unified_metrics(pred, gt)

        total_inter += m["inter"]
        total_pred += m["pred_sum"]
        total_gt += m["gt_sum"]
        lake_ious.append(m["lake_iou"])
        lake_bf1s.append(m["boundary_f1"])
        if m["bde_pixels"].size > 0:
            bde_chunks.append(m["bde_pixels"])

        per_lake_rows.append(
            {
                "lake_id": lake_id,
                "lake_iou": m["lake_iou"],
                "boundary_f1_3px": m["boundary_f1"],
                "bde_n": int(m["bde_pixels"].size),
            }
        )
        print(
            f"  lake {lake_id}: iou={m['lake_iou']:.4f} bf1={m['boundary_f1']:.4f} "
            f"bde_n={m['bde_pixels'].size}",
            flush=True,
        )

    hard_iou = (total_inter + EPS) / (total_pred + total_gt - total_inter + EPS)
    lake_iou = float(np.mean(lake_ious)) if lake_ious else 0.0
    bf1_mean = float(np.mean(lake_bf1s)) if lake_bf1s else 0.0

    if bde_chunks:
        bde_all = np.concatenate(bde_chunks)
        bde_mean = float(bde_all.mean())
        bde_median = float(np.median(bde_all))
        bde_p90 = float(np.percentile(bde_all, 90))
        bde_n = int(bde_all.size)
    else:
        bde_mean = bde_median = bde_p90 = 0.0
        bde_n = 0

    print(
        f"  => hard_IOU={hard_iou:.4f} lake_IOU={lake_iou:.4f} BF1={bf1_mean:.4f} "
        f"BDE mean/med/p90={bde_mean:.2f}/{bde_median:.2f}/{bde_p90:.2f} (n={bde_n})",
        flush=True,
    )

    return {
        "hard_iou": hard_iou,
        "lake_level_iou": lake_iou,
        "boundary_f1_mean_3px": bf1_mean,
        "mean_bde_px": bde_mean,
        "median_bde_px": bde_median,
        "p90_bde_px": bde_p90,
        "bde_pixel_count": bde_n,
        "per_lake": per_lake_rows,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    result = evaluate_v17()

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "version",
                "model",
                "hard_iou",
                "lake_level_iou",
                "boundary_f1_mean_3px",
                "mean_bde_px",
                "median_bde_px",
                "p90_bde_px",
                "bde_pixel_count",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "version": "V17",
                "model": "Ours",
                **{k: result[k] for k in w.fieldnames if k not in ("version", "model")},
            }
        )

    per_lake_csv = os.path.join(OUT_DIR, "test_metrics_unified_v17_per_lake.csv")
    with open(per_lake_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["lake_id", "lake_iou", "boundary_f1_3px", "bde_n"],
        )
        w.writeheader()
        w.writerows(result["per_lake"])

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("V17 Test metrics (unified)\n")
        f.write(f"method\t{os.path.join(METRICS_DIR, 'test_metrics_unified_METHOD.txt')}\n")
        f.write(f"mask_dir\t{TEST_MASK_DIR}\n\n")
        f.write(f"hard_IOU\t{result['hard_iou']:.6f}\n")
        f.write(f"lake_level_IOU\t{result['lake_level_iou']:.6f}\n")
        f.write(f"boundary_f1_mean_3px\t{result['boundary_f1_mean_3px']:.6f}\n")
        f.write(f"mean_bde_px\t{result['mean_bde_px']:.6f}\n")
        f.write(f"median_bde_px\t{result['median_bde_px']:.6f}\n")
        f.write(f"p90_bde_px\t{result['p90_bde_px']:.6f}\n")

    print(f"\nSaved:\n  {OUT_CSV}\n  {per_lake_csv}\n  {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
