# -*- coding: utf-8 -*-
"""
将 predict_stage2_output 中 GeoTIFF（像素值 1=水体）转为 Shapefile。

输入: predict_stage2_output/{train,val,test}/tif/{lake_id}.tif
输出: predict_stage2_output/{train,val,test}/vector/{split}_merged.shp

用法: python tiff_to_vector.py
"""
import os
import sys
import numpy as np

import rasterio
from rasterio.features import shapes
import fiona
from fiona.crs import CRS as FionaCRS
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

current_dir = os.path.dirname(os.path.abspath(__file__))
PREDICT_ROOT = os.path.join(current_dir, "predict_stage2_output")
SPLITS = ["train", "val", "test"]


def tiff_to_polygons(tif_path):
    """
    读取单波段 GeoTIFF，将像素值 == 1 的区域转为矢量多边形列表。
    返回: (polygon_list, crs)
        polygon_list: list of shapely Polygon/MultiPolygon
        crs: rasterio CRS 对象
    """
    with rasterio.open(tif_path) as ds:
        band = ds.read(1)  # [H, W] uint8
        transform = ds.transform
        crs = ds.crs

        # 生成 mask：只提取值 == 1 的区域
        mask = (band == 1).astype(np.uint8)

        polygons = []
        for geom, value in shapes(mask, mask=(mask == 1), transform=transform):
            if value == 1:
                poly = shape(geom)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)

    return polygons, crs


def merge_and_save_vector(tif_folder, out_shp_path, split_name):
    """
    将一个文件夹中所有 TIFF 转为矢量，合并后保存为 Shapefile。
    每个湖泊保留一个属性字段 lake_id。
    """
    if not os.path.isdir(tif_folder):
        print(f"  [{split_name}] TIFF 文件夹不存在: {tif_folder}")
        return

    tif_files = sorted([f for f in os.listdir(tif_folder) if f.lower().endswith(".tif")])
    if not tif_files:
        print(f"  [{split_name}] 未找到 TIFF 文件: {tif_folder}")
        return

    print(f"  [{split_name}] 共 {len(tif_files)} 个 TIFF 文件")

    all_records = []  # [(lake_id, polygon, crs), ...]
    output_crs = None
    n_ok, n_err, n_empty = 0, 0, 0

    for tif_name in tif_files:
        tif_path = os.path.join(tif_folder, tif_name)
        lake_id = os.path.splitext(tif_name)[0]
        try:
            polygons, crs = tiff_to_polygons(tif_path)
            if output_crs is None and crs is not None:
                output_crs = crs
            if polygons:
                # 合并同一湖泊的所有多边形为一个 MultiPolygon
                merged = unary_union(polygons)
                all_records.append((lake_id, merged))
                n_ok += 1
            else:
                n_empty += 1
        except Exception as e:
            n_err += 1
            print(f"    {tif_name}: 错误 - {e}")

    print(f"  [{split_name}] 成功: {n_ok}, 空(无水体): {n_empty}, 失败: {n_err}")

    if not all_records:
        print(f"  [{split_name}] 无有效矢量数据，跳过输出")
        return

    # 写入 Shapefile
    os.makedirs(os.path.dirname(out_shp_path), exist_ok=True)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "lake_id": "str",
        },
    }

    # 将 rasterio CRS 转为 fiona 可用的格式
    if output_crs is not None:
        crs_dict = output_crs.to_dict()
    else:
        crs_dict = {"init": "epsg:4326"}
        print(f"  [{split_name}] 警告: 未检测到 CRS，默认使用 EPSG:4326")

    with fiona.open(out_shp_path, "w", driver="ESRI Shapefile",
                    schema=schema, crs=crs_dict) as dst:
        for lake_id, geom in all_records:
            # fiona 写入时，MultiPolygon 也可以写入 Polygon 类型的 schema
            # 但为安全起见，如果是 MultiPolygon 拆成多条记录
            if geom.geom_type == "MultiPolygon":
                for sub_poly in geom.geoms:
                    dst.write({
                        "geometry": mapping(sub_poly),
                        "properties": {"lake_id": str(lake_id)},
                    })
            elif geom.geom_type == "Polygon":
                dst.write({
                    "geometry": mapping(geom),
                    "properties": {"lake_id": str(lake_id)},
                })

    print(f"  [{split_name}] 矢量已保存: {out_shp_path}")
    print(f"  [{split_name}] 共 {len(all_records)} 个湖泊的水体矢量")


def main():
    print("=" * 60)
    print("TIFF 栅格转矢量（值=1 → 多边形）")
    print(f"输入目录: {PREDICT_ROOT}")
    print("=" * 60)

    for split in SPLITS:
        tif_folder = os.path.join(PREDICT_ROOT, split, "tif")
        out_dir = os.path.join(PREDICT_ROOT, split, "vector")
        out_shp = os.path.join(out_dir, f"{split}_merged.shp")

        print(f"\n>>> {split} ...")
        merge_and_save_vector(tif_folder, out_shp, split)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("输出矢量文件：")
    for split in SPLITS:
        shp_path = os.path.join(PREDICT_ROOT, split, "vector", f"{split}_merged.shp")
        exists = "✓" if os.path.isfile(shp_path) else "✗"
        print(f"  {exists} {shp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
