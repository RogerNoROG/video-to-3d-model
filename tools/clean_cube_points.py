#!/usr/bin/env python3
"""从合并后的点云里剔掉「透过圆孔看到的台面」等外来几何（按颜色判）。

为什么需要这一步
--------------
B 段是**被摄物翻面后重拍**的，所以 B 段的支撑台面正好落在木块顶面的高度上。
透过木块的圆孔能看到台面，稠密重建把它一起建了出来 —— 合并后就成了一坨
**灰白色块嵌在孔里**，非常显眼。

为什么不能用连通分量删
--------------------
它在点云里与主体是**连通**的（实测连通分量只能删掉 0.80%）。体素预降噪也不行
（`voxel_down_sample` 会把它糊成更大的一整块）。

判据：归一化色度
--------------
    色度 = (R - B) / max(R, G, B)

- 榉木偏粉，归一化色度 ≈ **0.25**
- 灰白台面 ≈ **0.07**
- 除以 max 是为了**与亮度无关**：阴影里的木头 R-B 的绝对值也会变小，但归一化
  色度依然高，所以不会误删阴影（实测侧面木色中位 0.249）

用法
----
    clean_cube_points.py <输入.ply> <输出.ply> [--band 0.15] [--threshold 0.20]

- ``--band``：只清理 ``Y > ymax - band`` 的点。带子往下带得越深，凹槽边缘的
  台面残留也能一起清掉；但带得越深越可能误伤侧面的浅色木纹。
- ``--threshold``：色度低于它就当外来几何删掉。

实测（29.0 M 点合并云）：
    band 0.04 / thr 0.18  →  删 1.15%，凹槽边缘仍留一圈奶白
    band 0.15 / thr 0.20  →  删 2.06%，顶面干净，侧面木纹完好   ← 推荐
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402


def normalize_chroma(colors: np.ndarray) -> np.ndarray:
    """(R - B) / max(R, G, B)，与亮度无关的色度。"""
    return (colors[:, 0] - colors[:, 2]) / np.maximum(colors.max(axis=1), 1e-6)


def drop_foreign_by_color(
    cloud: o3d.geometry.PointCloud, band_depth: float, threshold: float
) -> o3d.geometry.PointCloud:
    point_type = o3d.geometry.PointCloud
    if not isinstance(cloud, point_type) or not cloud.has_colors():
        raise ValueError("点云必须带颜色（COLMAP 的 fused.ply 有）")
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    top = np.percentile(points[:, 1], 99.5)          # 顶面高度
    chroma = normalize_chroma(colors)

    band = points[:, 1] > top - band_depth
    drop = band & (chroma < threshold)
    print(f"顶面 Y={top:+.4f}   带深 {band_depth}   阈值 {threshold}")
    print(f"  带内 {int(band.sum()):,} 点，删 {int(drop.sum()):,} "
          f"（带内 {drop.sum() / max(band.sum(), 1) * 100:.1f}%，"
          f"全云 {drop.mean() * 100:.2f}%）")
    if drop.sum():
        print(f"  被删点平均色 RGB {np.round(colors[drop].mean(axis=0) * 255).astype(int)}"
              f"   中位归一化色度 {np.median(chroma[drop]):.3f}")
        print(f"  保留点中位归一化色度 {np.median(chroma[~drop]):.3f}（木色参考 0.25）")

    keep = ~drop
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
    result.colors = o3d.utility.Vector3dVector(colors[keep])
    if cloud.has_normals():
        result.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[keep])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入点云 PLY")
    parser.add_argument("output", help="输出点云 PLY")
    parser.add_argument("--band", type=float, default=0.15, help="只清理顶面往下这么深的范围")
    parser.add_argument("--threshold", type=float, default=0.20, help="归一化色度低于它就删")
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(args.input)
    print(f"读取 {args.input}：{len(cloud.points):,} 点")
    cleaned = drop_foreign_by_color(cloud, args.band, args.threshold)
    target = Path(args.output)
    o3d.io.write_point_cloud(str(target), cleaned, write_ascii=False)
    print(f"-> {target.name}  {len(cleaned.points):,} 点  "
          f"{target.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
