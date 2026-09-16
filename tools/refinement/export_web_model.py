#!/usr/bin/env python3
"""为静态展示页（docs/）导出一份网页版模型：对已批准基准做四边形简并。

为什么用简并而不是重新跑 finish_mesh
------------------------------------
基准已经过「贯穿圆孔保护 + 外平面平滑」，那是**网格级编辑**；重新从点云跑一遍
Poisson 会把这些编辑丢掉。简并只减三角面，几何形状与这些编辑都保留。

为什么不用 Open3D 原生写 GLB
---------------------------
Open3D 原生写出不合规，项目统一用 ``app/convert.py`` 的 ``write_glb``。

用法
----
    ./.venv/bin/python tools/refinement/export_web_model.py \\
        --input storage/<...>/merged_six_face_feature_protected_planes.glb \\
        --output docs/models/cube-approved-web.glb \\
        --triangles 1500000

⚠️ GitHub 单文件硬上限 **100 MB**，脚本会在超过时提醒；此时把 --triangles 调小。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from app.convert import transfer_colors, write_glb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出静态展示页用的轻量 GLB")
    parser.add_argument("--input", required=True, help="全量 GLB（建议用已批准基准）")
    parser.add_argument("--output", required=True, help="输出的网页版 GLB")
    parser.add_argument("--triangles", type=int, default=1_500_000,
                        help="目标三角面数；150 万约 36 MB，100 万约 24 MB")
    parser.add_argument("--color-samples", type=int, default=4_000_000,
                        help="简并丢掉顶点色时，从原网格采样这么多点做颜色转移")
    args = parser.parse_args()

    source = Path(args.input)
    mesh = o3d.io.read_triangle_mesh(str(source))
    if not len(mesh.triangles):
        raise SystemExit(f"空网格：{source}")
    print(f"原模型：{len(mesh.vertices):,} 顶点 / {len(mesh.triangles):,} 三角面  "
          f"{source.stat().st_size / 2**20:.1f} MB")

    if len(mesh.triangles) <= args.triangles:
        print(f"三角面已不多于目标 {args.triangles:,}，直接导出")
        small = mesh
    else:
        small = mesh.simplify_quadric_decimation(target_number_of_triangles=args.triangles)
        print(f"简并后：{len(small.vertices):,} 顶点 / {len(small.triangles):,} 三角面")

    colors = np.asarray(small.vertex_colors) if small.has_vertex_colors() else np.zeros((0, 3))
    if colors.size == 0 or not colors.any():
        print("  简并结果无顶点色 → 从原网格采样转移")
        transfer_colors(mesh.sample_points_uniformly(number_of_points=args.color_samples), small)
    else:
        print("  简并保留了顶点色")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_glb(small, output)
    size_mb = output.stat().st_size / 2**20
    print(f"写出 {output}  {size_mb:.1f} MB")
    if size_mb > 100:
        print("⚠️ 超过 GitHub 单文件 100 MB 上限：把 --triangles 调小后重跑")


if __name__ == "__main__":
    main()
