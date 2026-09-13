#!/usr/bin/env python3
"""把一个点云收尾成成品网格：Poisson → 补洞 → 轻度平滑 → 重算法线 → GLB。

为什么要有"补洞"这一步
--------------------
Poisson 在法线朝向不一致 / 点密度不均的地方会长出**不封闭**的表面（实测某次
39.9 M 三角面的网格留下 520 个开口、33,698 条边界边）。这些开口会让模型看起来
有撕裂缺口。这里按边界环逐个补扇形三角面：

- 补的面必须**反向遍历**共用边（原三角形用 a→b，补的面用 b→a），否则法线会翻
- **补完必须重算顶点法线** —— 只动拓扑不算法线会让整块模型明暗全错（踩过）

为什么不补会更好 / 哪些开口不该补
------------------------------
本例物体（木块 + 半圆凹槽 + 两个圆孔）的点云中，真实开口（凹槽、圆孔的洞）
在 Poisson 输出里**本来就是封闭的**（Poisson 会给它蒙一层皮），所以留下的边界环
基本都是缺陷。若换到其他物体上发现补完把真实特征堵住了，用 --max-hole-edges
限制只补小洞。

为什么平滑要克制
--------------
两段扫描之间约 0.0056 的残余错位是数据本身的精度上限；Poisson 叶节点比它还细时，
会把这两层之间的缝"雕"成薄片毛刺（实测深度 10 叶节点 0.0015 就出现这种情况）。
所以：要么叶节点别小于错位（用更低的 Poisson 深度），要么事后轻度平滑。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.convert import point_cloud_to_glb, write_glb  # noqa: E402


def count_boundary_edges(mesh: o3d.geometry.TriangleMesh) -> int:
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return 0
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def fill_boundary_holes(
    mesh: o3d.geometry.TriangleMesh, max_hole_edges: int = 0
) -> o3d.geometry.TriangleMesh:
    """补上所有边界开口（``max_hole_edges>0`` 时只补边数不超过它的开口）。"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    directed = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    sorted_edges = np.sort(directed, axis=1)
    _, inverse = np.unique(sorted_edges, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    is_border = counts[inverse] == 1
    border_directed = directed[is_border]
    border_sorted = sorted_edges[is_border]
    if len(border_directed) == 0:
        print("[finish] 没有边界边，网格已封闭")
        return mesh

    nodes: dict[int, int] = {}
    for pair in border_sorted:
        for value in pair:
            nodes.setdefault(int(value), len(nodes))
    rows = [nodes[int(a)] for a, b in border_sorted]
    cols = [nodes[int(b)] for a, b in border_sorted]
    graph = coo_matrix(
        (np.ones(len(border_sorted)), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    loop_count, loop_labels = connected_components(graph, directed=False)
    loop_sizes = np.bincount(loop_labels)

    new_vertices: list[np.ndarray] = []
    new_triangles: list[tuple[int, int, int]] = []
    center_of_loop: dict[int, int] = {}
    skipped = 0
    for a, b in border_directed:
        loop = loop_labels[nodes[int(a)]]
        if max_hole_edges > 0 and loop_sizes[loop] > max_hole_edges:
            skipped += 1
            continue
        if loop not in center_of_loop:
            members = np.array([v for v, group in nodes.items() if loop_labels[group] == loop])
            center_of_loop[loop] = len(vertices) + len(new_vertices)
            new_vertices.append(vertices[members].mean(axis=0))
        # 原三角形用 a→b，补的面必须用 b→a
        new_triangles.append((center_of_loop[loop], int(b), int(a)))

    print(
        f"[finish] 边界边 {len(border_directed):,}  开口 {loop_count} 个  "
        f"补上 {len(new_triangles):,} 个三角面"
        + (f"（跳过 {skipped:,} 条边上的大开口）" if skipped else "")
    )
    if not new_triangles:
        return mesh

    all_vertices = np.vstack([vertices, np.array(new_vertices)])
    all_triangles = np.vstack([triangles, np.array(new_triangles)])
    repaired = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(all_vertices),
        o3d.utility.Vector3iVector(all_triangles),
    )
    if colors is not None:
        padded = np.vstack([colors, np.tile(colors.mean(axis=0), (len(new_vertices), 1))])
        repaired.vertex_colors = o3d.utility.Vector3dVector(padded)
    return repaired


def clip_mesh_to_box(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    tolerance: float,
) -> o3d.geometry.TriangleMesh:
    """把伸出「点云 0.5%/99.5% 分位包围盒」超过 tolerance 的三角形删掉。

    物体已知是立方体时，两段扫描在棱/角处各留一层表面，Poisson 会沿棱起皱长出
    毛屑；这些毛屑大多伸到盒外 —— 直接按盒裁掉，比想办法在点云里分开两层简单得多。
    切出的小口交给后面的补洞步骤。

    ⚠ 裁完必须重算法线（跟补洞一样），否则明暗会错。
    """
    low = np.percentile(points, 0.5, axis=0) - tolerance
    high = np.percentile(points, 99.5, axis=0) + tolerance
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    inside = np.all((vertices >= low) & (vertices <= high), axis=1)
    keep = inside[triangles].all(axis=1)

    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles[keep])
    )
    if mesh.has_vertex_colors():
        result.vertex_colors = mesh.vertex_colors
    result.remove_unreferenced_vertices()
    print(
        f"[finish] 盒裁（容差 {tolerance:.4f}）：盒外顶点 {int((~inside).sum()):,} 个，"
        f"删掉 {len(triangles) - len(result.triangles):,} 三角面，"
        f"剩 {len(result.vertices):,} 顶点 / {len(result.triangles):,} 三角面"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入点云 PLY")
    parser.add_argument("--output", required=True, help="输出成品 GLB")
    parser.add_argument("--depth", type=int, default=10, help="Poisson 深度上限")
    parser.add_argument("--density-quantile", type=float, default=0.005)
    parser.add_argument("--smooth-iterations", type=int, default=3, help="Taubin 迭代次数，0=不平滑")
    parser.add_argument("--max-hole-edges", type=int, default=0, help="只补不超过这么多条边的开口，0=全补")
    parser.add_argument(
        "--pre-voxel", type=float, default=0.0,
        help="先按这个体素边长下采样（体素内取平均）来降噪，0=不降噪。"
             "实测两段扫描残余错位约 0.0056，是数据精度上限；体素平均能把随机噪声压下去，"
             "但体素小于错位时两层表面会糊成一条带，所以取 0.002~0.004 比较合适。"
             "⚠ 实测这条路对本项目更差（异物被糊成更大的整块），保留但别用。",
    )
    parser.add_argument(
        "--clip-tol", type=float, default=0.0,
        help="按输入点云的立方体包围盒裁网格：顶点伸出盒外超过该容差的三角形直接删掉，"
             "随后由补洞步骤把切出来的小口补平。0=不裁。\n"
             "用途：物体已知是立方体时，两段扫描在棱/角处各留一层表面，Poisson 会沿棱起"
             "皱长出毛屑，这些毛屑大多伸到盒外 —— 裁掉就能让棱变干净。",
    )
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    source = Path(args.input)
    target = Path(args.output)
    rough = target.with_suffix(".rough.glb")

    print(f"[finish] Poisson（深度上限 {args.depth}）...")
    if args.pre_voxel > 0:
        raw = o3d.io.read_point_cloud(str(source))
        before = len(raw.points)
        smoothed = raw.voxel_down_sample(args.pre_voxel)
        print(f"[finish] 体素降噪 {args.pre_voxel:.4f}: {before:,} -> {len(smoothed.points):,} 点")
        source = target.with_suffix(".pre.ply")
        o3d.io.write_point_cloud(str(source), smoothed, write_ascii=False)
        del raw, smoothed

    point_cloud_to_glb(
        source,
        rough,
        outlier_percentile=0.0,
        sor_neighbors=0,
        sor_std_ratio=0.0,
        min_component_ratio=0.01,
        density_quantile=args.density_quantile,
        object_warmth=-1.0,
        poisson_max_depth=args.depth,
        orient_normals_max_points=5_000_000,
    )

    mesh = o3d.io.read_triangle_mesh(str(rough))
    print(f"[finish] Poisson 输出 {len(mesh.vertices):,} 顶点 / {len(mesh.triangles):,} 三角面  "
          f"边界边 {count_boundary_edges(mesh):,}")

    if args.clip_tol > 0:
        box_points = np.asarray(o3d.io.read_point_cloud(str(source)).points)
        mesh = clip_mesh_to_box(mesh, box_points, args.clip_tol)
        del box_points

    mesh = fill_boundary_holes(mesh, args.max_hole_edges)
    print(f"[finish] 补洞后 边界边 {count_boundary_edges(mesh):,}")

    if args.smooth_iterations > 0:
        mark = time.monotonic()
        colors_before = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
        mesh.filter_smooth_taubin(number_of_iterations=args.smooth_iterations)
        if colors_before is not None:
            changed = not np.allclose(colors_before, np.asarray(mesh.vertex_colors))
            print(f"[finish] 平滑后顶点颜色是否被改动: {changed}")
        print(f"[finish] Taubin x{args.smooth_iterations} 耗时 {time.monotonic() - mark:.0f}s")

    mesh.compute_vertex_normals()          # 平滑后必须重算，否则明暗全错
    write_glb(mesh, target)
    print(f"[finish] -> {target.name}  {target.stat().st_size / 1024 / 1024:.0f} MB  "
          f"总耗时 {time.monotonic() - started:.0f}s")
    if not args.keep_temp:
        rough.unlink(missing_ok=True)
        if args.pre_voxel > 0:
            target.with_suffix(".pre.ply").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
