from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def _log(message: str) -> None:
    """网格重建是长时间无输出的黑盒，打点日志方便判断卡在哪一步。"""
    print(f"[mesh] {message}", file=sys.stderr, flush=True)

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UBYTE = 5121
_USHORT = 5123
_UINT = 5125


def _pad(data: bytes, fill: bytes) -> bytes:
    remainder = len(data) % 4
    if remainder:
        data += fill * (4 - remainder)
    return data


def write_glb(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None:
    """Write a triangle mesh as a standards-compliant GLB with a BIN chunk."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        raise RuntimeError("网格为空，无法导出 GLB")

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    index_dtype = np.uint16 if len(vertices) < 65536 else np.uint32
    indices = triangles.reshape(-1).astype(index_dtype)

    parts: list[bytes] = []
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, object]] = []

    def add_view(raw: bytes, target: int) -> int:
        parts.append(_pad(raw, b"\x00"))
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": sum(len(part) for part in parts[:-1]),
                "byteLength": len(raw),
                "target": target,
            }
        )
        return len(buffer_views) - 1

    def add_accessor(
        view: int,
        component_type: int,
        count: int,
        type_name: str,
        extra: dict[str, object] | None = None,
    ) -> int:
        accessor: dict[str, object] = {
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if extra:
            accessor.update(extra)
        accessors.append(accessor)
        return len(accessors) - 1

    position_view = add_view(vertices.tobytes(), _ARRAY_BUFFER)
    attributes = {
        "POSITION": add_accessor(
            position_view,
            _FLOAT,
            len(vertices),
            "VEC3",
            {
                "min": [float(value) for value in vertices.min(axis=0)],
                "max": [float(value) for value in vertices.max(axis=0)],
            },
        )
    }
    normal_view = add_view(normals.tobytes(), _ARRAY_BUFFER)
    attributes["NORMAL"] = add_accessor(normal_view, _FLOAT, len(normals), "VEC3")

    if colors is not None and len(colors) == len(vertices):
        color_bytes = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
        color_view = add_view(color_bytes.tobytes(), _ARRAY_BUFFER)
        attributes["COLOR_0"] = add_accessor(
            color_view, _UBYTE, len(colors), "VEC3", {"normalized": True}
        )

    index_view = add_view(indices.tobytes(), _ELEMENT_ARRAY_BUFFER)
    index_accessor = add_accessor(
        index_view, _USHORT if index_dtype is np.uint16 else _UINT, len(indices), "SCALAR"
    )

    binary = b"".join(parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "video-to-3d-model"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index_accessor, "mode": 4}]}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_bytes = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_bytes), b"JSON"))
        handle.write(json_bytes)
        handle.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        handle.write(binary)


def remove_sparse_outliers(
    cloud: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> o3d.geometry.PointCloud:
    """统计离群点移除（SOR）：删掉邻域平均距离明显偏大的稀疏点。

    分位数裁剪只能削掉盒子外面的点，盒子内部悬浮的稀疏点得靠 SOR。
    这些点正是 Poisson 在主体周围长出小碎块的原料。
    """
    if std_ratio <= 0 or len(cloud.points) <= nb_neighbors:
        return cloud
    filtered, keep = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    if len(keep) < 100:
        return cloud
    if len(filtered.points) != len(cloud.points):
        _log(
            f"SOR 移除稀疏离群点（{nb_neighbors} 邻域，std_ratio={std_ratio}）："
            f"{len(cloud.points):,} -> {len(filtered.points):,}"
        )
    return filtered


def keep_significant_components(mesh: o3d.geometry.TriangleMesh, min_ratio: float) -> o3d.geometry.TriangleMesh:
    """只保留足够大的连通分量。

    Poisson 总会在点密度不足的地方（背景、边缘、噪声）长出互不连通的小块，
    渲染出来就是主体周围的散点。按“不小于最大分量的 min_ratio”过滤，
    既能去掉碎块，又不会误删主体上真正独立的部分。
    """
    if min_ratio <= 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    keep = np.flatnonzero(counts >= counts.max() * min_ratio)
    before = int(counts.sum())
    mesh.remove_triangles_by_mask(~np.isin(labels, keep))
    mesh.remove_unreferenced_vertices()
    _log(
        f"连通分量共 {len(counts)} 个（最大 {int(counts.max()):,} 三角面），"
        f"保留 {len(keep)} 个不小于最大分量 {min_ratio:.0%} 的分量；"
        f"三角面 {before:,} -> {len(mesh.triangles):,}"
    )
    return mesh


def remove_dominant_planes(
    cloud: o3d.geometry.PointCloud,
    distance_threshold: float,
    max_planes: int = 1,
    min_inlier_ratio: float = 0.05,
) -> o3d.geometry.PointCloud:
    """用 RANSAC 检测并移除最大的几个平面（桌面、地面等支撑面）。

    桌面这类大面积平面常被 Poisson 重建成一层边缘撕裂的薄片，
    在主体周围形成一堆散点感。它往往与主体相连，所以连通分量过滤不了，
    只能在点云阶段就拿掉。

    ``distance_threshold`` 设 0 关闭。阈值应取点间距的几倍。
    注意：主体本身也有大平面，所以 ``max_planes`` 默认只去 1 个，
    判据是「内含点不得少于总数的 min_inlier_ratio」。
    """
    if distance_threshold <= 0 or len(cloud.points) < 100:
        return cloud
    working = cloud
    for index in range(max_planes):
        total = len(working.points)
        if total < 100:
            break
        try:
            plane, inliers = working.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception as exc:  # pragma: no cover - Open3D 在退化输入上会抛
            _log(f"平面检测失败，跳过去除：{exc}")
            break
        if len(inliers) < total * min_inlier_ratio:
            _log(f"平面 {index + 1} 只含 {len(inliers):,}/{total:,} 点，判为非支撑面，不处理")
            break
        working = working.select_by_index(inliers, invert=True)
        _log(
            f"移除支撑平面 {index + 1}：法向 {np.round(plane[:3], 3)} 偏移 {plane[3]:.3f}，"
            f"{total:,} -> {len(working.points):,} 点"
        )
    return working


def transfer_colors(cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh) -> None:
    """Assign each mesh vertex the color of its nearest point cloud point.

    用 ``scipy.spatial.cKDTree`` 批量查询。Open3D 的 ``KDTreeFlann`` 只能逐点查询，
    对百万级网格顶点就是纯 Python 循环，会慢到不可接受；批量查询走 C 且可多线程。
    """
    if not cloud.has_colors():
        return
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return
    cloud_points = np.asarray(cloud.points)
    cloud_colors = np.asarray(cloud.colors)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None and len(cloud_points) > 0:
        _, nearest = cKDTree(cloud_points).query(vertices, k=1, workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(cloud_colors[nearest])
        return
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    colors = np.zeros((len(vertices), 3))
    for index, point in enumerate(vertices):
        _, neighbors, _ = kdtree.search_knn_vector_3d(point, 1)
        colors[index] = cloud_colors[neighbors[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def estimate_point_spacing(points: np.ndarray, sample: int = 50_000, seed: int = 0) -> float:
    """用抽样点的最近邻距离中位数估计点间距。

    不要用 ``(体积 / 点数) ** (1/3)``：摄影测量的点云是**表面**分布而非填满体积，
    再有少量离群点撑大包围盒，体积法会把间距高估一到两个数量级
    （实测高估 72 倍：0.0792 vs 真实 0.0011），进而把 Poisson 深度选得过低、
    整个模型被抹成没有细节的糊块。
    """
    if len(points) < 2:
        return 0.0
    count = min(sample, len(points))
    if count < len(points):
        index = np.random.default_rng(seed).choice(len(points), count, replace=False)
    else:
        index = np.arange(len(points))
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distances, _ = cKDTree(points).query(points[index], k=2, workers=-1)
        return float(np.median(distances[:, 1]))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    nearest = []
    for point in points[index]:
        _, neighbours, distance = kdtree.search_knn_vector_3d(point, 2)
        if len(neighbours) > 1:
            nearest.append(math.sqrt(distance[1]))
    return float(np.median(nearest)) if nearest else 0.0


def crop_outliers(cloud: o3d.geometry.PointCloud, percentile: float) -> o3d.geometry.PointCloud:
    """按各轴分位数裁掉离群点。

    COLMAP 的 ``fused.ply`` 里总有一小部分深度估计失误的点飞到很远，
    把包围盒撑大十几倍。Poisson 的叶节点尺寸 = 包围盒最长边 / 2^depth，
    包围盒虚大就等于分辨率白降，所以建网格前必须先裁。
    """
    if percentile <= 0:
        return cloud
    points = np.asarray(cloud.points)
    if len(points) < 100:
        return cloud
    low = np.percentile(points, percentile, axis=0)
    high = np.percentile(points, 100 - percentile, axis=0)
    keep = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
    if len(keep) < 100 or len(keep) == len(points):
        return cloud
    selected = cloud.select_by_index(keep)
    _log(
        f"裁掉 {percentile:g}% 分位以外的离群点：{len(points):,} -> {len(keep):,} 个点，"
        f"包围盒 {np.round(cloud.get_max_bound() - cloud.get_min_bound(), 3)} -> "
        f"{np.round(selected.get_max_bound() - selected.get_min_bound(), 3)}"
    )
    return selected


def suggest_poisson_depth(
    cloud: o3d.geometry.PointCloud, spacing: float, max_depth: int = 9
) -> int:
    """按点云真实采样密度推荐 Poisson 八叉树深度。

    Poisson 的叶节点尺寸 = 包围盒最长边 / 2^depth。若它明显小于平均点间距，
    分辨率的提高纯属浪费：既拿不到数据里不存在的细节，又会让内存成倍增长
    （depth 每 +1，体素数约 ×8）。所以让两者匹配即可。
    """
    bounds = cloud.get_max_bound() - cloud.get_min_bound()
    max_extent = float(np.max(bounds))
    if max_extent <= 0 or spacing <= 0:
        return min(8, max_depth)
    ideal = int(math.floor(math.log2(max_extent / spacing)))
    depth = int(min(max(ideal, 6), max_depth))
    if ideal > max_depth:
        _log(
            f"点间距支持 depth={ideal}，但受内存上限限制取 {max_depth}"
            f"（叶节点会从 {max_extent / 2 ** ideal:.4f} 粗到 {max_extent / 2 ** max_depth:.4f}）"
        )
    return depth


def point_cloud_to_glb(
    point_cloud_path: Path,
    output_path: Path,
    voxel_size: float = 0.0,
    poisson_depth: int = 0,
    orient_normals_max_points: int = 5_000_000,
    outlier_percentile: float = 1.0,
    poisson_max_depth: int = 9,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    min_component_ratio: float = 0.02,
    density_quantile: float = 0.02,
    plane_max: int = 0,
    plane_distance_scale: float = 4.0,
) -> None:
    """Convert a COLMAP PLY point cloud into a triangle mesh GLB.

    ``poisson_depth`` 为 0 表示按采样密度自动选择（推荐）。
    ``outlier_percentile`` 先按各轴分位数裁掉离群点，否则少数飞点会撑大包围盒、
    把 Poisson 的分辨率拖垮（见 :func:`crop_outliers`）。
    ``sor_neighbors`` / ``sor_std_ratio`` 再用 SOR 清掉盒内悬浮的稀疏点
    （``sor_std_ratio<=0`` 关闭）；``min_component_ratio`` 最后按连通分量过滤碎块
    （``0`` 关闭），这两步共同决定主体周围是否还有散点。
    ``orient_normals_max_points`` 之上的点云会跳过全局法线定向：
    ``orient_normals_consistent_tangent_plane`` 要为每个点建 30 邻域图，
    在千万级点云上是主要的内存开销之一，而 COLMAP 的 ``fused.ply`` 本身已带法线。
    """
    started = time.monotonic()
    cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    if cloud.is_empty():
        raise RuntimeError("PLY 点云为空，无法转换为 GLB")
    _log(f"读取点云 {len(cloud.points):,} 个点，耗时 {time.monotonic() - started:.1f}s")

    cloud = crop_outliers(cloud, outlier_percentile)
    mark = time.monotonic()
    cloud = remove_sparse_outliers(cloud, sor_neighbors, sor_std_ratio)
    if sor_std_ratio > 0:
        _log(f"离群点清理耗时 {time.monotonic() - mark:.1f}s")
    spacing = estimate_point_spacing(np.asarray(cloud.points))
    _log(f"估计点间距 {spacing:.4f}")
    if plane_max > 0:
        mark = time.monotonic()
        cloud = remove_dominant_planes(
            cloud,
            distance_threshold=spacing * plane_distance_scale,
            max_planes=plane_max,
        )
        _log(f"支撑平面处理耗时 {time.monotonic() - mark:.1f}s")
    if len(cloud.points) < 100:
        raise RuntimeError("去掉支撑平面后点云过少，无法重建网格；可调小 poisson_remove_plane 或检查点云")
    bounds = cloud.get_max_bound() - cloud.get_min_bound()
    max_extent = float(np.max(bounds))
    _log(f"包围盒最长边 {max_extent:.4f}")

    if poisson_depth <= 0:
        poisson_depth = suggest_poisson_depth(cloud, spacing, max_depth=poisson_max_depth)
        _log(
            f"选择 Poisson 深度 {poisson_depth}（叶节点 {max_extent / 2 ** poisson_depth:.4f}，"
            f"点间距 {spacing:.4f}）"
        )

    if voxel_size > 0:
        mark = time.monotonic()
        cloud = cloud.voxel_down_sample(voxel_size)
        _log(f"体素降采样到 {len(cloud.points):,} 个点（voxel_size={voxel_size}），耗时 {time.monotonic() - mark:.1f}s")
    if len(cloud.points) < 100:
        raise RuntimeError("点云数量过少，无法稳定重建网格")

    if not cloud.has_normals():
        mark = time.monotonic()
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size * 4, 0.05),
                max_nn=30,
            )
        )
        _log(f"估计法线，耗时 {time.monotonic() - mark:.1f}s")
    else:
        _log("点云自带法线（COLMAP 输出的 fused.ply 通常已包含），跳过法线估计")

    if len(cloud.points) <= orient_normals_max_points:
        mark = time.monotonic()
        cloud.orient_normals_consistent_tangent_plane(30)
        _log(f"统一法线朝向，耗时 {time.monotonic() - mark:.1f}s")
    else:
        _log(
            f"点云 {len(cloud.points):,} 个点超过 orient_normals_max_points="
            f"{orient_normals_max_points:,}，跳过全局法线定向以控制内存"
        )

    mark = time.monotonic()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=poisson_depth,
    )
    _log(f"Poisson 重建完成，{len(mesh.vertices):,} 顶点 / {len(mesh.triangles):,} 三角面，耗时 {time.monotonic() - mark:.1f}s")

    mark = time.monotonic()
    density_values = np.asarray(densities)
    density_threshold = float(np.quantile(density_values, density_quantile))
    mesh.remove_vertices_by_mask(density_values < density_threshold)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    _log(
        f"剔除最低 {density_quantile:.0%} 密度的顶点后 {len(mesh.vertices):,} 顶点，"
        f"耗时 {time.monotonic() - mark:.1f}s"
    )

    mark = time.monotonic()
    mesh = keep_significant_components(mesh, min_component_ratio)
    _log(f"连通分量过滤耗时 {time.monotonic() - mark:.1f}s")
    mesh.compute_vertex_normals()

    mark = time.monotonic()
    transfer_colors(cloud, mesh)
    _log(f"颜色传递完成，耗时 {time.monotonic() - mark:.1f}s")

    mark = time.monotonic()
    write_glb(mesh, output_path)
    _log(f"写出 GLB（{output_path.stat().st_size / 1024 / 1024:.1f} MB），耗时 {time.monotonic() - mark:.1f}s")
    _log(f"网格重建总耗时 {time.monotonic() - started:.1f}s")
