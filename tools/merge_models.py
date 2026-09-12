#!/usr/bin/env python3
"""把两个稀疏模型里的被摄物合并成一个完整模型。

背景：一段视频里物体被拿起翻转过一次（第 670~673 帧），COLMAP 因此把它拆成两个
互不连通的稀疏模型（sparse/0 帧 1~670，sparse/1 帧 673~1335）。两段各自重建出
物体的一部分：A 段有顶面和 4 个侧面，B 段有底面和部分侧面。要合成完整物体必须
把两份点云配准后合并。

三个难点与对策
--------------
1. **没有公共坐标系，尺度不可观测** —— 每个 COLMAP 模型有独立 gauge，是 7 自由度
   相似变换。对策：各自按「到质心的中位距离」归一化（旋转不变），把尺度拉到 1，
   再用支持缩放的 ICP 精修。

2. **尺度精度要求苛刻** —— 点间距 0.0011、物体 2.5，1% 尺度误差 = 23 倍点间距的错位，
   目视就有明显接缝。要做到无缝，尺度需精确到 0.05~0.1%。

3. **箱体对称性歧义** —— 顶面和底面同尺寸，可能解出「上下颠倒」的错误对齐，
   而且 RMSE 也不会差太多。对策：多随机种子跑 RANSAC 收集候选，各自 ICP 精修，
   按「拟合度 + 重叠区颜色一致性」排序，最后目视确认。

用法：
    # 在稀疏点云上预演（快，用于验证算法）
    ./.venv/bin/python tools/merge_models.py --source sparse

    # 稠密点云（等 tools/build_dense_model.py --model 1 跑完）
    ./.venv/bin/python tools/merge_models.py --source dense
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.convert import crop_outliers, keep_object_by_color  # noqa: E402

DEFAULT_JOB = "e7f9f9e5d046490d93b34e4538bc9ef1"


# ---------------------------------------------------------------- 读取

def read_sparse_cameras(path: Path) -> np.ndarray:
    """读 images.bin 的旋转部分，返回 (N, 3, 3) 的「相机→世界」旋转矩阵。

    格式：image_id(I) + qvec(4d) + tvec(3d) + camera_id(I) + name(以 \\0 结尾)
    + num_points2D(Q) + 每个观测 (x d, y d, point3D_id Q) = **24 字节**。
    本机 CUDA 版 colmap 的 model_converter 会崩，所以自己解析。
    """
    data = path.read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    rotations = np.empty((count, 3, 3))
    for index in range(count):
        offset += 4
        quaternion = np.frombuffer(data, dtype=np.float64, count=4, offset=offset)
        offset += 32
        offset += 24  # tvec
        offset += 4   # camera_id
        end = data.index(b"\x00", offset)
        offset = end + 1
        (point_count,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + point_count * 24
        rotations[index] = quaternion_to_rotation(quaternion)
    return rotations


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """COLMAP 的四元数顺序是 (w, x, y, z)。"""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def gravity_up(images_path: Path) -> np.ndarray:
    """从相机姿态估计世界的「上」方向。

    COLMAP 的相机坐标是 **x 右、y 下、z 前**，所以相机自身的「上」是 (0,-1,0)；
    转到世界系就是 ``R @ (0,-1,0)``。

    **实测对这段素材不可靠**：手持环绕 + 翻动物体时相机自身倾斜很大，
    667 帧的「上」方向与平均方向夹角中位数达 57°，平均出来几乎没有意义。
    只在无法用桌面法向时兜底。
    """
    rotations = read_sparse_cameras(images_path)
    ups = rotations @ np.array([0.0, -1.0, 0.0])
    mean = ups.mean(axis=0)
    return mean / np.linalg.norm(mean)


def table_up(
    cloud: o3d.geometry.PointCloud, object_center: np.ndarray, object_extent: np.ndarray,
    warmth: float = 0.08, radius_scale: float = 3.0,
) -> tuple[np.ndarray, float]:
    """用桌面法向当「上」方向，返回 (单位法向, 桌面内点占比)。

    木块平放在桌上，底面必然与桌面平行，所以桌面法向就是重力的反方向。
    这比相机姿态可靠得多（见 :func:`gravity_up` 的说明）。
    ``cloud`` 用**未过滤的全景点云**（含物体和桌面），内部自己挑中性色的点。
    """
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    neutral = (colors[:, 0] - colors[:, 2]) < warmth
    # 只保留物体附近的桌面，远处墙面/天花板会把 RANSAC 引偏
    near = np.linalg.norm(points - object_center, axis=1) < float(np.max(object_extent)) * radius_scale
    selected = points[neutral & near]
    if len(selected) < 200:
        raise SystemExit("桌面点太少，无法拟合平面")
    plane_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
    plane, inliers = plane_cloud.segment_plane(
        distance_threshold=float(np.min(object_extent)) * 0.01,
        ransac_n=3,
        num_iterations=2000,
    )
    normal = np.array(plane[:3])
    normal /= np.linalg.norm(normal)
    # 法向朝上（指向物体所在的一侧）
    if float(np.dot(normal, object_center)) + plane[3] < 0:
        normal = -normal
    return normal, len(inliers) / len(selected)


def skew(vector: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """求把单位向量 source 转到 target 的旋转矩阵（罗德里格斯公式）。"""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if cosine > 1 - 1e-9:
        return np.eye(3)
    if cosine < -1 + 1e-9:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        rotation = skew(axis / np.linalg.norm(axis))
        return np.eye(3) + 2.0 * rotation @ rotation
    rotation = skew(cross)
    return np.eye(3) + rotation + rotation @ rotation * (1.0 / (1.0 + cosine))


def stand_upright(cloud: o3d.geometry.PointCloud, up: np.ndarray) -> o3d.geometry.PointCloud:
    """摆正并居中：重力反方向 → +Y，物体最长水平方向 → +X，几何中心 → 原点。"""
    gravity = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    aligned = (gravity @ np.asarray(cloud.points).T).T
    matrix = horizontal_spin(aligned) @ gravity
    cloud.rotate(matrix, center=(0.0, 0.0, 0.0))
    cloud.translate(-np.asarray(cloud.get_center()))
    return cloud


def horizontal_spin(aligned: np.ndarray) -> np.ndarray:
    """绕 +Y 旋转，把点云在水平面上的最长方向对齐 +X（只影响观感，不影响几何）。"""
    projected = aligned[:, [0, 2]]
    _, _, vectors = np.linalg.svd(projected - projected.mean(axis=0), full_matrices=False)
    angle = float(np.arctan2(vectors[0, 1], vectors[0, 0]))
    return np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])


def read_sparse_points(path: Path) -> o3d.geometry.PointCloud:
    """读 points3D.bin（含 RGB）。CUDA 版 colmap 的 model_converter 在本机会崩，自己解析。"""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    xyz = np.empty((count, 3), dtype=np.float64)
    rgb = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        offset += 8
        xyz[i] = struct.unpack_from("<3d", data, offset)
        offset += 24
        rgb[i] = np.frombuffer(data, dtype=np.uint8, count=3, offset=offset) / 255.0
        offset += 3 + 8
        (track,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + track * 8
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def load_full(job: Path, model: int, source: str) -> o3d.geometry.PointCloud:
    """读**未做颜色过滤**的完整点云（物体 + 背景），用于拟合桌面。"""
    if source == "sparse":
        return read_sparse_points(job / "sparse" / str(model) / "points3D.bin")
    name = "fused.ply" if model == 0 else f"fused_{model}.ply"
    return o3d.io.read_point_cloud(str(job / name))


def load_object(job: Path, model: int, source: str, warmth: float, cached: Path | None) -> o3d.geometry.PointCloud:
    if source == "sparse":
        cloud = read_sparse_points(job / "sparse" / str(model) / "points3D.bin")
    else:
        if cached is not None and cached.exists():
            cloud = o3d.io.read_point_cloud(str(cached))
        else:
            name = "fused.ply" if model == 0 else f"fused_{model}.ply"
            cloud = o3d.io.read_point_cloud(str(job / name))
    print(f"  模型{model}原始点云 {len(cloud.points):,} 点（{source}）")
    if warmth >= 0 and cloud.has_colors():
        cloud = keep_object_by_color(cloud, warmth)
    # 颜色过滤只能去掉中性色的背景；远处仍有暖色杂点（砖墙、木窗框等），
    # 它们会把包围盒捶到好几倍、把 FPFH 特征和归一化尺度全带偏，必须先裁。
    cloud = crop_outliers(cloud, 1.0)
    if len(cloud.points):
        size = cloud.get_max_bound() - cloud.get_min_bound()
        print(f"  裁剪后 {len(cloud.points):,} 点  尺寸 {np.round(size, 2)}")
    return cloud


# ---------------------------------------------------------------- 配准

def normalize(cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, float, np.ndarray]:
    """按「到质心的中位距离」归一化，返回 (中心, 尺度, 点)。中位距离对离群点稳健且旋转不变。"""
    points = np.asarray(cloud.points)
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale = float(np.median(distances))
    return center, scale, (points - center) / scale


def principal_axes(points: np.ndarray) -> np.ndarray:
    """点云的主轴（按特征值降序）。箱体类物体的主轴与它的面法向基本一致。"""
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    return vectors[:, np.argsort(values)[::-1]]


def cube_symmetries() -> list[np.ndarray]:
    """箱体的 24 个固有旋转（每行每列恰有一个 ±1，行列式 +1）。"""
    from itertools import permutations, product

    matrices = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, col in enumerate(perm):
                matrix[row, col] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """两个旋转之间的夹角（度）。"""
    delta = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """SO(3) 上的均匀随机旋转（四元数法）。"""
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_initial_transform(rotation: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """给定旋转，取使质心对齐的平移。"""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target.mean(axis=0) - rotation @ source.mean(axis=0)
    return transform


def icp_refine(
    source: np.ndarray, target: np.ndarray, transform: np.ndarray, voxel: float
) -> tuple[np.ndarray, float, float]:
    """多尺度 ICP 精修（允许缩放，两段尺度本就不完全一致）。

    评分用 ``evaluate_registration`` 在**统一而且合理的阈值**下重算：
    若沿用最后一轮 ICP 的极紧阈值（voxel*0.5，物体尺寸的千分之几），
    即使是正确对齐，稀疏点云的噪声也会让 fitness 掉到 0，无法区分好坏。
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
    for distance, iterations in ((voxel * 3, 60), (voxel * 1.2, 100), (voxel * 0.5, 150)):
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, distance, transform, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transform = result.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(src, tgt, voxel * 2, transform)
    return transform, float(evaluation.fitness), float(evaluation.inlier_rmse)


def generate_initial_candidates(
    source: np.ndarray, target: np.ndarray, random_count: int, seed: int
) -> list[tuple[str, np.ndarray]]:
    """产生初始位姿候选。

    为什么不用纯 FPFH：本物体是个箱体，6 个大平面的局部几何几乎一模一样，
    FPFH 在平面上无法区分，实测 6 个随机种子全部失败（fitness 0.000）。
    改用两条更契合箱体的路子：

    1. **主轴 + 箱体 24 个固有旋转**：物体是箱体，它的主轴与面法向一致，
       所以两段之间的旋转必然是这 24 个对称之一（叠加主轴自身的误差）。
    2. **随机旋转**：兜底，覆盖主轴估计偏差较大的情况。
    """
    candidates: list[tuple[str, np.ndarray]] = []

    axes_source = principal_axes(source)
    axes_target = principal_axes(target)
    for index, symmetry in enumerate(cube_symmetries()):
        rotation = axes_target @ symmetry @ axes_source.T
        candidates.append((f"axis{index}", build_initial_transform(rotation, source, target)))

    rng = np.random.default_rng(seed)
    for index in range(random_count):
        candidates.append((f"rand{index}", build_initial_transform(random_rotation(rng), source, target)))

    return candidates


def register_candidates(
    source_norm: np.ndarray,
    target_norm: np.ndarray,
    voxel: float,
    seeds: int,
    random_count: int = 400,
    keep: int = 6,
    min_fitness: float = 0.5,
) -> list[dict]:
    """把 source 配到 target：产生候选 → ICP 精修 → 去重 → 按分数排序。"""
    src_small = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_norm)).voxel_down_sample(voxel).points
    )
    tgt_small = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_norm)).voxel_down_sample(voxel).points
    )
    print(f"  下采样 voxel={voxel}: source {len(src_small):,}  target {len(tgt_small):,}")

    initial = generate_initial_candidates(src_small, tgt_small, random_count, seeds)
    print(f"  初始候选 {len(initial)} 个（24 箱体对称 + {random_count} 随机），逐个 ICP 精修…")

    started = time.monotonic()
    refined: list[dict] = []
    best = 0.0
    for label, transform in initial:
        transform, fitness, rmse = icp_refine(src_small, tgt_small, transform, voxel)
        best = max(best, fitness)
        if fitness >= min_fitness:
            refined.append({"label": label, "transform": transform, "fitness": fitness, "rmse": rmse})
    print(f"  精修完成 {time.monotonic() - started:.0f}s，最高 fitness {best:.3f}，"
          f"达到阈值 {min_fitness} 的有 {len(refined)} 个")

    # 去重：旋转差异小且平移差异也小的归为一类，保留分数最高的
    unique: list[dict] = []
    for item in sorted(refined, key=lambda c: (-c["fitness"], c["rmse"])):
        for other in unique:
            if rotation_angle(item["transform"], other["transform"]) < 8:
                break
        else:
            unique.append(item)

    result = []
    for item in unique[:keep]:
        rotation = item["transform"][:3, :3]
        result.append({
            "label": item["label"],
            "transform": item["transform"].tolist(),
            "fitness": item["fitness"],
            "rmse": item["rmse"],
            "scale": float(np.cbrt(abs(np.linalg.det(rotation)))),
        })
    return result


def color_consistency(source_cloud, target_cloud, transform, sample=8000, seed=0) -> float:
    """重叠区颜色一致性：把 source 变换过去后查最近邻，比较颜色差异的中位数。

    对称的箱体可能解出多个几何上都说得通的姿态。木纹颜色在空间上是唯一的，
    用它当额外判别依据 —— 错姿态（例如上下颠倒 180°）颜色往往对不上。
    """
    points = np.asarray(source_cloud.points)
    colors = np.asarray(source_cloud.colors)
    if len(points) == 0 or colors.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    index = rng.choice(len(points), min(sample, len(points)), replace=False)
    moved = (transform[:3, :3] @ points[index].T).T + transform[:3, 3]
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    target_colors = np.asarray(target_cloud.colors)
    differences = []
    for point, color in zip(moved, colors[index]):
        count, neighbours, _ = tree.search_knn_vector_3d(point, 1)
        if count:
            differences.append(float(np.linalg.norm(color - target_colors[neighbours[0]])))
    return float(np.median(differences)) if differences else float("nan")


def render_overlay(points_a: np.ndarray, points_b: np.ndarray, path: Path, subtitle: str) -> None:
    """三个正交视角的红/蓝叠加图：红=A，蓝=B。重合良好说明配准正确。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 6.5))
    for axis, (i, j, label) in zip(axes, [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        axis.scatter(points_a[:, i], points_a[:, j], s=0.4, c="#d62728", alpha=0.45, linewidths=0, label="A")
        axis.scatter(points_b[:, i], points_b[:, j], s=0.4, c="#1f77b4", alpha=0.45, linewidths=0, label="B")
        axis.set_aspect("equal")
        axis.set_title(label)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle(subtitle)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def scale_matrix(factor: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] *= factor
    return matrix


def translate_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = vector
    return matrix


def to_original_frame(
    transform_norm: np.ndarray,
    center_a: np.ndarray, scale_a: float,
    center_b: np.ndarray, scale_b: float,
) -> np.ndarray:
    """把归一化空间里的变换换回 A 的原始坐标系。

    归一化是 ``p' = (p - center) / scale``，所以
    ``p_a = scale_a * (T' @ ((p_b - center_b) / scale_b)) + center_a``。
    """
    return (
        translate_matrix(center_a)
        @ scale_matrix(scale_a)
        @ transform_norm
        @ scale_matrix(1.0 / scale_b)
        @ translate_matrix(-center_b)
    )


def paint(cloud: o3d.geometry.PointCloud, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """整体改色，用于诊断图里区分两段。"""
    import copy

    painted = copy.deepcopy(cloud)
    painted.paint_uniform_color(color)
    return painted


def auto_voxel(points: np.ndarray, max_points: int) -> float:
    """按「下采样后约 max_points 个点」反推体素边长。

    不能用体积估算：点云是**表面**分布不是填满体积（表面积开方才対）。
    稠密点云有 1500 万点，若不控制点数，一次 ICP 就要几十秒，几百个候选完全跑不完。
    """
    extent = np.sort(points.max(axis=0) - points.min(axis=0))[::-1]
    area = 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
    return float(np.sqrt(area / max(1, max_points)))


def downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    return np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        .voxel_down_sample(voxel)
        .points
    )


def render_turntable(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
    title: str,
    elevs: tuple[int, ...] = (-60, -20, 20, 60),
    azims: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
    max_points: int = 12_000,
) -> None:
    """环绕一圈多角度出图，用来判断合并后是不是**封闭**的箱体。

    这是不依赖坐标系的决定性判据：配准正确 → 六个面齐全，所有方向看都是实心的；
    配准把 B 的底面转到了 A 的顶面 → 总有一个方向能透过大洞看到箱体内部
    （只有四条棱的红色边，中间是空的）。PCA 定箱体朝向在正方体上不可靠，
    所以这里干脆不做任何朝向假设，全方向都看。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    if len(points) > max_points:
        index = rng.choice(len(points), max_points, replace=False)
        points, colors = points[index], colors[index]

    rows, cols = len(elevs), len(azims)
    figure = plt.figure(figsize=(2.4 * cols, 2.4 * rows), dpi=80)
    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    half = extent.max() / 2.0
    for i, elev in enumerate(elevs):
        for j, azim in enumerate(azims):
            axes = figure.add_subplot(rows, cols, i * cols + j + 1, projection="3d")
            axes.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=1.2, linewidths=0, depthshade=False,
            )
            axes.view_init(elev=elev, azim=azim)
            axes.set_xlim(center[0] - half, center[0] + half)
            axes.set_ylim(center[1] - half, center[1] + half)
            axes.set_zlim(center[2] - half, center[2] + half)
            axes.set_box_aspect((1, 1, 1))
            axes.set_axis_off()
            axes.set_title(f"elev {elev}  azim {azim}", fontsize=7)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def merge_and_export(
    cloud_a: o3d.geometry.PointCloud,
    cloud_b: o3d.geometry.PointCloud,
    transform: np.ndarray,
    out_dir: Path,
    tag: str,
) -> o3d.geometry.PointCloud:
    """把 B 变换到 A 的坐标系后合并，保存点云与红/蓝诊断模型。"""
    moved_b = paint(cloud_b, (0.1, 0.35, 0.9))
    moved_b.transform(transform)
    merged = paint(cloud_a, (0.9, 0.25, 0.15)) + moved_b

    ply_path = out_dir / f"merged_{tag}_diagnostic.ply"
    o3d.io.write_point_cloud(str(ply_path), merged)
    print(f"  诊断点云 {ply_path.name}  {len(merged.points):,} 点")

    for name, cloud in (("A", paint(cloud_a, (0.9, 0.25, 0.15))), ("B", moved_b)):
        path = out_dir / f"merged_{tag}_{name}.ply"
        o3d.io.write_point_cloud(str(path), cloud)
    return merged


def mesh_merged(ply_path: Path, glb_path: Path) -> None:
    """把合并后的点云跑 Poisson 出 GLB（直接复用正式流程的转换函数）。

    这里是红/蓝两色的诊断点云，所以出来的 GLB 也能直接看接缝：
    配准正确时两种颜色应各自覆盖箱体不同区域；配错时会出现
    一块区域红蓝重叠（双层面）而另一块区域两种颜色都没有。
    """
    from app.convert import point_cloud_to_glb

    point_cloud_to_glb(
        ply_path,
        glb_path,
        outlier_percentile=0.0,       # 上游已经裁过离群点
        sor_neighbors=0,
        sor_std_ratio=0.0,            # 上游已经做过 SOR
        min_component_ratio=0.02,
        density_quantile=0.02,
        object_warmth=-1.0,           # 上游已经按颜色提纯
        poisson_max_depth=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default=DEFAULT_JOB)
    parser.add_argument("--source", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--warmth-a", type=float, default=0.08)
    parser.add_argument("--warmth-b", type=float, default=0.08)
    parser.add_argument("--voxel", type=float, default=0.0, help="归一化尺度下的下采样体素；0 = 按 --max-points 自动推算")
    parser.add_argument("--max-points", type=int, default=15_000, help="粗尺度配准的下采样目标点数（稠密点云必须靠它控速）")
    parser.add_argument("--refine-voxel", type=float, default=0.0, help="细尺度重排的体素；0 = 自动（20 倍点数）")
    parser.add_argument("--refine-top", type=int, default=8, help="取前 N 个候选做细尺度重排（稀疏下所有候选 rmse 都在噪声底，只能在细尺度上区分）")
    parser.add_argument("--seeds", type=int, default=0, help="随机旋转的随机种子")
    parser.add_argument("--random", type=int, default=400, help="随机旋转候选个数")
    parser.add_argument("--min-fitness", type=float, default=0.5, help="判定候选合格的最低 fitness")
    parser.add_argument("--export", type=int, default=0, help="导出前 N 个候选的合并点云用于目视确认")
    parser.add_argument("--glb", type=int, default=0, help="把前 N 个候选的合并点云直接出成 GLB")
    parser.add_argument("--views", type=int, default=0, help="对前 N 个候选出环拍多角度图（判断箱体是否封闭）")
    parser.add_argument("--final", type=int, default=0, help="对第 N 名的候选出最终产物：全分辨率合并 → 摆正居中 → GLB")
    args = parser.parse_args()

    job = (Path("storage") / args.job).resolve()
    out_dir = job / "merge"
    out_dir.mkdir(exist_ok=True)

    print(f"=== 载入两段点云（{args.source}）===")
    cloud_a = load_object(job, 0, args.source, args.warmth_a, job / "object_A.ply")
    cloud_b = load_object(job, 1, args.source, args.warmth_b, None)
    for name, cloud in (("A", cloud_a), ("B", cloud_b)):
        if len(cloud.points) < 1000:
            raise SystemExit(f"段 {name} 提取后只剩 {len(cloud.points)} 点，颜色阈值可能不合适")

    # 把 A 放到原点附近，B 平移到远处，便于可视化时区分
    center_a, scale_a, points_a = normalize(cloud_a)
    center_b, scale_b, points_b = normalize(cloud_b)
    print(f"\nA 归一化: 中心 {np.round(center_a, 3)}  尺度 {scale_a:.4f}  点数 {len(points_a):,}")
    print(f"B 归一化: 中心 {np.round(center_b, 3)}  尺度 {scale_b:.4f}  点数 {len(points_b):,}")

    cloud_a_n = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_a))
    cloud_a_n.colors = cloud_a.colors
    cloud_b_n = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_b))
    cloud_b_n.colors = cloud_b.colors

    print(f"\n=== 配准（把 B 配到 A）===")
    coarse_voxel = args.voxel if args.voxel > 0 else auto_voxel(points_a, args.max_points)
    candidates = register_candidates(
        points_b, points_a, coarse_voxel, args.seeds,
        random_count=args.random, min_fitness=args.min_fitness,
    )
    if not candidates:
        raise SystemExit("没有候选达到 fitness 阈值，配准失败（可用 --min-fitness 调低看分布）")

    print(f"\n=== 候选评估（细尺度重排 + 重叠区颜色一致性）===")
    # 粗体素下所有候选的 rmse 都落在点间距量级（实测 0.0228~0.0260），完全分不出对错；
    # 只有在更细的体素上重跑 ICP，真正正确的那个才会明显拉开差距。
    fine_voxel = args.refine_voxel if args.refine_voxel > 0 else auto_voxel(points_a, args.max_points * 20)
    if fine_voxel < coarse_voxel and args.refine_top > 0:
        src_fine = downsample(points_b, fine_voxel)
        tgt_fine = downsample(points_a, fine_voxel)
        print(f"  细尺度 voxel={fine_voxel:.5f}: source {len(src_fine):,}  target {len(tgt_fine):,}"
              f"，重排前 {args.refine_top} 个候选")
        started = time.monotonic()
        for candidate in candidates[: args.refine_top]:
            transform, fitness, rmse = icp_refine(
                src_fine, tgt_fine, np.array(candidate["transform"]), fine_voxel
            )
            candidate["transform"] = transform.tolist()
            candidate["fine_fitness"] = fitness
            candidate["fine_rmse"] = rmse
            candidate["scale"] = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
        print(f"  细尺度重排完成 {time.monotonic() - started:.0f}s")
        # 颜色差异先算好，一并纳入排序（木纹颜色在空间上唯一，对称解往往对不上）
        for candidate in candidates[: args.refine_top]:
            candidate["color_diff"] = color_consistency(
                cloud_b_n, cloud_a_n, np.array(candidate["transform"])
            )
        candidates.sort(
            key=lambda item: (
                -item.get("fine_fitness", 0.0),
                item.get("fine_rmse", 9e9),
                item.get("color_diff", 9e9),
            )
        )

    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
        if "color_diff" not in candidate:
            candidate["color_diff"] = color_consistency(
                cloud_b_n, cloud_a_n, np.array(candidate["transform"])
            )
        fine = ("" if "fine_rmse" not in candidate else
                f"  细fitness {candidate['fine_fitness']:.3f}  细rmse {candidate['fine_rmse']:.5f}")
        print(f"  #{rank} {candidate['label']:<9} fitness {candidate['fitness']:.3f}  "
              f"rmse {candidate['rmse']:.4f}  缩放 {candidate['scale']:.4f}  "
              f"颜色差异 {candidate['color_diff']:.4f}{fine}")

    print(f"\n=== 生成叠加图（红=A，蓝=B，重合即为配准正确）===")
    red = np.tile(np.array([0.9, 0.25, 0.15]), (len(points_a), 1))
    for candidate in candidates:
        transform = np.array(candidate["transform"])
        moved = (transform[:3, :3] @ points_b.T).T + transform[:3, 3]
        path = out_dir / f"overlay_{args.source}_{candidate['rank']}.png"
        render_overlay(
            points_a, moved, path,
            f"{args.source} #{candidate['rank']} ({candidate['label']})  "
            f"fitness {candidate['fitness']:.3f}  rmse {candidate['rmse']:.4f}  "
            f"scale {candidate['scale']:.4f}  colorΔ {candidate['color_diff']:.4f}",
        )
        print(f"  {path.name}")

    if args.views:
        print(f"\n=== 环拍多角度图（任一方向透空 = 缺了一个面 = 配准错误）===")
        blue = np.tile(np.array([0.1, 0.35, 0.9]), (len(points_b), 1))
        for candidate in candidates[: args.views]:
            transform = np.array(candidate["transform"])
            moved = (transform[:3, :3] @ points_b.T).T + transform[:3, 3]
            path = out_dir / f"turntable_{args.source}_{candidate['rank']}.png"
            render_turntable(
                np.vstack([points_a, moved]), np.vstack([red, blue]), path,
                f"{args.source} #{candidate['rank']} ({candidate['label']})  "
                f"fitness {candidate['fitness']:.3f}  rmse {candidate['rmse']:.4f}  "
                f"scale {candidate['scale']:.4f}",
            )
            print(f"  {path.name}")

    # 把前几个候选在原始分辨率上合并，导出红/蓝诊断点云，用于目视确认（箱体对称，
    # 纯几何分不开「正确的」和「上下颠倒的」，必须看孔与缺口是否只出现一次）
    if args.export:
        print(f"\n=== 导出前 {args.export} 个候选的合并结果（A=红 B=蓝）===")
        for candidate in candidates[: args.export]:
            transform_norm = np.array(candidate["transform"])
            transform = to_original_frame(transform_norm, center_a, scale_a, center_b, scale_b)
            merge_and_export(
                cloud_a, cloud_b, transform, out_dir, f"{args.source}{candidate['rank']}",
            )

    for candidate in candidates[: args.glb]:
        tag = f"{args.source}{candidate['rank']}"
        ply_path = out_dir / f"merged_{tag}_diagnostic.ply"
        if not ply_path.exists():
            print(f"  跳过 {tag}：缺少 {ply_path.name}，请同时加 --export")
            continue
        print(f"\n=== 候选 {tag} 出 GLB ===")
        mesh_merged(ply_path, out_dir / f"merged_{tag}.glb")

    if args.final:
        candidate = candidates[args.final - 1]
        print(f"\n=== 最终产物：候选 #{candidate['rank']} ({candidate['label']}) ===")
        transform = to_original_frame(
            np.array(candidate["transform"]), center_a, scale_a, center_b, scale_b
        )
        moved_b = o3d.geometry.PointCloud(cloud_b)
        moved_b.transform(transform)
        merged = o3d.geometry.PointCloud(cloud_a) + moved_b
        print(f"  全分辨率合并 {len(merged.points):,} 点")
        points_a_full = np.asarray(cloud_a.points)
        up, plane_share = table_up(
            load_full(job, 0, args.source),
            points_a_full.mean(axis=0),
            points_a_full.max(axis=0) - points_a_full.min(axis=0),
        )
        print(f"  桌面法向（当作重力反方向）{np.round(up, 4)}  桌面内点占比 {plane_share:.1%}")
        stand_upright(merged, up)
        size = merged.get_max_bound() - merged.get_min_bound()
        print(f"  摆正居中后尺寸 {np.round(size, 4)}")
        final_ply = out_dir / "merged_final_upright.ply"
        o3d.io.write_point_cloud(str(final_ply), merged)
        print(f"  {final_ply.name}  {final_ply.stat().st_size / 1024 / 1024:.0f} MB")
        mesh_merged(final_ply, out_dir / "merged_final.glb")
        glb = out_dir / "merged_final.glb"
        print(f"  {glb.name}  {glb.stat().st_size / 1024 / 1024:.0f} MB")

    (out_dir / f"candidates_{args.source}.json").write_text(
        json.dumps(
            {"source": args.source, "voxel": args.voxel, "candidates": candidates},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n候选已写入 {out_dir / f'candidates_{args.source}.json'}")


if __name__ == "__main__":
    main()
