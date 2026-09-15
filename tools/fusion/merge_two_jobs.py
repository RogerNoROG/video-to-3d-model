#!/usr/bin/env python3
"""把两个**独立重建**的模型对齐并合并。

和 tools/fusion/merge_models.py 的区别：那一个处理的是同一段视频里的两个 COLMAP 子模型
（公共重叠大、尺度靠 ICP 估）；这里处理的是**两段不同视频**各建一次得到的两个模型，
公共重叠小、而且可能连拍摄方位都完全不同。

两个模型的「上」在各自视频里由桌面决定，但两段视频里物体放的姿势可能不同，
所以不能假设两者已经同向。

对齐思路
--------
1. **立方体框架**：实物是近正方体，用「最小体积包围盒」（SO(3) 粗采样 + 局部细化）
   求它的三个面法向 → 得到 cube frame。木块棱长就是物理尺度，
   两个模型各自除以**棱长**，尺度就统一了 —— 比用 ICP 估尺度准得多。
2. **候选姿态**：在 cube frame 下，两个模型的差只能是立方体自身的 24 个对称旋转
   （行列式为 +1 的正交矩阵）。逐个 ICP 精修。
3. **排序**：几何拟合度 + 重叠区颜色一致性。立方体接近对称，几何上常有好几个解
   都说得通，木纹颜色在空间上唯一，是决定性判据。

用法：
    ./.venv/bin/python tools/fusion/merge_two_jobs.py \
        --a storage/<old>/merge/merged_cube_final.glb \
        --b storage/<new>/fused.ply \
        --out storage/<new>/merge
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.convert import crop_outliers, keep_object_by_color  # noqa: E402
# 配准的通用部件全部复用 merge_models，别在这里再实现一遍：
# 立方体 24 对称、颜色一致性评分、多尺度 ICP（已加评分阈值与塔缩防护）、
# 4x4 拼接工具。这些都是一直在用的老代码，行为已经验证过。
from merge_models import (  # noqa: E402
    color_consistency,
    cube_symmetries,
    icp_refine,
    random_rotation,
    read_camera_centers,
    scale_matrix,
    translate_matrix,
)

_PROJECT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- 读取

def load_cloud(path: Path, max_points: int, seed: int = 0) -> o3d.geometry.PointCloud:
    lower = str(path).lower()
    if lower.endswith((".glb", ".gltf", ".obj", ".stl")):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0:
            raise SystemExit(f"空网格: {path}")
        cloud = mesh.sample_points_uniformly(number_of_points=max_points)
        if not cloud.has_colors():
            cloud.paint_uniform_color((0.8, 0.8, 0.8))
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise SystemExit(f"空点云: {path}")
    print(f"  读取 {path.name}: {len(cloud.points):,} 点")
    # 稠密点云常有数千万点。之前只对 GLB 的采样生效，PLY 会先经过颜色过滤与
    # 分位裁剪，短时间内同时持有数份千万级数组，合并还没开始就会被 OOM 杀掉。
    # 配准只需要均匀的表面样本；先确定性下采样，既保证内存上限也不改变候选排序。
    if len(cloud.points) > max_points:
        index = np.random.default_rng(seed).choice(len(cloud.points), max_points, replace=False)
        cloud = cloud.select_by_index(index.tolist())
        print(f"  配准采样: {len(index):,} 点（原始高密度点云保留在磁盘）")
    return cloud


# ---------------------------------------------------------------- 立方体框架

# （最小体积包围盒 / SO(3) 均匀采样曾在这里实现过一遍 —— 已删。
#   它对近正方体是病态解（体积对姿态的偏导≈0），已被面拟合的 face_frame 取代；
#   自检用的随机旋转直接复用 merge_models.random_rotation。）


def _best_plane(points: np.ndarray, tolerance: float, tries: int = 8, min_points: int = 400):
    """抽若干次，返回点最多的那个平面 (法向, 平面点, 内点索引)。

    ⚠️ Open3D 在退化点云（重复点/尺度异常）上偶尔会返回越界的 inlier 索引，
    直接拿去索引会崩成 "index 68720799217 is out of bounds"，所以要夹一下。
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    best = None
    for _ in range(tries):
        plane, inliers = cloud.segment_plane(tolerance, 3, 1200)
        inliers = np.asarray(inliers, dtype=np.int64)
        inliers = inliers[(inliers >= 0) & (inliers < len(points))]
        if len(inliers) < min_points:
            continue
        normal = np.asarray(plane[:3], dtype=np.float64)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        if best is None or len(inliers) > len(best[2]):
            best = (normal, points[inliers], inliers)
    return best


def face_frame(points: np.ndarray, tol_scale: float = 0.012, verbose: bool = True):
    """用**大平面拟合**求立方体框架：返回 (R, center, extents) 或 None。

    为什么不用最小体积包围盒：实物是**近正方体**，绕任何轴小角度转动时包围盒体积
    变化极小（体积对姿态的偏导接近 0）→ 解是病态的。实测同一块木块，A 定出棱长
    1.291、B（A 加了随机旋转）定出 1.317，差 2% —— 对 0.001 量级的点间距来说是灾难。

    大平面拟合则非常精确：一个平面上有几万个点，法向精度远优于 0.1°。
    先找最大的那个面拿到 n1，再在剩余点里找与 n1 垂直的面拿到 n2，n3 = n1 × n2。
    """
    size = float(np.max(points.max(axis=0) - points.min(axis=0)))
    tolerance = size * tol_scale
    centroid = points.mean(axis=0)

    found: list[np.ndarray] = []
    work = points
    step = 0
    attempts = 0
    while step < 2 and attempts < 30:
        attempts += 1
        if len(work) < 1000:
            break
        plane = _best_plane(work, tolerance)
        if plane is None:
            break
        normal, plane_points, _count = plane
        # 用**平面内点**的质心：不能用全体点投影的中位数（那落在物体中间的薄层上，
        # 物体只有表面没有内部，会取到圆孔/凹槽内壁的点）
        plane_centroid = plane_points.mean(axis=0)
        # 法向朝向物体外侧
        if float(np.dot(normal, centroid - plane_centroid)) < 0:
            normal = -normal
        offset = float(np.dot(plane_centroid, normal))

        if step == 1 and abs(float(np.dot(normal, found[0]))) > 0.25:
            # 找到的还是与 n1 同向的面（对侧那一面），剔掉它再试
            work = work[np.abs(work @ normal - offset) > tolerance * 2.0]
            continue
        if step == 1:
            normal = normal - float(np.dot(normal, found[0])) * found[0]
            normal /= np.linalg.norm(normal)

        found.append(normal)
        if verbose:
            print(f"    平面 {step + 1}: 法向 {np.round(normal, 4)}  "
                  f"内点 {len(_count):,}  平面位置 {offset:+.4f}")
        # 去掉这个面附近的点
        work = work[np.abs(work @ normal - offset) > tolerance * 2.0]
        step += 1

    if len(found) < 2:
        if verbose:
            print("    平面不足，退回最小体积包围盒")
        return None

    n1, n2 = found[0], found[1]
    n3 = np.cross(n1, n2)
    n3 /= np.linalg.norm(n3)
    n2 = np.cross(n3, n1)
    n2 /= np.linalg.norm(n2)
    rotation = np.stack([n1, n2, n3], axis=0)
    if verbose:
        print(f"    正交性 |n1·n2|={abs(float(np.dot(n1, n2))):.4f} "
              f"|n2·n3|={abs(float(np.dot(n2, n3))):.4f} "
              f"|n1·n3|={abs(float(np.dot(n1, n3))):.4f}")

    aligned = points @ rotation.T
    lo, hi = aligned.min(axis=0), aligned.max(axis=0)
    extents = hi - lo
    center = rotation.T @ ((lo + hi) * 0.5)
    return rotation, center, extents


def cube_frame(points: np.ndarray, verbose: bool = True):
    """求立方体框架，返回 (R, center, extents)；面拟合失败就退回轴对齐包围盒。"""
    frame = face_frame(points, verbose=verbose)
    if frame is not None:
        return frame
    if verbose:
        print("    面拟合失败，退回轴对齐包围盒")
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.eye(3), (lo + hi) * 0.5, hi - lo


# ---------------------------------------------------------------- 配准

def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由旋转 + 平移拼出 4x4。"""
    t = translate_matrix(np.asarray(translation, dtype=np.float64))
    t[:3, :3] = rotation
    return t


# ---------------------------------------------------------------- 主流程

def point_spacing(normalized_points: np.ndarray, count: int) -> float:
    """归一化尺度下点云的平均点间距（假设点均匀铺在六面表面积上）。"""
    extent = normalized_points.max(axis=0) - normalized_points.min(axis=0)
    area = 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
    return float(np.sqrt(area / max(1, count)))


def face_signature(points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   bins: int = 26) -> np.ndarray:
    """把点云的六个面各做成一张「凹陷深度图」，拼成一个特征向量。

    **为什么需要它**：木块近似正方体，纯几何拟合度分不出姿态。
    实测一个真错的姿态（凹槽方向错了 90°）照样拿到 fitness 0.91、
    最近邻中位 0.0009 —— 因为凹槽只占表面约 10%，其余平面全对得上，
    中位数和拟合度都被平面部分主导了。能区分姿态的**只有**凹槽和圆孔。

    ⚠️ 深度必须取「该格子内**沿该轴的最大坐标**」，不能取「距该面 40% 棱长
    以内的点的最大距离」—— 后者会把**侧面上的点**也卷进来，每个面的边界一圈
    格子都被算成"深"，六个面看起来都差不多，任何姿态的 IoU 都≈0.5，等于没判据
    （踩过：6 个姿态的特征差 0.503/0.505/0.507/0.531/0.544 挤成一团）。
    """
    size = float(np.max(hi - lo))
    result = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for sign in (1, -1):
            face = hi[axis] if sign > 0 else lo[axis]
            grid = np.full((bins, bins), np.nan)
            inside = (
                (points[:, other[0]] >= lo[other[0]]) & (points[:, other[0]] <= hi[other[0]])
                & (points[:, other[1]] >= lo[other[1]]) & (points[:, other[1]] <= hi[other[1]])
            )
            q = points[inside]
            if len(q) >= 50:
                i = np.clip(((q[:, other[0]] - lo[other[0]]) /
                             max(hi[other[0]] - lo[other[0]], 1e-9) * bins).astype(int), 0, bins - 1)
                j = np.clip(((q[:, other[1]] - lo[other[1]]) /
                             max(hi[other[1]] - lo[other[1]], 1e-9) * bins).astype(int), 0, bins - 1)
                # 沿该轴的最大坐标 = 这张面的表面高度；侧面的点坐标更小，不影响
                height = q[:, axis] * sign
                flat = (i * bins + j).ravel()
                # np.maximum.at 是这里唯一可用的向量化手段：同一个格子可能被命中
                # 很多次，需要「逐格取最大」而不是覆盖（bincount 只能求和）
                accumulator = np.full(bins * bins, -np.inf)
                np.maximum.at(accumulator, flat, height)
                grid = accumulator.reshape(bins, bins)
                grid[~np.isfinite(grid)] = np.nan
            result.append((face * sign - grid) / size)      # 相对该面的凹陷深度
    return np.stack(result)                                 # (6, bins, bins)


def signature_distance(sig_a: np.ndarray, sig_b: np.ndarray,
                       deep_fraction: float = 0.05) -> float:
    """两个六面深度图特征的差异（越小越像）。逐面算「凹陷掩码」的交并比再平均。

    ⚠️ 两个坑都踩过：
    1. **不能用平均绝对差**：凹槽/圆孔只占表面很小一部分，平均差会被大片平面
       稀释 —— 实测 6 个姿态是 0.0936/0.0951/0.0986/0.0987/0.1016/0.1023，挤成
       一团，等于没判据。
    2. **两面都是平面时必须判「一致」而不是「最差」**：那样每个物体都有 3~4 个
       面是平的，若按"没有共同特征 = 最差"处理，正确姿态也会被冤枉。
    """
    both = ~np.isnan(sig_a) & ~np.isnan(sig_b)
    mask_a = (sig_a > deep_fraction) & both
    mask_b = (sig_b > deep_fraction) & both
    scores = []
    for face in range(sig_a.shape[0]):
        a = mask_a[face]
        b = mask_b[face]
        union = int((a | b).sum())
        if union == 0:
            scores.append(0.0)                       # 两面都没有凹陷 -> 一致
        elif not a.any() or not b.any():
            scores.append(1.0)                       # 一边有凹陷一边没有 -> 完全不一致
        else:
            scores.append(1.0 - int((a & b).sum()) / union)
    return float(np.mean(scores))


def register(a: o3d.geometry.PointCloud, b: o3d.geometry.PointCloud, out: Path,
             voxel_refine: float, top: int, extra_spins: int = 0,
             coarse_points: int = 60000, fine_points: int = 600000,
             workers: int = 1):
    pa = np.asarray(a.points)
    pb = np.asarray(b.points)
    ca = np.asarray(a.colors) if a.has_colors() else np.zeros((len(pa), 3))
    cb = np.asarray(b.colors) if b.has_colors() else np.zeros((len(pb), 3))

    print("\n[A] 求立方体框架 ...")
    ra, cea, exa = cube_frame(pa)
    print(f"  棱长 {np.round(np.sort(exa)[::-1], 4)}")
    print("\n[B] 求立方体框架 ...")
    rb, ceb, exb = cube_frame(pb)
    print(f"  棱长 {np.round(np.sort(exb)[::-1], 4)}")

    # 用棱长（三向中位数）统一尺度 -> B 变换到「与 A 同尺度」
    sa = float(np.median(exa))
    sb = float(np.median(exb))
    print(f"\n棱长 A {sa:.4f}   B {sb:.4f}   尺度比 A/B = {sa / sb:.6f}")

    # 统一到「立方体框架、棱长归一」的空间
    a_norm = (pa - cea) @ ra.T / sa
    b_norm = (pb - ceb) @ rb.T / sb
    print(f"归一化后包围盒 A {np.round(a_norm.max(axis=0) - a_norm.min(axis=0), 4)}")
    print(f"归一化后包围盒 B {np.round(b_norm.max(axis=0) - b_norm.min(axis=0), 4)}")

    def take(points, colors, count, seed):
        if len(points) <= count:
            return points, colors
        idx = np.random.default_rng(seed).choice(len(points), count, replace=False)
        return points[idx], colors[idx]

    a_co, ca_co = take(a_norm, ca, coarse_points, 0)
    b_co, cb_co = take(b_norm, cb, coarse_points, 1)
    a_fi, ca_fi = take(a_norm, ca, fine_points, 2)
    b_fi, cb_fi = take(b_norm, cb, fine_points, 3)

    # 归一化后两者都是「立方体框架 + 棱长 1 + 居中」，所以参考盒就是 ±0.5
    box_lo, box_hi = np.full(3, -0.5), np.full(3, 0.5)
    sig_a = face_signature(a_co, box_lo, box_hi)

    # ★ 评分阈值必须跟**实际点间距**匹配：降到 6 万点时间距约 0.01，
    #   若还用 0.005 当阈值，即使完美对齐 fitness 也只有 0.4 左右，
    #   排序就失去意义了（这是第一版实现踩过的坑）。
    spacing_coarse = point_spacing(a_co, len(a_co))
    spacing_fine = point_spacing(a_fi, len(a_fi))
    thresh_coarse = spacing_coarse * 3.0
    thresh_fine = max(voxel_refine * 2.0, spacing_fine * 1.5)
    print(f"\n点间距：粗 {spacing_coarse:.5f}（阈值 {thresh_coarse:.4f}）  "
          f"细 {spacing_fine:.5f}（阈值 {thresh_fine:.4f}）")
    print("ICP 用 merge_models.icp_refine（多尺度 + 评分阈值 + 塔缩防护），刚性不做尺度优化")

    symmetries = cube_symmetries()
    spins = []
    if extra_spins:
        for k in range(1, extra_spins + 1):
            angle = 2 * np.pi * k / (4 * (extra_spins + 1))
            spins.append(np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]))

    tasks = [
        (index, spin @ symmetry)
        for index, symmetry in enumerate(symmetries)
        for spin in [np.eye(3)] + spins
    ]
    total = len(tasks)
    worker_count = max(1, min(int(workers), total))
    print(f"\n【粗搜】{total} 个姿态（并发 {worker_count}） ...")

    def refine_coarse(task: tuple[int, np.ndarray]) -> dict:
        index, rotation = task
        # 每个任务在 Open3D 内新建自己的临时 PointCloud；a_co / b_co 只读共享，
        # 因此 2~4 路同时 ICP 不会改变任一候选的结果。
        transform = make_transform(rotation, np.zeros(3))
        transform, fitness, rmse = icp_refine(
            b_co, a_co, transform, voxel=thresh_coarse,
            threshold=thresh_coarse, allow_scale=False,
        )
        moved = (transform[:3, :3] @ b_co.T).T + transform[:3, 3]
        return {
            "symmetry_index": index,
            "transform": transform,
            "fitness": fitness,
            "rmse": rmse,
            "signature": signature_distance(sig_a, face_signature(moved, box_lo, box_hi)),
        }

    if worker_count == 1:
        coarse = []
        for done, task in enumerate(tasks, 1):
            coarse.append(refine_coarse(task))
            print(f"  {done}/{total}", end="\r", flush=True)
    else:
        # Open3D 的 ICP 在 C++ 中释放 GIL；线程池避免为每一条候选复制点云。
        # 限制为调用方指定的小并发数，避免每个 Open3D 调用的内部线程过度抢占。
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            pending = [pool.submit(refine_coarse, task) for task in tasks]
            coarse = []
            for done, future in enumerate(as_completed(pending), 1):
                coarse.append(future.result())
                print(f"  {done}/{total}", end="\r", flush=True)
    print()
    # ★ 按**六面特征差异**排序，不是按 fitness。立方体对称下 fitness 分不开姿态：
    #   实测一个凹槽方向错了 90° 的错解照样 fitness 0.91。
    coarse.sort(key=lambda item: (item["signature"], -item["fitness"]))
    print("粗搜前 6（按六面特征差异排序，越小越像）：")
    for item in coarse[:6]:
        print(f"    symmetry #{item['symmetry_index']:>2}  特征差 {item['signature']:.5f}  "
              f"fitness {item['fitness']:.4f}  rmse {item['rmse']:.5f}")

    print(f"\n【精修】用 {len(a_fi):,} 点重算前 {min(top, len(coarse))} 名 ...")
    # 颜色一致性要传 o3d 点云（merge_models 里的那份签名），在这里建一次复用
    a_cloud_fi = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a_fi))
    b_cloud_fi = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b_fi))
    a_cloud_fi.colors = o3d.utility.Vector3dVector(ca_fi)
    b_cloud_fi.colors = o3d.utility.Vector3dVector(cb_fi)
    def refine_fine(item: dict) -> dict:
        transform, fitness, rmse = icp_refine(
            b_fi, a_fi, item["transform"], voxel=thresh_fine,
            threshold=thresh_fine, allow_scale=False,
        )
        score = color_consistency(b_cloud_fi, a_cloud_fi, np.asarray(transform), sample=15000)
        moved_fi = (transform[:3, :3] @ b_fi.T).T + transform[:3, 3]
        signature = signature_distance(sig_a, face_signature(moved_fi, box_lo, box_hi))
        scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
        # 尺度已由棱长归一（误差 0.02%），剩下的只该是千分之几的修正。
        # 用带尺度的 ICP 时踩过退化解：把源点云塔缩成一个点，fitness 反而是 1.0、
        # rmse 掉到 1e-18。所以这里同时(1) 用刚性 ICP、(2) 把尺度异常的直接禁掉。
        plausible = 0.85 <= scale <= 1.18
        return {
            "symmetry_index": item["symmetry_index"],
            "coarse_fitness": item["fitness"],
            "coarse_signature": item["signature"],
            "transform": transform.tolist(),
            "fitness": fitness if plausible else 0.0,
            "raw_fitness": fitness,
            "rmse": rmse,
            "color": score,
            "signature": signature,
            "scale": scale,
            "plausible": bool(plausible),
            "threshold": thresh_fine,
        }

    fine_candidates = coarse[:max(1, top)]
    if worker_count == 1:
        results = []
        for done, item in enumerate(fine_candidates, 1):
            results.append(refine_fine(item))
            print(f"  精修 {done}/{len(fine_candidates)}", end="\r", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(fine_candidates))) as pool:
            pending = [pool.submit(refine_fine, item) for item in fine_candidates]
            results = []
            for done, future in enumerate(as_completed(pending), 1):
                results.append(future.result())
                print(f"  精修 {done}/{len(fine_candidates)}", end="\r", flush=True)
    print()
    # 主排序用六面特征差异，fitness 只当辅助（它分不开对称姿态）
    results.sort(key=lambda item: (item["signature"], -item["fitness"]))
    print("\n精修后排名（主排序 = 六面特征差异）：")
    print("  排名  特征差    fitness    rmse    颜色差   尺度   合理?")
    for rank, item in enumerate(results, 1):
        print(f"  {rank:>3}  {item['signature']:.5f}  {item['fitness']:.4f}  "
              f"{item['rmse']:.5f}  {item['color']:.4f}  {item['scale']:.4f}   "
              f"{'✓' if item['plausible'] else '✗'}")
    if results and not results[0]["plausible"]:
        print("⚠️ 最佳候选的尺度不合理（可能是退化解），结果不可信！")

    return {
        "job_a_norm": (ra, cea, sa),
        "job_b_norm": (rb, ceb, sb),
        "results": results,
        "a_norm": a_norm, "b_norm": b_norm,
        "ca": ca, "cb": cb,
        "a_fine": a_fi, "ca_fine": ca_fi,
        "b_fine": b_fi, "cb_fine": cb_fi,
        "thresh_fine": thresh_fine,
    }


def normalized_to_original(transform, ra, cea, sa, rb, ceb, sb) -> np.ndarray:
    """把「归一化立方体框架里」的变换换回 A 的原始坐标系。

    a_norm = ra @ (pa - cea) / sa      b_norm = rb @ (pb - ceb) / sb
    a_norm = T @ b_norm
    => pa = cea + sa * ra.T @ (T @ (rb @ (pb - ceb) / sb))

    实测（B = A 经已知相似变换）：搬回去后到 A 的最近邻中位 0.00004，
    矩阵尺度误差 0.003% —— 正确性已经验过。
    """
    def rotate(rotation):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        return matrix

    return (
        translate_matrix(np.asarray(cea, dtype=np.float64))
        @ scale_matrix(sa)
        @ rotate(ra.T)
        @ np.asarray(transform, dtype=np.float64)
        @ scale_matrix(1.0 / sb)
        @ rotate(rb)
        @ translate_matrix(-np.asarray(ceb, dtype=np.float64))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="老模型（glb 或 ply）")
    parser.add_argument("--b", required=True, help="新模型（fused.ply 或 glb）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--max-points", type=int, default=1_200_000)
    parser.add_argument("--refine-voxel", type=float, default=0.0025,
                        help="评分阈值的下限（归一化棱长）；实际阈值会再按点间距放大")
    parser.add_argument("--coarse-points", type=int, default=60000)
    parser.add_argument("--fine-points", type=int, default=600000)
    parser.add_argument("--extra-spins", type=int, default=0,
                        help="每个对称姿态再加几个小自旋（立方体框架不准时用）")
    parser.add_argument("--warmth-a", type=float, default=-1.0)
    parser.add_argument("--warmth-b", type=float, default=-1.0)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--glb", type=int, default=0,
                        help="把前 N 个候选各出一份合并 GLB 供目视挑选（本物体纯几何"
                             "分不开姿态，最终要靠眼睛定）")
    parser.add_argument("--glb-depth", type=int, default=7,
                        help="候选 GLB 的 Poisson 深度。目视挑选用 7 就够（约 30 MB），"
                             "9 会到 200~300 MB，浏览器很卡")
    parser.add_argument("--jitter", type=int, default=0,
                        help="自检：用该种子对 B 施加随机相似变换（旋转+缩放+平移），"
                             "正确的配准应仍然拿到接近 1.0 的 fitness")
    parser.add_argument("--workers", type=int, default=1,
                        help="同时精修的候选数；Open3D 的单次 ICP 已使用内部并行，默认 1")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = load_cloud(Path(args.a), args.max_points)
    b = load_cloud(Path(args.b), args.max_points)
    if args.warmth_a >= 0:
        a = keep_object_by_color(a, args.warmth_a)
    if args.warmth_b >= 0:
        b = keep_object_by_color(b, args.warmth_b)
    a = crop_outliers(a, 1.0)
    b = crop_outliers(b, 1.0)
    print(f"  清理后 A {len(a.points):,} 点   B {len(b.points):,} 点")

    if args.jitter:
        rng = np.random.default_rng(args.jitter)
        R = random_rotation(rng)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        s = round(float(0.5 + rng.random()), 3)
        t = np.round((rng.random(3) - 0.5) * 0.9, 3)
        extent = float(np.max(a.get_max_bound() - a.get_min_bound()))
        t = t * extent
        pts = np.asarray(b.points) @ R.T * s + t
        b.points = o3d.utility.Vector3dVector(pts)
        print(f"  [自检] 对 B 施加随机相似变换：旋转 {angle:.1f}°（轴随机）"
              f" 缩放 {s:.3f} 平移 {np.round(t, 3)}")
        print("         正确的配准应仍然拿到 >= 0.95 的 fitness；若最高只有 0.3~0.6，"
                             "说明立方体框架或对称姿态搜索有问题。")
    state = register(a, b, out, args.refine_voxel, args.top, args.extra_spins,
                     args.coarse_points, args.fine_points, args.workers)

    (out / "merge_two_jobs.json").write_text(json.dumps({
        "job_a_norm": [state["job_a_norm"][0].tolist(),
                       state["job_a_norm"][1].tolist(),
                       state["job_a_norm"][2]],
        "job_b_norm": [state["job_b_norm"][0].tolist(),
                       state["job_b_norm"][1].tolist(),
                       state["job_b_norm"][2]],
        "results": state["results"][:args.top],
    }, indent=2))
    print(f"\n写出 {out / 'merge_two_jobs.json'}")

    # 导出前 N 个候选，供目视挑选 —— 本物体是近正方体，纯几何分不开姿态（已实测：
    # 凹槽错 90° 的错解照样 fitness 0.91），最终只能靠眼睛看凹槽/圆孔对不对。
    # 出图直接复用 merge_models 里既有的 auto_voxel / mesh_merged，不另写一套。
    if args.glb > 0:
        from merge_models import auto_voxel, mesh_merged

        ra, cea, sa = state["job_a_norm"]
        rb, ceb, sb = state["job_b_norm"]
        print(f"\n导出前 {args.glb} 个候选 GLB（depth {args.glb_depth}）...")
        for rank, item in enumerate(state["results"][:args.glb], 1):
            matrix = normalized_to_original(
                np.asarray(item["transform"]), ra, cea, sa, rb, ceb, sb
            )
            moved = o3d.geometry.PointCloud(b)
            moved.transform(matrix)
            merged = a + moved
            voxel = auto_voxel(np.asarray(merged.points), 3_000_000)
            small = merged.voxel_down_sample(voxel)
            ply = out / f"candidate_{rank}.ply"
            o3d.io.write_point_cloud(str(ply), small)
            mesh_merged(ply, out / f"candidate_{rank}.glb", depth=args.glb_depth)
            print(f"  候选 {rank}: 特征差 {item['signature']:.4f}  fitness {item['fitness']:.4f} "
                  f"-> candidate_{rank}.glb")
        print("用浏览器打开这些候选逐一看：凹槽/圆孔的轮廓要与旧模型完全对上，"
              "错姿态会看到表面起泡或特征错位。")
    else:
        print("下一步：加 --glb N 导出前 N 个候选，用眼睛挑一个正确的。")


if __name__ == "__main__":
    main()
