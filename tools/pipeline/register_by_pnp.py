#!/usr/bin/env python3
"""绕开 COLMAP 的注册器，自己算补拍帧的相机位姿（DLT + RANSAC）。

为什么要自己写
------------
本机 COLMAP 4.3.0.dev0 构建里 `matches_importer` **完全不工作**：
无论 `--match_type pairs` 还是 `inliers`，连 image_id=1 都报 "not found in database"
（在未经修改的原始库上同样失败，指定 `--FeatureMatching.type` 也没用）。
而 ALIKED 匹配器的门限参数 `--AlikedMatching.brute_force_min_cossim` 又是死的
（0.85 扫到 0.30，匹配对数全程不变），所以跨场次匹配没法走 COLMAP 自己的路。

既然匹配已经自己算出来了（tools/pipeline/match_cross_session.py），位姿也自己算：
二维点来自补拍帧的关键点，三维点来自原模型 points3D 的 TRACK，
用已知内参做 DLT + RANSAC 求相机位姿，再把新图写进模型。

用法
----
    ./.venv/bin/python tools/pipeline/register_by_pnp.py --job <任务id> --model sparse/0
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COLMAP = str(Path.home() / ".local/bin/colmap")
CAMERA_MODELS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL"}


def log(message: str) -> None:
    print(f"[pnp] {message}", flush=True)


# ---------------------------------------------------------------- 模型读写


def to_txt(model_bin: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    result = subprocess.run(
        [COLMAP, "model_converter", "--input_path", str(model_bin),
         "--output_path", str(target), "--output_type", "TXT"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit("模型转 TXT 失败")
    return target


def read_images_txt(path: Path):
    """返回 {image_id: (qvec, tvec, camera_id, name, points2d)}，points2d 是 (N,3) 的 (x,y,p3d)。"""
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    out = {}
    for i in range(0, len(lines), 2):
        f = lines[i].split()
        image_id = int(f[0])
        qvec = np.array([float(v) for v in f[1:5]])
        tvec = np.array([float(v) for v in f[5:8]])
        values = np.array([float(v) for v in lines[i + 1].split()]).reshape(-1, 3)
        out[image_id] = (qvec, tvec, int(f[8]), f[9], values)
    return out


def read_points3d_txt(path: Path):
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        out[int(f[0])] = np.array([float(v) for v in f[1:4]])
    return out


def read_camera(path: Path):
    # TXT 格式里模型写的是**名字**（SIMPLE_RADIAL…），不是编号
    row = [l for l in path.read_text().splitlines()
           if l.strip() and not l.startswith("#")][0].split()
    model = row[1]
    params = np.array([float(v) for v in row[4:]])
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        k = 0.0
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
    else:
        raise SystemExit(f"暂不支持相机模型 {model}")
    return f, cx, cy, k


# ---------------------------------------------------------------- 几何


def undistort(points: np.ndarray, f: float, cx: float, cy: float, k: float) -> np.ndarray:
    """SIMPLE_RADIAL 去畸变，返回归一化坐标 (N,2)。"""
    x = (points[:, 0] - cx) / f
    y = (points[:, 1] - cy) / f
    if k != 0:
        # x_d = x_u (1 + k r_u^2)，两轮不动点迭代即可（k 很小）
        for _ in range(3):
            r2 = x * x + y * y
            scale = 1.0 + k * r2
            x = (points[:, 0] - cx) / f / scale
            y = (points[:, 1] - cy) / f / scale
    return np.column_stack([x, y])


def dlt(bearings: np.ndarray, world: np.ndarray) -> np.ndarray | None:
    """已知内参的 DLT：bearing 已是归一化坐标，返回 3x4 的 [R|t]。"""
    n = len(bearings)
    if n < 6:
        return None
    rows = np.zeros((2 * n, 12))
    ones = np.ones(n)
    homog = np.column_stack([world, ones])
    for i in range(n):
        x, y = bearings[i]
        rows[2 * i, 0:4] = homog[i]
        rows[2 * i, 8:12] = -x * homog[i]
        rows[2 * i + 1, 4:8] = homog[i]
        rows[2 * i + 1, 8:12] = -y * homog[i]
    _, _, vt = np.linalg.svd(rows)
    P = vt[-1].reshape(3, 4)
    rotation = P[:, :3]
    scale = np.linalg.norm(rotation) / np.sqrt(3.0)
    if scale < 1e-12:
        return None
    rotation = rotation / scale
    translation = P[:, 3] / scale
    # 把 R 正交化（DLT 解一般只是近似正交）
    u, _, vt2 = np.linalg.svd(rotation)
    rotation = u @ vt2
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
        translation = -translation
    return np.column_stack([rotation, translation])


def reprojection_error(pose: np.ndarray, bearings: np.ndarray, world: np.ndarray) -> np.ndarray:
    homog = np.column_stack([world, np.ones(len(world))])
    projected = homog @ pose.T                      # (N,3) 相机坐标
    depth = projected[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    x = projected[:, 0] / safe
    y = projected[:, 1] / safe
    return np.linalg.norm(np.column_stack([x, y]) - bearings, axis=1)


def ransac_pnp(bearings: np.ndarray, world: np.ndarray, threshold: float,
               iterations: int = 3000, seed: int = 0):
    """返回 (pose 3x4, 内点掩码)。threshold 是归一化坐标下的重投影误差。"""
    n = len(bearings)
    if n < 6:
        return None, None
    rng = np.random.default_rng(seed)
    best_pose, best_mask = None, None
    for _ in range(iterations):
        sample = rng.choice(n, 6, replace=False)
        pose = dlt(bearings[sample], world[sample])
        if pose is None:
            continue
        error = reprojection_error(pose, bearings, world)
        mask = np.abs(error) < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_pose, best_mask = pose, mask
            if mask.sum() > 0.9 * n:
                break
    if best_pose is None or best_mask.sum() < 6:
        return None, None
    # 用全部内点重解一次
    refined = dlt(bearings[best_mask], world[best_mask])
    if refined is not None:
        error = reprojection_error(refined, bearings, world)
        mask = np.abs(error) < threshold
        if mask.sum() >= 6:
            return refined, mask
    return best_pose, best_mask


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """COLMAP 的四元数是 (w, x, y, z)。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """COLMAP 用 (w, x, y, z)。"""
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------- 主流程


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", default="sparse/0")
    parser.add_argument("--matches", default="supp/cross_matches.txt")
    parser.add_argument("--output", default="supp/model_supplemented")
    parser.add_argument("--pixel-threshold", type=float, default=4.0,
                        help="RANSAC 内点判据（像素）")
    parser.add_argument("--min-inliers", type=int, default=12)
    args = parser.parse_args()

    root = Path(args.job).resolve()
    model_bin = root / args.model
    started = time.monotonic()

    text = to_txt(model_bin, root / "supp" / "model_txt")
    f, cx, cy, k = read_camera(text / "cameras.txt")
    images = read_images_txt(text / "images.txt")
    points3d = read_points3d_txt(text / "points3D.txt")
    log(f"原模型 {len(images)} 张图 / {len(points3d)} 个三维点，内参 f={f:.1f} k={k:.5f}")

    # 旧图：feature 下标 -> 三维点
    feature_to_point: dict[int, dict[int, int]] = {}
    for image_id, (_, _, _, _, values) in images.items():
        mapping = {}
        for index, (_, _, point_id) in enumerate(values):
            point_id = int(point_id)
            if point_id > 0:
                mapping[index] = point_id
        feature_to_point[image_id] = mapping

    # 补拍帧的关键点
    import sqlite3

    database = sqlite3.connect(root / "supp" / "database.db")
    names = {name: image_id for image_id, name in
             database.execute("select image_id, name from images")}
    supp_ids = {image_id: name for name, image_id in names.items() if name.startswith("supp_")}

    def keypoints(image_id: int) -> np.ndarray:
        """keypoints 表每行是 6 个 float32（x, y, size, …），共 24 字节。

        ⚠️ 注意和 descriptors 表不一致：descriptors 的 cols 记的是**字节数**
        （512 = 128 维 float32），keypoints 的 cols 记的是**元素个数**（6）。
        """
        row = database.execute(
            "select rows, cols, data from keypoints where image_id=?", (image_id,)
        ).fetchone()
        values = np.frombuffer(row[2], dtype=np.float32).reshape(row[0], row[1])
        return values[:, :2]

    # 读跨场次匹配，按补拍帧聚合
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    with (root / args.matches).open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            a, b, count = int(fields[0]), int(fields[1]), int(fields[2])
            values = [int(v) for v in fields[3:3 + 2 * count]]
            supp_is_a = a in supp_ids
            supp_image = a if supp_is_a else b
            old_image = b if supp_is_a else a
            for i in range(count):
                supp_index = values[2 * i] if supp_is_a else values[2 * i + 1]
                old_index = values[2 * i + 1] if supp_is_a else values[2 * i]
                grouped.setdefault(supp_image, []).append((old_image, supp_index, old_index))
    log(f"读到 {len(grouped)} 张补拍帧的跨场次匹配")

    threshold = args.pixel_threshold / f
    # 旧图位姿：用作补拍帧的初值
    old_poses = {
        iid: np.column_stack([quaternion_to_rotation(q), t])
        for iid, (q, t, _, _, _) in images.items()
    }

    def refine(pose, bearings, world, start_px, end_px, rounds=6):
        """从 pose 出发迭代收紧内点并重解，返回 (pose, 内点掩码)。

        ⚠️ 这里的阈值单位是**像素**：reprojection_error 返回的是归一化坐标下的
        误差，要乘 f 才是像素。早先把像素误差和归一化阈值直接比，导致几乎筛不出
        任何内点，一张都配不上。
        """
        for step in range(rounds):
            limit = start_px + (end_px - start_px) * step / max(rounds - 1, 1)
            error = reprojection_error(pose, bearings, world) * f
            mask = error < limit
            if mask.sum() < 6:
                return pose, mask
            better = dlt(bearings[mask], world[mask])
            if better is None:
                return pose, mask
            pose = better
        error = reprojection_error(pose, bearings, world) * f
        return pose, error < end_px

    poses: dict[int, tuple[np.ndarray, np.ndarray, dict[int, int]]] = {}
    for supp_image, entries in sorted(grouped.items()):
        votes: dict[int, dict[int, int]] = {}
        support: dict[int, int] = {}
        for old_image, supp_index, old_index in entries:
            point_id = feature_to_point.get(old_image, {}).get(old_index)
            if point_id is None:
                continue
            votes.setdefault(supp_index, {})
            votes[supp_index][point_id] = votes[supp_index].get(point_id, 0) + 1
            support[old_image] = support.get(old_image, 0) + 1
        if len(votes) < 6:
            continue
        pixels = keypoints(supp_image)
        bearings, world, keep = [], [], {}
        for supp_index, tally in votes.items():
            point_id = max(tally, key=tally.get)
            point = points3d.get(point_id)
            if point is None:
                continue
            keep[supp_index] = point_id
            bearings.append(pixels[supp_index])
            world.append(point)
        if len(bearings) < 6:
            continue
        bearings = undistort(np.array(bearings, dtype=np.float64), f, cx, cy, k)
        world = np.array(world)

        # 关键：用「匹配最多的那张旧帧」的位姿当初值。
        # 补拍帧与它视角相近，初值下正确对应关系就已经接近重合，
        # 迭代收紧阈值即可把它们分离出来 —— 比盲抽 6 点做 RANSAC 稳得多
        # （实测内点率只有两成左右，盲抽 6 点连中的概率约 6e-5，抽不到）。
        best = None
        for old_image in sorted(support, key=support.get, reverse=True)[:5]:
            pose = old_poses.get(old_image)
            if pose is None:
                continue
            candidate, mask = refine(pose, bearings, world, 40.0, args.pixel_threshold)
            count = int(mask.sum())
            if best is None or count > best[1]:
                best = (candidate, count, mask)
        if best is None or best[1] < args.min_inliers:
            continue
        pose, count, mask = best
        inliers = {index: keep[index] for index, flag in zip(keep, mask) if flag}
        poses[supp_image] = (pose[:, :3], pose[:, 3], inliers)

    log(f"配准成功 {len(poses)} / {len(grouped)} 张（阈值 {args.pixel_threshold} 像素，"
        f"≥{args.min_inliers} 内点），耗时 {time.monotonic() - started:.0f}s")
    if not poses:
        raise SystemExit("一张都没配上")

    # 相机位置是否构成一个合理的环绕轨道？——这是没有真值时的最佳自检
    centers = np.array([-r.T @ t for r, t, _ in poses.values()])
    target = np.mean([points3d[p] for _, _, inl in poses.values() for p in inl.values()], axis=0)
    radius = np.linalg.norm(centers - target, axis=1)
    log(f"相机到物体中心距离：中位 {np.median(radius):.4f}  最小 {radius.min():.4f}  "
        f"最大 {radius.max():.4f}（离散度 {radius.std() / radius.mean() * 100:.1f}%）")

    # 写回模型
    output = root / args.output
    if output.exists():
        shutil.rmtree(output)
    work = root / "supp" / "model_pnp_txt"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(text, work)

    # 复制已在内存里的内容，追加新图
    with (work / "images.txt").open("a") as handle:
        for supp_image, (rotation, translation, inliers) in sorted(poses.items()):
            image_id = supp_ids[supp_image]
            q = rotation_to_quaternion(rotation)
            pixels = keypoints(image_id)
            line = [str(image_id)] + [f"{v:.17g}" for v in q] + \
                   [f"{v:.17g}" for v in translation] + ["1", supp_image]
            handle.write(" ".join(line) + "\n")
            entries = []
            for index in range(len(pixels)):
                point_id = inliers.get(index, 0)
                entries.extend([f"{pixels[index, 0]:.6f}", f"{pixels[index, 1]:.6f}", str(point_id)])
            handle.write(" ".join(entries) + "\n")

    frame_lines = (work / "frames.txt").read_text().splitlines()
    header = [l for l in frame_lines if l.startswith("#")]
    body = [l for l in frame_lines if l.strip() and not l.startswith("#")]
    next_frame = 1 + max(int(l.split()[0]) for l in body)
    with (work / "frames.txt").open("w") as handle:
        handle.write("\n".join(header + body) + "\n")
        for offset, (supp_image, (rotation, translation, _)) in enumerate(sorted(poses.items())):
            image_id = supp_ids[supp_image]
            q = rotation_to_quaternion(rotation)
            row = [str(next_frame + offset), "1"] + [f"{v:.17g}" for v in q] + \
                  [f"{v:.17g}" for v in translation] + ["1", "CAMERA", "1", str(image_id)]
            handle.write(" ".join(row) + "\n")

    output.mkdir(parents=True)
    result = subprocess.run(
        [COLMAP, "model_converter", "--input_path", str(work),
         "--output_path", str(output), "--output_type", "BIN"],
        capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-2000:])
        raise SystemExit("写回模型失败")
    log(f"新模型：{output}（{len(images)} + {len(poses)} = {len(images) + len(poses)} 张图）")


if __name__ == "__main__":
    main()
