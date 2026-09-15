#!/usr/bin/env python3
"""六面物块的朝向审计与安全融合辅助工具。

这个工具刻意把“判定姿态”和“写入合并结果”分开：近似立方体在几何上有 24 个
等价旋转，ICP 的高分不能证明凹槽、圆孔和木纹处在正确的物理面。先用 ``preview``
输出每个候选相对已批准基准的截图和原始坐标变换，审核通过后才能把该变换交给
后续的面级融合步骤。

示例：
    ./.venv/bin/python tools/fusion/six_face_pipeline.py preview \
      --base storage/.../merged_cube_final.glb \
      --source storage/.../object_new.ply \
      --candidates storage/.../merge_two_jobs.json \
      --out storage/yellow-cube/six-face-v1/orientation/scan-a
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
# Matplotlib 默认写 ~/.config；在受限环境中会退回临时目录、每个子进程都重建缓存。
# 把缓存放到项目存储中，环拍截图的并行子进程就无需重复初始化。
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "storage" / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "tools" / "fusion"))

from merge_models import render_overlay, render_turntable  # noqa: E402
from merge_two_jobs import normalized_to_original  # noqa: E402


def load_cloud(path: Path, maximum: int, seed: int) -> o3d.geometry.PointCloud:
    """Load a colored mesh/PLY and deterministically keep a diagnostic sample."""
    if path.suffix.lower() in {".glb", ".gltf", ".obj", ".stl"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not len(mesh.vertices):
            raise SystemExit(f"空网格: {path}")
        cloud = mesh.sample_points_uniformly(number_of_points=maximum)
        if mesh.has_vertex_colors():
            # Uniform sampling retains colours in current Open3D, but keep a safe fallback.
            if not cloud.has_colors():
                cloud.paint_uniform_color((0.72, 0.55, 0.28))
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise SystemExit(f"空点云: {path}")
    if len(cloud.points) > maximum:
        indices = np.random.default_rng(seed).choice(len(cloud.points), maximum, replace=False)
        cloud = cloud.select_by_index(indices.tolist())
    if not cloud.has_colors():
        cloud.paint_uniform_color((0.72, 0.55, 0.28))
    return cloud


def candidate_matrix(data: dict, rank: int) -> np.ndarray:
    results = data.get("results", [])
    if not 1 <= rank <= len(results):
        raise SystemExit(f"候选 rank {rank} 不存在（文件内共 {len(results)} 个）")
    ra, cea, sa = data["job_a_norm"]
    rb, ceb, sb = data["job_b_norm"]
    return normalized_to_original(
        np.asarray(results[rank - 1]["transform"], dtype=float),
        np.asarray(ra, dtype=float), np.asarray(cea, dtype=float), float(sa),
        np.asarray(rb, dtype=float), np.asarray(ceb, dtype=float), float(sb),
    )


def render_preview_candidate(payload: tuple) -> int:
    """Render one candidate in an isolated process; Matplotlib is not thread-safe."""
    (rank, item, matrix, base_points, base_colours, source_points, source_colours,
     output_directory, turntable_points) = payload
    destination = Path(output_directory)
    moved = source_points @ matrix[:3, :3].T + matrix[:3, 3]
    title = (
        f"candidate {rank}: symmetry {item.get('symmetry_index')} | "
        f"signature {item.get('signature', float('nan')):.4f} | "
        f"fitness {item.get('fitness', float('nan')):.4f}"
    )
    render_overlay(base_points, moved, destination / f"candidate_{rank:02d}_overlay.png", title)
    render_turntable(
        np.concatenate((base_points, moved), axis=0),
        np.concatenate((base_colours, source_colours), axis=0),
        destination / f"candidate_{rank:02d}_turntable.png", title,
        max_points=turntable_points,
    )
    return rank


def preview(args: argparse.Namespace) -> int:
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    base = load_cloud(Path(args.base), args.max_points, 0)
    source = load_cloud(Path(args.source), args.max_points, 1)
    base_points = np.asarray(base.points)
    base_colours = np.asarray(base.colors)
    source_points = np.asarray(source.points)
    source_colours = np.asarray(source.colors)

    report = {
        "status": "orientation_review_required",
        "approved": False,
        "base": str(Path(args.base)),
        "source": str(Path(args.source)),
        "candidate_file": str(Path(args.candidates)),
        "base_sample_points": int(len(base_points)),
        "source_sample_points": int(len(source_points)),
        "candidates": [],
        "review_rule": (
            "截图中凹槽、圆孔及木纹须落在同一物理面；仅 ICP/fitness 高分不能批准。"
        ),
    }
    count = min(args.top, len(data.get("results", [])))
    jobs = []
    for rank in range(1, count + 1):
        item = data["results"][rank - 1]
        matrix = candidate_matrix(data, rank)
        jobs.append((rank, item, matrix, base_points, base_colours, source_points, source_colours,
                     str(destination), args.turntable_points))
        report["candidates"].append({
            "rank": rank,
            "symmetry_index": item.get("symmetry_index"),
            "signature": item.get("signature"),
            "fitness": item.get("fitness"),
            "rmse": item.get("rmse"),
            "color": item.get("color"),
            "matrix_source_to_base": matrix.tolist(),
            "overlay": f"candidate_{rank:02d}_overlay.png",
            "turntable": f"candidate_{rank:02d}_turntable.png",
        })
    workers = max(1, min(args.workers, count))
    print(f"候选截图：{count} 组，渲染进程 {workers}")
    if workers == 1:
        for done, job in enumerate(jobs, 1):
            rank = render_preview_candidate(job)
            print(f"已出图 {done}/{count}: candidate_{rank:02d}_overlay.png", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(render_preview_candidate, job) for job in jobs]
            for done, future in enumerate(as_completed(pending), 1):
                rank = future.result()
                print(f"已出图 {done}/{count}: candidate_{rank:02d}_overlay.png", flush=True)
    (destination / "orientation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"朝向审计写入 {destination / 'orientation_audit.json'}")
    return 0


def face_depth_grid(
    points: np.ndarray, lo: np.ndarray, hi: np.ndarray, axis: int, sign: int, bins: int
) -> np.ndarray:
    """Return a face-aligned depth image in the base coordinate system.

    The face plane comes from the approved base rather than the candidate itself.  A
    wrong 90-degree rotation therefore leaves the quarter-ring on a different panel
    instead of making it appear superficially correct through re-normalisation.
    """
    other = [item for item in range(3) if item != axis]
    in_bounds = (
        (points[:, other[0]] >= lo[other[0]]) & (points[:, other[0]] <= hi[other[0]])
        & (points[:, other[1]] >= lo[other[1]]) & (points[:, other[1]] <= hi[other[1]])
    )
    selected = points[in_bounds]
    grid = np.full((bins, bins), np.nan)
    if not len(selected):
        return grid
    rows = np.clip(
        ((selected[:, other[0]] - lo[other[0]]) /
         max(hi[other[0]] - lo[other[0]], 1e-9) * bins).astype(int), 0, bins - 1
    )
    columns = np.clip(
        ((selected[:, other[1]] - lo[other[1]]) /
         max(hi[other[1]] - lo[other[1]], 1e-9) * bins).astype(int), 0, bins - 1
    )
    heights = selected[:, axis] * sign
    flat = rows * bins + columns
    accumulator = np.full(bins * bins, -np.inf)
    np.maximum.at(accumulator, flat, heights)
    face = hi[axis] if sign > 0 else -lo[axis]
    grid = (face - accumulator.reshape(bins, bins)) / float(np.max(hi - lo))
    grid[~np.isfinite(grid)] = np.nan
    # Points far behind the visible outer surface do not describe this face.  They
    # are often the opposite face projected through a hole in a partial scan.
    grid[(grid < -0.04) | (grid > 0.32)] = np.nan
    return grid


def face_review(args: argparse.Namespace) -> int:
    """Render six physical base faces next to a transformed candidate's faces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    base = load_cloud(Path(args.base), args.max_points, 10)
    source = load_cloud(Path(args.source), args.max_points, 11)
    base_points = np.asarray(base.points)
    source_points = np.asarray(source.points)
    lo, hi = np.percentile(base_points, (0.5, 99.5), axis=0)
    names = ((0, 1, "X+"), (0, -1, "X-"), (1, 1, "Y+"),
             (1, -1, "Y-"), (2, 1, "Z+"), (2, -1, "Z-"))
    result = []
    for rank in args.ranks:
        if rank < 1 or rank > len(data.get("results", [])):
            raise SystemExit(f"无效 rank: {rank}")
        item = data["results"][rank - 1]
        matrix = candidate_matrix(data, rank)
        moved = source_points @ matrix[:3, :3].T + matrix[:3, 3]
        figure, axes = plt.subplots(6, 2, figsize=(8.4, 21), dpi=125)
        for row, (axis, sign, label) in enumerate(names):
            for column, (title, points) in enumerate((("approved base", base_points),
                                                        ("candidate source", moved))):
                grid = face_depth_grid(points, lo, hi, axis, sign, args.bins)
                image = axes[row, column].imshow(
                    grid.T, origin="lower", cmap="magma", vmin=0.0, vmax=0.25,
                    interpolation="nearest",
                )
                axes[row, column].set_title(f"{label} — {title}")
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.55,
                        label="inward depth / cube size")
        figure.suptitle(
            f"candidate {rank}, symmetry {item.get('symmetry_index')} | "
            f"signature {item.get('signature', float('nan')):.4f}", y=0.998
        )
        figure.tight_layout()
        output = destination / f"candidate_{rank:02d}_six_faces.png"
        figure.savefig(output)
        plt.close(figure)
        result.append({"rank": rank, "face_review": output.name})
        print(f"已出六面比对图: {output.name}")
    (destination / "face_review.json").write_text(
        json.dumps({
            "approved": False,
            "base_bounds_p005_p995": [lo.tolist(), hi.tolist()],
            "rank_outputs": result,
            "review_rule": "四分之一圆环及孔洞须在左右两栏的同一物理面、同一位置后才能批准。",
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def strip_plane(args: argparse.Namespace) -> int:
    """Remove one overwhelmingly large planar support/background sheet.

    This is deliberately limited to one plane.  Repeatedly deleting planes from a
    cube scan would start deleting actual cube faces; callers must review the report
    and only use it where a non-object plane dominates the scan.
    """
    input_path = Path(args.input)
    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points)
    if len(points) < 1_000:
        raise SystemExit("点云太小，无法可靠识别支撑面")
    sample_size = min(len(points), args.sample_points)
    if sample_size < len(points):
        indices = np.random.default_rng(args.seed).choice(len(points), sample_size, replace=False)
        sample = cloud.select_by_index(indices.tolist())
    else:
        sample = cloud
    extent = points.max(axis=0) - points.min(axis=0)
    threshold = float(np.max(extent) * args.distance_ratio)
    plane, inliers = sample.segment_plane(threshold, 3, args.iterations)
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    # Classify every original point using the sampled plane; this avoids feeding a
    # 20+ million point cloud into RANSAC while still removing the entire sheet.
    distance = np.abs(points @ normal + float(plane[3]))
    remove = distance <= threshold
    ratio = float(remove.mean())
    if ratio < args.minimum_ratio:
        raise SystemExit(
            f"最大平面只占 {ratio:.1%}，低于安全下限 {args.minimum_ratio:.1%}；拒绝删除。"
        )
    cleaned = cloud.select_by_index(np.flatnonzero(~remove).tolist())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(destination), cleaned, write_ascii=False)
    report = {
        "input": str(input_path),
        "output": str(destination),
        "operation": "remove_one_dominant_non_object_plane",
        "approved": False,
        "sample_points": int(sample_size),
        "distance_threshold": threshold,
        "plane": [float(value) for value in plane],
        "removed_points": int(remove.sum()),
        "removed_ratio": ratio,
        "remaining_points": int((~remove).sum()),
        "safety_note": "只去掉一张占比异常大的平面；其余面仍须经姿态和面级核验。",
    }
    report_path = Path(args.report) if args.report else destination.with_suffix(".plane_audit.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="六面物块的安全朝向审计")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("preview", help="将候选姿态出成叠加和环拍审计截图")
    command.add_argument("--base", required=True, help="已批准基准模型或点云")
    command.add_argument("--source", required=True, help="单次扫描中已提取的物块点云")
    command.add_argument("--candidates", required=True, help="merge_two_jobs.py 的候选 JSON")
    command.add_argument("--out", required=True, help="只写入新的审计目录")
    command.add_argument("--top", type=int, default=6, help="出图的候选数")
    command.add_argument("--workers", type=int, default=4,
                         help="独立渲染进程数；默认 4，Matplotlib 截图可安全并行")
    command.add_argument("--max-points", type=int, default=35_000, help="每个模型的截图采样上限")
    command.add_argument("--turntable-points", type=int, default=18_000, help="环拍图采样上限")
    command.set_defaults(handler=preview)
    faces = subparsers.add_parser("face-review", help="按六个物理面输出基准/候选深度图")
    faces.add_argument("--base", required=True)
    faces.add_argument("--source", required=True)
    faces.add_argument("--candidates", required=True)
    faces.add_argument("--out", required=True)
    faces.add_argument("--ranks", type=int, nargs="+", required=True)
    faces.add_argument("--max-points", type=int, default=250_000)
    faces.add_argument("--bins", type=int, default=72)
    faces.set_defaults(handler=face_review)
    plane = subparsers.add_parser("strip-plane", help="仅移除一张占比异常大的支撑/背景平面")
    plane.add_argument("--input", required=True)
    plane.add_argument("--output", required=True)
    plane.add_argument("--report", help="默认写到 output 同名 .plane_audit.json")
    plane.add_argument("--sample-points", type=int, default=500_000)
    plane.add_argument("--distance-ratio", type=float, default=0.008)
    plane.add_argument("--iterations", type=int, default=1_500)
    plane.add_argument("--minimum-ratio", type=float, default=0.30)
    plane.add_argument("--seed", type=int, default=0)
    plane.set_defaults(handler=strip_plane)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
