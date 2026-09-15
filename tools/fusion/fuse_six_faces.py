#!/usr/bin/env python3
"""安全合并三次扫描的五面点云。

输出以已批准基准点云为锚。每个补充扫描先变换到基准坐标，再删除指定的
物理支撑面；只有同时满足表面距离、法线方向、颜色一致性和立方体外壳约束的
点才会加入。脚本不会修改基准文件，也不会把任何未通过核验的面写入结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "fusion"))
from merge_two_jobs import normalized_to_original  # noqa: E402
from merge_models import render_turntable  # noqa: E402


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    if path.suffix.lower() in {".glb", ".gltf", ".obj", ".stl"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not len(mesh.vertices):
            raise SystemExit(f"空网格: {path}")
        cloud = mesh.sample_points_uniformly(number_of_points=1_500_000)
        if mesh.has_vertex_colors() and not cloud.has_colors():
            cloud.paint_uniform_color((0.72, 0.55, 0.28))
    else:
        cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise SystemExit(f"空点云: {path}")
    if not cloud.has_colors():
        cloud.paint_uniform_color((0.72, 0.55, 0.28))
    return cloud


def matrix_from_json(path: Path, rank: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not 1 <= rank <= len(results):
        raise SystemExit(f"{path}: 候选 rank {rank} 不存在")
    ra, ca, sa = data["job_a_norm"]
    rb, cb, sb = data["job_b_norm"]
    return normalized_to_original(
        np.asarray(results[rank - 1]["transform"], dtype=float),
        np.asarray(ra, dtype=float), np.asarray(ca, dtype=float), float(sa),
        np.asarray(rb, dtype=float), np.asarray(cb, dtype=float), float(sb),
    )


def parse_face(face: str) -> tuple[int, int]:
    if len(face) != 2 or face[0].lower() not in "xyz" or face[1] not in "+-":
        raise SystemExit(f"无效支撑面 {face}，应为 x+、x-、y+、y-、z+ 或 z-")
    return {"x": 0, "y": 1, "z": 2}[face[0].lower()], 1 if face[1] == "+" else -1


def transform_source(cloud: o3d.geometry.PointCloud, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.014, max_nn=50)
        )
    points = np.asarray(cloud.points) @ matrix[:3, :3].T + matrix[:3, 3]
    normals = np.asarray(cloud.normals) @ matrix[:3, :3].T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, normals, np.asarray(cloud.colors)


def accept_source(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    base_points: np.ndarray,
    base_normals: np.ndarray,
    base_colors: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    drop_face: str,
    face_depth: float,
    max_distance: float,
    min_normal_dot: float,
    max_color_distance: float,
    novel_distance: float,
    voxel: float,
    chunk_size: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    lo, hi = bounds
    axis, sign = parse_face(drop_face)
    # Index order is x-, y-, z-, x+, y+, z+.
    outer = np.concatenate((points - lo, hi - points), axis=1)
    face_distance = outer.min(axis=1)
    face_index = outer.argmin(axis=1)
    # 0..2 are negative faces, 3..5 are positive faces.
    dropped_index = axis if sign < 0 else axis + 3
    shell = face_distance <= face_depth
    shell &= face_index != dropped_index
    shell_count = int(shell.sum())
    if shell_count == 0:
        return o3d.geometry.PointCloud(), {
            "input_points": int(len(points)), "shell_points": 0,
            "accepted_points": 0, "accepted_ratio": 0.0,
            "dropped_face": drop_face,
        }
    tree = cKDTree(base_points)
    indices = np.flatnonzero(shell)
    chosen_points: list[np.ndarray] = []
    chosen_normals: list[np.ndarray] = []
    chosen_colors: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    normal_dots: list[np.ndarray] = []
    color_distances: list[np.ndarray] = []
    valid_count = distance_count = normal_count = color_count = 0
    # Querying 25 million points in one cKDTree call allocates several temporary
    # output arrays.  Fixed chunks keep resident memory predictable; SciPy still
    # uses all CPU cores inside each chunk.
    for start in range(0, len(indices), chunk_size):
        local = indices[start:start + chunk_size]
        distance, nearest = tree.query(points[local], k=1, workers=-1)
        normal_dot = np.abs(np.einsum("ij,ij->i", normals[local], base_normals[nearest]))
        color_distance = np.linalg.norm(colors[local] - base_colors[nearest], axis=1)
        valid = (
            (distance <= max_distance)
            & (normal_dot >= min_normal_dot)
            & (color_distance <= max_color_distance)
            # Points already represented at the base sampling density need not be copied;
            # retaining only a narrow novel band avoids double surfaces along every edge.
            & (distance >= novel_distance)
        )
        distance_count += int((distance <= max_distance).sum())
        normal_count += int((normal_dot >= min_normal_dot).sum())
        color_count += int((color_distance <= max_color_distance).sum())
        valid_count += int(valid.sum())
        if valid.any():
            selected_normals = normals[local][valid].copy()
            selected_dot = np.einsum("ij,ij->i", selected_normals, base_normals[nearest[valid]])
            selected_normals[selected_dot < 0] *= -1
            chosen_points.append(points[local][valid])
            chosen_normals.append(selected_normals)
            chosen_colors.append(colors[local][valid])
            distances.append(distance[valid])
            normal_dots.append(normal_dot[valid])
            color_distances.append(color_distance[valid])
    selected_points = np.concatenate(chosen_points) if chosen_points else np.empty((0, 3))
    selected_normals = np.concatenate(chosen_normals) if chosen_normals else np.empty((0, 3))
    selected_colors = np.concatenate(chosen_colors) if chosen_colors else np.empty((0, 3))
    output = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_points))
    output.colors = o3d.utility.Vector3dVector(selected_colors)
    output.normals = o3d.utility.Vector3dVector(selected_normals)
    if voxel > 0 and len(output.points) > 100:
        output = output.voxel_down_sample(voxel)
    report = {
        "input_points": int(len(points)),
        "shell_points": shell_count,
        "dropped_face_points": int((~shell).sum()),
        "distance_valid_points": distance_count,
        "normal_valid_points": normal_count,
        "color_valid_points": color_count,
        "accepted_before_voxel": valid_count,
        "accepted_points": int(len(output.points)),
        "accepted_ratio": float(len(output.points) / max(len(points), 1)),
        "distance_median": float(np.median(np.concatenate(distances))) if distances else None,
        "normal_dot_median": float(np.median(np.concatenate(normal_dots))) if normal_dots else None,
        "color_distance_median": float(np.median(np.concatenate(color_distances))) if color_distances else None,
        "dropped_face": drop_face,
        "face_depth": face_depth,
        "max_distance": max_distance,
        "min_normal_dot": min_normal_dot,
        "max_color_distance": max_color_distance,
        "novel_distance": novel_distance,
        "voxel": voxel,
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description="以批准基准为锚安全合并三次五面扫描")
    parser.add_argument("--base", required=True, help="批准基准点云 PLY")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--source", action="append", nargs=3, metavar=("PLY", "CANDIDATES_JSON", "RANK"),
        required=True, help="可重复三次：点云、候选 JSON、候选排名（通常为 1）",
    )
    parser.add_argument("--drop-face", action="append", default=[],
                        help="与 --source 一一对应的物理支撑面，默认每次 y-")
    parser.add_argument("--face-depth", type=float, default=0.12)
    parser.add_argument("--max-distance", type=float, default=0.025)
    parser.add_argument("--min-normal-dot", type=float, default=0.55)
    parser.add_argument("--max-color-distance", type=float, default=0.35)
    parser.add_argument("--novel-distance", type=float, default=0.0012)
    parser.add_argument("--voxel", type=float, default=0.0010)
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="每次最近邻核验的点数，控制高密度扫描的内存峰值")
    parser.add_argument("--preview-points", type=int, default=30_000)
    args = parser.parse_args()
    # 第一段视频已经是批准基准（含它自己的两个子模型），故仅需要传入另外两段
    # 扫描；基准本身也会在审计中作为三段输入中的第一段记录。
    if len(args.source) != 2:
        raise SystemExit("基准代表第一段视频；请提供另外两组 --source")
    drops = args.drop_face or ["y-"] * 2
    if len(drops) != 2:
        raise SystemExit("--drop-face 必须提供两次，或完全省略以全部使用 y-")

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    base_cloud = load_cloud(Path(args.base))
    base_points = np.asarray(base_cloud.points)
    if not base_cloud.has_normals():
        base_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=40)
        )
    base_normals = np.asarray(base_cloud.normals)
    base_colors = np.asarray(base_cloud.colors)
    lo, hi = np.percentile(base_points, (0.5, 99.5), axis=0)
    bounds = (lo, hi)
    merged_points = [base_points]
    merged_normals = [base_normals]
    merged_colors = [base_colors]
    audit = {
        "approved": False,
        "base": str(Path(args.base)),
        "base_points": int(len(base_points)),
        "base_source_role": "视频 1：已批准的高密度核心立方体合并结果",
        "base_bounds_p005_p995": [lo.tolist(), hi.tolist()],
        "sources": [],
        "rule": "基准保留完整六面；每个补充源先排除指定支撑面，再通过距离/法线/颜色/外壳核验。",
    }
    for index, (source_path, candidate_path, rank_text) in enumerate(args.source, 1):
        source_cloud = load_cloud(Path(source_path))
        matrix = matrix_from_json(Path(candidate_path), int(rank_text))
        points, normals, colors = transform_source(source_cloud, matrix)
        accepted, report = accept_source(
            points, normals, colors, base_points, base_normals, base_colors, bounds,
            drops[index - 1], args.face_depth, args.max_distance, args.min_normal_dot,
            args.max_color_distance, args.novel_distance, args.voxel,
            args.chunk_size,
        )
        report.update({
            "source": source_path,
            "candidate_file": candidate_path,
            "candidate_rank": int(rank_text),
            "matrix_source_to_base": matrix.tolist(),
        })
        audit["sources"].append(report)
        if len(accepted.points):
            merged_points.append(np.asarray(accepted.points))
            merged_normals.append(np.asarray(accepted.normals))
            merged_colors.append(np.asarray(accepted.colors))
            o3d.io.write_point_cloud(str(destination / f"source_{index}_accepted.ply"), accepted,
                                     write_ascii=False)
        print(f"源 {index}: {report['input_points']:,} -> {report['accepted_points']:,} 点通过核验")

    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(merged_points)))
    result.normals = o3d.utility.Vector3dVector(np.concatenate(merged_normals))
    result.colors = o3d.utility.Vector3dVector(np.concatenate(merged_colors))
    output = destination / "merged_six_face_points.ply"
    o3d.io.write_point_cloud(str(output), result, write_ascii=False)
    audit["merged_points"] = int(len(result.points))
    audit["supplement_points"] = int(len(result.points) - len(base_points))
    audit["output"] = str(output)
    audit["approved"] = False
    (destination / "six_face_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 低密度环拍仅用于快速发现空洞/双层，绝不作为批准依据。
    sample_index = np.random.default_rng(0).choice(
        len(result.points), min(args.preview_points, len(result.points)), replace=False
    )
    render_turntable(
        np.asarray(result.points)[sample_index], np.asarray(result.colors)[sample_index],
        destination / "merged_six_face_turntable.png", "six-face merge (audit pending)",
        max_points=args.preview_points,
    )
    print(f"写出 {output}（{len(result.points):,} 点）")
    print(f"审计报告 {destination / 'six_face_audit.json'}；当前 approved=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
