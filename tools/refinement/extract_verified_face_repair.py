#!/usr/bin/env python3
"""Extract only a verified supplementary scan patch for one physical cube face.

This script deliberately fails closed.  A near-cubical object has several
geometrically plausible 90-degree rotations, so a global ICP result alone is
not enough evidence to copy a face.  Every exported supplement point must
agree with the accepted model in position, normal direction, and colour, and
must belong to the requested face volume.

The output is a point-cloud patch only.  It never overwrites the accepted
model or a production GLB; mesh reconstruction is a separate, reviewable
step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


FACE_AXES = {"x": 0, "y": 1, "z": 2}


def parse_face(token: str) -> tuple[int, int]:
    token = token.lower().strip()
    if len(token) != 2 or token[0] not in FACE_AXES or token[1] not in "+-":
        raise ValueError("face 必须是 x+、x-、y+、y-、z+ 或 z-")
    return FACE_AXES[token[0]], 1 if token[1] == "+" else -1


def transform_points_and_normals(
    cloud: o3d.geometry.PointCloud, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the recorded B-to-A transform and estimate source normals."""
    if not cloud.has_colors():
        raise RuntimeError("补拍点云没有颜色，无法做颜色一致性核验")
    if not cloud.has_normals():
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=40)
        )
    points = np.asarray(cloud.points) @ matrix[:3, :3].T + matrix[:3, 3]
    normals = np.asarray(cloud.normals) @ matrix[:3, :3].T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, normals, np.asarray(cloud.colors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mesh", required=True, help="已验收的 GLB，用于逐点核验")
    parser.add_argument("--supplement", required=True, help="最新导入、已提取的物块点云")
    parser.add_argument("--transform-json", required=True, help="记录 B 到 A 配准矩阵的 JSON")
    parser.add_argument("--candidate-rank", type=int, default=0,
                        help="从 merge_two_jobs.json 读取第 N 个候选，并换算为原始坐标")
    parser.add_argument("--face", required=True, help="只允许导出的物理面，例如 y+")
    parser.add_argument("--output", required=True, help="通过核验的局部补面 PLY")
    parser.add_argument("--report", required=True, help="审计 JSON")
    parser.add_argument("--face-depth", type=float, default=0.36,
                        help="从面平面向物体内部允许的最大深度")
    parser.add_argument("--max-distance", type=float, default=0.012,
                        help="到定稿表面的最大距离")
    parser.add_argument("--min-normal-dot", type=float, default=0.75,
                        help="源/目标法线绝对点积下限")
    parser.add_argument("--max-color-distance", type=float, default=0.22,
                        help="RGB 欧氏距离上限（颜色范围 0~1）")
    parser.add_argument("--min-points", type=int, default=1000,
                        help="不足此数量时拒绝导出")
    args = parser.parse_args()

    axis, direction = parse_face(args.face)
    mesh = o3d.io.read_triangle_mesh(args.base_mesh)
    if len(mesh.vertices) == 0 or not mesh.has_vertex_colors():
        raise SystemExit("定稿 GLB 必须包含带颜色的网格")
    mesh.compute_vertex_normals()
    target_points = np.asarray(mesh.vertices)
    target_normals = np.asarray(mesh.vertex_normals)
    target_colors = np.asarray(mesh.vertex_colors)

    transform_data = json.loads(Path(args.transform_json).read_text())
    if args.candidate_rank:
        if "job_a_norm" not in transform_data or "job_b_norm" not in transform_data:
            raise SystemExit("candidate JSON 缺少归一化框架信息")
        candidates = transform_data.get("results", [])
        if not 1 <= args.candidate_rank <= len(candidates):
            raise SystemExit("candidate-rank 超出候选范围")
        item = candidates[args.candidate_rank - 1]
        rotation_a = np.asarray(transform_data["job_a_norm"][0], dtype=float)
        center_a, scale_a, rotation_b, center_b, scale_b = (
            np.asarray(transform_data["job_a_norm"][1], dtype=float),
            float(transform_data["job_a_norm"][2]),
            np.asarray(transform_data["job_b_norm"][0], dtype=float),
            np.asarray(transform_data["job_b_norm"][1], dtype=float),
            float(transform_data["job_b_norm"][2]),
        )
        normalized = np.asarray(item["transform"], dtype=float)
        # Row-vector form used by merge_two_jobs:
        # a_norm = (a - center_a) @ rotation_a.T / scale_a
        # b_norm = (b - center_b) @ rotation_b.T / scale_b
        # Therefore b_original @ matrix.T + matrix.t = a_original.
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_a.T @ normalized[:3, :3] @ rotation_b * (scale_a / scale_b)
        trans_norm_row = (
            (-center_b @ rotation_b.T / scale_b) @ normalized[:3, :3].T
            + normalized[:3, 3]
        )
        matrix[:3, 3] = trans_norm_row * scale_a @ rotation_a + center_a
    else:
        matrix = np.asarray(transform_data["matrix"], dtype=float)
    if matrix.shape != (4, 4):
        raise SystemExit("配准矩阵不是 4×4")
    supplement = o3d.io.read_point_cloud(args.supplement)
    if len(supplement.points) == 0:
        raise SystemExit("补拍点云为空")
    points, normals, colors = transform_points_and_normals(supplement, matrix)

    distance, nearest = cKDTree(target_points).query(points, k=1, workers=-1)
    normal_dot = np.abs(np.einsum("ij,ij->i", normals, target_normals[nearest]))
    color_distance = np.linalg.norm(colors - target_colors[nearest], axis=1)

    # The groove is recessed.  Membership is therefore based on the matched
    # target vertex rather than just source coordinates: this includes the
    # inner wall while excluding points from the opposite or unrelated faces.
    if direction > 0:
        face_plane = float(np.percentile(target_points[:, axis], 99.5))
        on_face = target_points[nearest, axis] >= face_plane - args.face_depth
    else:
        face_plane = float(np.percentile(target_points[:, axis], 0.5))
        on_face = target_points[nearest, axis] <= face_plane + args.face_depth

    valid = (
        (distance <= args.max_distance)
        & (normal_dot >= args.min_normal_dot)
        & (color_distance <= args.max_color_distance)
        & on_face
    )
    chosen = np.flatnonzero(valid)
    report = {
        "face": args.face,
        "input_points": int(len(points)),
        "accepted_points": int(len(chosen)),
        "accepted_ratio": float(valid.mean()),
        "thresholds": {
            "face_depth": args.face_depth,
            "max_distance": args.max_distance,
            "min_normal_dot": args.min_normal_dot,
            "max_color_distance": args.max_color_distance,
        },
        "accepted_distance_median": float(np.median(distance[valid])) if len(chosen) else None,
        "accepted_distance_p90": float(np.percentile(distance[valid], 90)) if len(chosen) else None,
        "accepted_normal_dot_median": float(np.median(normal_dot[valid])) if len(chosen) else None,
        "accepted_color_distance_median": float(np.median(color_distance[valid])) if len(chosen) else None,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if len(chosen) < args.min_points:
        raise SystemExit("通过三重核验的局部点不足，拒绝复制任何面")

    # Make normal signs agree with the target surface before a later Poisson
    # reconstruction.  We retain only the already-validated points above.
    selected_normals = normals[chosen].copy()
    target_dot = np.einsum("ij,ij->i", selected_normals, target_normals[nearest[chosen]])
    selected_normals[target_dot < 0] *= -1
    output = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[chosen]))
    output.colors = o3d.utility.Vector3dVector(colors[chosen])
    output.normals = o3d.utility.Vector3dVector(selected_normals)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(target), output, write_ascii=False)
    print(f"已写出经核验的 {args.face} 面补面点：{target}（{len(chosen):,} 点）")


if __name__ == "__main__":
    main()
