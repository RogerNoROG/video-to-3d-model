#!/usr/bin/env python3
"""安全模型精修：真实局部证据加权与特征保护的外平面平滑。

两个子命令都只写出新的、待审核候选：

``region-evidence``
    将同一坐标系的批准网格局部表面采样为真实点云证据，以有限权重融入候选点云，
    供后续统一 Poisson 重建使用；不拼网格、不生成贴图或扇形补洞。

``smooth-planes``
    只在外平面内部、稳定法线区域轻微平滑。棱、孔口、孔壁、凹槽和用户提供的保护盒
    均不移动；不改变颜色或网格拓扑。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.convert import write_glb  # noqa: E402


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise SystemExit(f"空点云：{path}")
    if not cloud.has_colors():
        cloud.paint_uniform_color((0.72, 0.55, 0.28))
    return cloud


def region_evidence(args: argparse.Namespace) -> int:
    low, high = np.asarray(args.box[:3], float), np.asarray(args.box[3:], float)
    if np.any(high <= low) or args.samples < 1_000 or args.repeats < 1:
        raise SystemExit("区域或采样参数无效")
    candidate = load_cloud(Path(args.input))
    reference = o3d.io.read_triangle_mesh(args.reference, enable_post_processing=False)
    vertices, triangles = np.asarray(reference.vertices), np.asarray(reference.triangles)
    if not len(triangles):
        raise SystemExit("参考网格为空")
    # Slight overlap makes this evidence constrain the transition band rather than
    # creating a separately stitched patch at the visible defect boundary.
    selected = np.all((vertices[triangles].mean(axis=1) >= low - 0.012)
                      & (vertices[triangles].mean(axis=1) <= high + 0.012), axis=1)
    if not selected.any():
        raise SystemExit("参考模型在指定区域没有三角面")
    region = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.copy()),
        o3d.utility.Vector3iVector(triangles[selected].copy()),
    )
    if reference.has_vertex_colors():
        region.vertex_colors = o3d.utility.Vector3dVector(np.asarray(reference.vertex_colors).copy())
    region.remove_unreferenced_vertices()
    region.compute_vertex_normals()
    evidence = region.sample_points_uniformly(number_of_points=args.samples, use_triangle_normal=False)
    if not evidence.has_normals():
        evidence.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    c_points, e_points = np.asarray(candidate.points), np.asarray(evidence.points)
    c_colors, e_colors = np.asarray(candidate.colors), np.asarray(evidence.colors)
    output = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate((c_points, *([e_points] * args.repeats)))))
    output.colors = o3d.utility.Vector3dVector(np.concatenate((c_colors, *([e_colors] * args.repeats))))
    if candidate.has_normals():
        output.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(candidate.normals), *([np.asarray(evidence.normals)] * args.repeats))))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(target), output, write_ascii=False)
    report = {
        "approved": False, "operation": "approved_local_surface_evidence_reconstruction",
        "input": args.input, "reference": args.reference,
        "box": [low.tolist(), high.tolist()], "reference_samples": args.samples,
        "real_evidence_repeats": args.repeats, "input_points": int(len(c_points)),
        "output_points": int(len(output.points)), "output": str(target),
        "rule": "局部点来自同一物体的批准真实网格；仅提高重建权重，不拼网格、不生成贴图、放射线补洞或定稿。",
    }
    Path(args.audit).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_box(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.asarray(values[:3], float), np.asarray(values[3:], float)
    if np.any(high <= low):
        raise ValueError("保护区域无效")
    return low, high


def mesh_adjacency(triangles: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    pairs = np.vstack((pairs, pairs[:, ::-1]))
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    return np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=count))], pairs[:, 1]


def smooth_planes(args: argparse.Namespace) -> int:
    if args.iterations < 1 or not 0 < args.weight <= 1 or args.max_step <= 0:
        raise SystemExit("平滑参数无效")
    mesh = o3d.io.read_triangle_mesh(args.input, enable_post_processing=False)
    if not len(mesh.triangles):
        raise SystemExit("输入网格为空")
    mesh.compute_vertex_normals()
    points, normals, triangles = np.asarray(mesh.vertices).copy(), np.asarray(mesh.vertex_normals), np.asarray(mesh.triangles)
    low, high = np.percentile(points, (0.5, 99.5), axis=0)
    boxes = [parse_box(box) for box in args.protect_box]
    protected = np.zeros(len(points), bool)
    for box_low, box_high in boxes:
        protected |= np.all((points >= box_low - 0.018) & (points <= box_high + 0.018), axis=1)
    eligible, face_ids = np.zeros(len(points), bool), np.full(len(points), -1, np.int8)
    for axis in range(3):
        other = [item for item in range(3) if item != axis]
        interior = np.ones(len(points), bool)
        for coordinate in other:
            interior &= (points[:, coordinate] > low[coordinate] + args.edge_margin) & (points[:, coordinate] < high[coordinate] - args.edge_margin)
        for direction, plane in ((-1, low[axis]), (1, high[axis])):
            mask = (np.abs(points[:, axis] - plane) <= args.face_band) & (normals[:, axis] * direction >= args.normal_dot) & interior & ~protected
            eligible |= mask
            face_ids[mask] = axis * 2 + int(direction > 0)
    offsets, neighbours = mesh_adjacency(triangles, len(points))
    initial, moved = points.copy(), np.zeros(len(points), bool)
    for _ in range(args.iterations):
        updated = points.copy()
        for index in np.flatnonzero(eligible):
            linked = neighbours[offsets[index]:offsets[index + 1]]
            linked = linked[eligible[linked] & (face_ids[linked] == face_ids[index])]
            if len(linked) < 3:
                continue
            delta = (points[linked].mean(axis=0) - points[index]) * args.weight
            length = np.linalg.norm(delta)
            if length > args.max_step:
                delta *= args.max_step / length
            updated[index] += delta
            moved[index] |= np.any(delta)
        points = updated
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.compute_vertex_normals()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_glb(mesh, target)
    report = {
        "approved": False, "operation": "masked_outer_plane_smoothing", "input": args.input,
        "output": str(target), "triangles_before_after": [int(len(triangles)), int(len(triangles))],
        "vertices_before_after": [int(len(initial)), int(len(points))], "eligible_vertices": int(eligible.sum()),
        "moved_vertices": int(moved.sum()), "max_displacement": float(np.linalg.norm(points - initial, axis=1).max()),
        "protected_vertices": int(protected.sum()), "protected_boxes": [[a.tolist(), b.tolist()] for a, b in boxes],
        "parameters": {"edge_margin": args.edge_margin, "face_band": args.face_band, "normal_dot": args.normal_dot, "iterations": args.iterations, "lambda": args.weight, "max_step": args.max_step},
        "rule": "仅移动外平面内部的稳定法线顶点；棱、孔口、孔壁、凹槽和颜色均不改动，未新增或删除任何三角面。",
    }
    Path(args.audit).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="安全模型精修（全部输出为待审核候选）")
    commands = parser.add_subparsers(dest="command", required=True)
    region = commands.add_parser("region-evidence", help="把批准网格的真实局部表面作为重建证据")
    region.add_argument("--input", required=True); region.add_argument("--reference", required=True)
    region.add_argument("--box", type=float, nargs=6, required=True, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    region.add_argument("--samples", type=int, default=300_000); region.add_argument("--repeats", type=int, default=3)
    region.add_argument("--output", required=True); region.add_argument("--audit", required=True); region.set_defaults(handler=region_evidence)
    planes = commands.add_parser("smooth-planes", help="仅平滑受保护特征之外的外平面内部")
    planes.add_argument("--input", required=True); planes.add_argument("--output", required=True); planes.add_argument("--audit", required=True)
    planes.add_argument("--protect-box", action="append", type=float, nargs=6, default=[], metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    planes.add_argument("--edge-margin", type=float, default=0.035); planes.add_argument("--face-band", type=float, default=0.017)
    planes.add_argument("--normal-dot", type=float, default=0.965); planes.add_argument("--iterations", type=int, default=2)
    planes.add_argument("--lambda", dest="weight", type=float, default=0.30); planes.add_argument("--max-step", type=float, default=0.0018)
    planes.set_defaults(handler=smooth_planes)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
