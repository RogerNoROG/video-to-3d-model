#!/usr/bin/env python3
"""从已核验的真实点云中抽取并加权立方体棱角证据。

这不是网格补洞工具。它完全不创建顶点、不外推平面：输出的每一个点都来自已通过
三扫描融合距离/法线/颜色检查的补充点云。用途是给后续 Poisson 重建增加棱角处的
真实观测权重，减少当前模型在角和棱上的低密度残留。

安全约束：
* 由批准基准的包围盒定义 12 条物理棱；补充来源不能移动这组边界；
* 只允许指定的非支撑面边带；
* 每条棱按长度分箱；没有足够覆盖的棱不写入输出；
* 输出的 audit 明确列出缺失棱，调用者不得把它们当作已经补全。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


AXIS = "xyz"


def parse_face(face: str) -> tuple[int, int]:
    if len(face) != 2 or face[0].lower() not in AXIS or face[1] not in "+-":
        raise SystemExit(f"无效支撑面 {face}，应为 x+、x-、y+、y-、z+ 或 z-")
    return AXIS.index(face[0].lower()), 1 if face[1] == "+" else -1


def load(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise SystemExit(f"空点云：{path}")
    if not cloud.has_colors():
        cloud.paint_uniform_color((0.72, 0.55, 0.28))
    return cloud


def edge_name(axis: int, signs: tuple[int, int]) -> str:
    other = [value for value in range(3) if value != axis]
    return f"{AXIS[other[0]]}{'+' if signs[0] > 0 else '-'}_{AXIS[other[1]]}{'+' if signs[1] > 0 else '-'}"


def collect_edge_indices(
    points: np.ndarray, lo: np.ndarray, hi: np.ndarray, axis: int,
    signs: tuple[int, int], width: float,
) -> np.ndarray:
    """Return only points observed in one physical edge band.

    The third coordinate must remain inside the reference cube, which prevents a
    support/background sheet just outside a face from being admitted as an edge.
    """
    other = [value for value in range(3) if value != axis]
    edge = np.ones(len(points), dtype=bool)
    for coordinate, sign in zip(other, signs):
        edge &= (points[:, coordinate] - lo[coordinate] if sign < 0
                 else hi[coordinate] - points[:, coordinate]) <= width
    edge &= points[:, axis] >= lo[axis] - width
    edge &= points[:, axis] <= hi[axis] + width
    return np.flatnonzero(edge)


def bins_for_edge(points: np.ndarray, indices: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  axis: int, bins: int) -> np.ndarray:
    length = max(float(hi[axis] - lo[axis]), 1e-9)
    locations = ((points[indices, axis] - lo[axis]) / length * bins).astype(int)
    return np.bincount(np.clip(locations, 0, bins - 1), minlength=bins)


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实补充点云抽取立方体棱角补强层")
    parser.add_argument("--base", required=True, help="批准基准点云，只用来定义物理边界")
    parser.add_argument("--source", action="append", required=True,
                        help="已通过融合核验的补充点云；可重复")
    parser.add_argument("--drop-face", action="append", default=[],
                        help="每个来源对应的支撑面；相邻棱一律不作为真实修复证据，默认均为 y-")
    parser.add_argument("--out", required=True, help="只写入新目录")
    parser.add_argument("--edge-width", type=float, default=0.030,
                        help="从两相邻外表面向内的边带宽度")
    parser.add_argument("--bins", type=int, default=72, help="每条棱的覆盖分箱数")
    parser.add_argument("--min-points-per-bin", type=int, default=12,
                        help="一个分箱被视为有真实覆盖的最低点数")
    parser.add_argument("--min-covered-ratio", type=float, default=0.55,
                        help="低于该覆盖比例的整条棱不纳入补强层")
    parser.add_argument("--voxel", type=float, default=0.0010,
                        help="输出补强层的体素去重大小，0 禁用")
    parser.add_argument("--merge-input", default="",
                        help="可选：原五面融合点云。提供后写出带真实棱角加权的新点云")
    parser.add_argument("--reinforce-repeats", type=int, default=0,
                        help="每个真实棱角点在新点云中的额外权重次数；0=仅写证据层")
    args = parser.parse_args()

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    base = load(Path(args.base))
    base_points = np.asarray(base.points)
    lo, hi = np.percentile(base_points, (0.5, 99.5), axis=0)
    sources = [load(Path(item)) for item in args.source]
    dropped = args.drop_face or ["y-"] * len(sources)
    if len(dropped) != len(sources):
        raise SystemExit("--drop-face 必须与 --source 数量相同，或完全省略")
    dropped_faces = [parse_face(value) for value in dropped]
    point_groups = [np.asarray(cloud.points) for cloud in sources]
    normal_groups = [np.asarray(cloud.normals) if cloud.has_normals() else None for cloud in sources]
    colour_groups = [np.asarray(cloud.colors) for cloud in sources]

    selections: list[tuple[int, np.ndarray]] = []
    report: dict[str, object] = {
        "approved": False,
        "operation": "real_data_edge_evidence_only",
        "base": str(Path(args.base)),
        "sources": [str(Path(item)) for item in args.source],
        "dropped_support_faces": dropped,
        "bounds_p005_p995": [lo.tolist(), hi.tolist()],
        "edge_width": args.edge_width,
        "bins": args.bins,
        "min_points_per_bin": args.min_points_per_bin,
        "min_covered_ratio": args.min_covered_ratio,
        "edges": {},
        "rule": "每个输出点均来自已通过距离、法线和颜色核验的真实补充扫描；低覆盖棱不补。",
    }
    for axis in range(3):
        for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            name = edge_name(axis, signs)
            other = [value for value in range(3) if value != axis]
            # A physical edge touches two faces.  If any input scan had either face
            # on its support plane, that scan cannot prove the edge geometry.  The
            # output is deliberately conservative: it rejects the entire edge
            # rather than retaining a few corner points that may belong to support.
            touches_support = any(
                dropped_axis in other and signs[other.index(dropped_axis)] == dropped_sign
                for dropped_axis, dropped_sign in dropped_faces
            )
            if touches_support:
                report["edges"][name] = {
                    "edge_axis": AXIS[axis],
                    "source_points": [0] * len(sources),
                    "covered_bins": 0,
                    "total_bins": args.bins,
                    "covered_ratio": 0.0,
                    "accepted_for_reinforcement": False,
                    "rejection": "touches_declared_support_face",
                }
                continue
            combined: list[tuple[int, np.ndarray]] = []
            coverage = np.zeros(args.bins, dtype=np.int64)
            counts: list[int] = []
            for source_index, points in enumerate(point_groups):
                indices = collect_edge_indices(points, lo, hi, axis, signs, args.edge_width)
                combined.append((source_index, indices))
                coverage += bins_for_edge(points, indices, lo, hi, axis, args.bins)
                counts.append(int(len(indices)))
            ratio = float((coverage >= args.min_points_per_bin).mean())
            accepted = ratio >= args.min_covered_ratio
            report["edges"][name] = {
                "edge_axis": AXIS[axis],
                "source_points": counts,
                "covered_bins": int((coverage >= args.min_points_per_bin).sum()),
                "total_bins": args.bins,
                "covered_ratio": ratio,
                "accepted_for_reinforcement": accepted,
            }
            if accepted:
                selections.extend((source_index, indices) for source_index, indices in combined if len(indices))

    if selections:
        points = np.concatenate([point_groups[index][indices] for index, indices in selections])
        colours = np.concatenate([colour_groups[index][indices] for index, indices in selections])
        have_normals = all(normal_groups[index] is not None for index, _ in selections)
        normals = (np.concatenate([normal_groups[index][indices] for index, indices in selections])
                   if have_normals else None)
    else:
        points = np.empty((0, 3))
        colours = np.empty((0, 3))
        normals = None
    evidence = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    evidence.colors = o3d.utility.Vector3dVector(colours)
    if normals is not None:
        evidence.normals = o3d.utility.Vector3dVector(normals)
    before_voxel = len(evidence.points)
    if args.voxel > 0 and before_voxel:
        evidence = evidence.voxel_down_sample(args.voxel)
    evidence_path = destination / "edge_real_evidence.ply"
    o3d.io.write_point_cloud(str(evidence_path), evidence, write_ascii=False)
    report["evidence_points_before_voxel"] = int(before_voxel)
    report["evidence_points"] = int(len(evidence.points))
    report["output"] = str(evidence_path)
    report["accepted_edges"] = [key for key, value in report["edges"].items()
                                if value["accepted_for_reinforcement"]]
    report["unresolved_edges"] = [key for key, value in report["edges"].items()
                                  if not value["accepted_for_reinforcement"]]
    if args.merge_input:
        if args.reinforce_repeats < 1:
            raise SystemExit("指定 --merge-input 时 --reinforce-repeats 必须至少为 1")
        merged = load(Path(args.merge_input))
        original_points = np.asarray(merged.points)
        original_colours = np.asarray(merged.colors)
        original_normals = np.asarray(merged.normals) if merged.has_normals() else None
        evidence_points = np.asarray(evidence.points)
        evidence_colours = np.asarray(evidence.colors)
        evidence_normals = np.asarray(evidence.normals) if evidence.has_normals() else None
        reinforced = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate((
            original_points, *([evidence_points] * args.reinforce_repeats),
        ))))
        reinforced.colors = o3d.utility.Vector3dVector(np.concatenate((
            original_colours, *([evidence_colours] * args.reinforce_repeats),
        )))
        if original_normals is not None and evidence_normals is not None:
            reinforced.normals = o3d.utility.Vector3dVector(np.concatenate((
                original_normals, *([evidence_normals] * args.reinforce_repeats),
            )))
        reinforced_path = destination / "merged_six_face_with_real_edge_evidence.ply"
        o3d.io.write_point_cloud(str(reinforced_path), reinforced, write_ascii=False)
        report["reinforcement"] = {
            "input": str(Path(args.merge_input)),
            "real_evidence_repeats": args.reinforce_repeats,
            "input_points": int(len(original_points)),
            "output_points": int(len(reinforced.points)),
            "output": str(reinforced_path),
            "note": "重复仅表示 Poisson 的观测权重；没有生成、平移或外推任何点。",
        }
    (destination / "edge_repair_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"真实棱角证据：{before_voxel:,} -> {len(evidence.points):,} 点")
    print("可补强棱：", ", ".join(report["accepted_edges"]) or "无")
    print("缺少真实覆盖：", ", ".join(report["unresolved_edges"]) or "无")
    if args.merge_input:
        print(f"已写入带真实棱角权重的点云：{report['reinforcement']['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
