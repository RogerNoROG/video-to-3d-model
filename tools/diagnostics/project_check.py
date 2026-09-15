#!/usr/bin/env python3
"""统一的状态、点云诊断和安全颜色清理工具。

把原来的 ``dense_status.py``、``face_maps.py``、``clean_cube_points.py`` 合并为
一个入口，避免项目中散落三个只做读取/诊断的小脚本。

子命令：
    python tools/diagnostics/project_check.py status [--quiet]
    python tools/diagnostics/project_check.py face-map --input model.ply [--bins 46] [--faces "z+ z-"]
    python tools/diagnostics/project_check.py clean --input in.ply --output out.ply [--band 0.15] [--threshold 0.20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
RAMP = "-=+*#%@"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def detect_pass(pass_dir: Path) -> int:
    try:
        names = os.listdir(pass_dir)
    except OSError:
        return 0
    if any(name.endswith("geometric.bin") for name in names):
        return 2
    if any(name.endswith("photometric.bin") for name in names):
        return 1
    return 0


def report_status(quiet: bool = False) -> int:
    states = sorted(ROOT.glob("logs/dense_model*.json"))
    if not states:
        if not quiet:
            print("稠密构建: (无记录，用 tools/pipeline/build_dense_model.py 启动后会出现在这里)")
        return 0
    for state_path in states:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model = state.get("model")
        job = state.get("job", "")
        total = int(state.get("total") or 0)
        passes = int(state.get("passes") or 1)
        depth_dir = ROOT / "storage" / job / f"dense_{model}" / "stereo" / "depth_maps"
        fused = ROOT / "storage" / job / f"fused_{model}.ply"
        pid_path = state_path.with_suffix(".pid")
        try:
            pid = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            pid = None
        running = pid is not None and alive(pid)
        try:
            entries = [entry for entry in os.scandir(depth_dir) if entry.name.endswith(".bin")]
        except OSError:
            entries = []
        done = len(entries)
        label = f"模型{model}"
        if state.get("finished_at"):
            size = fused.stat().st_size / 1024 / 1024 if fused.exists() else 0
            print(f"{label:<6} 已完成        {fused.name} {size:.0f} MB")
            continue
        if not running:
            print(f"{label:<6} 已停止 (pid {pid})   产物 {done:,}/{total:,} —— 重新运行同一命令可断点续跑")
            continue
        if done == 0:
            print(f"{label:<6} 运行中 (pid {pid})   正在准备/读取工作区…")
            continue
        percent = done / total * 100 if total else 0
        rate = eta = ""
        if len(entries) >= 2:
            times = sorted(entry.stat().st_mtime for entry in entries)
            span = times[-1] - times[0]
            if span > 0:
                per_min = (done - 1) / span * 60
                remaining = (total - done) / per_min if per_min > 0 else float("inf")
                rate = f"{per_min:.1f}/分钟"
                if remaining != float("inf"):
                    eta = f"剩余 {remaining:.0f} 分钟 (约 {(datetime.now() + timedelta(minutes=remaining)):%H:%M})"
        phase = f"第{detect_pass(depth_dir)}遍/共{passes}遍  " if passes > 1 and detect_pass(depth_dir) else ""
        print(f"{label:<6} 运行中 (pid {pid})   {phase}产物 {done:,}/{total:,} ({percent:.1f}%)   {rate}   {eta}")
    return 0


def load_points(path: str, max_points: int = 3_000_000, seed: int = 0) -> np.ndarray:
    lower = path.lower()
    if lower.endswith((".glb", ".gltf", ".obj", ".stl")):
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            raise SystemExit(f"empty mesh: {path}")
        cloud = mesh.sample_points_uniformly(number_of_points=max_points)
        points = np.asarray(cloud.points, dtype=np.float64)
    else:
        cloud = o3d.io.read_point_cloud(path)
        points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise SystemExit(f"empty cloud: {path}")
    if len(points) > max_points:
        index = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points = points[index]
    return points


def height_map(points: np.ndarray, axis: int, sign: int, bins: int, size: float) -> np.ndarray:
    other = [a for a in range(3) if a != axis]
    values = points[:, axis] * sign
    face = values.max()
    bounds = [(points[:, a].min(), points[:, a].max()) for a in other]
    grid = np.full((bins, bins), np.nan)
    i = np.clip(((points[:, other[0]] - bounds[0][0]) / max(bounds[0][1] - bounds[0][0], 1e-9) * bins).astype(int), 0, bins - 1)
    j = np.clip(((points[:, other[1]] - bounds[1][0]) / max(bounds[1][1] - bounds[1][0], 1e-9) * bins).astype(int), 0, bins - 1)
    flat = i * bins + j
    order = np.argsort(flat, kind="stable")
    flat_sorted, values_sorted = flat[order], values[order]
    unique, starts = np.unique(flat_sorted, return_index=True)
    ends = np.append(starts[1:], len(flat_sorted))
    for cell, start, end in zip(unique, starts, ends):
        grid[cell // bins, cell % bins] = values_sorted[start:end].max()
    return (face - grid) / size


def print_map(relative: np.ndarray, axis_name: str, sign_name: str, other_names: list[str]) -> None:
    bins = relative.shape[0]
    print(f"\n=== {axis_name}{sign_name} face  (rows = {other_names[0]}, cols = {other_names[1]}) ===")
    print("    '.' = at the face plane, deeper = - = + * # % @ (up to 0.25 x size), ' ' = no points")
    print("     " + "".join(str(i % 10) for i in range(bins)))
    for row in range(bins - 1, -1, -1):
        chars = []
        for value in relative[row]:
            if np.isnan(value) or value > 0.25:
                chars.append(" ")
            else:
                chars.append(RAMP[int(np.clip(value / 0.25, 0, 0.999) * len(RAMP))])
        print(f"{row:3d}  {''.join(chars)}")


def report_faces(args: argparse.Namespace) -> int:
    points = load_points(args.input, args.max_points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = float(np.max(hi - lo))
    print(f"points {len(points)}   extent {np.round(hi - lo, 4)}   size {size:.4f}")
    names = {0: "X", 1: "Y", 2: "Z"}
    for token in args.faces.split():
        if len(token) != 2 or token[0].lower() not in "xyz" or token[1] not in "+-":
            raise SystemExit(f"无效面标记: {token}（应为 x+、x-、y+、y-、z+ 或 z-）")
        axis = {"x": 0, "y": 1, "z": 2}[token[0].lower()]
        sign = 1 if token[1] == "+" else -1
        other = [a for a in range(3) if a != axis]
        print_map(height_map(points, axis, sign, args.bins, size), names[axis], token[1], [names[other[0]], names[other[1]]])
    return 0


def normalize_chroma(colors: np.ndarray) -> np.ndarray:
    return (colors[:, 0] - colors[:, 2]) / np.maximum(colors.max(axis=1), 1e-6)


def drop_foreign_by_color(cloud: o3d.geometry.PointCloud, band_depth: float, threshold: float) -> o3d.geometry.PointCloud:
    if not cloud.has_colors():
        raise ValueError("点云必须带颜色（COLMAP 的 fused.ply 有）")
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    top = np.percentile(points[:, 1], 99.5)
    chroma = normalize_chroma(colors)
    band = points[:, 1] > top - band_depth
    drop = band & (chroma < threshold)
    print(f"顶面 Y={top:+.4f}   带深 {band_depth}   阈值 {threshold}")
    print(f"  带内 {int(band.sum()):,} 点，删 {int(drop.sum()):,}（带内 {drop.sum() / max(band.sum(), 1) * 100:.1f}%，全云 {drop.mean() * 100:.2f}%）")
    keep = ~drop
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
    result.colors = o3d.utility.Vector3dVector(colors[keep])
    if cloud.has_normals():
        result.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[keep])
    return result


def clean_cloud(args: argparse.Namespace) -> int:
    cloud = o3d.io.read_point_cloud(args.input)
    print(f"读取 {args.input}：{len(cloud.points):,} 点")
    cleaned = drop_foreign_by_color(cloud, args.band, args.threshold)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(target), cleaned, write_ascii=False)
    print(f"-> {target.name}  {len(cleaned.points):,} 点")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="统一状态、面特征图和颜色清理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="报告独立稠密构建状态")
    status.add_argument("--quiet", action="store_true")
    face = subparsers.add_parser("face-map", help="输出各面的 ASCII 凹槽深度图")
    face.add_argument("--input", required=True)
    face.add_argument("--bins", type=int, default=46)
    face.add_argument("--max-points", type=int, default=3_000_000)
    face.add_argument("--faces", default="z+ z- x+ x- y+ y-")
    clean = subparsers.add_parser("clean", help="按顶面色度清理外来几何")
    clean.add_argument("--input", required=True)
    clean.add_argument("--output", required=True)
    clean.add_argument("--band", type=float, default=0.15)
    clean.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    if args.command == "status":
        return report_status(args.quiet)
    if args.command == "face-map":
        return report_faces(args)
    return clean_cloud(args)


if __name__ == "__main__":
    sys.exit(main())
