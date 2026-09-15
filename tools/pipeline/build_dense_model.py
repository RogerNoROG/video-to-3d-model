#!/usr/bin/env python3
"""为指定的稀疏模型单独跑稠密重建，输出到独立的 dense_<N>/ 与 fused_<N>.ply。

现有流水线（app/main.py）只用 select_best_model() 选中的那一个模型，并把结果写进
dense/ 与 fused.ply。要处理第二个模型必须另开一份工作区，否则会覆盖已有成果。

用法：
    ./.venv/bin/python tools/pipeline/build_dense_model.py --model 1
    ./.venv/bin/python tools/pipeline/build_dense_model.py --model 1 --max-image-size 960   # 快速验证
    ./.venv/bin/python tools/pipeline/build_dense_model.py --model 1 --no-geom-consistency  # 单遍，耗时减半

可重复运行：patch_match_stereo 会跳过 depth_maps 里已存在的视角，中断后直接重跑即可续上。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.main import settings, subprocess_env  # noqa: E402

POLL_SECONDS = 3.0


def write_state(path: Path, **fields: object) -> None:
    """写进度状态文件，供 tools/diagnostics/project_check.py 与 ./start.sh status 读取。"""
    payload: dict[str, object] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], cwd: Path, log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=settings.command_timeout_seconds,
            env=subprocess_env(),
        )
    if completed.returncode != 0:
        raise SystemExit(f"[{label}] 命令失败（退出码 {completed.returncode}），详见 {log_path}")


def watch_depth_maps(depth_dir: Path, total: int, label: str, stop: threading.Event) -> None:
    last = -1
    started = time.monotonic()
    while not stop.wait(POLL_SECONDS):
        try:
            done = sum(1 for name in os.listdir(depth_dir) if name.endswith(".bin"))
        except OSError:
            continue
        if done == last:
            continue
        last = done
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 and done else 0
        remain = (total - done) / rate / 60 if rate > 0 else float("inf")
        print(
            f"[{label}] {done:,}/{total:,} 视角文件"
            f"  速率 {rate * 60:.1f}/分钟  预计剩余 {remain:.0f} 分钟",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="e7f9f9e5d046490d93b34e4538bc9ef1")
    parser.add_argument("--model", type=int, required=True, help="sparse/ 下的模型下标，例如 1")
    parser.add_argument("--max-image-size", type=int, default=None,
                        help="稠密分辨率上限，越小越快（默认用配置里的值）")
    parser.add_argument("--no-geom-consistency", action="store_true",
                        help="关闭几何一致性过滤，只跑一遍，耗时减半")
    args = parser.parse_args()

    root = (settings.storage_dir / args.job).resolve()
    model = root / "sparse" / str(args.model)
    frames = root / "frames"
    dense = root / f"dense_{args.model}"
    fused = root / f"fused_{args.model}.ply"
    log_path = Path("logs") / f"dense_model{args.model}.log"
    state_path = Path("logs") / f"dense_model{args.model}.json"

    if not model.exists():
        raise SystemExit(f"找不到稀疏模型 {model}")
    if not frames.exists():
        raise SystemExit(f"找不到抽帧目录 {frames}")

    max_size = args.max_image_size if args.max_image_size is not None else settings.patch_match_max_image_size
    geom = not args.no_geom_consistency and settings.patch_match_geom_consistency

    print(f"任务 {args.job}  模型 sparse/{args.model}")
    print(f"工作区 {dense}")
    print(f"输出点云 {fused.name}")
    print(f"稠密分辨率上限 {'原始' if max_size <= 0 else max_size}   几何一致性 {geom}")

    # ---- 1. image_undistorter ----
    dense.mkdir(parents=True, exist_ok=True)
    marker = dense / "sparse" / "images.bin"
    if marker.exists():
        print("\n[1/3] image_undistorter 已完成，跳过")
    else:
        print("\n[1/3] image_undistorter ...", flush=True)
        run(
            [
                settings.colmap_binary, "image_undistorter",
                "--image_path", str(frames),
                "--input_path", str(model),
                "--output_path", str(dense),
                "--output_type", "COLMAP",
            ],
            root, log_path, "undistorter",
        )
        print("      完成")

    # ---- 2. patch_match_stereo ----
    depth_dir = dense / "stereo" / "depth_maps"
    views = 0
    config = dense / "stereo" / "patch-match.cfg"
    if config.exists():
        views = len([l for l in config.read_text(encoding="utf-8").splitlines() if l.strip()]) // 2
    passes = 2 if geom else 1
    total = max(views, 1) * passes
    Path(f"logs/dense_model{args.model}.pid").write_text(str(os.getpid()), encoding="utf-8")
    write_state(
        state_path,
        job=args.job,
        model=args.model,
        views=views,
        passes=passes,
        total=total,
        max_image_size=max_size,
        geom_consistency=geom,
        started_at=datetime.now().isoformat(timespec="seconds"),
        finished_at=None,
    )
    print(f"\n[2/3] patch_match_stereo（{views} 视角 x {passes} 遍 = {total:,} 个产物）", flush=True)

    command = [
        settings.colmap_binary, "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
        # 与网页主流程保持一致：单个 COLMAP 作业用满可用 CPU，缓存上限
        # 由 settings 控制，避免默认 32GB 在 24GB 主机上触发换页。
        "--PatchMatchStereo.num_threads", str(settings.colmap_num_threads),
        "--PatchMatchStereo.cache_size", str(settings.colmap_cache_size_gb),
    ]
    if max_size > 0:
        command += ["--PatchMatchStereo.max_image_size", str(max_size)]

    stop = threading.Event()
    watcher = threading.Thread(target=watch_depth_maps, args=(depth_dir, total, "稠密", stop), daemon=True)
    watcher.start()
    try:
        run(command, root, log_path, "patch_match_stereo")
    finally:
        stop.set()
        watcher.join(timeout=POLL_SECONDS * 2)

    # ---- 3. stereo_fusion ----
    print("\n[3/3] stereo_fusion ...", flush=True)
    run(
        [
            settings.colmap_binary, "stereo_fusion",
            "--workspace_path", str(dense),
            "--workspace_format", "COLMAP",
            "--output_path", str(fused),
            "--StereoFusion.min_num_pixels", str(settings.stereo_fusion_min_num_pixels),
            "--StereoFusion.num_threads", str(settings.colmap_num_threads),
            "--StereoFusion.use_cache", "1",
            "--StereoFusion.cache_size", str(settings.colmap_cache_size_gb),
        ] + (
            ["--StereoFusion.max_image_size", str(settings.stereo_fusion_max_image_size)]
            if settings.stereo_fusion_max_image_size > 0 else []
        ),
        root, log_path, "stereo_fusion",
    )

    size = fused.stat().st_size if fused.exists() else 0
    write_state(state_path, finished_at=datetime.now().isoformat(timespec="seconds"), size_bytes=size)
    print(f"\n完成：{fused}  {size / 1024 / 1024:.0f} MB")
    print(f"日志：{log_path}")


if __name__ == "__main__":
    main()
