#!/usr/bin/env python3
"""把补拍帧并入稠密重建 —— 只算新增视角，旧视角的深度图直接复用。

为什么能省这么多
--------------
`patch_match_stereo` 会**跳过 depth_maps 里已存在的视角**。所以只要把旧工作区的
`depth_maps/`、`normal_maps/` 软链接进新的工作区，它就只算补拍那批新视角。

实测规模：原来 667 视角（几何一致性 = 两遍）跑了约 7 小时；补拍 286 视角只需约 3 小时。

前提
----
- 新模型是在**同一个坐标系**下增量注册出来的（tools/pipeline/register_supplement.py），
  旧图像位姿一点没动 —— 否则旧深度图不能复用
- 相机内参没变（同相机、同分辨率）。软链接时要顺手核对图像名一一对应

用法
----
    ./.venv/bin/python tools/pipeline/build_dense_supplement.py --job <任务id>
    ./.venv/bin/python tools/pipeline/build_dense_supplement.py --job <任务id> --no-geom-consistency
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.main import settings, subprocess_env  # noqa: E402

POLL_SECONDS = 5.0


def run(command: list[str], cwd: Path, log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   check=False, timeout=settings.command_timeout_seconds,
                                   env=subprocess_env())
    if completed.returncode != 0:
        raise SystemExit(f"[{label}] 失败（退出码 {completed.returncode}），详见 {log_path}")


def watch(depth_dir: Path, total_done: int, total: int, label: str, stop: threading.Event) -> None:
    started = time.monotonic()
    last = -1
    while not stop.wait(POLL_SECONDS):
        try:
            done = sum(1 for name in os.listdir(depth_dir) if name.endswith(".bin"))
        except OSError:
            continue
        if done == last:
            continue
        last = done
        elapsed = time.monotonic() - started
        rate = (done - total_done) / elapsed if elapsed > 0 and done > total_done else 0
        remain = (total - done) / rate / 60 if rate > 0 else float("inf")
        print(f"[{label}] {done:,}/{total:,} 个产物  本批已算 {done - total_done:,}"
              f"  速率 {rate * 60:.1f}/分钟  预计剩余 {remain:.0f} 分钟", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="e7f9f9e5d046490d93b34e4538bc9ef1")
    parser.add_argument("--model", default="supp/model_supplemented",
                        help="增量注册后的模型（相对任务目录）")
    parser.add_argument("--base-dense", default="dense", help="旧稠密工作区（相对任务目录）")
    parser.add_argument("--out-dense", default="supp/dense", help="新稠密工作区（相对任务目录）")
    parser.add_argument("--max-image-size", type=int, default=None)
    parser.add_argument("--no-geom-consistency", action="store_true")
    args = parser.parse_args()

    root = (settings.storage_dir / args.job).resolve()
    model = root / args.model
    base_dense = root / args.base_dense
    dense = root / args.out_dense
    all_frames = root / "supp/all_frames"
    fused = root / "supp/fused_supplemented.ply"
    log_path = Path("logs") / "dense_supplement.log"

    for required in (model, base_dense / "stereo/depth_maps", root / "frames",
                     root / "supp/new_frames"):
        if not required.exists():
            raise SystemExit(f"缺少 {required}")

    max_size = (args.max_image_size if args.max_image_size is not None
                else settings.patch_match_max_image_size)
    geom = not args.no_geom_consistency and settings.patch_match_geom_consistency
    print(f"任务 {args.job}")
    print(f"增量模型 {model}")
    print(f"旧工作区 {base_dense}   新工作区 {dense}")
    print(f"稠密分辨率上限 {'原始' if max_size <= 0 else max_size}   几何一致性 {geom}\n")

    # ---- 0. 新旧帧放在同一个目录（软链接），让 undistorter 一次处理全部 ----
    all_frames.mkdir(parents=True, exist_ok=True)
    linked = 0
    for source_dir in (root / "frames", root / "supp/new_frames"):
        for image in source_dir.glob("*.jpg"):
            target = all_frames / image.name
            if not target.exists():
                target.symlink_to(image)
                linked += 1
    print(f"[0/4] 新旧帧目录 {all_frames}（新建 {linked} 个软链接，"
          f"共 {len(list(all_frames.glob('*.jpg')))} 张）", flush=True)

    # ---- 1. image_undistorter ----
    if (dense / "sparse" / "images.bin").exists():
        print("[1/4] image_undistorter 已完成，跳过")
    else:
        print("[1/4] image_undistorter ...", flush=True)
        run([settings.colmap_binary, "image_undistorter",
             "--image_path", str(all_frames),
             "--input_path", str(model),
             "--output_path", str(dense),
             "--output_type", "COLMAP"], root, log_path, "undistorter")
        print("      完成")

    # ---- 2. 复用旧视角的深度图 / 法线图（patch_match 见到文件存在就会跳过） ----
    depth_dir = dense / "stereo" / "depth_maps"
    normal_dir = dense / "stereo" / "normal_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    reused = 0
    for source_dir, target_dir in ((base_dense / "stereo/depth_maps", depth_dir),
                                   (base_dense / "stereo/normal_maps", normal_dir)):
        for artifact in source_dir.glob("frame_*.bin"):
            target = target_dir / artifact.name
            if not target.exists():
                target.symlink_to(artifact)
                reused += 1
    existing = sum(1 for name in os.listdir(depth_dir) if name.endswith(".bin"))
    total_views = sum(1 for line in (dense / "stereo/patch-match.cfg").read_text().splitlines()
                      if line.strip()) // 2
    print(f"[2/4] 复用旧视角产物：新建 {reused} 个软链接，"
          f"depth_maps 现有 {existing:,} 个（{total_views} 视角 × {'2' if geom else '1'} 遍 "
          f"= {total_views * (2 if geom else 1):,} 个产物）", flush=True)

    # ---- 3. patch_match_stereo（只算新增视角） ----
    print("\n[3/4] patch_match_stereo ...", flush=True)
    command = [settings.colmap_binary, "patch_match_stereo",
               "--workspace_path", str(dense), "--workspace_format", "COLMAP",
               "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
               "--PatchMatchStereo.num_threads", str(settings.colmap_num_threads),
               "--PatchMatchStereo.cache_size", str(settings.colmap_cache_size_gb)]
    if max_size > 0:
        command += ["--PatchMatchStereo.max_image_size", str(max_size)]
    total = total_views * (2 if geom else 1)
    stop = threading.Event()
    watcher = threading.Thread(target=watch,
                               args=(depth_dir, existing, total, "稠密", stop), daemon=True)
    watcher.start()
    try:
        run(command, root, log_path, "patch_match_stereo")
    finally:
        stop.set()
        watcher.join(timeout=POLL_SECONDS * 2)

    # ---- 4. stereo_fusion（全部视角一起融合） ----
    print("\n[4/4] stereo_fusion ...", flush=True)
    run([settings.colmap_binary, "stereo_fusion",
         "--workspace_path", str(dense), "--workspace_format", "COLMAP",
         "--output_path", str(fused),
         "--StereoFusion.min_num_pixels", str(settings.stereo_fusion_min_num_pixels),
         "--StereoFusion.num_threads", str(settings.colmap_num_threads),
         "--StereoFusion.use_cache", "1",
         "--StereoFusion.cache_size", str(settings.colmap_cache_size_gb)]
        + (["--StereoFusion.max_image_size", str(settings.stereo_fusion_max_image_size)]
           if settings.stereo_fusion_max_image_size > 0 else []),
        root, log_path, "stereo_fusion")

    size_mb = fused.stat().st_size / 1024 / 1024
    print(f"\n完成 -> {fused}（{size_mb:.0f} MB）")
    print("下一步：用 tools/fusion/merge_models.py 重新合并（A 段换成这个更完整的点云），"
          "再走立方体对齐 + 清理链")


if __name__ == "__main__":
    main()
