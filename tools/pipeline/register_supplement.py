#!/usr/bin/env python3
"""把**补拍片段**增量注册进已有的稀疏模型（方案 A）。

为什么用这条路
------------
已有模型的坐标系不能动，但又想补拍某个区域。`colmap image_registrator` 能在
**保持原有位姿完全不变**的前提下给新帧估位姿（实测：原 606 张的位姿改动恰好 0.000000，
新注册的位姿与真值只差 0.17%/0.05°）。

两个必须知道的坑
--------------
1. **内参必须一致**。补拍要用同一台设备、同一分辨率；如果原素材和补拍都是 4K，
   统一降到 1080p 处理即可。本脚本用 `--ImageReader.camera_params` 强制复用原有相机。

2. **穷举匹配会制造伪匹配，把无关的图像也注册进来**。
   本项目的 `sparse/1`（翻面那一段）与 `sparse/0` 没有共享图像，穷举匹配之后
   `image_registrator` 会顺带把 sparse/1 的 564 张也注册进去（实测模型从 667 涨到 1272）。
   所以注册完**必须过滤模型**，只保留「原模型里的图像 + 本次新增的帧」。

流程
----
    feature_extractor（只提新帧） → exhaustive_matcher（全库，实测 94.6 万对约 16 分钟）
    → image_registrator → 过滤模型 → point_triangulator（给新帧三角化新点）

产物都在 `<job>/supp/` 下，**原始 database.db 和 sparse/ 不会被改动**。

用法
----
    register_supplement.py --video 补拍.mp4 --job storage/<任务id> --model sparse/0
    register_supplement.py --frames <已抽好的帧目录> --job storage/<任务id> --model sparse/0
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
COLMAP = os.environ.get("MODEL_API_COLMAP_BINARY") or str(Path.home() / ".local/bin/colmap")
FEATURE_TYPE = "ALIKED_N16ROT"
MATCHER_TYPE = "ALIKED_BRUTEFORCE"
CAMERA_MODELS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL"}


def log(message: str) -> None:
    print(f"[supp] {message}", flush=True)


def subprocess_env() -> dict:
    """ALIKED 走 GPU 需要 cuDNN 9，它装在 venv 里，得手动加进 LD_LIBRARY_PATH。"""
    env = os.environ.copy()
    cudnn = REPO / ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
    if cudnn.is_dir():
        env["LD_LIBRARY_PATH"] = f"{cudnn}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run(command: list[str], what: str) -> None:
    log(f"{what} ...")
    started = time.monotonic()
    result = subprocess.run(command, env=subprocess_env(), capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"{what} 失败（退出码 {result.returncode}）")
    log(f"{what} 完成，耗时 {time.monotonic() - started:.0f}s")


def to_txt(model_bin: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    run([COLMAP, "model_converter", "--input_path", str(model_bin),
         "--output_path", str(target), "--output_type", "TXT"], f"模型 -> TXT（{model_bin.name}）")
    return target


def to_bin(model_txt: Path, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    run([COLMAP, "model_converter", "--input_path", str(model_txt),
         "--output_path", str(target), "--output_type", "BIN"], f"TXT -> 模型（{target.name}）")
    return target


def count_images(model_dir: Path) -> int:
    import re

    result = subprocess.run([COLMAP, "model_analyzer", "--path", str(model_dir)],
                            env=subprocess_env(), capture_output=True, text=True)
    # 输出形如 `I0913 ... model.cc:445] Images: 667`；⚠️ colmap 把日志写到 stderr
    match = re.search(r"\]\s*Images:\s*(\d+)", result.stdout + result.stderr)
    return int(match.group(1)) if match else -1


def names_in_model(model_dir: Path, cache: Path) -> set[str]:
    text = to_txt(model_dir, cache)
    lines = (text / "images.txt").read_text().splitlines()
    # images.txt 每张图占**两行**（元数据行 + POINTS2D 行），只能取偶数行
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    return {body[i].split()[-1] for i in range(0, len(body), 2)}


def filter_model(model_bin: Path, keep_names: set[str], target: Path, scratch: Path) -> Path:
    """把模型裁到只剩 ``keep_names`` 里的图像（连同它们的观测点）。

    ⚠️ 必须同步过滤 points3D.txt 的 TRACK 与 frames.txt，否则读模型时会报
    "Image with ID xxx does not exist" 直接 abort。
    """
    text = to_txt(model_bin, scratch)

    lines = (text / "images.txt").read_text().splitlines()
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    kept, dropped_ids = [], set()
    for index in range(0, len(body), 2):
        fields = body[index].split()
        if fields[-1] in keep_names:
            kept.extend([body[index], body[index + 1]])
        else:
            dropped_ids.add(int(fields[0]))

    if not dropped_ids:
        log("模型无需过滤")
        return to_bin(text, target)

    header = [line for line in lines if line.startswith("#")]
    (text / "images.txt").write_text("\n".join(header + kept) + "\n")

    point_lines = (text / "points3D.txt").read_text().splitlines()
    point_header = [line for line in point_lines if line.startswith("#")]
    kept_points = 0
    with (text / "points3D.txt").open("w") as handle:
        handle.write("\n".join(point_header) + "\n")
        for line in point_lines:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            track = fields[8:]
            remaining = [
                (int(track[i]), track[i + 1]) for i in range(0, len(track), 2)
                if int(track[i]) not in dropped_ids
            ]
            if not remaining:
                continue
            handle.write(" ".join(fields[:8] + [str(x) for p in remaining for x in p]) + "\n")
            kept_points += 1

    frame_lines = (text / "frames.txt").read_text().splitlines()
    frame_header = [line for line in frame_lines if line.startswith("#")]
    kept_frames = [
        line for line in frame_lines
        if line.strip() and not line.startswith("#") and int(line.split()[-1]) not in dropped_ids
    ]
    (text / "frames.txt").write_text("\n".join(frame_header + kept_frames) + "\n")

    log(f"过滤掉 {len(dropped_ids)} 张非目标图像，保留 {len(kept) // 2} 张；"
        f"points3D 保留 {kept_points} 个")
    return to_bin(text, target)


def force_camera(database: Path, prefix: str, target_camera_id: int) -> None:
    """把新帧归一化到目标模型的相机与 rig 上。

    ⚠️ 这一步不能省，而且要动四张表。COLMAP 4.x 的数据库有 rig 概念：
    提特征时如果相机参数和库里已有的不吻合，它会**新建一台相机 + 新建一个 rig**。
    本项目就踩到了：新帧挂到 rig 2，而 rig 2 的 `ref_sensor_id` 指向那台新相机；
    一旦把多出来的相机删掉，`image_registrator` 读模型时就在
    `Reconstruction::AddRig()` 里 abort。

    顺序很关键：`frames.rig_id` 是 `ON DELETE CASCADE`，**必须先改 frames 再删 rig**，
    否则新帧会被级联删掉。
    """
    import sqlite3

    connection = sqlite3.connect(database)
    changed = connection.execute(
        "update images set camera_id = ? where name like ? and camera_id != ?",
        (target_camera_id, f"{prefix}%", target_camera_id),
    ).rowcount
    frames_moved = connection.execute(
        "update frames set rig_id = (select min(rig_id) from rigs) "
        "where rig_id != (select min(rig_id) from rigs)"
    ).rowcount
    sensors_fixed = connection.execute(
        "update frame_data set sensor_id = ?, sensor_type = 0 where sensor_id != ?",
        (target_camera_id, target_camera_id),
    ).rowcount
    rigs_removed = connection.execute(
        "delete from rigs where rig_id != (select min(rig_id) from rigs)"
    ).rowcount
    orphans = connection.execute(
        "delete from cameras where camera_id not in (select distinct camera_id from images)"
    ).rowcount
    connection.commit()
    connection.close()
    log(f"归一化：{changed} 张改相机  {frames_moved} 帧改 rig  {sensors_fixed} 条改传感器  "
        f"删掉 {rigs_removed} 个多余 rig、{orphans} 台孤儿相机")


def read_camera(database: Path) -> tuple[str, str, int, int]:
    import sqlite3

    connection = sqlite3.connect(database)
    row = connection.execute(
        "select model, width, height, params from cameras order by camera_id limit 1"
    ).fetchone()
    connection.close()
    if row is None:
        raise SystemExit("数据库里没有相机")
    model_id, width, height, blob = row
    params = np.frombuffer(blob, dtype=np.float64)
    return CAMERA_MODELS[model_id], ",".join(f"{v:g}" for v in params), width, height


def model_camera(model_dir: Path, scratch: Path) -> tuple[int, str, str, int, int]:
    """读**模型**里的相机：(camera_id, model_name, 参数串, 宽, 高)。

    ⚠️ 必须用模型里的值，不能用数据库里的：数据库存的是**未优化**的提取参数
    （本项目 f=2304），而模型里是光束法平差**优化后**的（f=1864.27）。
    """
    text = to_txt(model_dir, scratch)
    rows = [
        line for line in (text / "cameras.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(rows) != 1:
        raise SystemExit(f"目标模型有 {len(rows)} 台相机，本工具只支持单相机模型")
    fields = rows[0].split()
    return int(fields[0]), fields[1], ",".join(fields[4:]), int(fields[2]), int(fields[3])


def extract_frames(video: Path, out_dir: Path, fps: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("supp_*.jpg"):
        old.unlink()
    # 补拍常见是 4K：统一降到 1080p，才能复用原素材的相机内参
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", f"scale=1920:1080,fps={fps}", "-q:v", "2", str(out_dir / "supp_%06d.jpg")],
        check=True,
    )
    return len(list(out_dir.glob("supp_*.jpg")))


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="补拍视频")
    source.add_argument("--frames", help="已抽好的补拍帧目录")
    parser.add_argument("--job", required=True, help="任务目录（含 database.db / sparse/）")
    parser.add_argument("--model", default="sparse/0", help="注册进哪个稀疏模型")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧帧率")
    parser.add_argument("--min-registered", type=int, default=8, help="少于这么多张算失败")
    parser.add_argument("--skip-matching", action="store_true", help="复用已有匹配（调试用）")
    args = parser.parse_args()

    job = Path(args.job).resolve()
    work = job / "supp"
    work.mkdir(parents=True, exist_ok=True)
    model_in = job / args.model
    started = time.monotonic()

    # ---------- 0. 数据库与相机 ----------
    database = work / "database.db"
    if not database.exists():
        log(f"复制数据库 -> {database}")
        shutil.copy(job / "database.db", database)
    else:
        log(f"复用已有数据库 {database}")
    model_name, camera_params, width, height = read_camera(database)
    log(f"数据库相机：{model_name} {width}x{height} params=[{camera_params}]")
    # ⚠️ 提特征要用**模型里**的相机参数，不能用数据库里的（数据库是未优化的提取值）
    model_camera_id, model_name, camera_params, width, height = model_camera(
        model_in, work / "camera_txt"
    )
    log(f"模型相机 #{model_camera_id}：{model_name} {width}x{height} params=[{camera_params}]")

    # ---------- 1. 抽帧 ----------
    frames_dir = work / "new_frames"
    if args.video:
        count = extract_frames(Path(args.video), frames_dir, args.fps)
        log(f"抽出 {count} 帧（1080p / {args.fps}fps）")
    else:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(Path(args.frames).iterdir()):
            if source_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy(source_file, frames_dir / source_file.name)
        log(f"沿用 {len(list(frames_dir.iterdir()))} 帧 -> {frames_dir}")

    # ---------- 2. 新帧特征 ----------
    run([
        COLMAP, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", model_name,
        "--ImageReader.camera_params", camera_params,
        "--FeatureExtraction.type", FEATURE_TYPE,
        "--FeatureExtraction.use_gpu", "1",
    ], "提取新帧特征")

    # 强制复用目标模型的相机（数据库里的内参是未优化值，不修的话会建出第二台相机）
    force_camera(database, "supp_", model_camera_id)

    # ---------- 3. 穷举匹配（实测 94.6 万对约 16 分钟，比词汇树方案简单可靠） ----------
    if not args.skip_matching:
        run([
            COLMAP, "exhaustive_matcher",
            "--database_path", str(database),
            "--FeatureMatching.type", MATCHER_TYPE,
            "--FeatureMatching.use_gpu", "1",
        ], "穷举匹配")

    # ---------- 4. 增量注册 ----------
    # ⚠️ colmap 要求 output_path 是**已存在的目录**，否则直接报错退出
    registered = work / "registered_raw"
    if registered.exists():
        shutil.rmtree(registered)
    registered.mkdir(parents=True)
    run([
        COLMAP, "image_registrator",
        "--database_path", str(database),
        "--input_path", str(model_in),
        "--output_path", str(registered),
    ], "增量注册新帧")

    # ---------- 5. 过滤：只留「原模型图像 + 新帧」 ----------
    import sqlite3

    connection = sqlite3.connect(database)
    new_names = {
        name for (name,) in connection.execute("select name from images")
        if name.startswith("supp_")
    }
    connection.close()
    original_names = names_in_model(model_in, work / "original_txt")
    filtered = filter_model(registered, original_names | new_names,
                            work / "model_filtered", work / "filtered_txt")

    before, after = count_images(model_in), count_images(filtered)
    added = after - before
    log(f"注册结果：{before} -> {after} 张（新增 {added} 张）")
    if added < args.min_registered:
        raise SystemExit(f"只注册上 {added} 张（< {args.min_registered}）→ 补拍与旧模型匹配不足。")

    # ---------- 6. 给新帧三角化新点 ----------
    triangulated = work / "model_supplemented"
    if triangulated.exists():
        shutil.rmtree(triangulated)
    triangulated.mkdir(parents=True)
    run([
        COLMAP, "point_triangulator",
        "--database_path", str(database),
        "--image_path", str(frames_dir),
        "--input_path", str(filtered),
        "--output_path", str(triangulated),
    ], "为新帧三角化新点")

    log("")
    log(f"完成，总耗时 {time.monotonic() - started:.0f}s")
    log(f"补充后的模型：{triangulated}（{count_images(triangulated)} 张图像）")
    log("下一步（新旧帧放进同一目录，软链接即可）：")
    log(f"  colmap image_undistorter --image_path <新旧帧目录> --input_path {triangulated} "
        f"--output_path {work}/dense --output_type COLMAP")
    log(f"  colmap patch_match_stereo --workspace_path {work}/dense "
        f"--workspace_format COLMAP --PatchMatchStereo.geom_consistency true")
    log(f"  colmap stereo_fusion --workspace_path {work}/dense --workspace_format COLMAP "
        f"--input_type geometric --output_path {work}/fused_supplemented.ply")


if __name__ == "__main__":
    main()
