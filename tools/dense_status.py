#!/usr/bin/env python3
"""报告独立稠密重建（tools/build_dense_model.py）的进度。

这些重建跑在各自的会话里，不属于后端任务，所以 ./start.sh status 原本看不到它们。
进度直接数 depth_maps 里的产物文件，速率用最早/最新产物的 mtime 推算 ——
不依赖日志解析，中途重启也能给出正确速率。

输出示例：
  模型1  运行中 (pid 445810)   产物 56/1326 (4.2%)   2.7/分钟   剩余 466 分钟 (约 06:45)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def detect_pass(pass_dir: Path) -> int:
    """判断这一遍是否已经开始产出，用于区分「第一遍」和「第二遍」。

    geom_consistency=true 时 COLMAP 先写 .photometric.bin 再写 .geometric.bin，
    所以出现 geometric 文件就说明进入第二遍了。
    """
    try:
        names = os.listdir(pass_dir)
    except OSError:
        return 0
    if any(n.endswith("geometric.bin") for n in names):
        return 2
    if any(n.endswith("photometric.bin") for n in names):
        return 1
    return 0


def main() -> int:
    states = sorted(ROOT.glob("logs/dense_model*.json"))
    if not states:
        if "--quiet" not in sys.argv:
            print("稠密构建: (无记录，用 tools/build_dense_model.py 启动后会出现在这里)")
        return 0

    for state_path in states:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        model = state.get("model")
        job = state.get("job", "")
        total = int(state.get("total") or 0)
        views = int(state.get("views") or 0)
        passes = int(state.get("passes") or 1)
        depth_dir = ROOT / "storage" / job / f"dense_{model}" / "stereo" / "depth_maps"
        fused = ROOT / "storage" / job / f"fused_{model}.ply"

        pid_file = Path(str(state_path).replace(".json", ".pid"))
        pid = None
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except ValueError:
                pid = None
        running = pid is not None and alive(pid)

        try:
            entries = [e for e in os.scandir(depth_dir) if e.name.endswith(".bin")]
        except OSError:
            entries = []
        done = len(entries)

        label = f"模型{model}"
        if state.get("finished_at"):
            size = fused.stat().st_size / 1024 / 1024 if fused.exists() else 0
            print(f"{label:<6} 已完成        {fused.name} {size:.0f} MB")
            continue

        if not running:
            print(f"{label:<6} 已停止 (pid {pid})  产物 {done:,}/{total:,} —— 重新运行同一命令可断点续跑")
            continue

        phase = detect_pass(depth_dir)
        if done == 0:
            print(f"{label:<6} 运行中 (pid {pid})   正在准备/读取工作区…")
            continue

        percent = done / total * 100 if total else 0
        rate = eta_text = ""
        if len(entries) >= 2:
            times = sorted(e.stat().st_mtime for e in entries)
            span = times[-1] - times[0]
            if span > 0:
                per_min = (done - 1) / span * 60
                remaining = (total - done) / per_min if per_min > 0 else float("inf")
                rate = f"{per_min:.1f}/分钟"
                eta_text = (
                    f"剩余 {remaining:.0f} 分钟 (约 {(datetime.now() + timedelta(minutes=remaining)):%H:%M})"
                    if remaining != float("inf")
                    else ""
                )
        stage = f"第{phase}遍/共{passes}遍  " if passes > 1 and phase else ""
        print(f"{label:<6} 运行中 (pid {pid})   {stage}产物 {done:,}/{total:,} ({percent:.1f}%)   {rate}   {eta_text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
