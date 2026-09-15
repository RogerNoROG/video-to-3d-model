#!/usr/bin/env python3
"""跨场次匹配：绕开 COLMAP 自带的匹配器，自己算描述子相似度再导入。

为什么必须绕开
------------
补拍视频和原素材是**两次不同的拍摄**（光照/距离/白平衡都变了），物体本身还是同一块木头。
实测同一张补拍帧与最佳旧帧的互近邻匹配：
    相似度≥0.85 → 0 个    ≥0.75 → 0~1 个    ≥0.70 → 3~9 个
    ≥0.65 → 20~37 个      ≥0.60 → 76~157 个
而 ALIKED 匹配器默认门限就是 **0.85** → 一个都匹配不上。

致命的是：`--AlikedMatching.brute_force_min_cossim` **改不动**（0.85 扫到 0.30，
同一批的匹配对数全程不变，说明该参数没生效），所以没法靠调参解决。
**结论：不是视频不能用，是工具链没配对。**

做法
----
自己按定义算：描述子归一化 → 点积 → **双向互为最近邻** + 阈值过滤。
互近邻天然排除大量误匹配，剩下的交给 COLMAP 的几何验证（RANSAC）兜底。

输出 COLMAP `matches_importer --match_type inliers` 的格式：
    image_id1 image_id2 num_matches feature_idx1 feature_idx2 ...

用法
----
    ./.venv/bin/python tools/pipeline/match_cross_session.py --job <任务id>
    # 然后
    colmap matches_importer --database_path <job>/supp/database.db \\
        --match_list_path <job>/supp/cross_matches.txt --match_type inliers
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def log(message: str) -> None:
    print(f"[cross] {message}", flush=True)


def load_descriptors(database: Path) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(database)
    rows = list(connection.execute(
        "select i.image_id, i.name, d.rows, d.cols, d.data "
        "from images i join descriptors d on d.image_id = i.image_id"
    ))
    connection.close()
    table: dict[str, np.ndarray] = {}
    for _, name, count, cols, blob in rows:
        table[name] = np.frombuffer(blob, dtype=np.float32).reshape(count, cols // 4)
    return table


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def mutual_matches(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    """返回 (idx_a, idx_b)：双向互为最近邻且相似度都过阈值的配对。"""
    similarity = a @ b.T
    forward = similarity.argmax(axis=1)
    backward = similarity.argmax(axis=0)
    index_a = np.arange(len(a))
    best = similarity[index_a, forward]
    keep = (backward[forward] == index_a) & (best >= threshold)
    return np.column_stack([index_a[keep], forward[keep]]).astype(np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="余弦相似度门限。实测 0.65 时每对能有 20~37 个互近邻")
    parser.add_argument("--min-matches", type=int, default=12,
                        help="低于这么多互近邻的图对不写（COLMAP Mapper.min_num_matches 默认 15）")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.job).resolve()
    database = root / "supp" / "database.db"
    output = root / "supp" / "cross_matches.txt"

    started = time.monotonic()
    table = load_descriptors(database)
    connection = sqlite3.connect(database)
    ids = {name: image_id for image_id, name in
           connection.execute("select image_id, name from images")}
    model_images = {
        name for (name,) in connection.execute(
            "select name from images where name not like 'supp_%'")
    }
    connection.close()

    supp_names = sorted(n for n in table if n.startswith("supp_"))
    reference = sorted(n for n in table if n in model_images)
    log(f"载入 {len(table)} 张描述子（补拍 {len(supp_names)}，旧帧 {len(reference)}），"
        f"耗时 {time.monotonic() - started:.0f}s")

    reference_normalized = {name: normalize(table[name]) for name in reference}

    def match_one(name: str):
        a = normalize(table[name])
        found = []
        for other in reference:
            pairs = mutual_matches(a, reference_normalized[other], args.threshold)
            if len(pairs) >= args.min_matches:
                found.append((other, pairs))
        return name, found

    found_total = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, output.open("w") as handle:
        for done, (name, found) in enumerate(pool.map(match_one, supp_names), start=1):
            for other, pairs in found:
                handle.write(
                    f"{ids[name]} {ids[other]} {len(pairs)} "
                    + " ".join(f"{a} {b}" for a, b in pairs) + "\n"
                )
                found_total += 1
            if done % 20 == 0 or done == len(supp_names):
                elapsed = time.monotonic() - started
                log(f"  {done}/{len(supp_names)} 帧  已写出 {found_total:,} 对  "
                    f"预计剩余 {elapsed / done * (len(supp_names) - done):.0f}s")

    log(f"完成：{found_total:,} 个跨场次匹配对 -> {output}")
    log(f"接下来：colmap matches_importer --database_path {database} "
        f"--match_list_path {output} --match_type inliers")


if __name__ == "__main__":
    main()
