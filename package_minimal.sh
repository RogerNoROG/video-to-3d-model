#!/usr/bin/env bash
# ===========================================================================
# 生成「最小可复现集」：网页前后端可运行 + 建模主链可从视频重跑。
#
# 不含稠密重建中间态（storage/*/dense*/stereo 的 238 GB），不含 .venv
# （由 bootstrap.sh + requirements.lock.txt 重建）。内容边界与恢复步骤见 PACKAGING.md。
#
# 用法：
#   ./package_minimal.sh                        # 默认输出 /mnt/e/video-to-3d-model-minimal
#   ./package_minimal.sh /some/where            # 指定输出目录
#   ./package_minimal.sh --dry-run /some/where  # 只列出要复制什么，不写任何文件
#
# 幂等：基于 rsync 增量同步，可重复执行；不会删除目标目录里已有的额外文件。
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

DEST="/mnt/e/video-to-3d-model-minimal"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "未知参数：$arg" >&2; exit 2 ;;
    *) DEST="$arg" ;;
  esac
done

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

command -v rsync > /dev/null 2>&1 || die "找不到 rsync"

# 优先用项目自己的 venv，没有就退回系统 python3
PY_BIN="$ROOT/.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN="$(command -v python3 || true)"
[ -n "$PY_BIN" ] || die "找不到可用的 Python 解释器"

printf '\033[1m最小可复现集打包\033[0m\n'
echo "  源目录  : $ROOT"
echo "  目标目录: $DEST"
[ "$DRY_RUN" = 1 ] && echo "  模式    : --dry-run（不写入任何文件）"

[ -f storage/projects.json ] || die "storage/projects.json 不存在，是否在项目根目录执行？"
[ -d "$ROOT/app" ] || die "缺少 app/ 目录"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
FILELIST="$WORK/filelist.txt"
ARTIFACTS="$WORK/artifacts.txt"

# ---------------------------------------------------------------------------
# 1. 组装文件清单
# ---------------------------------------------------------------------------
step "1/6  组装文件清单"

# 源码与文档是固定的一小撮，单独走一次 rsync（顺带排除 logs/*.pid）
SRC_ITEMS=(app tools docs video_upload.html start.sh bootstrap.sh
           package_minimal.sh PACKAGING.md requirements.txt requirements.lock.txt
           README.md DEVELOPMENT.md .gitignore)
MISSING=()
for item in "${SRC_ITEMS[@]}"; do
  [ -e "$ROOT/$item" ] || MISSING+=("$item")
done
[ ${#MISSING[@]} -gt 0 ] && die "缺少：${MISSING[*]}"

# 任务数据与展示产物合成一张清单，交给**单次** rsync。
# 为什么要合成一次：-H（保留硬链接）只对同一次传输内的文件生效，而部分任务的
# normalized_1080p30.mp4 是 input.mp4 的硬链接；分开传会变成两份实体，白占一倍空间。
printf 'storage/projects.json\n' > "$FILELIST"
JOBS=()
while IFS= read -r d; do
  JOBS+=("$(basename "$d")")
done < <(find storage -mindepth 2 -maxdepth 2 -name job.json -printf '%h\n' | sort)

for job in "${JOBS[@]}"; do
  for name in job.json input.mp4 normalized_1080p30.mp4 pipeline.log frames; do
    [ -e "storage/$job/$name" ] && printf 'storage/%s/%s\n' "$job" "$name" >> "$FILELIST"
  done
done

# 展示产物：按 projects.json 登记的原相对路径取，保证网页能按原索引直接读取
"$PY_BIN" - "$ROOT" <<'PY' > "$ARTIFACTS"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for project in json.loads((root / "storage/projects.json").read_text())["projects"]:
    for art in project.get("artifacts", []):
        if (root / art["path"]).is_file():
            print(art["path"])
PY
cat "$ARTIFACTS" >> "$FILELIST"

ARTIFACT_COUNT="$(wc -l < "$ARTIFACTS" | tr -d ' ')"
[ "$ARTIFACT_COUNT" -gt 0 ] || die "projects.json 里没有可用的产物文件"

echo "  建模任务 : ${#JOBS[@]} 个"
printf '             %s\n' "${JOBS[@]}"
echo "  展示产物 : $ARTIFACT_COUNT 个"
echo "  清单条目 : $(wc -l < "$FILELIST" | tr -d ' ')"
ok "源码 ${#SRC_ITEMS[@]} 项 + 任务数据 + 展示产物"

# ---------------------------------------------------------------------------
# 2. 复制
# ---------------------------------------------------------------------------
step "2/6  复制到目标目录"

if [ "$DRY_RUN" = 1 ]; then
  echo "  清单前 10 条（实际不会复制）："
  head -n 10 "$FILELIST" | sed 's/^/      /'
  echo "      ... 共 $(wc -l < "$FILELIST" | tr -d ' ') 条"
  echo "  + 源码：${SRC_ITEMS[*]}"
else
  mkdir -p "$DEST" || die "无法创建目标目录 $DEST"

  rsync -aH "${SRC_ITEMS[@]}" "$DEST/" || die "源码复制失败"
  ok "源码与文档 ${#SRC_ITEMS[@]} 项"

  if compgen -G "logs/*.log" > /dev/null 2>&1; then
    mkdir -p "$DEST/logs"
    rsync -aH --include='*/' --include='*.log' --exclude='*' logs/ "$DEST/logs/" \
      || warn "logs 复制失败"
    ok "logs/*.log（已排除过期 *.pid，否则会拖累目标机的 ./start.sh start）"
  fi

  echo "  复制任务数据与展示产物（含视频，是本包最慢的一步）..."
  # ⚠️ 必须显式写 -r：--files-from 会**关掉 -a 隐含的递归**，清单里的目录只会被建成
  #    空目录。踩过：frames/ 建成空目录，0 张抽帧，而外部检查只看目录是否存在，静默通过。
  rsync -aH -r --files-from="$FILELIST" "$ROOT/" "$DEST/" || die "任务数据复制失败"
  ok "$(wc -l < "$FILELIST" | tr -d ' ') 个条目"
fi

# ---------------------------------------------------------------------------
# 3. 外部依赖（在项目目录之外，缺了跑不出同样结果）
# ---------------------------------------------------------------------------
step "3/6  外部依赖（COLMAP 二进制 + ALIKED ONNX 权重）"

COLMAP_SRC="${MODEL_API_COLMAP_BINARY:-}"
[ -z "$COLMAP_SRC" ] && [ -x "$HOME/.local/bin/colmap" ] && COLMAP_SRC="$HOME/.local/bin/colmap"
[ -z "$COLMAP_SRC" ] && COLMAP_SRC="$(command -v colmap 2> /dev/null || true)"

if [ -n "$COLMAP_SRC" ] && [ -x "$COLMAP_SRC" ]; then
  # ⚠️ colmap 的输出必须落盘再读，不能接管道：管道提前关闭会让它收到 SIGPIPE 被杀，
  #    若当时正在写模型就会把结果丢掉（本项目踩过这个坑）。
  CM_INFO="$WORK/colmap-help.txt"
  "$COLMAP_SRC" -h > "$CM_INFO" 2>&1
  if grep -q 'with CUDA' "$CM_INFO"; then
    ok "COLMAP：$(head -n 1 "$CM_INFO")"
  else
    warn "COLMAP 不带 CUDA（$(head -n 1 "$CM_INFO")）—— 目标机无法用 GPU 重建"
  fi
  if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$DEST/external"
    rsync -aH "$COLMAP_SRC" "$DEST/external/colmap" || die "colmap 复制失败"
    chmod +x "$DEST/external/colmap"
  fi
  ok "→ external/colmap（$(du -h "$COLMAP_SRC" | cut -f1)，来自 $COLMAP_SRC）"
else
  warn "本机找不到 colmap，external/colmap 未生成 —— 目标机需自备自编译 CUDA 版"
fi

ONNX_SRC="$HOME/.cache/colmap"
if compgen -G "$ONNX_SRC/*.onnx" > /dev/null 2>&1; then
  if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$DEST/external/cache-colmap"
    rsync -aH "$ONNX_SRC"/*.onnx "$DEST/external/cache-colmap/" || warn "ONNX 权重复制失败"
  fi
  ok "→ external/cache-colmap/（$(find "$ONNX_SRC" -maxdepth 1 -name '*.onnx' | wc -l) 个）"
  echo "     缺了它且目标机断网时，后端会静默回退 SIFT，重建结果差异极大"
else
  warn "本机没有 $ONNX_SRC/*.onnx —— 目标机首次运行需联网下载"
fi

# ---------------------------------------------------------------------------
# 4. 生成 PACKAGE-README.md（用实测数据填充清单）
# ---------------------------------------------------------------------------
step "4/6  生成 PACKAGE-README.md"

if [ "$DRY_RUN" = 1 ]; then
  echo "  （--dry-run 跳过）"
else
  "$PY_BIN" - "$ROOT" "$DEST" <<'PY' || die "生成说明失败"
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root, dest = Path(sys.argv[1]), Path(sys.argv[2])


def iter_files(path: Path):
    if not path.exists():
        return
    if path.is_file():
        yield path
        return
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            f = Path(dirpath) / name
            if f.is_file():
                yield f


_claimed: set = set()


def collect(paths) -> tuple:
    """返回 (字节数, 文件数)。按 (dev, inode) 去重：硬链接只在最先归类的类别里计一次，
    否则 normalized_1080p30.mp4 与 input.mp4 会被算成两份，把清单数字撑虚。"""
    size = 0
    count = 0
    for item in paths:
        for f in iter_files(dest / item):
            try:
                st = f.stat()
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in _claimed:
                continue
            _claimed.add(key)
            size += st.st_size
            count += 1
    return size, count


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


jobs = sorted(p.name for p in (root / "storage").iterdir()
              if p.is_dir() and (p / "job.json").exists())

# 逐项用「目标目录里的实际体量」统计，确保清单反映的是包内真实内容
CATEGORIES = [
    ("源码与文档", ["app", "tools", "docs", "video_upload.html", "start.sh", "bootstrap.sh",
                    "package_minimal.sh", "PACKAGING.md", "PACKAGE-README.md",
                    "requirements.txt", "requirements.lock.txt",
                    "README.md", "DEVELOPMENT.md", ".gitignore"]),
    ("运行日志", ["logs"]),
    ("项目与任务索引", ["storage/projects.json"] + [f"storage/{j}/job.json" for j in jobs]),
    ("原始视频", [f"storage/{j}/input.mp4" for j in jobs]),
    ("归一化视频", [f"storage/{j}/normalized_1080p30.mp4" for j in jobs]),
    ("抽帧结果", [f"storage/{j}/frames" for j in jobs]),
    ("命令行日志", [f"storage/{j}/pipeline.log" for j in jobs]),
    ("外部依赖", ["external"]),
]

rows = []
for label, items in CATEGORIES:
    size, count = collect(items)
    rows.append((label, count, size))

# 展示产物散落在 storage/ 与 docs/ 之外的地方，取所有尚未被归类的文件
art_size, art_count = 0, 0
for _f in iter_files(dest):
    try:
        _st = _f.stat()
    except OSError:
        continue
    _key = (_st.st_dev, _st.st_ino)
    if _key in _claimed:
        continue
    _claimed.add(_key)
    art_size += _st.st_size
    art_count += 1

# 与 projects.json 的登记数量交叉核对：不一致说明清单漏了或多了文件
declared = sum(len(p.get("artifacts", [])) for p in
               json.loads((root / "storage/projects.json").read_text())["projects"])
note = "" if art_count == declared else f"（projects.json 登记 {declared} 个，不一致需检查）"
rows.append((f"展示产物（模型文件）{note}", art_count, art_size))

total_size = sum(size for _, _, size in rows)
total_count = sum(count for _, count, _ in rows)
disk_usage = subprocess.run(["du", "-sh", str(dest)], capture_output=True,
                            text=True).stdout.split()
disk_str = disk_usage[0] if disk_usage else "?"

env = [
    f"- 生成时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
    f"- 源目录：`{root}`",
    f"- 建模任务：{len(jobs)} 个 —— " + "、".join(f"`{j[:8]}`" for j in jobs),
]
for cmd in (["python3.12", "--version"], ["ffmpeg", "-version"],
            ["bash", "-c", ". /etc/os-release && echo $PRETTY_NAME"]):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
        if out:
            env.append(f"- {out[0][:100]}")
    except OSError:
        pass

manifest = ["", "### 生成信息", ""] + env
manifest += ["", "### 逐项体量", "", "| 内容 | 文件数 | 体积 |", "| --- | ---: | ---: |"]
for label, count, size in rows:
    if count or size:
        manifest.append(f"| {label} | {count:,} | {human(size)} |")
manifest.append(f"| **合计** | **{total_count:,}** | **{human(total_size)}** |")
manifest += ["", f"磁盘实际占用 **{disk_str}** —— 比表内合计略小，因为部分任务的 "
             "`normalized_1080p30.mp4` 是 `input.mp4` 的硬链接（打包时用 `rsync -H` 保留），"
             "同一份数据只在磁盘上存一次。"]
manifest += [
    "", "### 与完整项目的对比", "",
    "- 完整项目（不含视频）：约 274 GB",
    f"- 本包：**{human(total_size)}**（约 {total_size / (274 * 1024 ** 3) * 100:.1f}%）",
    "- 差异主要是 `storage/*/dense*/stereo/` 的 238 GB 深度图/法线图，见正文 §3",
    "", "### 目标机上的恢复步骤（详见正文 §4）", "",
    "```bash",
    "cp external/colmap ~/.local/bin/colmap && chmod +x ~/.local/bin/colmap   # ① 自编译 CUDA 版",
    "mkdir -p ~/.cache/colmap && cp external/cache-colmap/*.onnx ~/.cache/colmap/  # ② ALIKED 权重",
    "./bootstrap.sh --mirror      # ③ 重建 venv 并逐项校验",
    "./start.sh start             # ④ 启动 → http://localhost:5500/video_upload.html",
    "```", "",
]

doc = (root / "PACKAGING.md").read_text()
if "<!-- MANIFEST -->" in doc:
    out = doc.replace("<!-- MANIFEST -->", "\n".join(manifest))
else:
    print("  ! PACKAGING.md 缺少 <!-- MANIFEST --> 占位符，改为直接追加")
    out = doc + "\n".join(manifest) + "\n"
(dest / "PACKAGE-README.md").write_text(out)
print(f"  \033[32m✓\033[0m PACKAGE-README.md")
print(f"    合计 {human(total_size)}，{total_count:,} 个文件")
PY
fi

# ---------------------------------------------------------------------------
# 5. 校验
# ---------------------------------------------------------------------------
step "5/6  校验"

if [ "$DRY_RUN" = 1 ]; then
  echo "  （--dry-run 跳过）"
else
  PROBLEMS=0

  # 网页能否展示，取决于 projects.json 里每个相对路径在目标目录中都存在
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    [ -f "$DEST/$rel" ] || { warn "缺失：$rel"; PROBLEMS=$((PROBLEMS + 1)); }
  done < "$ARTIFACTS"
  [ "$PROBLEMS" = 0 ] && ok "projects.json 登记的 $ARTIFACT_COUNT 个产物在目标目录中均可找到"

  for item in bootstrap.sh start.sh requirements.lock.txt PACKAGE-README.md \
              app/main.py video_upload.html storage/projects.json; do
    [ -e "$DEST/$item" ] || { warn "缺少 $item"; PROBLEMS=$((PROBLEMS + 1)); }
  done

  JOB_N_SRC="$(find storage -mindepth 2 -maxdepth 2 -name job.json | wc -l)"
  JOB_N_DST="$(find "$DEST/storage" -mindepth 2 -maxdepth 2 -name job.json 2> /dev/null | wc -l)"
  if [ "$JOB_N_SRC" = "$JOB_N_DST" ]; then
    ok "任务目录完整：$JOB_N_DST/$JOB_N_SRC"
  else
    warn "任务目录数量不符：目标 $JOB_N_DST / 源 $JOB_N_SRC"
    PROBLEMS=$((PROBLEMS + 1))
  fi

  # 抽帧必须逐任务比对**文件数**：只检查目录存在是不够的 —— --files-from 不递归时
  # 会建出空目录，只查 -d 会静默放过。
  FRAMES_BAD=0
  FRAMES_TOTAL=0
  for job in "${JOBS[@]}"; do
    SRC_N="$(find "storage/$job/frames" -type f 2> /dev/null | wc -l)"
    DST_N="$(find "$DEST/storage/$job/frames" -type f 2> /dev/null | wc -l)"
    FRAMES_TOTAL=$((FRAMES_TOTAL + DST_N))
    if [ "$SRC_N" -gt 0 ] && [ "$SRC_N" = "$DST_N" ]; then
      :
    else
      warn "任务 $job 抽帧不完整：源 $SRC_N 张 / 目标 $DST_N 张"
      FRAMES_BAD=$((FRAMES_BAD + 1))
    fi
  done
  if [ "$FRAMES_BAD" = 0 ]; then
    ok "每个任务的抽帧都完整（共 $FRAMES_TOTAL 张，逐任务比对文件数）"
  else
    PROBLEMS=$((PROBLEMS + FRAMES_BAD))
  fi

  # 视频同样逐任务比对大小，防止清单写错导致静默漏掉
  VIDEO_BAD=0
  for job in "${JOBS[@]}"; do
    for name in input.mp4 pipeline.log job.json; do
      A="$(stat -c%s "storage/$job/$name" 2> /dev/null || echo -1)"
      B="$(stat -c%s "$DEST/storage/$job/$name" 2> /dev/null || echo -2)"
      [ "$A" = "$B" ] || { warn "任务 $job 的 $name 大小不符（源 $A / 目标 $B）"; VIDEO_BAD=$((VIDEO_BAD + 1)); }
    done
  done
  [ "$VIDEO_BAD" = 0 ] && ok "每个任务的视频/命令行日志/任务元数据大小一致" \
    || PROBLEMS=$((PROBLEMS + VIDEO_BAD))

  if [ -x "$DEST/external/colmap" ]; then
    ok "external/colmap 可执行"
  else
    warn "external/colmap 缺失或不可执行"
    PROBLEMS=$((PROBLEMS + 1))
  fi

  # 包内不应有 .pid：那些 pid 在目标机上是过期的，会让 ./start.sh start 误判服务状态
  if find "$DEST" -name '*.pid' -print -quit | grep -q .; then
    warn "包内存在 .pid 文件，目标机上 ./start.sh 可能误判服务状态"
  else
    ok "无过期 .pid 文件"
  fi

  # 稠密中间产物绝不该进这个包（进来说明 --files-from 清单写错了）
  if [ -d "$DEST/storage" ] && find "$DEST/storage" -type d -name stereo -print -quit | grep -q .; then
    warn "包内出现了 dense*/stereo/ 目录，本包不应包含稠密中间产物"
    PROBLEMS=$((PROBLEMS + 1))
  else
    ok "未包含稠密中间产物（dense*/stereo/）"
  fi

  [ "$PROBLEMS" = 0 ] && ok "校验全部通过" || warn "$PROBLEMS 项待确认（见上）"
fi

# ---------------------------------------------------------------------------
# 6. 汇总
# ---------------------------------------------------------------------------
step "6/6  完成"
if [ "$DRY_RUN" = 1 ]; then
  echo "  --dry-run 结束，未写入任何文件。去掉 --dry-run 即可实际打包。"
else
  echo "  输出    : $DEST"
  echo "  体积    : $(du -sh "$DEST" | cut -f1)   （$(find "$DEST" -type f | wc -l) 个文件）"
  echo
  echo "  目标机上的下一步："
  echo "    1) cp external/colmap ~/.local/bin/colmap && chmod +x ~/.local/bin/colmap"
  echo "    2) mkdir -p ~/.cache/colmap && cp external/cache-colmap/*.onnx ~/.cache/colmap/"
  echo "    3) ./bootstrap.sh --mirror"
  echo "    4) ./start.sh start    →  http://localhost:5500/video_upload.html"
  echo
  echo "  详细说明见包内 PACKAGE-README.md"
fi
