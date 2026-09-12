#!/usr/bin/env bash
# 把「迁移到另一台电脑所需的最小文件集」打包成一个 tar.gz。
#
# 体积取舍（源机器实测）：
#   仅源码             ~250 KB   —— 目标机器重新跑一遍全部流程
#   + 交付产物         ~4.6 GB   —— 保留 model.glb / fused.ply / 稀疏模型，可直接 remesh
#   + 稠密中间结果     ~47 GB    —— 保留深度图，可重跑融合而不用重跑 patch_match（省 7 小时）
#
# 用法：
#   ./export_for_migration.sh                  # 默认：源码 + 交付产物（约 4.6 GB）
#   ./export_for_migration.sh --with-dense     # 额外包含稠密中间结果（约 47 GB）
#   ./export_for_migration.sh --source-only    # 只打包源码（约 250 KB）
set -euo pipefail

cd "$(dirname "$0")"
WITH_DENSE=0
SOURCE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-dense) WITH_DENSE=1 ;;
    --source-only) SOURCE_ONLY=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d-%H%M)"
OUT="migration-${STAMP}.tar.gz"

# 源码（版本库已跟踪的文件），始终包含
mapfile -t FILES < <(git ls-files 2>/dev/null || printf '%s\n' \
  README.md requirements.txt start.sh video_upload.html app)

if [ "$SOURCE_ONLY" -eq 0 ]; then
  JOB=e7f9f9e5d046490d93b34e4538bc9ef1
  if [ -d "storage/$JOB" ]; then
    echo "包含任务 $JOB 的交付产物（约 4.6 GB）"
    # 体积小但难重建的产物：交付模型、融合点云、稀疏模型、特征库、抽帧
    for item in job.json model.glb fused.ply sparse frames database.db; do
      [ -e "storage/$JOB/$item" ] && FILES+=("storage/$JOB/$item")
    done
    # input.mp4 是原始素材，重跑必需（也可从原始来源重新拷贝）
    [ -e "storage/$JOB/input.mp4" ] && FILES+=("storage/$JOB/input.mp4")
    if [ "$WITH_DENSE" -eq 1 ]; then
      echo "额外包含稠密中间结果 dense/（约 42 GB，保留可重跑融合）"
      FILES+=("storage/$JOB/dense")
    else
      echo "跳过 dense/（42 GB）。目标机器将无法重跑 stereo_fusion，"
      echo "但 remesh 只需要 fused.ply，不受影响。需要时用 --with-dense 重新导出。"
    fi
  else
    echo "警告：找不到 storage/$JOB，只打包源码"
  fi
fi

echo "打包 ${#FILES[@]} 项 -> $OUT"
tar -czf "$OUT" -- "${FILES[@]}"
ls -lh "$OUT"
echo
echo "迁移到目标机器后："
echo "  tar -xzf $OUT -C /path/to/video-to-3d-model"
echo "  然后按 HANDOVER.md 第 3 节搭建环境"
