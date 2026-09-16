#!/usr/bin/env bash
# 启动后端与静态前端，并让它们真正脱离终端。
#
# 为什么必须脱离终端：uvicorn 用 BackgroundTasks 跑重建，子进程与后端同生共死。
# 后端一旦被杀，正在跑的 colmap 会一起死，几小时的稠密重建就白做了。
#
# 为什么只用 nohup 不够：
#   nohup 只把 SIGHUP 设为忽略，进程仍留在原终端所属的**会话**里
#   （`ps -o sid` 可验证）。VS Code 关闭终端时若定向发 SIGTERM，照样杀得掉 —— 本项目
#   已被这样坑过一次（退出码 143）。setsid 会新建会话、脱离控制终端，才真正安全。
# 所以这里用：setsid（新会话）+ nohup（忽略 SIGHUP）+ </dev/null（不占 stdin）。
#
# 用法：
#   ./start.sh            # 启动（若已在跑则跳过）
#   ./start.sh status     # 查看前后台是否仍在运行 + 任务进度 + 独立稠密构建进度
#   ./start.sh watch      # 每 30 秒刷新一次 status（Ctrl-C 退出，不影响后台）
#   ./start.sh restart    # 重启后端并自动续跑未完成的稠密阶段
#   ./start.sh stop       # 停止后端与前端（不会动独立稠密构建）
#   ./start.sh stop-dense # 显式停止独立稠密构建（产物保留，可断点续跑）
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"
BACKEND_PID="logs/backend.pid"
FRONTEND_PID="logs/frontend.pid"
# 项目 API 与任务主链统一在 8000（原先另有 8001 的 ETA API，已合并回来）。
# ⚠️ 8000 上可能有正在跑的 COLMAP 任务：重启前先确认没有 processing/queued 的任务，
#    否则会中断数小时的稠密重建。

export PATH="$HOME/.local/bin:$PATH"
export MODEL_API_COLMAP_BINARY="$HOME/.local/bin/colmap"
export MODEL_API_USE_GPU=true
export MODEL_API_PREFER_CUDA=true
export MODEL_API_ALLOW_CPU_FALLBACK=false
export MODEL_API_FEATURE_EXTRACTOR=ALIKED_N16ROT
# 单个 COLMAP 稠密任务使用全部核心；缓存固定在可用内存范围内，避免默认 32GB
# 缓存触发换页反而让 CPU 空转。需要为别的程序预留核心时可在启动前覆盖这两个变量。
export MODEL_API_COLMAP_NUM_THREADS="${MODEL_API_COLMAP_NUM_THREADS:--1}"
export MODEL_API_COLMAP_CACHE_SIZE_GB="${MODEL_API_COLMAP_CACHE_SIZE_GB:-8}"

# 抽帧率。它同时决定重建规模和耗时（稠密约 23 秒/视角/遍，开 geom_consistency 跑两遍，
# 所以总时长 ≈ 帧数 × 2 × 23 秒）。默认 3 与原始素材一致；长视频可用环境变量覆盖：
#   MODEL_API_FRAME_FPS=1 ./start.sh restart
export MODEL_API_FRAME_FPS="${MODEL_API_FRAME_FPS:-3}"

detach() {  # detach <logfile> <command...>
  local log="$1"; shift
  setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
  echo $!
}

alive() {  # alive <pidfile> -> 0 存活 / 1 不存在或已死
  local pidfile="$1"
  [ -f "$pidfile" ] || return 1
  local pid; pid="$(cat "$pidfile")"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

listening() {  # listening <port> -> 0 表示已有服务监听，避免误重启正在跑的建模后端
  ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
}

# 独立稠密构建（tools/pipeline/build_dense_model.py）跑在各自的会话里，与后端无关。
# 它们各自有一个 logs/dense_model*.pid，且会话号(SID)等于该 python 进程的 PID。
# 收集这些会话号，用于在 stop 时**避免误杀**它们正在跑的子 colmap。
dense_build_pids() {
  local p
  for f in logs/dense_model*.pid; do
    [ -f "$f" ] || continue
    p="$(cat "$f" 2>/dev/null)"
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
  done
}

stop_colmap_except_dense() {
  local protected; protected="$(dense_build_pids)"
  local killed=0 skipped=0 pid sid keep
  for pid in $(pgrep -f 'colmap' 2>/dev/null); do
    sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')"
    keep=0
    for p in $protected; do [ "$sid" = "$p" ] && keep=1 && break; done
    if [ "$keep" = 1 ]; then
      skipped=$((skipped + 1))
      continue
    fi
    kill "$pid" 2>/dev/null && killed=$((killed + 1))
  done
  if [ "$killed" -gt 0 ] || [ "$skipped" -gt 0 ]; then
    for _ in $(seq 1 40); do
      pgrep -f 'colmap patch_match_stereo' > /dev/null || break
      sleep 0.5
    done
    echo "colmap：已清理 $killed 个遗留进程；保留 $skipped 个属于独立稠密构建的进程"
  fi
}

start() {
  if listening 8000; then
    echo "后端端口 8000 已有服务监听；保留现有建模进程"
  elif alive "$BACKEND_PID"; then
    echo "后端已在运行 (pid $(cat "$BACKEND_PID"))"
  else
    detach "$BACKEND_LOG" ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
      > "$BACKEND_PID"
    echo "后端已启动 (pid $(cat "$BACKEND_PID"))，日志 $BACKEND_LOG"
  fi
  if listening 5500; then
    echo "前端端口 5500 已有服务监听；保留现有网页服务"
  elif alive "$FRONTEND_PID"; then
    echo "前端已在运行 (pid $(cat "$FRONTEND_PID"))"
  else
    detach "$FRONTEND_LOG" python3 -m http.server 5500 > "$FRONTEND_PID"
    echo "前端已启动 (pid $(cat "$FRONTEND_PID")): http://localhost:5500/video_upload.html"
  fi
  # 后端冷启动约 2-3 秒，轮询等待而不是 sleep 固定时长
  for _ in $(seq 1 40); do
    curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1 && break
    sleep 0.25
  done
  status
}

stop() {
  if alive "$BACKEND_PID"; then
    kill "$(cat "$BACKEND_PID")" 2>/dev/null || true
    for _ in $(seq 1 20); do
      alive "$BACKEND_PID" || break
      sleep 0.25
    done
    alive "$BACKEND_PID" && kill -9 "$(cat "$BACKEND_PID")" 2>/dev/null || true
  fi
  if alive "$FRONTEND_PID"; then
    kill "$(cat "$FRONTEND_PID")" 2>/dev/null || true
  fi
  # 关键：uvicorn 优雅退出**不会**杀死 BackgroundTasks 起的子进程，colmap 会变成孤儿
  # 继续跑，与下一次启动的 colmap 抢 GPU、抢同一批输出文件。必须显式清掉。
  # 但要避开 tools/pipeline/build_dense_model.py 起的 colmap —— 那是另开的长期任务，
  # 一刀切 pkill 会把几小时的稠密重建打掉。
  stop_colmap_except_dense
  rm -f "$BACKEND_PID" "$FRONTEND_PID"
  echo "已停止。产物保留在 storage/<job_id>/，重启后用 ./start.sh restart 自动续跑。"
}

stop_dense() {
  local p found=0
  for f in logs/dense_model*.pid; do
    [ -f "$f" ] || continue
    p="$(cat "$f" 2>/dev/null)"
    [ -n "$p" ] || continue
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
      echo "已停止稠密构建 pid $p ($(basename "$f" .pid))"
      found=1
    fi
  done
  [ "$found" = 0 ] && echo "没有正在运行的独立稠密构建"
  echo "产物保留在 storage/<job_id>/dense_*/ 与 fused_*.ply，重新运行同一命令可断点续跑。"
}

resume_unfinished() {
  local ids
  ids=$(curl -fsS http://127.0.0.1:8000/api/v1/jobs 2>/dev/null |
    python3 -c 'import json,sys
try: jobs = json.load(sys.stdin)
except Exception: sys.exit(0)
for j in jobs:
    if j.get("status") in ("processing", "queued") and not j.get("result_url"):
        print(j["id"])' 2>/dev/null)
  [ -z "$ids" ] && { echo "没有需要续跑的任务"; return; }
  for id in $ids; do
    echo "续跑 $id ..."
    curl -fsS -X POST "http://127.0.0.1:8000/api/v1/jobs/$id/resume" > /dev/null \
      && echo "  已提交" || echo "  提交失败（可能 dense/ 不完整，需要重试整个任务）"
  done
}

status() {
  local bpid fpid
  if alive "$BACKEND_PID"; then
    printf '后端    : 运行中 (pid %s)  %s\n' "$(cat "$BACKEND_PID")" "$(curl -fsS http://127.0.0.1:8000/health 2>/dev/null || echo '端口未响应')"
    bpid="$(cat "$BACKEND_PID")"
    printf '  会话  : SID=%s  TT=%s  (TT 为空 = 已脱离终端)\n' \
      "$(ps -o sid= -p "$bpid" | tr -d ' ')" "$(ps -o tty= -p "$bpid" | tr -d ' ')"
  else
    echo "后端    : 未运行"
  fi
  if alive "$FRONTEND_PID"; then
    echo "前端    : 运行中 (pid $(cat "$FRONTEND_PID"))  http://localhost:5500/video_upload.html"
  else
    echo "前端    : 未运行"
  fi
  echo -n "colmap  : "
  local n; n=$(pgrep -cf 'colmap patch_match_stereo')
  if [ "${n:-0}" -eq 0 ]; then
    echo "未运行"
  elif [ "$n" -eq 1 ]; then
    echo "运行中 (pid $(pgrep -f 'colmap patch_match_stereo' | head -n1))"
  else
    echo "⚠️ 有 $n 个实例同时在跑，会互相抢 GPU 和输出文件！执行 ./start.sh restart 清理"
  fi
  curl -fsS http://127.0.0.1:8000/api/v1/jobs 2>/dev/null |
    python3 -c 'import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if not jobs:
    print("任务    : (无)")
for j in jobs[:3]:
    print("任务    : %-10s %3d%%  %s  (%s)" % (
        j.get("status", ""), j.get("progress", 0), j.get("stage", "")[:58], j.get("id", "")[:8]))'

  # 独立稠密构建（tools/pipeline/build_dense_model.py）不属于后端任务，得单独报
  if [ -x ./.venv/bin/python ]; then
    ./.venv/bin/python tools/diagnostics/project_check.py status 2>/dev/null || true
  else
    python3 tools/diagnostics/project_check.py status 2>/dev/null || true
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  watch)
    while true; do
      clear 2>/dev/null || printf '\033[2J\033[H'
      date '+%Y-%m-%d %H:%M:%S  （每 30 秒刷新，Ctrl-C 退出，不影响后台）'
      echo
      status
      sleep 30
    done
    ;;
  restart) stop; start; resume_unfinished ;;
  resume) resume_unfinished ;;
  stop-dense) stop_dense ;;
  *) echo "用法: $0 [start|stop|status|watch|restart|resume|stop-dense]" >&2; exit 2 ;;
esac
