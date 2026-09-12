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
#   ./start.sh status     # 查看前后台是否仍在运行 + 任务进度
#   ./start.sh restart    # 重启并自动续跑未完成的稠密阶段
#   ./start.sh stop       # 停止
set -uo pipefail

cd "$(dirname "$0")"
mkdir -p logs
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"
BACKEND_PID="logs/backend.pid"
FRONTEND_PID="logs/frontend.pid"

export PATH="$HOME/.local/bin:$PATH"
export MODEL_API_COLMAP_BINARY="$HOME/.local/bin/colmap"
export MODEL_API_USE_GPU=true
export MODEL_API_PREFER_CUDA=true
export MODEL_API_ALLOW_CPU_FALLBACK=false
export MODEL_API_FEATURE_EXTRACTOR=ALIKED_N16ROT

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

start() {
  if alive "$BACKEND_PID"; then
    echo "后端已在运行 (pid $(cat "$BACKEND_PID"))"
  else
    detach "$BACKEND_LOG" ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
      > "$BACKEND_PID"
    echo "后端已启动 (pid $(cat "$BACKEND_PID"))，日志 $BACKEND_LOG"
  fi
  if alive "$FRONTEND_PID"; then
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
  if pgrep -f 'colmap' > /dev/null; then
    pkill -f 'colmap' 2>/dev/null || true
    for _ in $(seq 1 40); do
      pgrep -f 'colmap patch_match_stereo' > /dev/null || break
      sleep 0.5
    done
    pgrep -f 'colmap patch_match_stereo' > /dev/null && pkill -9 -f 'colmap patch_match_stereo' 2>/dev/null || true
    echo "已清理遗留的 colmap 进程（当前视角会重跑，已完成的视角不受影响）"
  fi
  rm -f "$BACKEND_PID" "$FRONTEND_PID"
  echo "已停止。产物保留在 storage/<job_id>/，重启后用 ./start.sh restart 自动续跑。"
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
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) stop; start; resume_unfinished ;;
  resume) resume_unfinished ;;
  *) echo "用法: $0 [start|stop|status|restart|resume]" >&2; exit 2 ;;
esac

