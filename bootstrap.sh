#!/usr/bin/env bash
# ===========================================================================
# 在全新环境里重建本项目的 Python 运行环境，并校验所有外部依赖。
#
# 为什么需要这个脚本：
#   start.sh 直接调用 ./.venv/bin/python，仓库里**没有任何**创建 venv 或安装依赖的
#   逻辑；requirements.txt 也只有 5 条范围约束，缺 scipy 与 nvidia-cudnn-cu12
#   （这两者都不在任何已声明依赖的传递链上）。缺 cuDNN 时 ALIKED 走 GPU 会直接
#   abort 而不是回退 CPU。所以裸搬到新机器上会失败，且症状隐蔽（setsid nohup 把
#   错误吞进 logs/backend.log，表现为"前端能开、接口全 404"）。
#
# 用法：
#   ./bootstrap.sh                # 建/更新 .venv，然后校验（推荐首次用）
#   ./bootstrap.sh --check        # 只校验现状，不创建、不安装任何东西
#   ./bootstrap.sh --mirror       # 用清华 TUNA 镜像安装（国内网络推荐）
#   PYTHON=python3.12 ./bootstrap.sh
#   MODEL_API_COLMAP_BINARY=/path/to/colmap ./bootstrap.sh
#
# 退出码：0 = 关键项全部通过（可能有警告）；1 = 有关键项失败，无法运行建模主链
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"

VENV=".venv"
LOCK="requirements.lock.txt"
FALLBACK="requirements.txt"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PY="${PYTHON:-python3.12}"

CHECK_ONLY=0
USE_MIRROR=0
for arg in "$@"; do
  case "$arg" in
    --check)  CHECK_ONLY=1 ;;
    --mirror) USE_MIRROR=1 ;;
    -h|--help) sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$arg（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

FAILED=0
WARNED=0
TMP=""    # 惰性创建，供需要落盘的命令使用（见下方 colmap 说明）

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; WARNED=$((WARNED + 1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; FAILED=$((FAILED + 1)); }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

cleanup() {
  if [ -n "$TMP" ] && [ -e "$TMP" ]; then
    rm -rf "$TMP"
  fi
  return 0
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Python 解释器
# ---------------------------------------------------------------------------
head_ "1. Python 解释器"

if ! command -v "$PY" > /dev/null 2>&1; then
  bad "找不到 $PY。请安装 Python 3.12，或用 PYTHON=<路径> 指定"
  echo
  echo "推荐组合是 Ubuntu 24.04 自带的 Python 3.12（本项目实测 3.12.3）。"
  echo "venv 里编译好的 .so（open3d/uvloop/cudnn）与原解释器 ABI 绑定，"
  echo "换 Python 大版本必须重建 venv，不能直接改软链。"
  exit 1
fi

PYVER="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
PYMAJOR="$("$PY" -c 'import sys; print(sys.version_info[0])')"
PYMINOR="$("$PY" -c 'import sys; print(sys.version_info[1])')"

if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 10 ]; }; then
  bad "Python $PYVER 过旧，本项目需要 >= 3.10（实测组合为 3.12.3）"
  exit 1
elif [ "$PYMAJOR" -ne 3 ] || [ "$PYMINOR" -ne 12 ]; then
  warn "Python $PYVER != 3.12；锁定文件是在 3.12.3 上验证的，可能出现 ABI/二进制轮子差异"
else
  ok "Python $PYVER ($PY)"
fi

# ---------------------------------------------------------------------------
# 2. 虚拟环境
# ---------------------------------------------------------------------------
head_ "2. 虚拟环境 $VENV/"

if [ -x "$VENV/bin/python" ]; then
  VENVVER="$("$VENV/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2> /dev/null || echo "?")"
  if [ "$VENVVER" = "$PYVER" ]; then
    ok "已存在，Python $VENVVER，与解释器一致"
  else
    bad "已存在的 $VENV 是 Python $VENVVER，与 $PY 的 $PYVER 不一致"
    echo "     venv 内的 .so 与解释器 ABI 绑定，必须删除重建：rm -rf $VENV"
    exit 1
  fi
elif [ "$CHECK_ONLY" = 1 ]; then
  bad "缺少 $VENV/（--check 模式不会创建它；去掉 --check 即可自动创建）"
else
  echo "  创建中 ..."
  if ! "$PY" -m venv "$VENV"; then
    bad "创建 venv 失败。Ubuntu 需要先装 python3.12-venv：sudo apt install python3.12-venv"
    exit 1
  fi
  ok "已创建"
fi

VPY="$VENV/bin/python"

# ---------------------------------------------------------------------------
# 3. 安装依赖
# ---------------------------------------------------------------------------
head_ "3. Python 依赖"

PIP_ARGS=(--disable-pip-version-check)
if [ "$USE_MIRROR" = 1 ]; then
  PIP_ARGS+=(-i "$MIRROR")
  echo "  使用镜像：$MIRROR"
fi

if [ "$CHECK_ONLY" = 1 ]; then
  echo "  --check 模式，跳过安装"
elif [ -f "$LOCK" ]; then
  echo "  按 $LOCK 安装（全量锁定）..."
  # nvidia-cudnn-cu12 体积较大（数百 MB），是这里最慢的一步
  if "$VPY" -m pip install "${PIP_ARGS[@]}" -r "$LOCK"; then
    ok "依赖安装完成"
  else
    bad "依赖安装失败。若卡在 nvidia-cudnn-cu12，可加 --mirror 重试"
    exit 1
  fi
elif [ -f "$FALLBACK" ]; then
  warn "缺少 $LOCK，退回 $FALLBACK（范围约束，且缺 scipy 与 cudnn，可能不完整）"
  "$VPY" -m pip install "${PIP_ARGS[@]}" -r "$FALLBACK" \
    || warn "$FALLBACK 安装未完全成功，继续尝试补装关键包"
  # requirements.txt 不含 scipy / nvidia-cudnn-cu12，必须补装
  echo "  补装 requirements.txt 缺失的 scipy 与 nvidia-cudnn-cu12 ..."
  "$VPY" -m pip install "${PIP_ARGS[@]}" scipy nvidia-cudnn-cu12 \
    || bad "补装失败，ALIKED 走 GPU 与 tools/ 下的 scipy 调用都会不可用"
else
  bad "既没有 $LOCK 也没有 $FALLBACK"
  exit 1
fi

# ---------------------------------------------------------------------------
# 4. Python 依赖可用性（实际 import，而非只看已安装列表）
# ---------------------------------------------------------------------------
head_ "4. Python 依赖可用性"

"$VPY" - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

# (import 名, 发行包名) —— 发行包名与 import 名不一致的要显式给出
REQUIRED = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("pydantic_settings", "pydantic-settings"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("open3d", "open3d"),
]
# 必须在这一步就能暴露，不能等到运行期：open3d 的 import 会加载其原生扩展
OPTIONAL = [("nvidia.cudnn", "nvidia-cudnn-cu12")]

failed = 0
for module, dist in REQUIRED:
    try:
        __import__(module)
        print(f"  \033[32m✓\033[0m {dist}=={version(dist)}")
    except Exception as exc:                     # noqa: BLE001
        print(f"  \033[31m✗\033[0m {dist} 无法导入：{type(exc).__name__}: {exc}")
        failed += 1

for module, dist in OPTIONAL:
    try:
        __import__(module)
        print(f"  \033[32m✓\033[0m {dist}=={version(dist)}")
    except Exception as exc:                     # noqa: BLE001
        print(f"  \033[31m✗\033[0m {dist} 缺失：{type(exc).__name__} → ALIKED 走 GPU 会 abort")
        failed += 1

sys.exit(1 if failed else 0)
PY
if [ $? -ne 0 ]; then
  FAILED=$((FAILED + 1))
else
  ok "核心包均可导入"
fi

# ---------------------------------------------------------------------------
# 5. cuDNN 动态库（ALIKED 走 GPU 的硬前提）
# ---------------------------------------------------------------------------
head_ "5. cuDNN 动态库"

CUDNN_DIR="$("$VPY" -c '
import sysconfig
from pathlib import Path
p = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn" / "lib"
print(p if p.is_dir() else "")
' 2> /dev/null)"

if [ -n "$CUDNN_DIR" ] && [ -n "$(find "$CUDNN_DIR" -maxdepth 1 -name 'libcudnn.so.*' -print -quit 2> /dev/null)" ]; then
  ok "libcudnn.so.9 位于 ${CUDNN_DIR#$PWD/}"
  echo "     无需设置 MODEL_API_CUDNN_LIB_DIR：app/main.py 的 cudnn_library_dir() 会自动探测"
else
  bad "未找到 libcudnn.so.*（ALIKED 走 GPU 时 colmap 会直接 abort，不是回退 CPU）"
  echo "     安装：$VPY -m pip install nvidia-cudnn-cu12"
  echo "     国内网络可加镜像：-i $MIRROR"
fi

# ---------------------------------------------------------------------------
# 6. COLMAP 二进制 —— 最关键、也最容易配错的一项
# ---------------------------------------------------------------------------
head_ "6. COLMAP 二进制"

COLMAP_BIN="${MODEL_API_COLMAP_BINARY:-}"
if [ -z "$COLMAP_BIN" ]; then
  if [ -x "$HOME/.local/bin/colmap" ]; then
    COLMAP_BIN="$HOME/.local/bin/colmap"
  else
    COLMAP_BIN="$(command -v colmap 2> /dev/null || true)"
  fi
fi

if [ -z "$COLMAP_BIN" ] || ! [ -x "$COLMAP_BIN" ]; then
  bad "找不到 colmap 可执行文件"
  echo "     设 MODEL_API_COLMAP_BINARY=<路径>，或放到 ~/.local/bin/colmap"
else
  # ⚠️ 绝不能用管道接 colmap 的输出：管道这一侧提前关闭会让 colmap 收到 SIGPIPE
  #    被杀掉（本项目曾因此在模型写盘前中断）。统一落到临时文件再解析。
  TMP="$(mktemp -d)"
  CM_VER_FILE="$TMP/colmap-version.txt"
  CM_FE_FILE="$TMP/colmap-feature-extractor.txt"

  "$COLMAP_BIN" -h > "$CM_VER_FILE" 2>&1
  CM_VER_LINE="$(head -n 1 "$CM_VER_FILE")"

  if grep -qi 'with CUDA' "$CM_VER_FILE"; then
    ok "$CM_VER_LINE"
  else
    bad "$CM_VER_LINE  ← 没有 CUDA，稠密重建会退化到 CPU（本项目实测需数天而非数小时）"
  fi

  "$COLMAP_BIN" feature_extractor -h > "$CM_FE_FILE" 2>&1
  if grep -qE 'ALIKED_N16ROT' "$CM_FE_FILE"; then
    ok "支持 ALIKED_N16ROT（本项目默认特征提取器）"
  else
    bad "不支持 ALIKED —— 多半是较旧的 COLMAP 或未编入 ONNX 支持"
    echo "     本项目要求 4.3.0.dev0（commit 8b9936c3，自编译 CUDA 版）。"
    echo "     Ubuntu 自带的 colmap 是 3.9.1 且无 CUDA，既不认 ALIKED，"
    echo "     参数名前缀也是旧的 SiftExtraction.*，用不了。"
  fi

  # 4.3 把 GPU 开关等参数改到了 FeatureExtraction./FeatureMatching. 前缀下
  if grep -qE '\-\-FeatureExtraction\.' "$CM_FE_FILE"; then
    ok "参数前缀为 4.3 风格的 --FeatureExtraction.*"
  else
    warn "未见 --FeatureExtraction.* 前缀；若为 4.3 之前的版本，app/main.py 里的参数名会不被识别"
  fi

  if [ -x /usr/bin/colmap ] && [ "$COLMAP_BIN" != "/usr/bin/colmap" ]; then
    warn "系统另有 /usr/bin/colmap，注意别被 PATH 顺序误导（start.sh 已把 ~/.local/bin 前置）"
  fi
fi

# ---------------------------------------------------------------------------
# 7. ALIKED 的 ONNX 权重缓存
# ---------------------------------------------------------------------------
head_ "7. ALIKED ONNX 权重缓存"

ONNX_COUNT="$(find "$HOME/.cache/colmap" -maxdepth 1 -name '*.onnx' 2> /dev/null | wc -l)"
if [ "$ONNX_COUNT" -ge 2 ]; then
  ok "已缓存 $ONNX_COUNT 个（aliked-n16rot + bruteforce-matcher）"
elif [ "$ONNX_COUNT" -ge 1 ]; then
  warn "只缓存了 $ONNX_COUNT 个，首次用到另一个时会联网下载"
else
  warn "无缓存 —— 首次运行建模时会联网下载；若目标机断网则会自动回退 SIFT，重建结果与预期不同"
  echo "     从旧机器拷贝即可：~/.cache/colmap/*.onnx（约 2.9 MB）"
fi

# ---------------------------------------------------------------------------
# 8. FFmpeg
# ---------------------------------------------------------------------------
head_ "8. FFmpeg / FFprobe"

for tool in ffmpeg ffprobe; do
  if command -v "$tool" > /dev/null 2>&1; then
    ok "$("$tool" -version 2> /dev/null | head -n 1 | cut -c1-60)"
  else
    bad "找不到 $tool（抽帧与视频规格探测依赖它）"
  fi
done

# ---------------------------------------------------------------------------
# 9. 项目自检
# ---------------------------------------------------------------------------
head_ "9. 项目自检"

if compgen -G "app/*.py" > /dev/null 2>&1; then
  if "$VPY" -m py_compile app/*.py $(find tools -name '*.py' 2> /dev/null); then
    ok "app/*.py 与 tools/**/*.py 语法检查通过"
  else
    bad "py_compile 失败"
  fi
else
  warn "app/*.py 不存在，跳过（是否只打包了部分内容？）"
fi

if [ -f tools/diagnostics/project_check.py ]; then
  if "$VPY" tools/diagnostics/project_check.py --help > /dev/null 2>&1; then
    ok "project_check.py 可运行"
  else
    bad "project_check.py 无法运行（注意它会 import scipy）"
  fi
else
  warn "缺少 tools/diagnostics/project_check.py"
fi

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
head_ "汇总"
if [ "$FAILED" -eq 0 ]; then
  printf '  \033[32m关键项全部通过\033[0m'
  [ "$WARNED" -gt 0 ] && printf '（%d 条警告）' "$WARNED"
  echo
  echo
  echo "下一步：./start.sh start"
  echo "  后端 http://127.0.0.1:8000/health"
  echo "  前端 http://localhost:5500/video_upload.html"
  exit 0
else
  printf '  \033[31m%d 个关键项失败\033[0m' "$FAILED"
  [ "$WARNED" -gt 0 ] && printf '，另有 %d 条警告' "$WARNED"
  echo
  echo
  echo "请先解决上面标 ✗ 的项，否则 ./start.sh start 会导致："
  echo "  前端能打开但模型列表为空、所有接口 404（后端其实没起来，"
  echo "  错误被 setsid nohup 吞进了 logs/backend.log）"
  exit 1
fi
