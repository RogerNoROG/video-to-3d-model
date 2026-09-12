from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sysconfig
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Iterator

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.convert import point_cloud_to_glb


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_API_")

    storage_dir: Path = Path("./storage")
    max_upload_size_mb: int = Field(default=8192, gt=0)
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    colmap_binary: str = "colmap"
    pipeline: str = "colmap"
    frame_fps: int = Field(default=3, gt=0, le=30)
    normalize_width: int = Field(default=1920, gt=0)
    normalize_height: int = Field(default=1080, gt=0)
    normalize_fps: int = Field(default=30, gt=0, le=60)
    command_timeout_seconds: int = Field(default=86400, gt=0)
    # Poisson 八叉树深度。0 = 按点云采样密度自动推导（推荐）。
    # 不要盲目调高：叶节点尺寸 = 包围盒最长边 / 2^depth，若它远小于平均点间距，
    # 就是在重建数据里不存在的细节，而内存会随 depth 每 +1 约 ×8 —— 本项目曾因
    # 硬编码 9 在 1854 万点上把 24GB 内存耗尽、swap 打满而卡死。
    poisson_depth: int = Field(default=0, ge=0, le=12)
    # 网格重建前的体素降采样尺寸，0 = 不降采样。点云过密时设成平均点间距的 1~2 倍
    # 可大幅降内存且几乎不损细节
    poisson_voxel_size: float = Field(default=0.0, ge=0.0)
    # 建网格前按各轴分位数裁掉离群点（百分比）。COLMAP 的点云总有少量飞点
    # 把包围盒撑大十几倍，而 Poisson 分辨率 = 包围盒最长边 / 2^depth，不裁就白降分辨率
    poisson_outlier_percentile: float = Field(default=1.0, ge=0.0, lt=50.0)
    # 再用 SOR（统计离群点移除）清掉盒内悬浮的稀疏点：分位数裁剪只能削掉盒子外面的点，
    # 盒内稀疏点是 Poisson 长出小碎块的原料。sor_std_ratio 设 0 关闭
    poisson_sor_neighbors: int = Field(default=20, gt=0)
    poisson_sor_std_ratio: float = Field(default=2.0, ge=0.0)
    # 最后按连通分量去掉碎块，只保留不小于最大分量该比例的部分。0 = 不过滤。
    # 主体周围的散点就是这些互不连通的小块
    mesh_min_component_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    # 剔除 Poisson 结果里密度最低的部分（分位数）。桌面等薄片边缘的撕裂与尖刺来自
    # 低密度区域，调高可修边（代价是极少量细节被削掉）
    mesh_density_quantile: float = Field(default=0.02, ge=0.0, lt=0.5)
    # RANSAC 移除几个最大的支撑平面（桌面/地面），只留被摄主体。0 = 不移除。
    # 桌面常被重建成边缘撕裂的薄片，且与主体连通（连通分量过滤不掉），只能在这里去。
    # 注意主体自身也有大平面，所以默认只去 1 个，别盲目调高
    poisson_remove_planes: int = Field(default=0, ge=0, le=4)
    # 平面判定的距离阈值 = 点间距 × 该系数
    poisson_plane_distance_scale: float = Field(default=4.0, gt=0.0)
    # Poisson 深度的上限（内存约束）。八叉树叶节点数约为 8^depth，每 +1 内存约 ×8，
    # 24GB 内存下 9 已是上限（实测 9 在 1854 万点上会耗尽内存）
    poisson_max_depth: int = Field(default=9, ge=5, le=12)
    # 超过这个点数就跳过全局法线定向（该步骤要为每个点建 30 邻域图，是主要内存开销）
    poisson_orient_max_points: int = Field(default=5_000_000, gt=0)
    use_gpu: bool = True
    gpu_index: int = Field(default=0, ge=0)
    prefer_cuda: bool = True
    allow_cpu_fallback: bool = True
    stale_job_timeout_seconds: int = Field(default=3600, gt=0)
    min_registered_images: int = Field(default=20, ge=1)
    # 视频抽帧来自同一台相机，必须共用一套内参，否则标定会被过度参数化
    single_camera: bool = True
    # 特征提取器：ALIKED_N16ROT（默认，学习型，GPU 约 65ms/图，低纹理更稳健）
    # 也可换成 SIFT（无 ONNX 依赖）、ALIKED_N32、LOMA
    feature_extractor: str = "ALIKED_N16ROT"
    # 匹配器，留空则按特征提取器自动选择
    matcher_type: str = ""
    # 学习型特征由 ONNX Runtime 驱动，走 GPU 需要 cuDNN；缺失时会自动降级为 CPU
    learned_extractor_gpu: bool = True
    # cuDNN 动态库目录，留空则自动探测虚拟环境里 pip 安装的 cuDNN
    cudnn_lib_dir: str = ""
    # SIFT 调参（实测：过激的参数在低纹理物体上收益有限，默认保持保守稳妥）
    max_num_features: int = Field(default=16384, gt=0)
    sift_peak_threshold: float = Field(default=0.00667, gt=0)
    sift_estimate_affine_shape: bool = False
    sift_domain_size_pooling: bool = False
    # 匹配调参：视频抽帧相邻帧位移较大，需要更大的重叠窗口
    sequential_overlap: int = Field(default=15, gt=0)
    guided_matching: bool = True
    # 融合调参
    stereo_fusion_min_num_pixels: int = Field(default=3, gt=0)
    # 稠密重建的分辨率上限。稠密耗时与像素数近似成正比，调低可大幅提速（代价是细节减少）
    # -1 表示使用原始分辨率（1920x1080）；常用值 1600 / 1280 / 960
    patch_match_max_image_size: int = Field(default=-1, ge=-1)
    stereo_fusion_max_image_size: int = Field(default=-1, ge=-1)
    # 几何一致性过滤。开启后 patch_match_stereo 会把全部视角跑两遍
    # （第一遍写 .photometric.bin，第二遍写 .geometric.bin），稠密阶段耗时翻倍，
    # 但会剔除跨视角不一致的深度估计，低纹理表面上通常能明显减少漂浮噪点。
    patch_match_geom_consistency: bool = True
    # 让光束法平差走 GPU（mapper 阶段加速，部分 COLMAP 构建不支持）
    mapper_ba_use_gpu: bool = False


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result_url: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        now = datetime.now(timezone.utc)
        job = Job(
            id=uuid.uuid4().hex,
            filename=filename,
            status=JobStatus.queued,
            progress=0,
            stage="等待处理",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def register(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            job_file = settings.storage_dir / job_id / "job.json"
            try:
                job = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                job = None
            if job is not None:
                with self._lock:
                    self._jobs[job.id] = job
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        job = self.recover_stale(job)
        return job

    def recover_stale(self, job: Job) -> Job:
        if job.status not in (JobStatus.queued, JobStatus.processing):
            return job
        # 取日志 mtime 与任务 updated_at 的较新者：不同阶段“有活性”的迹象不同——
        # 外部命令阶段写 pipeline.log，而 Open3D 网格这类无日志的步骤靠 update_stage
        # 或 log_heartbeat 刷新。只看其中一个会在长步骤上误判。
        candidates = [job.updated_at.timestamp()]
        log_file = settings.storage_dir / job.id / "pipeline.log"
        try:
            candidates.append(log_file.stat().st_mtime)
        except OSError:
            pass
        age = datetime.now(timezone.utc).timestamp() - max(candidates)
        if age <= settings.stale_job_timeout_seconds:
            return job
        resumable = (settings.storage_dir / job.id / "dense" / "stereo").exists()
        hint = (
            "。中间产物仍在，用 POST /api/v1/jobs/{}/resume 可以续跑，不必重做整个任务".format(job.id)
            if resumable
            else "，请重试任务"
        )
        failed = job.model_copy(
            update={
                "status": JobStatus.failed,
                "stage": "处理失败",
                "error": f"建模进程长时间没有活动，可能已异常退出{hint}",
                "updated_at": datetime.now(timezone.utc),
            }
        )
        with self._lock:
            self._jobs[job.id] = failed
        persist_job(failed)
        return failed

    def update(self, job_id: str, **changes: object) -> Job:
        with self._lock:
            current = self._jobs.get(job_id)
        if current is None:
            job_file = settings.storage_dir / job_id / "job.json"
            try:
                current = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise KeyError(f"任务不存在或已被删除: {job_id}") from None
        with self._lock:
            updated = current.model_copy(
                update={"updated_at": datetime.now(timezone.utc), **changes}
            )
            self._jobs[job_id] = updated
            return updated

    def list(self) -> list[Job]:
        jobs: dict[str, Job] = {}
        with self._lock:
            jobs.update({
                job.id: job
                for job in self._jobs.values()
                if (settings.storage_dir / job.id / "job.json").exists()
            })
        for job_file in settings.storage_dir.glob("*/job.json"):
            try:
                job = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            jobs.setdefault(job.id, job)
        normalized: list[Job] = []
        for job in jobs.values():
            job = self.recover_stale(job)
            if job.status == JobStatus.completed and not result_file(job.id).exists():
                job = job.model_copy(update={"result_url": None, "error": "结果文件不存在，请重试任务"})
            elif job.status == JobStatus.completed and job.error:
                job = job.model_copy(update={"error": None})
            normalized.append(job)
        return sorted(normalized, key=lambda job: job.updated_at, reverse=True)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


store = JobStore()
pipeline_lock = threading.Lock()
app = FastAPI(title="Video to 3D Model API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def job_dir(job_id: str) -> Path:
    return settings.storage_dir / job_id


def result_file(job_id: str) -> Path:
    return job_dir(job_id).resolve() / "model.glb"


def point_cloud_file(job_id: str) -> Path:
    return job_dir(job_id).resolve() / "fused.ply"


def persist_job(job: Job) -> None:
    job_dir(job.id).mkdir(parents=True, exist_ok=True)
    (job_dir(job.id) / "job.json").write_text(
        job.model_dump_json(indent=2), encoding="utf-8"
    )


def cudnn_library_dir() -> Path | None:
    """定位 pip 安装的 cuDNN，ONNX Runtime 的 CUDA provider 依赖它。"""
    candidates: list[Path] = []
    if settings.cudnn_lib_dir:
        candidates.append(Path(settings.cudnn_lib_dir))
    candidates.append(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn" / "lib")
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("libcudnn.so.*"), None) is not None:
            return candidate
    return None


def subprocess_env() -> dict[str, str]:
    """把 cuDNN 目录加入动态库搜索路径，否则 ONNX 的 CUDA provider 加载失败。"""
    env = os.environ.copy()
    cudnn_dir = cudnn_library_dir()
    if cudnn_dir is not None:
        parts = [str(cudnn_dir), *[p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def learned_gpu_enabled() -> bool:
    """学习型特征走 GPU 需要 cuDNN；缺失 cuDNN 时 CUDA provider 会让进程 abort，必须降级。"""
    return settings.learned_extractor_gpu and cudnn_library_dir() is not None


def run_command(command: list[str], cwd: Path) -> None:
    log_path = cwd / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
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
        raise RuntimeError(f"命令执行失败，详见 {log_path}: {' '.join(command)}")


PROGRESS_POLL_SECONDS = 2.0
PROGRESS_TAIL_BYTES = 262144


@contextlib.contextmanager
def log_heartbeat(cwd: Path, interval: float = 30.0) -> Iterator[None]:
    """定期 touch 一下 ``pipeline.log``。

    陈旧任务判定看的是 ``pipeline.log`` 的 mtime。有些步骤（Open3D 的 Poisson 网格、
    长时间无输出的外部命令）会连续几十分钟不写日志，看着就像挂了，从而被误判为失败。
    """
    stop = threading.Event()
    log_path = cwd / "pipeline.log"

    def beat() -> None:
        while not stop.wait(interval):
            try:
                os.utime(log_path, None)
            except OSError:
                pass

    watcher = threading.Thread(target=beat, name="log-heartbeat", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=interval)


def parse_log_tail_progress(
    log_path: Path, pattern: re.Pattern[str]
) -> tuple[int, int | None] | None:
    """从日志尾部解析最近一次进度，返回 ``(当前, 总数)``；总数未知时为 ``None``。

    只读尾部固定长度，避免每个轮询周期都扫整个日志文件。

    ``re.findall`` 在正则只有 1 个捕获组时返回字符串列表而不是元组，
    必须先归一化，否则 ``len("480") == 3`` 会被误判成「有两个捕获组」。
    """
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - PROGRESS_TAIL_BYTES))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    found = pattern.findall(tail)
    if not found:
        return None
    groups = found[-1]
    if isinstance(groups, str):
        groups = (groups,)
    if len(groups) >= 2:
        total = int(groups[1])
        return (int(groups[0]), total) if total > 0 else None
    return int(groups[0]), None


def count_dense_depth_maps(dense_dir: Path) -> int:
    """统计已生成的稠密深度图数量，作为 ``patch_match_stereo`` 的进度来源。

    不解析日志而数文件，是因为开启 ``geom_consistency`` 后 COLMAP 会把
    ``Processing view 1..N`` 跑两遍（先写 ``.photometric.bin``，再写 ``.geometric.bin``），
    日志里的视角编号会从 N 跳回 1，用它汇报进度会出现进度条倒退。
    按文件计数天然单调递增，且能同时覆盖两遍。
    """
    depth_maps = dense_dir / "stereo" / "depth_maps"
    try:
        return sum(1 for name in os.listdir(depth_maps) if name.endswith(".bin"))
    except OSError:
        return 0


def run_command_with_progress(
    command: list[str],
    cwd: Path,
    job_id: str,
    stage_label: str,
    progress_from: int,
    progress_to: int,
    pattern: re.Pattern[str] | None = None,
    counter: Callable[[], tuple[int, int]] | None = None,
) -> None:
    """运行命令，并周期性把进度写回任务阶段，让长耗时阶段能看到具体走到第几步。

    进度来源二选一：
    - ``counter``：返回 ``(当前, 总数)`` 的回调，适用于日志进度会倒退的场景（见
      :func:`count_dense_depth_maps`）；
    - ``pattern``：从 ``pipeline.log`` 尾部解析。带 2 个捕获组视为 ``当前/总数``；
      带 1 个捕获组只更新计数文案，进度值停在 ``progress_from``（总数未知时不做虚假推进）。
    """
    stop = threading.Event()
    log_path = cwd / "pipeline.log"

    def monitor() -> None:
        last_text = ""
        while not stop.wait(PROGRESS_POLL_SECONDS):
            if counter is not None:
                try:
                    current, total = counter()
                except Exception:
                    continue
            else:
                assert pattern is not None
                parsed = parse_log_tail_progress(log_path, pattern)
                if parsed is None:
                    continue
                current, total = parsed

            if total is not None and total > 0:
                current = min(current, total)
                percent = progress_from + round((progress_to - progress_from) * current / total)
                text = f"{stage_label}（{current}/{total}）"
            else:
                percent = progress_from
                text = f"{stage_label}（已完成 {current}）"
            if text == last_text:
                continue
            last_text = text
            try:
                update_stage(job_id, percent, text)
            except Exception:
                continue

    watcher = threading.Thread(target=monitor, name=f"progress-{job_id[:8]}", daemon=True)
    watcher.start()
    try:
        run_command(command, cwd)
    finally:
        stop.set()
        watcher.join(timeout=PROGRESS_POLL_SECONDS * 2)


def colmap_supports_cuda() -> bool:
    result = subprocess.run(
        [settings.colmap_binary, "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "without CUDA" not in f"{result.stdout}\n{result.stderr}"


def validate_video(video: Path) -> None:
    if video.stat().st_size == 0:
        raise ValueError("视频文件为空")
    if shutil.which(settings.ffprobe_binary) is None:
        return
    result = subprocess.run(
        [settings.ffprobe_binary, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无法读取视频流"
        raise ValueError(f"视频文件无效：{detail}")


def colmap_gpu_flags(feature: bool) -> list[str]:
    option = "FeatureExtraction" if feature else "FeatureMatching"
    return [f"--{option}.use_gpu", "1", f"--{option}.gpu_index", str(settings.gpu_index)]


def is_learned_extractor(feature_type: str) -> bool:
    """学习型特征（ALIKED/LOMA）由 ONNX Runtime 驱动，不是 COLMAP 自带的 CUDA 实现。"""
    return feature_type.upper().startswith(("ALIKED", "LOMA"))


def resolve_matcher_type(feature_type: str) -> str:
    if settings.matcher_type:
        return settings.matcher_type
    if feature_type.upper().startswith("ALIKED"):
        return "ALIKED_BRUTEFORCE"
    if feature_type.upper().startswith("LOMA"):
        return "LOMA_BRUTEFORCE"
    return "SIFT_BRUTEFORCE"


def build_feature_command(
    database: Path, frames: Path, feature_type: str, gpu_enabled: bool
) -> list[str]:
    command = [
        settings.colmap_binary,
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(frames),
        "--FeatureExtraction.type", feature_type,
    ]
    if settings.single_camera:
        command.extend(["--ImageReader.single_camera", "1"])
    if is_learned_extractor(feature_type):
        command.extend(["--FeatureExtraction.use_gpu", "1" if learned_gpu_enabled() else "0"])
        command.extend(["--AlikedExtraction.max_num_features", str(settings.max_num_features)])
    else:
        command.extend(
            [
                "--SiftExtraction.max_num_features", str(settings.max_num_features),
                "--SiftExtraction.peak_threshold", f"{settings.sift_peak_threshold:g}",
            ]
        )
        if settings.sift_estimate_affine_shape:
            command.extend(["--SiftExtraction.estimate_affine_shape", "1"])
        if settings.sift_domain_size_pooling:
            command.extend(["--SiftExtraction.domain_size_pooling", "1"])
        if gpu_enabled:
            command.extend(colmap_gpu_flags(feature=True))
    return command


def build_matcher_command(database: Path, gpu_enabled: bool, feature_type: str) -> list[str]:
    command = [
        settings.colmap_binary,
        "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", str(settings.sequential_overlap),
        "--FeatureMatching.type", resolve_matcher_type(feature_type),
    ]
    if is_learned_extractor(feature_type):
        command.extend(["--FeatureMatching.use_gpu", "1" if learned_gpu_enabled() else "0"])
    else:
        if gpu_enabled:
            command.extend(colmap_gpu_flags(feature=False))
        if settings.guided_matching:
            command.extend(["--FeatureMatching.guided_matching", "1"])
    return command


def model_stats(model_dir: Path) -> tuple[int, int]:
    """Return (registered images, sparse points) of a COLMAP sparse model."""
    result = subprocess.run(
        [settings.colmap_binary, "model_analyzer", "--path", str(model_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    registered = 0
    points = 0
    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        match = re.search(r"Registered images:\s*(\d+)", line)
        if match:
            registered = int(match.group(1))
            continue
        if registered == 0:
            match = re.search(r"\bImages:\s*(\d+)", line)
            if match:
                registered = int(match.group(1))
                continue
        match = re.search(r"\bPoints:\s*(\d+)", line)
        if match:
            points = int(match.group(1))
    return registered, points


def select_best_model(sparse_root: Path) -> tuple[Path, int, int, int]:
    """COLMAP 可能因视角断裂生成多个互不连通的模型，取注册图像最多的那个。

    返回 (模型目录, 注册图像数, 稀疏点数, 模型总数)。
    """
    models = sorted(p for p in sparse_root.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError("COLMAP 未生成稀疏模型，请检查视频是否包含足够视角重叠")
    best: tuple[Path, int, int] | None = None
    for model in models:
        registered, points = model_stats(model)
        if best is None or registered > best[1]:
            best = (model, registered, points)
    assert best is not None
    return best[0], best[1], best[2], len(models)


def normalize_video(video: Path, normalized: Path, root: Path) -> None:
    video_filter = (
        "hwdownload,format=nv12,"
        f"scale={settings.normalize_width}:{settings.normalize_height},"
        f"fps={settings.normalize_fps},format=yuv420p"
    )
    cpu_filter = f"scale={settings.normalize_width}:{settings.normalize_height},fps={settings.normalize_fps}"
    common = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(video),
        "-vf",
        cpu_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(normalized),
    ]
    if not settings.prefer_cuda:
        run_command(common, root)
        return

    cuda_command = common.copy()
    cuda_command[2:2] = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    filter_index = cuda_command.index("-vf") + 1
    cuda_command[filter_index] = video_filter
    try:
        run_command(cuda_command, root)
    except Exception:
        normalized.unlink(missing_ok=True)
        if not settings.allow_cpu_fallback:
            raise
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write("CUDA 预处理失败，回退到 CPU 1080p30 预处理\n")
        run_command(common, root)


def extract_frames(video: Path, frames: Path, root: Path) -> None:
    output = str(frames / "frame_%06d.jpg")
    run_command([settings.ffmpeg_binary, "-y", "-i", str(video), "-vf", f"fps={settings.frame_fps}", output], root)


def update_stage(job_id: str, progress: int, stage: str) -> None:
    updated = store.update(job_id, status=JobStatus.processing, progress=progress, stage=stage)
    persist_job(updated)


def run_colmap_pipeline(job_id: str, gpu_enabled: bool) -> tuple[int, int]:
    root = job_dir(job_id).resolve()
    input_files = list(root.glob("input.*"))
    if not input_files:
        raise RuntimeError("找不到上传的视频文件")
    frames = root / "frames"
    database = root / "database.db"
    sparse = root / "sparse"
    dense = root / "dense"
    normalized = root / "normalized_1080p30.mp4"
    frames.mkdir(exist_ok=True)

    update_stage(job_id, 10, "FFmpeg 统一转码为 1080p30")
    normalize_video(input_files[0], normalized, root)
    update_stage(job_id, 20, f"FFmpeg 按 {settings.frame_fps} FPS 抽取关键帧")
    extract_frames(normalized, frames, root)
    total_frames = len(list(frames.glob("*.jpg")))
    if total_frames == 0:
        raise RuntimeError("未能从视频中抽取到任何帧，请检查视频内容是否有效")
    update_stage(
        job_id, 30, f"COLMAP 提取图像特征（{settings.feature_extractor}，共 {total_frames} 帧）"
    )
    feature_type = settings.feature_extractor
    try:
        run_command_with_progress(
            build_feature_command(database, frames, feature_type, gpu_enabled),
            root,
            job_id,
            f"COLMAP 提取图像特征（{feature_type}）",
            30,
            45,
            re.compile(r"Processed file \[(\d+)/(\d+)\]"),
        )
    except RuntimeError:
        if not is_learned_extractor(feature_type):
            raise
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write("学习型特征提取失败（通常是 ONNX 模型不可用），回退到 SIFT\n")
        feature_type = "SIFT"
        run_command_with_progress(
            build_feature_command(database, frames, feature_type, gpu_enabled),
            root,
            job_id,
            "COLMAP 提取图像特征（SIFT 回退）",
            30,
            45,
            re.compile(r"Processed file \[(\d+)/(\d+)\]"),
        )
    update_stage(job_id, 45, "COLMAP 匹配相邻帧")
    run_command(build_matcher_command(database, gpu_enabled, feature_type), root)
    update_stage(job_id, 60, "COLMAP 估计相机位姿")
    sparse.mkdir(exist_ok=True)
    mapper_command = [
        settings.colmap_binary,
        "mapper",
        "--database_path", str(database),
        "--image_path", str(frames),
        "--output_path", str(sparse),
    ]
    if settings.mapper_ba_use_gpu and gpu_enabled:
        mapper_command.extend(
            ["--Mapper.ba_use_gpu", "1", "--Mapper.ba_gpu_index", str(settings.gpu_index)]
        )
    run_command_with_progress(
        mapper_command,
        root,
        job_id,
        "COLMAP 估计相机位姿（已注册图像）",
        60,
        60,
        re.compile(r"num_reg_frames=(\d+)"),
    )
    model, registered, sparse_points, model_count = select_best_model(sparse)
    with (root / "pipeline.log").open("a", encoding="utf-8") as log:
        if model_count > 1:
            log.write(
                f"COLMAP 生成了 {model_count} 个互不连通的稀疏模型（相机运动过快或视角断裂），"
                f"已选用注册图像最多的 {model.name}\n"
            )
        log.write(f"重建统计: 注册 {registered}/{total_frames} 张图像, {sparse_points} 个稀疏点\n")
    if registered < settings.min_registered_images:
        detail = f"{model_count} 个互不连通的模型，最大者仅注册 {registered}" if model_count > 1 else f"仅注册 {registered}"
        raise RuntimeError(
            f"COLMAP 重建质量不合格：{detail}/{total_frames} 张图像（稀疏点 {sparse_points}）。"
            "常见原因是相机移动过快导致相邻帧重叠不足、被摄物在移动，或场景纹理过少。"
            "请保持被摄物静止、相机缓慢平稳环绕（相邻帧重叠 70% 以上），并让物体始终占据画面主要区域后重试"
        )
    update_stage(job_id, 75, f"COLMAP 生成稠密点云（注册 {registered}/{total_frames} 张）")
    run_command([settings.colmap_binary, "image_undistorter", "--image_path", str(frames), "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP"], root)
    run_dense_stage(job_id, root, dense, registered)
    return registered, total_frames


def count_dense_views(dense: Path) -> int:
    """从 ``patch-match.cfg`` 读稠密工作区的视角数（每个视角占 2 行：图像名 + 匹配配置）。"""
    config = dense / "stereo" / "patch-match.cfg"
    try:
        lines = [line for line in config.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return 0
    return len(lines) // 2


def mesh_and_export(job_id: str, root: Path, point_cloud_path: Path) -> None:
    """点云 → Poisson 网格 → GLB。正式流程与续跑共用。"""
    update_stage(job_id, 95, "Open3D 点云重建网格并导出 GLB")
    with log_heartbeat(root):
        point_cloud_to_glb(
            point_cloud_path,
            result_file(job_id),
            voxel_size=settings.poisson_voxel_size,
            poisson_depth=settings.poisson_depth,
            orient_normals_max_points=settings.poisson_orient_max_points,
            outlier_percentile=settings.poisson_outlier_percentile,
            poisson_max_depth=settings.poisson_max_depth,
            sor_neighbors=settings.poisson_sor_neighbors,
            sor_std_ratio=settings.poisson_sor_std_ratio,
            min_component_ratio=settings.mesh_min_component_ratio,
            density_quantile=settings.mesh_density_quantile,
            plane_max=settings.poisson_remove_planes,
            plane_distance_scale=settings.poisson_plane_distance_scale,
        )


def run_dense_stage(job_id: str, root: Path, dense: Path, registered: int) -> None:
    """稠密重建：深度图 → 融合 → 网格 → GLB。假定 ``dense/`` 已由 image_undistorter 准备好。

    **可重复调用**：``patch_match_stereo`` 会跳过 ``depth_maps`` 里已存在的视角。
    后端进程中途死亡后，可以靠这个特性从断点续跑，而不必重做整个稀疏重建。
    """
    geom_consistency = settings.patch_match_geom_consistency
    patch_match_command = [
        settings.colmap_binary,
        "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true" if geom_consistency else "false",
    ]
    if settings.patch_match_max_image_size > 0:
        patch_match_command.extend(
            ["--PatchMatchStereo.max_image_size", str(settings.patch_match_max_image_size)]
        )
    # 开启 geom_consistency 时 COLMAP 会跑两遍全部视角，总数要按两遍算
    views = count_dense_views(dense) or registered
    dense_total_views = views * (2 if geom_consistency else 1)
    dense_pass_note = "光度+几何一致性两遍" if geom_consistency else "单遍"
    run_command_with_progress(
        patch_match_command,
        root,
        job_id,
        f"COLMAP 生成稠密点云（{dense_pass_note}）",
        75,
        90,
        counter=lambda: (count_dense_depth_maps(dense), dense_total_views),
    )
    update_stage(job_id, 90, "COLMAP 融合点云")
    run_command_with_progress(
        [
            settings.colmap_binary,
            "stereo_fusion",
            "--workspace_path", str(dense),
            "--workspace_format", "COLMAP",
            "--output_path", str(point_cloud_file(job_id)),
            "--StereoFusion.min_num_pixels", str(settings.stereo_fusion_min_num_pixels),
        ]
        + (
            ["--StereoFusion.max_image_size", str(settings.stereo_fusion_max_image_size)]
            if settings.stereo_fusion_max_image_size > 0
            else []
        ),
        root,
        job_id,
        "COLMAP 融合点云（已融合视角）",
        90,
        95,
        re.compile(r"Fusing image \[(\d+)/(\d+)\]"),
    )
    update_stage(job_id, 95, "Open3D 点云重建网格并导出 GLB")
    mesh_and_export(job_id, root, point_cloud_file(job_id))


def finish_job(job_id: str, stage: str) -> None:
    completed = store.update(
        job_id,
        status=JobStatus.completed,
        progress=100,
        stage=stage,
        result_url=f"/api/v1/jobs/{job_id}/result",
        error=None,
    )
    persist_job(completed)


def fail_job(job_id: str, error: str) -> None:
    try:
        failed = store.update(job_id, status=JobStatus.failed, stage="处理失败", error=error)
        persist_job(failed)
    except KeyError:
        pass


def resume_dense_reconstruction(job_id: str, resume_from: int = 0) -> None:
    """续跑未完成的稠密阶段与出模型步骤（后端中途死亡时用）。

    ``resume_from`` 是调用时任务的进度，必须由调用方在**改写进度之前**传入：
    融合阶段是 ``update_stage(90)`` → 跑 stereo_fusion → ``update_stage(95)``，
    所以进度 ≥ 95 就证明融合曾经成功返回过，``fused.ply`` 是完整的，不必重跑。

    自动判断从哪一步接：
    - 已有 ``fused.ply`` 且进度到过 95 → 只补网格和 GLB；
    - 否则从 ``patch_match_stereo`` 跑起（它自己会跳过已存在的深度图）。
    """
    try:
        root = job_dir(job_id).resolve()
        dense = root / "dense"
        if not (dense / "sparse").exists() or not (dense / "stereo").exists():
            raise RuntimeError(
                "dense/ 目录不完整（缺少 image_undistorter 的输出），无法续跑，请重试整个任务"
            )
        _, registered, _, _ = select_best_model(root / "sparse")
        frames = root / "frames"
        total_frames = len(list(frames.glob("*.jpg"))) or registered
        fused = point_cloud_file(job_id)
        fusion_done = resume_from >= 95 and fused.exists() and fused.stat().st_size > 1_000_000
        if fusion_done:
            update_stage(
                job_id,
                95,
                f"复用已有的 {fused.name}（{fused.stat().st_size / 1024 / 1024:.0f} MB），直接重建网格并导出 GLB",
            )
            mesh_and_export(job_id, root, fused)
        else:
            done = count_dense_depth_maps(dense)
            update_stage(job_id, 75, f"续跑稠密重建（已有 {done} 个视角文件，已完成的视角会自动跳过）")
            with pipeline_lock:
                run_dense_stage(job_id, root, dense, registered)
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）")
    except Exception as exc:
        fail_job(job_id, str(exc))


def remesh_from_point_cloud(job_id: str) -> None:
    """用已有的 fused.ply 重新生成网格与 GLB。"""
    try:
        root = job_dir(job_id).resolve()
        _, registered, _, _ = select_best_model(root / "sparse")
        frames = root / "frames"
        total_frames = len(list(frames.glob("*.jpg"))) or registered
        mesh_and_export(job_id, root, point_cloud_file(job_id))
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）")
    except Exception as exc:
        fail_job(job_id, str(exc))


def run_reconstruction(job_id: str) -> None:
    try:
        if settings.pipeline != "colmap":
            raise RuntimeError(f"不支持的 pipeline: {settings.pipeline}，当前仅支持 colmap")
        if shutil.which(settings.ffmpeg_binary) is None:
            raise RuntimeError(f"找不到 FFmpeg: {settings.ffmpeg_binary}")
        if shutil.which(settings.colmap_binary) is None:
            raise RuntimeError(f"找不到 COLMAP: {settings.colmap_binary}")
        gpu_enabled = settings.use_gpu and colmap_supports_cuda()
        if settings.use_gpu and not gpu_enabled:
            if not settings.allow_cpu_fallback:
                raise RuntimeError("已开启 GPU，但当前 COLMAP 是 CPU 版本（without CUDA），请安装 CUDA 版 COLMAP")
            update_stage(job_id, 5, "COLMAP 不支持 CUDA，回退到 CPU")
        if pipeline_lock.locked():
            update_stage(job_id, 1, "排队等待 GPU 资源")
        with pipeline_lock:
            registered, total_frames = run_colmap_pipeline(job_id, gpu_enabled=gpu_enabled)
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）")
    except Exception as exc:
        fail_job(job_id, str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="待重建的视频文件")],
) -> Job:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in settings.allowed_video_extensions:
        raise HTTPException(status_code=415, detail="不支持的视频格式")

    job = store.create(video.filename or f"input{suffix}")
    destination = job_dir(job.id) / f"input{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size_mb * 1024 * 1024:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"视频超过大小限制：上限 {settings.max_upload_size_mb / 1024:g} GB，"
                            f"当前已接收 {size / 1024 / 1024 / 1024:.2f} GB。"
                            "可通过 MODEL_API_MAX_UPLOAD_SIZE_MB 调整上限，或先用 ffmpeg 裁剪/压缩视频"
                        ),
                    )
                output.write(chunk)
    except Exception:
        shutil.rmtree(job_dir(job.id), ignore_errors=True)
        store.remove(job.id)
        raise
    finally:
        await video.close()

    try:
        validate_video(destination)
    except ValueError as exc:
        shutil.rmtree(job_dir(job.id), ignore_errors=True)
        store.remove(job.id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    persist_job(job)
    background_tasks.add_task(run_reconstruction, job.id)
    return job


@app.get("/api/v1/jobs", response_model=list[Job])
def list_jobs() -> list[Job]:
    return store.list()


@app.post("/api/v1/jobs/{job_id}/retry", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def retry_job(job_id: str, background_tasks: BackgroundTasks) -> Job:
    old_job = store.get(job_id)
    if old_job.status not in (JobStatus.failed, JobStatus.completed):
        raise HTTPException(status_code=409, detail="任务正在处理中，不能重复提交")
    input_files = list(job_dir(job_id).glob("input.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="找不到原始视频文件")
    now = datetime.now(timezone.utc)
    retry = Job(
        id=uuid.uuid4().hex,
        filename=old_job.filename,
        status=JobStatus.queued,
        progress=0,
        stage="等待处理",
        created_at=now,
        updated_at=now,
    )
    destination = job_dir(retry.id)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_files[0], destination / input_files[0].name)
    try:
        validate_video(destination / input_files[0].name)
    except ValueError as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    persist_job(retry)
    store.register(retry)
    background_tasks.add_task(run_reconstruction, retry.id)
    return retry


@app.post(
    "/api/v1/jobs/{job_id}/resume", response_model=Job, status_code=status.HTTP_202_ACCEPTED
)
def resume_job(job_id: str, background_tasks: BackgroundTasks) -> Job:
    """续跑稠密阶段。

    后端进程被关闭（例如终端被关掉）会遗弃正在跑的任务，稀疏重建的成果却还在磁盘上。
    只要 ``dense/`` 完整，这个接口就能从 ``patch_match_stereo`` 的断点接着跑完融合和出模型，
    省下整段稀疏重建（本例中约 3.5 小时）。
    """
    job = store.get(job_id)
    if not (job_dir(job_id) / "dense" / "sparse").exists():
        raise HTTPException(status_code=409, detail="该任务没有可续跑的稠密中间产物（dense/ 不存在）")
    if result_file(job_id).exists():
        raise HTTPException(status_code=409, detail="该任务已经生成过模型，无需续跑")
    if pipeline_lock.locked():
        raise HTTPException(status_code=409, detail="已有任务在使用 GPU，请稍后重试")
    # 注意：不要在这里写 progress。它是判断“从哪一步续跑”的依据
    # （≥95 = 融合已完成，可只补网格），覆盖掉会导致白白重跑一遍融合。
    resume_from = job.progress
    updated = store.update(
        job_id,
        status=JobStatus.processing,
        stage=f"准备续跑（上次停在 {resume_from}%）",
        error=None,
    )
    persist_job(updated)
    background_tasks.add_task(resume_dense_reconstruction, job_id, resume_from)
    return updated


@app.post(
    "/api/v1/jobs/{job_id}/remesh", response_model=Job, status_code=status.HTTP_202_ACCEPTED
)
def remesh_job(job_id: str, background_tasks: BackgroundTasks) -> Job:
    """用已有的 ``fused.ply`` 重新生成网格与 GLB，不重跑稠密重建。

    调了网格参数（Poisson 深度 / 离群点裁剪 / 体素降采样）后用它，一分钟就能重新出模型，
    而不必再花几小时跑 patch_match_stereo。
    """
    fused = point_cloud_file(job_id)
    if not fused.exists() or fused.stat().st_size == 0:
        raise HTTPException(status_code=409, detail="没有可复用的 fused.ply，请先跑完稠密重建")
    if pipeline_lock.locked():
        raise HTTPException(status_code=409, detail="已有任务在使用 GPU，请稍后重试")
    updated = store.update(
        job_id,
        status=JobStatus.processing,
        progress=95,
        stage=f"准备用现有 {fused.name} 重建网格",
        error=None,
        result_url=None,  # 先摘掉旧链接，避免下载到半成品
    )
    persist_job(updated)
    background_tasks.add_task(remesh_from_point_cloud, job_id)
    return updated


@app.delete("/api/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str) -> None:
    job = store.get(job_id)
    if job.status in (JobStatus.queued, JobStatus.processing):
        raise HTTPException(status_code=409, detail="任务正在处理中，暂不能删除")
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    store.remove(job_id)


@app.get("/api/v1/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    return store.get(job_id)


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job.status != JobStatus.completed or not result_file(job_id).exists():
        raise HTTPException(status_code=409, detail="任务尚未生成结果")
    return FileResponse(result_file(job_id), media_type="model/gltf-binary", filename="model.glb")
