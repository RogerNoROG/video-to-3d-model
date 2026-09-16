from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
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
import open3d as o3d

from app.convert import crop_outliers, keep_object_by_color, point_cloud_to_glb


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
    # 只保留「暖色」被摄物，剔除白/灰的支撑面与背景。-1 = 关闭（默认）。
    # 判据是 R-B：棕木材/陶土 R 明显大于 B，白桌面/水泥/天空 R≈G≈B。
    # 0 = Otsu 自动（注意：实测会偏大、削到物体暗部，需目视确认）；
    # >0 = 显式阈值。阈值是**场景相关**的，本项目实测 0.08 最佳。
    # 这是唯一可靠的分离手段：几何上物块与桌面完全连通、密度上桌面更密、
    # RANSAC 平面会命中物体自己的面
    mesh_object_warmth: float = Field(default=-1.0, ge=-1.0, le=1.0)
    # 项目任务会额外保存一份已分离主体的点云。黄色物块在现有数据中以 0.08
    # 可稳定排除灰白背景；原始 fused.ply 始终保留，任何过滤异常都会回退原始点云。
    project_object_warmth: float = Field(default=0.08, ge=0.0, le=1.0)
    # Poisson 深度的上限（内存约束）。八叉树叶节点数约为 8^depth，每 +1 内存约 ×8，
    # 24GB 内存下 9 已是上限（实测 9 在 1854 万点上会耗尽内存）
    poisson_max_depth: int = Field(default=9, ge=5, le=12)
    # 超过这个点数就跳过全局法线定向（该步骤要为每个点建 30 邻域图，是主要内存开销）
    poisson_orient_max_points: int = Field(default=5_000_000, gt=0)
    use_gpu: bool = True
    gpu_index: int = Field(default=0, ge=0)
    # 资源用量控制：**CPU 用满，只压内存**。
    # num_threads = -1 是 COLMAP 默认值，即用满所有核 —— 保持不动。
    # 真正要收的是 cache_size：它默认 32 GB，比这台机器 24 GB 的物理内存还大，
    # 会把内存吃千并触发 WSL 虚拟机的内存回收。
    colmap_num_threads: int = Field(default=-1, ge=-1)
    colmap_cache_size_gb: int = Field(default=8, ge=1, le=128)
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
    # 融合时每张图参考多少张邻居做一致性检查（COLMAP 默认 50）。
    # ★ 这是 fusion 阶段内存和 IO 的**总开关**：每张图要读 1+check_num_images
    #   份深度/法线图（本任务单份 16 MB）。50 → 816 MB/张，1325 张共 ~550 GB
    #   读取量，工作集远超 24 GB 物理内存，实测退化成 ~2 分钟/张（预计 44 小时）
    #   且 CPU 只用 1 个核（其余全在等换页）。
    #   视频抽帧相邻帧高度冗余，50 张参考里有 40 张几乎重复，降到 10 既保住质量，
    #   又把工作集压到 ~176 MB/张，缓存命中率接近 100%，IO 降到 82 GB（只读一遍）。
    stereo_fusion_check_num_images: int = Field(default=10, ge=1, le=200)
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


class StepEta(BaseModel):
    """一个重建步骤的只读进度预测。"""

    key: str
    label: str
    status: str
    eta_seconds: int | None = None


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
    # 项目是可选的，以兼容历史任务和当前正在运行的旧任务。
    project_id: str | None = None
    # 在项目中的提交顺序。用于区分首段扫描与后续补拍，即使用户在首段完成前继续上传。
    project_sequence: int | None = None
    # 以下字段由读取接口实时推导，不能写进 job.json：输出速度会持续变化。
    eta_seconds: int | None = None
    stage_eta_seconds: int | None = None
    step_etas: list[StepEta] = Field(default_factory=list)


class Artifact(BaseModel):
    name: str
    path: str
    kind: str = "model"
    approved: bool = False


class Project(BaseModel):
    id: str
    name: str
    description: str = ""
    job_ids: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


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


class ProjectStore:
    """轻量项目索引；任务目录保持原位，避免影响正在运行的 pipeline。"""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._lock = threading.Lock()
        self._load()

    @property
    def index_file(self) -> Path:
        return settings.storage_dir / "projects.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else raw.get("projects", [])
            for item in entries:
                project = Project.model_validate(item)
                self._projects[project.id] = project
        except (OSError, ValueError, TypeError, AttributeError):
            return

    def _persist(self) -> None:
        self.index_file.write_text(
            json.dumps({"projects": [p.model_dump(mode="json") for p in self._projects.values()]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[Project]:
        with self._lock:
            return sorted(self._projects.values(), key=lambda p: p.updated_at, reverse=True)

    def get(self, project_id: str) -> Project:
        with self._lock:
            project = self._projects.get(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="项目不存在")
        return project

    def create(self, name: str, description: str = "", project_id: str | None = None) -> Project:
        now = datetime.now(timezone.utc)
        project = Project(id=project_id or uuid.uuid4().hex, name=name.strip() or "未命名项目", description=description, created_at=now, updated_at=now)
        with self._lock:
            self._projects[project.id] = project
            self._persist()
        return project

    def update(self, project_id: str, **changes: object) -> Project:
        with self._lock:
            current = self._projects.get(project_id)
            if current is None:
                raise HTTPException(status_code=404, detail="项目不存在")
            updated = current.model_copy(update={"updated_at": datetime.now(timezone.utc), **changes})
            self._projects[project_id] = updated
            self._persist()
            return updated

    def add_job(self, project_id: str, job: Job) -> Project:
        project = self.get(project_id)
        ids = list(project.job_ids)
        if job.id not in ids:
            ids.append(job.id)
        artifacts = list(project.artifacts)
        model_path = job_dir(job.id) / "merge" / "merged_cube_final.glb"
        if model_path.exists() and not any(a.path == str(model_path) for a in artifacts):
            artifacts.append(Artifact(name="黄色正方体定稿", path=str(model_path), approved=True))
        return self.update(project_id, job_ids=ids, artifacts=artifacts)

    def add_artifact(self, project_id: str, artifact: Artifact) -> Project:
        project = self.get(project_id)
        # 以路径去重，允许登记时更新同一产物的名称或审核状态。
        artifacts = [item for item in project.artifacts if item.path != artifact.path]
        artifacts.append(artifact)
        return self.update(project_id, artifacts=artifacts)


def seed_projects() -> None:
    """建立历史黄色物块项目索引；不写入任何任务目录。"""
    if projects.list():
        return
    project = projects.create("黄色正方体物块", "黄色/木色立方体及四分之一圆环凹槽的全部扫描、修复和交付产物。", project_id="yellow-cube")
    historical = ["e7f9f9e5d046490d93b34e4538bc9ef1", "1a479e264ad34e54bf33957ff9841254", "078ba28f089846bca6c5fd2d8456a4d4"]
    ids = [jid for jid in historical if (settings.storage_dir / jid / "job.json").exists()]
    artifacts = list(project.artifacts)
    final_path = settings.storage_dir / historical[0] / "merge" / "merged_cube_final.glb"
    if final_path.exists():
        artifacts.append(Artifact(name="干净定稿模型", path=str(final_path), approved=True))
    projects.update(project.id, job_ids=ids, artifacts=artifacts)


def reconcile_project_index() -> None:
    """把已完成的新任务加入黄色物块项目，并登记独立候选产物。

    只修改项目索引，不改任务目录；候选默认未批准，避免前端误把错误面当定稿。
    """
    project = projects.get("yellow-cube")
    new_job_id = "0965f558d11a4c7b9ce96ac869ca5368"
    job_file = settings.storage_dir / new_job_id / "job.json"
    if not job_file.exists():
        return
    try:
        job = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if new_job_id not in project.job_ids:
        project = projects.update(project.id, job_ids=[*project.job_ids, new_job_id])
    candidate_dir = settings.storage_dir / "yellow-cube" / "merge" / "latest-0965"
    candidates = list(project.artifacts)
    for index in (1, 2, 3):
        path = candidate_dir / f"candidate_{index}.glb"
        if path.exists() and not any(item.path == str(path) for item in candidates):
            candidates.append(Artifact(name=f"新视频候选合并 {index}", path=str(path), approved=False))
    # 兼容 2026-09-16 首次自动补拍的历史输出。那次子进程以任务目录为 cwd，
    # 相对的 storage/... 被解析成了 <job>/storage/...；文件本身有效，只是索引
    # 原先找不到它们。后续任务会走 run_project_postprocess 的绝对输出路径。
    legacy_job_id = "a0769748882b4d569e4114a7dd48de47"
    legacy_dir = (
        settings.storage_dir / legacy_job_id / "storage" / "yellow-cube" / "automation" / legacy_job_id
    )
    for index in (1, 2, 3):
        path = legacy_dir / f"candidate_{index}.glb"
        if path.exists() and not any(item.path == str(path) for item in candidates):
            candidates.append(Artifact(
                name=f"补拍对齐候选 {index} · 1000059278_1080p30_10bit.mp4",
                path=str(path),
                kind="alignment-candidate",
                approved=False,
            ))
    six_face = settings.storage_dir / "yellow-cube" / "six-face-v1" / "final-v3" / "merged_six_face_depth9.glb"
    if six_face.exists() and not any(item.path == str(six_face) for item in candidates):
        candidates.append(Artifact(
            name="三次扫描五面融合高密度候选",
            path=str(six_face),
            kind="candidate",
            approved=False,
        ))
    edge_repair = settings.storage_dir / "yellow-cube" / "six-face-v1" / "edge-repair-v1" / "merged_six_face_real_edges_depth9.glb"
    if edge_repair.exists() and not any(item.path == str(edge_repair) for item in candidates):
        candidates.append(Artifact(
            name="真实数据棱角补强候选",
            path=str(edge_repair),
            kind="candidate",
            approved=False,
        ))
    hole_repair = settings.storage_dir / "yellow-cube" / "six-face-v1" / "hole-repair-v1" / "merged_six_face_hole_repaired_depth9.glb"
    if hole_repair.exists() and not any(item.path == str(hole_repair) for item in candidates):
        candidates.append(Artifact(
            name="真实数据局部底面修复候选",
            path=str(hole_repair),
            kind="candidate",
            approved=False,
        ))
    feature_planes = settings.storage_dir / "yellow-cube" / "six-face-v1" / "feature-and-plane-v1" / "merged_six_face_feature_protected_planes.glb"
    if feature_planes.exists() and not any(item.path == str(feature_planes) for item in candidates):
        candidates.append(Artifact(
            name="贯穿圆孔保护与外平面平滑候选",
            path=str(feature_planes),
            kind="candidate",
            approved=False,
        ))
    if candidates != project.artifacts:
        projects.update(project.id, artifacts=candidates)


store = JobStore()
projects = ProjectStore()
pipeline_lock = threading.Lock()
app = FastAPI(title="Video to 3D Model API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class ArtifactCreate(BaseModel):
    """登记已生成的项目产物；仅允许 storage/ 下的现有文件。"""

    name: str = Field(min_length=1, max_length=160)
    path: str = Field(min_length=1)
    kind: str = Field(default="model", max_length=40)
    approved: bool = False


class ArtifactPatch(BaseModel):
    """人工审核结论：只改批准状态，不允许改路径或名称。"""

    approved: bool


class AssistantMessage(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=20000)


class AssistantChatRequest(BaseModel):
    """网页 AI 助手的一次请求。

    接口地址与密钥由网页从本机 localStorage 读出来随请求带上：
    密钥只在这一次请求里透传（不落盘、不进项目文件），而转发由本机后端完成，
    可以避开第三方接口的浏览器跨域限制。
    """

    endpoint: str = Field(min_length=1, max_length=500)
    api_key: str = Field(default="", max_length=500)
    model: str = Field(default="", max_length=200)
    project_id: str = Field(default="", max_length=80)
    messages: list[AssistantMessage] = Field(min_length=1, max_length=40)


def job_dir(job_id: str) -> Path:
    return settings.storage_dir / job_id


def result_file(job_id: str) -> Path:
    return job_dir(job_id).resolve() / "model.glb"


# 只创建/更新 projects.json；历史任务目录与运行中的任务完全不动。
seed_projects()
reconcile_project_index()


def point_cloud_file(job_id: str) -> Path:
    return job_dir(job_id).resolve() / "fused.ply"


ETA_STEPS: tuple[tuple[str, str], ...] = (
    ("prepare", "检查并规范化视频"),
    ("frames", "抽取关键帧"),
    ("features", "提取图像特征"),
    ("matching", "匹配相邻帧"),
    ("poses", "估计相机位姿"),
    ("dense", "生成稠密深度图"),
    ("fusion", "融合点云"),
    ("mesh", "重建网格并导出 GLB"),
    ("subject", "分离目标物体"),
    ("project_merge", "对齐项目模型并补充细节"),
)


def _stage_key(job: Job) -> str | None:
    """从正在显示的阶段文字判定流程步骤；文字同时兼容旧任务。"""
    stage = job.stage
    if job.status == JobStatus.completed:
        return None
    if "项目内" in stage or "对齐项目" in stage or "补充细节" in stage:
        return "project_merge"
    if "分离目标" in stage or "主体清理" in stage:
        return "subject"
    if "网格" in stage or "GLB" in stage or "重建模型" in stage:
        return "mesh"
    if "融合" in stage:
        return "fusion"
    if "稠密" in stage or "patch_match" in stage:
        return "dense"
    if "位姿" in stage or "注册图像" in stage:
        return "poses"
    if "匹配" in stage:
        return "matching"
    if "特征" in stage:
        return "features"
    if "抽取" in stage:
        return "frames"
    if "视频" in stage or "转码" in stage or "规格" in stage:
        return "prepare"
    return None


def _dense_stage_progress(job: Job, dense: Path) -> tuple[int, int] | None:
    """优先使用阶段中的计数；续跑的旧阶段则从已写的深度图推断。"""
    match = re.search(r"[（(]\s*(\d+)\s*/\s*(\d+)\s*[）)]", job.stage)
    if match:
        return int(match.group(1)), int(match.group(2))
    views = count_dense_views(dense)
    if not views:
        return None
    total = views * (2 if settings.patch_match_geom_consistency else 1)
    return min(count_dense_depth_maps(dense), total), total


def _dense_rate_seconds_per_output(dense: Path) -> float | None:
    """根据最近完成的深度图文件估计真实速率，忽略启动阶段的波动。"""
    maps = dense / "stereo" / "depth_maps"
    try:
        times = sorted(path.stat().st_mtime for path in maps.glob("*.bin"))
    except OSError:
        return None
    # 末尾 20 个输出能跟上当前 GPU/CPU 状态，又不会被最初缓存预热拉偏。
    sample = times[-20:]
    if len(sample) < 2 or sample[-1] <= sample[0]:
        return None
    return (sample[-1] - sample[0]) / (len(sample) - 1)


def _step_duration_estimates(job: Job) -> dict[str, int]:
    """保守的计划时长；稠密阶段开始后会被真实输出速率替代。"""
    root = job_dir(job.id)
    frames = sum(1 for _ in (root / "frames").glob("*.jpg"))
    dense = root / "dense"
    dense_progress = _dense_stage_progress(job, dense)
    dense_outputs = dense_progress[1] if dense_progress else max(frames * (2 if settings.patch_match_geom_consistency else 1), 1)
    views = max(dense_outputs // (2 if settings.patch_match_geom_consistency else 1), frames, 1)
    # 这些是无实时计数时的初始计划值。它们会在任务进入对应步骤后由输出/日志
    # 更新；没有足够样本时宁可保守些，也不把预计时间伪装成精确时间。
    return {
        "prepare": 90,
        "frames": max(45, int(max(frames, 90) * 0.25)),
        "features": max(90, int(max(frames, 90) * 0.8)),
        "matching": max(120, int(max(frames, 90) * 1.2)),
        "poses": max(180, int(max(frames, 90) * 1.5)),
        "dense": max(300, dense_outputs * 25),
        "fusion": max(180, views * 7),
        "subject": 300,
        "mesh": 900,
        "project_merge": 1800,
    }


def with_eta(job: Job) -> Job:
    """为 API 响应附加 ETA，不修改内存任务或磁盘上的 job.json。"""
    estimates = _step_duration_estimates(job)
    active = _stage_key(job)
    dense = job_dir(job.id) / "dense"
    stage_eta: int | None = None

    if job.status == JobStatus.processing and active:
        if active == "dense":
            progress = _dense_stage_progress(job, dense)
            seconds_per_output = _dense_rate_seconds_per_output(dense)
            if progress and seconds_per_output is not None:
                done, total = progress
                stage_eta = max(0, round((total - done) * seconds_per_output))
            elif progress:
                done, total = progress
                stage_eta = max(0, (total - done) * 25)
        else:
            stage_eta = estimates[active]

    steps: list[StepEta] = []
    active_index = next((i for i, (key, _) in enumerate(ETA_STEPS) if key == active), None)
    for index, (key, label) in enumerate(ETA_STEPS):
        if job.status == JobStatus.completed:
            step_status, eta = "done", None
        elif job.status == JobStatus.failed:
            step_status = "failed" if key == active else ("done" if active_index is not None and index < active_index else "pending")
            eta = None if step_status != "pending" else estimates[key]
        elif active_index is None:
            step_status, eta = "pending", estimates[key]
        elif index < active_index:
            step_status, eta = "done", None
        elif index == active_index:
            step_status, eta = "active", stage_eta
        else:
            step_status, eta = "pending", estimates[key]
        steps.append(StepEta(key=key, label=label, status=step_status, eta_seconds=eta))

    total_eta: int | None = None
    if job.status == JobStatus.processing and active_index is not None and stage_eta is not None:
        total_eta = stage_eta + sum(estimates[key] for key, _ in ETA_STEPS[active_index + 1:])
    return job.model_copy(update={"eta_seconds": total_eta, "stage_eta_seconds": stage_eta, "step_etas": steps})


def persist_job(job: Job) -> None:
    job_dir(job.id).mkdir(parents=True, exist_ok=True)
    (job_dir(job.id) / "job.json").write_text(
        job.model_dump_json(indent=2, exclude={"eta_seconds", "stage_eta_seconds", "step_etas"}), encoding="utf-8"
    )


def object_cloud_file(job_id: str) -> Path:
    """项目自动合并唯一允许读取的、已分离主体点云。"""
    return job_dir(job_id).resolve() / "project" / "object_only.ply"


def _project_path(path: Path) -> str:
    """项目索引使用工作区相对路径，保证服务重启后仍能读取。"""
    return str(path.resolve().relative_to(Path.cwd().resolve()))


def _write_project_audit(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_project_object(job_id: str) -> Path:
    """从单次稠密点云得到项目级主体来源；原始点云从不修改。"""
    source = point_cloud_file(job_id)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("缺少 fused.ply，无法分离项目主体")
    cloud = o3d.io.read_point_cloud(str(source))
    if not len(cloud.points):
        raise RuntimeError("fused.ply 为空，无法分离项目主体")
    before = len(cloud.points)
    cleaned = crop_outliers(cloud, settings.poisson_outlier_percentile)
    # 这是一项有意保守的颜色分离：keep_object_by_color 在比例异常时会原样返回，
    # 避免把没有明显色差的物体裁残。黄色物块已用 0.08 验证能去掉灰白背景。
    cleaned = keep_object_by_color(cleaned, settings.project_object_warmth)
    output = object_cloud_file(job_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output), cleaned, write_ascii=False)
    _write_project_audit(output.with_name("object_extraction.json"), {
        "job": job_id,
        "source": _project_path(source),
        "output": _project_path(output),
        "points_before": int(before),
        "points_after": int(len(cleaned.points)),
        "warmth_threshold": settings.project_object_warmth,
        "object_extent": [float(value) for value in (cleaned.get_max_bound() - cleaned.get_min_bound())],
        "rule": "原始 fused.ply 保留不动；项目合并只读取 object_only.ply，颜色分离比例异常时自动回退为未按色删除的点云。",
    })
    return output


def _project_reference(project: Project, exclude_job_id: str) -> Path | None:
    """优先使用已批准模型；新项目没有批准模型时使用最早的主体点云。"""
    for artifact in project.artifacts:
        if artifact.approved:
            candidate = Path(artifact.path).resolve()
            if candidate.is_file():
                return candidate
    prior = sorted(
        (
            job for job in store.list()
            if job.project_id == project.id
            and job.id != exclude_job_id
            and job.status == JobStatus.completed
        ),
        key=lambda item: (item.project_sequence or 0, item.created_at),
    )
    for job in prior:
        candidate = object_cloud_file(job.id)
        if candidate.is_file():
            return candidate
    return None


def run_project_postprocess(job_id: str) -> str:
    """首段做主体交付；补拍做独立坐标系对齐候选，不自动覆盖批准模型。"""
    job = store.get(job_id)
    if not job.project_id:
        return ""
    project = projects.get(job.project_id)
    object_cloud = object_cloud_file(job_id)
    if not object_cloud.is_file():
        update_stage(job_id, 97, "项目内分离目标物体并保留原始点云")
        object_cloud = extract_project_object(job_id)
    reference = _project_reference(project, job_id)
    is_first = job.project_sequence == 1 or reference is None
    if is_first:
        object_model = result_file(job_id)
        projects.add_artifact(project.id, Artifact(
            name=f"首段主体模型 · {job.filename}",
            path=_project_path(object_model), kind="model", approved=False,
        ))
        return "；已分离首段主体，原始背景结果已保留作追溯"

    # run_command 以任务目录为 cwd；输出必须绝对化，防止在任务目录内再创建一层
    # storage/，并确保候选可被下面的项目索引立即登记。
    output = (settings.storage_dir / project.id / "automation" / job_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    update_stage(job_id, 99, "项目内对齐既有模型并生成补细节候选")
    command = [
        sys.executable, str(Path("tools/fusion/merge_two_jobs.py").resolve()),
        "--a", str(reference), "--b", str(object_cloud), "--out", str(output),
        "--top", "4", "--glb", "3", "--warmth-a", "-1", "--warmth-b", "-1",
    ]
    run_command(command, job_dir(job_id).resolve())
    candidates = []
    for rank in range(1, 4):
        path = output / f"candidate_{rank}.glb"
        if not path.is_file():
            continue
        projects.add_artifact(project.id, Artifact(
            name=f"补拍对齐候选 {rank} · {job.filename}",
            path=_project_path(path), kind="alignment-candidate", approved=False,
        ))
        candidates.append(_project_path(path))
    _write_project_audit(output / "project_merge_audit.json", {
        "project": project.id,
        "job": job_id,
        "base": _project_path(reference),
        "source": _project_path(object_cloud),
        "candidates": candidates,
        "approved": False,
        "rule": "24 个立方体对称姿态均已评分和 ICP 精修。所有输出都是待审核候选；未验证的面、错误姿态或背景绝不自动写入已批准模型。",
    })
    if not candidates:
        return "；已完成对齐评分，但没有通过导出的补细节候选"
    return f"；已生成 {len(candidates)} 个对齐补细节候选，等待审核后再采用"


def project_postprocess_summary(job_id: str) -> str:
    """项目自动后处理不能让已完成的单次重建被标记为失败。"""
    try:
        return run_project_postprocess(job_id)
    except Exception as exc:
        root = job_dir(job_id)
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write(f"项目自动后处理失败（原始单次结果保留）：{exc}\n")
        return "；项目自动处理未完成，原始单次结果和点云均已保留，可安全重试"


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


def probe_video(video: Path) -> dict:
    """用 ffprobe 读视频流参数；失败返回空 dict（那就按老路老老实实转码）。

    返回键：``width``/``height``/``fps``/``codec_name``/``pix_fmt``。
    """
    if shutil.which(settings.ffprobe_binary) is None:
        return {}
    try:
        result = subprocess.run(
            [
                settings.ffprobe_binary, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,codec_name,pix_fmt",
                "-of", "json", str(video),
            ],
            capture_output=True, text=True, check=False, timeout=30,
        )
        streams = json.loads(result.stdout).get("streams") or []
    except Exception:
        return {}
    if not streams:
        return {}
    stream = streams[0]
    numerator, _, denominator = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except ValueError:
        fps = 0.0
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "codec_name": stream.get("codec_name", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
    }


def already_normalized(info: dict) -> bool:
    """源视频是否已经是目标规格，不必再转一遍。

    多转一轮不只是浪费时间：本项目素材有 10bit HEVC 的，转成 8bit H.264
    是实打实的精度损失。规格已经对上就直接拿原文件去抽帧。
    """
    if not info:
        return False
    return (
        info["width"] == settings.normalize_width
        and info["height"] == settings.normalize_height
        and abs(info["fps"] - settings.normalize_fps) < 0.01
    )


def normalize_video(video: Path, normalized: Path, root: Path) -> None:
    # 源视频已经是目标分辨率/帧率就直接复用，不再转码（见 already_normalized 的说明）。
    # 用硬链接而不是复制：同一文件系统下零拷贝、不占额外磁盘。
    info = probe_video(video)
    if already_normalized(info):
        normalized.unlink(missing_ok=True)
        try:
            normalized.hardlink_to(video)
        except OSError:
            shutil.copy2(video, normalized)
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write(
                f"源视频已是 {info['width']}x{info['height']}@{info['fps']:g} "
                f"({info['codec_name']}/{info['pix_fmt']})，跳过转码，直接用它抽帧\n"
            )
        return
    if info:
        with (root / "pipeline.log").open("a", encoding="utf-8") as log:
            log.write(
                f"源视频 {info['width']}x{info['height']}@{info['fps']:g} "
                f"({info['codec_name']}/{info['pix_fmt']})，转码为 "
                f"{settings.normalize_width}x{settings.normalize_height}@{settings.normalize_fps}\n"
            )

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

    update_stage(job_id, 10, "检查视频规格（已是 1080p30 则跳过转码）")
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
    source = point_cloud_path
    job = store.get(job_id)
    # 首段扫描直接从已分离的主体点云出模型。原始 fused.ply 未改动，可随时复核
    # 或调整阈值后重新导出；后续补拍的单次模型保持原样，仅把清理点云用于候选融合。
    if job.project_id and job.project_sequence == 1:
        update_stage(job_id, 96, "项目内分离目标物体，首段模型不包含背景")
        source = extract_project_object(job_id)
    with log_heartbeat(root):
        point_cloud_to_glb(
            source,
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
            object_warmth=settings.mesh_object_warmth,
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
        # 资源用量：**CPU 用满，只压内存**。num_threads=-1 就是用满所有核；
        # 要收的是 cache_size —— 它默认 32 GB，比这台机器 24 GB 物理内存还大，
        # 会把内存吃千并触发 WSL 虚拟机的内存回收。
        "--PatchMatchStereo.num_threads", str(settings.colmap_num_threads),
        "--PatchMatchStereo.cache_size", str(settings.colmap_cache_size_gb),
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
            "--StereoFusion.num_threads", str(settings.colmap_num_threads),
            "--StereoFusion.check_num_images", str(settings.stereo_fusion_check_num_images),
            # use_cache 默认是 0（关闭）—— 此时 COLMAP 会把**全部** 1325 张图的
            # 深度图+法线图一次性读进内存（1325 x 16 MB ~= 21 GB），24 GB 的机器
            # 装不下就会触发 WSL 内存回收，表现为 WSL 反复重启、终端连不上。
            # 打开缓存把驻留量封在 cache_size 以内。
            "--StereoFusion.use_cache", "1",
            "--StereoFusion.cache_size", str(settings.colmap_cache_size_gb),
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
    if completed.project_id:
        projects.add_job(completed.project_id, completed)


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
        with pipeline_lock:
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
                run_dense_stage(job_id, root, dense, registered)
            note = project_postprocess_summary(job_id)
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）{note}")
    except Exception as exc:
        fail_job(job_id, str(exc))


def remesh_from_point_cloud(job_id: str) -> None:
    """用已有的 fused.ply 重新生成网格与 GLB。"""
    try:
        root = job_dir(job_id).resolve()
        _, registered, _, _ = select_best_model(root / "sparse")
        frames = root / "frames"
        total_frames = len(list(frames.glob("*.jpg"))) or registered
        with pipeline_lock:
            mesh_and_export(job_id, root, point_cloud_file(job_id))
            note = project_postprocess_summary(job_id)
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）{note}")
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
            note = project_postprocess_summary(job_id)
        finish_job(job_id, f"处理完成（注册 {registered}/{total_frames} 张图像）{note}")
    except Exception as exc:
        fail_job(job_id, str(exc))


def _finish_legacy_project_jobs(job_ids: set[str]) -> None:
    """兼容旧后端已启动的项目任务；只观察启动时仍在运行的这一批。"""
    pending = set(job_ids)
    while pending:
        for job_id in tuple(pending):
            try:
                # 任务由另一台仍在运行的旧后端更新；不能使用本进程启动时缓存的
                # Job，而必须每轮读它刚写入的 job.json。
                job = Job.model_validate_json(
                    (job_dir(job_id) / "job.json").read_text(encoding="utf-8")
                )
                store.register(job)
            except (OSError, ValueError):
                pending.discard(job_id)
                continue
            if job.status in (JobStatus.failed, JobStatus.queued, JobStatus.processing):
                continue
            pending.discard(job_id)
            if job.status != JobStatus.completed or not point_cloud_file(job_id).is_file():
                continue
            # 旧后端的任务已经完成全部 COLMAP 写入；此时独占本服务的项目后处理，
            # 既不改它的原始视频/稠密目录，也不会与 patch_match_stereo 并发。
            with pipeline_lock:
                note = project_postprocess_summary(job_id)
            finish_job(job_id, f"{job.stage}{note}")
        if pending:
            threading.Event().wait(15)


@app.on_event("startup")
def watch_legacy_project_jobs() -> None:
    """在无重启迁移期间，为旧 8000 后端的当前项目任务补上自动流程。"""
    pending = {
        job.id for job in store.list()
        if job.project_id and job.project_sequence is None
        and job.status in (JobStatus.queued, JobStatus.processing)
    }
    if not pending:
        return
    watcher = threading.Thread(
        target=_finish_legacy_project_jobs,
        args=(pending,), name="legacy-project-postprocess", daemon=True,
    )
    watcher.start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/projects", response_model=list[Project])
def list_projects() -> list[Project]:
    return projects.list()


@app.post("/api/v1/projects", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate) -> Project:
    return projects.create(payload.name, payload.description)


@app.get("/api/v1/projects/{project_id}", response_model=Project)
def get_project(project_id: str) -> Project:
    return projects.get(project_id)


@app.patch("/api/v1/projects/{project_id}", response_model=Project)
def patch_project(project_id: str, payload: ProjectPatch) -> Project:
    changes = payload.model_dump(exclude_unset=True)
    return projects.update(project_id, **changes) if changes else projects.get(project_id)


@app.get("/api/v1/projects/{project_id}/jobs", response_model=list[Job])
def list_project_jobs(project_id: str) -> list[Job]:
    project = projects.get(project_id)
    jobs = {job.id: job for job in store.list()}
    return [with_eta(jobs[jid]) for jid in project.job_ids if jid in jobs]


@app.get("/api/v1/projects/{project_id}/artifacts", response_model=list[Artifact])
def list_project_artifacts(project_id: str) -> list[Artifact]:
    return projects.get(project_id).artifacts


@app.post(
    "/api/v1/projects/{project_id}/artifacts",
    response_model=Project,
    status_code=status.HTTP_201_CREATED,
)
def create_project_artifact(project_id: str, payload: ArtifactCreate) -> Project:
    path = Path(payload.path).resolve()
    storage_root = settings.storage_dir.resolve()
    if storage_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=422, detail="产物必须是 storage/ 内已存在的文件")
    # 存相对路径，避免项目索引依赖当前工作目录的绝对位置。
    artifact = Artifact(
        name=payload.name,
        path=str(path.relative_to(Path.cwd().resolve())),
        kind=payload.kind,
        approved=payload.approved,
    )
    return projects.add_artifact(project_id, artifact)


@app.patch("/api/v1/projects/{project_id}/artifacts/{artifact_index}", response_model=Project)
def patch_project_artifact(
    project_id: str, artifact_index: int, payload: ArtifactPatch
) -> Project:
    """人工审核后写入批准状态。

    项目语义是「只有一个已批准基准」：把某一条设为批准时会同时撤回其它条目，
    这样网页上的「已批准基准模型」始终唯一，候选也不会互相争抢基准位。
    只允许改 approved —— 路径与名称由登记时决定，审核阶段不允许篡改。
    """
    project = projects.get(project_id)
    if artifact_index < 0 or artifact_index >= len(project.artifacts):
        raise HTTPException(status_code=404, detail="产物不存在")
    updated = [
        artifact.model_copy(
            update={"approved": payload.approved if index == artifact_index else False}
        )
        for index, artifact in enumerate(project.artifacts)
    ]
    return projects.update(project_id, artifacts=updated)


# ---------------------------------------------------------------- AI 助手
# 助手要能“看项目”，所以后端给它两样东西：
#   1. 每次请求都附一份项目快照（项目/产物/任务/已批准基准），相当于代码聊天框的仓库摘要；
#   2. 一组**只读**工具，模型可以按需再查（列目录、数几何、读审计 JSON）。
# 工具全部限制在 storage/ 内且不写任何文件。

ASSISTANT_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "list_projects", "description": "列出全项目及各自的模型数与任务数",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_project", "description": "读取一个项目的全部产物（名称/路径/类型/批准状态/文件大小）与视频任务状态",
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string", "description": "项目 ID"}}, "required": ["project_id"]}}},
    {"type": "function", "function": {
        "name": "inspect_mesh", "description": "读取 storage/ 下某个 GLB/PLY 的几何统计：点数/顶点数/三角面数/边界边数/包围盒/中心",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "storage/ 下的相对路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_storage", "description": "列出 storage/ 下某个目录的内容（审计产物、中间点云、日志通常在这里）",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "storage/ 下的相对目录"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_storage_json", "description": "读取 storage/ 下的审计 JSON（如 merge_two_jobs.json、*_audit.json、fill_report.json）",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "description": "最多返回多少字符，默认 4000"}},
            "required": ["path"]}}},
]

ASSISTANT_RULES = """\
你在一个本地摄影测量项目（FastAPI + COLMAP + Open3D 的“video-to-3d-model”）里充当助手的角色。

项目要点（回答时请遵守）：
- 一个“项目”下挂多个视频任务，每个任务产出 fused.ply 与 model.glb；项目产物（artifacts）才是交付物。
- **已批准基准永远唯一**：批准某条产物时会同时撤回其它条目。任何候选都不会自动覆盖已批准模型。
- 目录约定：storage/<job_id>/ 为单次任务中间产物（frames、sparse/、dense/、fused.ply、model.glb）；
  storage/yellow-cube/six-face-v1/ 下的子目录是各类审计与候选（orientation/、*-repair-v1/、*-fill-*/ 等）。
- 木块是近正方体，存在 24 重对称歧义：**fitness 与最近邻距离分不出姿态**，必须靠凹槽/圆孔
  落在同一物理面来目视判定。不要仅凭数值高就断言姿态正确。
- “局部修补”的正当做法是只取有真实观测证据的部分（按棱 repair_cube_edges.py / 按面
  extract_verified_face_repair.py），不得创建顶点或外推平面；孔、凹槽、棱不可被通用补洞处理。

工作方式：
- 先用给定工具查清事实再回答；不确定就说不确定，不要编造路径或数字。
- 引用数字时说明它来自哪个文件或工具调用。
- 回答用中文，简洁、直说结论，必要时给可执行命令。
"""


def _storage_path(raw: str) -> Path:
    """把用户/模型给的路径限制到 storage/ 之内（只读）。"""
    root = settings.storage_dir.resolve()
    candidate = Path(raw)
    path = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=422, detail=f"只允许访问 storage/ 内的路径：{raw}")
    return path


def _artifact_facts(artifact: Artifact) -> dict:
    path = _storage_path(artifact.path)
    exists = path.is_file()
    return {
        "name": artifact.name, "path": artifact.path, "kind": artifact.kind,
        "approved": artifact.approved, "exists": exists,
        "size_mb": round(path.stat().st_size / 2**20, 1) if exists else None,
    }


def _status_text(status: object) -> str:
    return str(getattr(status, "value", status))


def _project_snapshot(project_id: str) -> str:
    """项目摘要：让助手一开始就知道桌上有什么，不必先追问。"""
    lines: list[str] = []
    try:
        all_projects = projects.list()
    except Exception:  # noqa: BLE001
        all_projects = []
    lines.append("已登记的项目：" + ("、".join(p.id for p in all_projects) or "（无）"))
    target = next((p for p in all_projects if p.id == project_id), None) if project_id else None
    if target is None and all_projects:
        target = all_projects[0]
    if target is None:
        return "\n".join(lines)
    lines.append(f"\n当前项目：{target.id}（{target.name}）")
    lines.append(f"描述：{target.description or '（无）'}")
    lines.append(f"任务数：{len(target.job_ids)}")
    fact_jobs = {job.id: job for job in store.list()}
    for job_id in target.job_ids:
        job = fact_jobs.get(job_id)
        if job:
            lines.append(f"  - {job_id}: {job.filename} · {_status_text(job.status)} {job.progress}% · {job.stage}")
    lines.append(f"产物数：{len(target.artifacts)}")
    for artifact in target.artifacts:
        fact = _artifact_facts(artifact)
        mark = "★已批准基准" if artifact.approved else "待审核"
        lines.append(f"  - [{mark}] {fact['name']} · {fact['kind']} · {fact['size_mb']}MB · {fact['path']}")
    return "\n".join(lines)


def _run_assistant_tool(name: str, arguments: dict) -> dict:
    """执行一个只读工具。任何异常都会变成 {error: ...} 交回给模型继续推理。"""
    if name == "list_projects":
        return {"projects": [
            {"id": project.id, "name": project.name, "description": project.description,
             "artifacts": len(project.artifacts), "jobs": len(project.job_ids),
             "approved": [a.path for a in project.artifacts if a.approved]}
            for project in projects.list()
        ]}
    if name == "get_project":
        project = projects.get(str(arguments.get("project_id", "")))
        jobs = {job.id: job for job in store.list()}
        return {
            "id": project.id, "name": project.name, "description": project.description,
            "artifacts": [_artifact_facts(a) for a in project.artifacts],
            "jobs": [
                {"id": jid, "filename": jobs[jid].filename, "status": _status_text(jobs[jid].status),
                 "progress": jobs[jid].progress, "stage": jobs[jid].stage}
                for jid in project.job_ids if jid in jobs
            ],
        }
    if name == "list_storage":
        path = _storage_path(str(arguments.get("path", "storage")))
        if not path.exists():
            return {"error": f"路径不存在：{arguments.get('path')}"}
        if path.is_file():
            return {"path": str(path), "kind": "file", "size_mb": round(path.stat().st_size / 2**20, 2)}
        entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name))[:200]
        return {"path": str(path), "entries": [
            {"name": item.name, "dir": item.is_dir(),
             "size_mb": round(item.stat().st_size / 2**20, 2) if item.is_file() else None}
            for item in entries
        ]}
    if name == "read_storage_json":
        path = _storage_path(str(arguments.get("path", "")))
        if not path.is_file():
            return {"error": f"文件不存在：{arguments.get('path')}"}
        limit = int(arguments.get("max_chars") or 4000)
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"path": str(path), "text": text[:limit]}
        compact = json.dumps(data, ensure_ascii=False, indent=1)
        return {"path": str(path), "json": compact[:limit], "truncated": len(compact) > limit}
    if name == "inspect_mesh":
        import open3d as o3d  # 延迟导入：只有真要看几何时才付这个代价
        import numpy as np

        path = _storage_path(str(arguments.get("path", "")))
        if not path.is_file():
            return {"error": f"文件不存在：{arguments.get('path')}"}
        suffix = path.suffix.lower()
        if suffix in {".glb", ".gltf", ".obj", ".stl", ".ply"}:
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.triangles):
                triangles = np.asarray(mesh.triangles)
                facts = {
                    "path": str(path), "kind": "mesh",
                    "vertices": int(len(mesh.vertices)), "triangles": int(len(triangles)),
                    "bounds": [np.round(mesh.get_min_bound(), 4).tolist(),
                               np.round(mesh.get_max_bound(), 4).tolist()],
                }
                if len(triangles) <= 4_000_000:
                    edges = np.sort(np.vstack([
                        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
                    unique, counts = np.unique(edges, axis=0, return_counts=True)
                    facts["boundary_edges"] = int((counts == 1).sum())
                else:
                    facts["boundary_edges"] = None
                    facts["note"] = "三角面超过 400 万，跳过边界边统计"
                return facts
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if not len(points):
            return {"error": "既不是网格也不是点云"}
        return {
            "path": str(path), "kind": "pointcloud", "points": int(len(points)),
            "bounds": [np.round(points.min(axis=0), 4).tolist(),
                       np.round(points.max(axis=0), 4).tolist()],
            "center": np.round((points.min(axis=0) + points.max(axis=0)) / 2, 4).tolist(),
            "has_colors": bool(cloud.has_colors()),
        }
    return {"error": f"未知工具：{name}"}


def _post_chat_completions(url: str, headers: dict, messages: list[dict],
                           model: str, tools: list[dict] | None) -> dict:
    import urllib.error
    import urllib.request

    body: dict = {"messages": messages}
    if model:
        body["model"] = model
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise HTTPException(status_code=502, detail=f"上游返回 {error.code}：{detail}") from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"无法访问上游接口：{error}") from error


@app.post("/api/v1/assistant/chat")
def assistant_chat(payload: AssistantChatRequest) -> dict:
    """带项目上下文的助手对话：注入项目快照 + 只读工具循环。

    同步函数：FastAPI 会把它放到线程池里执行，不会卡住事件循环。
    工具全部只读且限制在 storage/ 内；助手不会写任何文件、也不会批准任何候选。
    """
    base = payload.endpoint.strip().rstrip("/")
    if not base.endswith("/chat/completions"):
        base = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if payload.api_key:
        headers["Authorization"] = f"Bearer {payload.api_key}"

    messages: list[dict] = [
        {"role": "system", "content": f"{ASSISTANT_RULES}\n\n=== 当前项目快照 ===\n{_project_snapshot(payload.project_id)}"}
    ] + [message.model_dump() for message in payload.messages]

    trace: list[dict] = []
    for _ in range(5):
        data = _post_chat_completions(base, headers, messages, payload.model, ASSISTANT_TOOLS)
        choices = data.get("choices") or []
        if not choices:
            raise HTTPException(status_code=502, detail=f"上游响应不符合 OpenAI 规范：{str(data)[:300]}")
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            return {"content": message.get("content") or "", "tool_calls": trace}
        messages.append(message)
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            try:
                result = _run_assistant_tool(name, arguments if isinstance(arguments, dict) else {})
                trace.append({"name": name, "arguments": arguments, "ok": "error" not in result})
            except Exception as error:  # noqa: BLE001
                result = {"error": str(error)}
                trace.append({"name": name, "arguments": arguments, "ok": False})
            messages.append({
                "role": "tool", "tool_call_id": call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False)[:6000],
            })
    return {"content": "（工具调用已达 5 轮上限，请把问题缩小一点再问）", "tool_calls": trace}


@app.get("/api/v1/projects/{project_id}/artifacts/{artifact_index}")
def get_project_artifact(project_id: str, artifact_index: int) -> FileResponse:
    artifacts = projects.get(project_id).artifacts
    if artifact_index < 0 or artifact_index >= len(artifacts):
        raise HTTPException(status_code=404, detail="产物不存在")
    path = Path(artifacts[artifact_index].path).resolve()
    storage_root = settings.storage_dir.resolve()
    if storage_root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="产物文件不存在")
    return FileResponse(path, media_type="model/gltf-binary", filename=path.name)


async def _create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    project_id: str | None = None,
) -> Job:
    project: Project | None = None
    if project_id is not None:
        project = projects.get(project_id)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in settings.allowed_video_extensions:
        raise HTTPException(status_code=415, detail="不支持的视频格式")

    job = store.create(video.filename or f"input{suffix}")
    if project_id:
        # 上传时就固定序号；若用户连续上传多个视频，不能等前一个完成后再猜谁是首段。
        job = store.update(
            job.id,
            project_id=project_id,
            project_sequence=len(project.job_ids) + 1 if project is not None else None,
        )
    destination = job_dir(job.id) / f"input{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size_mb * 1024 * 1024:
                    destination.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"视频超过大小限制：上限 {settings.max_upload_size_mb / 1024:g} GB")
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
    if project_id:
        projects.add_job(project_id, job)
    background_tasks.add_task(run_reconstruction, job.id)
    return job


@app.post("/api/v1/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="待重建的视频文件")],
) -> Job:
    return with_eta(await _create_job(background_tasks, video))


@app.post("/api/v1/projects/{project_id}/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
async def create_project_job(
    project_id: str,
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="项目中的视频任务")],
) -> Job:
    return with_eta(await _create_job(background_tasks, video, project_id))


@app.get("/api/v1/jobs", response_model=list[Job])
def list_jobs() -> list[Job]:
    return [with_eta(job) for job in store.list()]


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
        project_id=old_job.project_id,
        project_sequence=old_job.project_sequence,
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
    if retry.project_id:
        projects.add_job(retry.project_id, retry)
    background_tasks.add_task(run_reconstruction, retry.id)
    return with_eta(retry)


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
    return with_eta(updated)


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
    return with_eta(updated)


@app.delete("/api/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str) -> None:
    job = store.get(job_id)
    if job.status in (JobStatus.queued, JobStatus.processing):
        raise HTTPException(status_code=409, detail="任务正在处理中，暂不能删除")
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    store.remove(job_id)


@app.get("/api/v1/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    return with_eta(store.get(job_id))


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job.status != JobStatus.completed or not result_file(job_id).exists():
        raise HTTPException(status_code=409, detail="任务尚未生成结果")
    return FileResponse(result_file(job_id), media_type="model/gltf-binary", filename="model.glb")
