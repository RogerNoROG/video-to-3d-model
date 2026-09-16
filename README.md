# Video to 3D Model

将环绕拍摄的视频重建为可预览、可下载的 GLB，并以“项目”而不是“单个视频”为管理单位。一个项目可包含多次扫描、候选合并、审核结果与最终交付；任何候选都不会自动覆盖已批准模型。

## 设计原则

- **项目优先**：`storage/projects.json` 将任务与产物挂到项目下；历史任务目录保持原位即可纳入项目。
- **真实证据优先**：补细节只能使用同一物体、同一坐标系中通过位置、法线、颜色核验的扫描数据，或已批准模型中的真实表面采样。
- **候选先审计**：工具默认写入新的目录并标记 `approved: false`。只有人工视觉确认后，才可更新项目索引中的批准状态。
- **保护真实特征**：孔、弯曲贯穿通道、凹槽与棱不可被通用“补洞”或全局平滑处理。任何自动操作均需显式保护区和审计结果。
- **长任务可恢复**：稠密重建的中间产物、进度与日志保存在任务目录；中断后重跑同一命令即可跳过已完成的深度图。

## 架构

```text
浏览器展示页 (video_upload.html)
            │ HTTP
FastAPI 服务 (app/main.py)
 ├─ 项目索引与候选登记     storage/projects.json
 ├─ 任务调度与状态         storage/<job-id>/job.json
 └─ 视频重建主链           COLMAP → fused.ply → Open3D → model.glb
            │
            ├─ 核心离线流程：稠密重建、增量注册、配准与融合
            └─ 辅助离线流程：质检、六面审计、真实证据精修、网格收尾
```

### 运行时组件

| 位置 | 作用 | 何时使用 |
| --- | --- | --- |
| `app/main.py` | API、上传校验、任务状态、COLMAP 调度、项目/产物索引 | 网页上传、查询、续跑、重新出网格 |
| `app/convert.py` | 点云清理、法线处理、Poisson、颜色转移、GLB 写入 | 被 API 与网格工具共用；不直接维护项目状态 |
| `video_upload.html` | 项目与任务上传、候选预览与下载、产物审核（批准/撤回）、AI 助手（OpenAI 兼容） | 本地展示页 |
| `start.sh` | 持久启动/停止/状态查看；隔离长任务会话 | 本机日常入口 |
| `storage/` | 视频、帧、COLMAP 中间产物、点云、GLB、审计 JSON | 所有可恢复工作数据；不作为代码目录 |

### 离线工具分类

| 分类 | 工具 | 职责与边界 |
| --- | --- | --- |
| 稠密重建 | `tools/pipeline/build_dense_model.py`、`build_dense_supplement.py` | 分别从完整任务或补拍增量工作区生成稠密点云；可断点续跑。|
| 补拍注册 | `tools/pipeline/register_supplement.py` | 常规增量注册新拍帧到既有 COLMAP 模型。|
| 兼容回退 | `tools/pipeline/match_cross_session.py`、`register_by_pnp.py` | 当本机构建的 COLMAP 跨场次匹配失效时，使用互近邻描述子和 PnP；只处理相机位姿，不生成定稿。|
| 配准与融合 | `tools/fusion/merge_models.py`、`merge_two_jobs.py`、`fuse_six_faces.py` | 前者合并同一任务内子模型；后两者处理不同视频/不同扫描。坐标系假设不同，不能互换。|
| 朝向与数据质检 | `tools/fusion/six_face_pipeline.py`、`tools/diagnostics/project_check.py` | 前者审计近立方体的 24 种姿态、六面深度图和异常支撑面；后者汇总任务状态、ASCII 面深度图和颜色清理。|
| 网格收尾 | `tools/refinement/finish_mesh.py` | 点云统一 Poisson、可控密度过滤、边界处理、夹盒与基础平滑，输出 GLB。真实孔槽存在时必须用 `--max-hole-edges -1`。|
| 展示页导出 | `tools/refinement/export_web_model.py` | 对已批准基准做四边形简并，产出 ≤100 MB 的网页版 GLB 供 `docs/` 静态站使用；几何与网格级特征编辑都保留。|
| 真实数据精修 | `tools/refinement/model_refinement.py`、`repair_cube_edges.py`、`extract_verified_face_repair.py` | 用真实表面证据补强局部区域/棱；受保护外平面平滑；或从补拍中导出经过位置、法线、颜色三重核验的面。均只产生候选。|

删除了已审计为非流形的 `hybrid_face_cleanup.py`，以及会以历史硬编码参数自动补洞、可能误伤真实通道的 `night_pipeline.py`。它们的产物保留在 `storage/` 供审计，但不再提供可误用的执行入口。

本轮重构将现存工具脚本从 **17 个减至 14 个**（连同此前已移除的 2 个历史诊断入口，历史入口总数为 19 个）：状态、面特征图、颜色清理已在 `project_check.py` 中统一；真实局部证据与特征保护平滑已在 `model_refinement.py` 中统一。说明文件从 4 份收拢为本 README，历史决策、运行方式和安全边界不再分散维护。

## 快速开始

```bash
./start.sh start
./start.sh status
```

打开 <http://localhost:5500/video_upload.html>，API 文档位于 <http://localhost:8000/docs>。

若当前终端的隔离策略会在会话结束时回收监听进程，请在自己的终端运行上述命令；`start.sh` 会使用独立会话保证正常关闭 VS Code 终端后长任务仍可继续。

想看前后端是**怎么一步步做出来的** —— 架构决策、踩坑记录与取舍理由，见 [`DEVELOPMENT.md`](DEVELOPMENT.md)（前端 / 后端分述，含关键决策一览与已知边界）。

## 网页结构

```text
第一行   项目（选择 / 新建 / 项目卡片）        导入视频
第二行   AI 助手                              视频任务
第三行   当前模型展示（含「模型版本」侧栏）
底部     设置
```

- 项目 API 与任务主链**统一在 8000**（原先另有 8001 的 ETA API，已合并回来），网页不再需要第二个端口。
- 每个模型卡片下方有 **「批准为基准」/「撤回批准」**（`PATCH /api/v1/projects/{id}/artifacts/{index}`）。
  批准是**互斥**的：设某条为批准时会同时撤回其它条目，保证「已批准基准」唯一；
  审核阶段只允许改批准状态，路径与名称不可篡改。
- **AI 助手**使用 OpenAI 兼容接口。底部「设置」里填接口地址、API Key 与可选的模型名，
  保存在浏览器 `localStorage`；聊天的请求经本机后端 `POST /api/v1/assistant/chat` 转发，
  这样既避开第三方接口的浏览器跨域限制，密钥也不会写进任何项目文件。
- 助手**能读项目本身的数据**，不是单纯的聊天转发：
  - 每次请求自动附带一份**项目快照**：项目列表、每个视频任务的状态/进度/阶段、
    全部产物及其批准状态、文件大小与路径（相当于代码聊天框里的仓库摘要）；
  - 另外给一组**只读工具**（OpenAI function calling），它可以按需再查：
    `list_projects`、`get_project`、`inspect_mesh`（顶点数/三角面数/边界边/包围盒/中心）、
    `list_storage`、`read_storage_json`（各项审计 JSON）；
  - 工具**全部限制在 `storage/` 内、且不写任何文件** —— 助手不能批准候选、不能改模型、
    也不能碰 storage/ 之外的路径；
  - 回复下方会列出它实际调用过的工具与参数，方便核对它是否真在“看数据”。

## 主流程

1. 在网页中新建或选择项目并上传视频。服务把任务写入 `storage/<job-id>/`。
2. API 统一视频规格并抽帧，运行 COLMAP 特征、匹配、位姿、稠密深度和融合，生成 `fused.ply`。

### 同一项目的连续补拍

通过网页上传到同一项目的视频会按提交顺序处理。首段视频在融合完成后会从原始
`fused.ply` 中分离目标物体：原始点云保留不动，`project/object_only.ply` 是仅供项目
模型和后续配准使用的主体来源；黄色物块默认采用已验证的暖色阈值 `R-B >= 0.08`，用于
排除灰白背景和支撑面。首段的 `model.glb` 也由这份主体点云导出。

后续视频完成自己的重建后，系统会自动：

1. 生成该视频独立的 `project/object_only.ply`，不会把背景带入合并；
2. 针对立方体的 24 个可能朝向进行特征、颜色与 ICP 对齐评分；
3. 在 `storage/<project>/automation/<job>/` 写出前三个独立 GLB 候选及审计 JSON。

自动流程只生成 `alignment-candidate` 待审核产物，绝不覆盖已批准模型，也不会直接采用
未核验的面。请在网页中检查圆孔、四分之一圆环凹槽和棱的对应关系；确认正确后再将候选
登记为批准版本。原始视频、原始 `fused.ply`、单次 `model.glb` 和所有候选都会保留。
3. Open3D 将点云转为 GLB；任务状态、日志和可下载产物同步更新。
4. 对多次扫描，先用六面朝向审计确认凹槽、孔与木纹属于同一个物理面，再进行跨视频融合。
5. 所有融合/精修输出作为候选登记到项目；人工检查后才允许批准。

### 恢复与重网格

```bash
# COLMAP 稠密阶段中断后：由服务续跑已有工作区
curl -X POST http://127.0.0.1:8000/api/v1/jobs/<job-id>/resume

# 已有 fused.ply，仅重新出网格
curl -X POST http://127.0.0.1:8000/api/v1/jobs/<job-id>/remesh

# 独立稠密重建的状态
./.venv/bin/python tools/diagnostics/project_check.py status
```

不要删除 `queued` 或 `processing` 任务的目录。要停止服务请使用 `./start.sh stop`；它会避免误杀独立稠密重建进程。

## 多扫描融合与特征安全

近似立方体具有多个几何上等价的 90° 旋转，ICP 高分不足以证明物理面正确。融合前必须对候选姿态出图并核验：圆孔、四分之一圆环槽、木纹和棱应同时落在正确的物理面和相对位置。

```bash
./.venv/bin/python tools/fusion/six_face_pipeline.py preview \
  --base storage/<base>/merge/merged_cube_final.glb \
  --source storage/<scan>/merge/object_new.ply \
  --candidates storage/<scan>/merge/merge_two_jobs.json \
  --out storage/<project>/orientation/review

./.venv/bin/python tools/fusion/six_face_pipeline.py face-review \
  --base storage/<base>/merge/merged_cube_final.glb \
  --source storage/<scan>/merge/object_new.ply \
  --candidates storage/<scan>/merge/merge_two_jobs.json \
  --out storage/<project>/orientation/review --ranks 1 2
```

### 真实数据局部补强

`model_refinement.py` 统一了两种精修操作。两者都不能修改批准模型路径。

```bash
# 将批准网格指定区域采样为真实证据，有限权重并入候选点云
./.venv/bin/python tools/refinement/model_refinement.py region-evidence \
  --input candidate.ply --reference approved.glb \
  --box X0 Y0 Z0 X1 Y1 Z1 --samples 300000 --repeats 3 \
  --output review/evidence.ply --audit review/evidence.json

# 对明确避开所有孔、凹槽、棱的外平面做小幅平滑
./.venv/bin/python tools/refinement/model_refinement.py smooth-planes \
  --input review/raw.glb --output review/planes.glb --audit review/planes.json \
  --protect-box X0 Y0 Z0 X1 Y1 Z1
```

对于贯穿孔或弯曲通道，需同时保护两个端口、孔壁与弯折区；建议在平滑前后对两个端口进行射线命中对比。禁止使用扇形补洞、直接拼接网格片或放射线状贴图替代真实数据。

## 当前黄色物块项目

已批准基准：

`storage/e7f9f9e5d046490d93b34e4538bc9ef1/merge/merged_cube_final.glb`

最近的“贯穿圆孔保护与外平面平滑候选”：

`storage/yellow-cube/six-face-v1/feature-and-plane-v1/merged_six_face_feature_protected_planes.glb`

该候选保持 `approved: false`。其审计文件记录了：网格三角形数量未变、约 29 万个外平面稳定顶点被轻微平滑、主通道两端的 2,809 条测试射线命中掩码一致且首个命中点变化为零。

## 性能与资源策略

- COLMAP 稠密重建应单实例运行，默认使用全部 CPU 核；缓存限制为 8 GiB，避免 24 GiB 物理内存下的换页。
- 稠密融合读取大量深度图，`stereo_fusion_check_num_images=10` 是质量与 IO 的平衡；盲目提高到默认 50 会显著增加工作集。
- ICP 与 Poisson 已在原生层多线程，外层同时启动多个重建通常更慢且更容易耗尽内存。
- 六面候选截图可并行，内存充足时可提高 `six_face_pipeline.py preview --workers`；不要与 Poisson 同时运行。
- Poisson 深度受点间距与内存共同约束。该项目通常将深度限制在 9；更高深度不会凭空恢复拍摄中不存在的细节，反而会放大错位毛刺。

## GitHub Pages 展示页（`docs/`）

`docs/` 是一份**纯静态展示页**：只做模型浏览与下载，**不依赖任何后端**。
部署：仓库 **Settings → Pages → Deploy from a branch → Branch `master` + 目录 `/docs` → Save**，
约 1~2 分钟后上线于 `https://<你的用户名>.github.io/video-to-3d-model/`。

```text
docs/
├── index.html                    # 展示页：model-viewer + 模型切换 + X±/Y±/Z± 坐标标记
├── models.json                   # 模型清单：title / note / models[{name, path, note}]
├── .nojekyll                     # 跳过 Jekyll，文件原样发布
└── models/cube-approved-web.glb  # 网页版模型（四边形简并）
```

- **只发布 `docs/`**：`app/`、`tools/`、`start.sh`、`video_upload.html` 都不受影响，
  访客也访问不到它们。本地完整 App 与公网展示页互不干扰。
- **网页版模型是简并版**：GitHub 单文件**硬上限 100 MB**，所以把已批准基准从
  136 MB 压到 36 MB（150 万三角面），几何形状与「贯穿圆孔保护 + 外平面平滑」
  这些**网格级编辑都保留**（简并只减面，不重建）：
  ```bash
  ./.venv/bin/python tools/refinement/export_web_model.py \
      --input storage/yellow-cube/six-face-v1/feature-and-plane-v1/merged_six_face_feature_protected_planes.glb \
      --output docs/models/cube-approved-web.glb --triangles 1500000
  ```
  ⚠️ 不要用 Open3D 原生 `write_triangle_mesh` 写 GLB（不合规），本项目统一用 `app/convert.py` 的 `write_glb`。
- **新增模型**：GLB 放进 `docs/models/`，在 `models.json` 的 `models` 里追加一条，`git push` 后 Pages 自动重新发布。
- **限额与前提**：单文件 ≤100 MB（超了 push 直接被拒）、站点 ≤1 GB、月流量 100 GB 软上限；
  仓库需为**公开**（私有仓库的 Pages 需要付费）。
- **静态站上跑不了的**：上传视频、COLMAP 稠密重建、候选审核、AI 助手 —— 它们都需要本机的
  Python + CUDA 环境与数小时算力，继续用 `video_upload.html` + `./start.sh start`。

## API 摘要

```text
GET   /api/v1/projects
POST  /api/v1/projects
GET   /api/v1/projects/{id}
PATCH /api/v1/projects/{id}
GET   /api/v1/projects/{id}/jobs
GET   /api/v1/projects/{id}/artifacts
POST  /api/v1/projects/{id}/jobs       multipart: video=<file>
POST  /api/v1/projects/{id}/artifacts  {name, path, approved}
PATCH /api/v1/projects/{id}/artifacts/{index}  {approved}   人工审核结论（互斥）
POST  /api/v1/assistant/chat           {endpoint, api_key, model, project_id, messages}
```

`PATCH .../artifacts/{index}` 只改批准状态：设为 true 时会同时撤回其它条目，所以
「已批准基准」始终唯一。

`POST /api/v1/assistant/chat` 是 AI 助手的入口：密钥只在这一次请求里透传、不落盘；
后端会注入项目快照并执行**只读工具循环**（最多 5 轮），返回 `{content, tool_calls}`，
其中 `tool_calls` 记录了模型实际读取过哪些项目数据。

旧的 `/api/v1/jobs`、重试、续跑、重网格与结果下载接口保持兼容。未指定项目的上传会显示在“未归档任务”。

## 验证修改

```bash
# ⚠️ tools/ 下的工具已按用途分到子目录，tools/*.py 匹配不到任何文件（会返回非零码）
./.venv/bin/python -m py_compile app/*.py $(find tools -name "*.py")
./.venv/bin/python tools/diagnostics/project_check.py --help
./.venv/bin/python tools/refinement/model_refinement.py --help
git diff --check
```
