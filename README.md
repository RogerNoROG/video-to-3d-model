# Video to 3D Model Backend

这是一个可供独立前端接入的 FastAPI 后端。任务处理器已接入 FFmpeg + COLMAP + Open3D：先从视频抽帧，再完成特征匹配、相机定位、稠密点云，最后通过 Poisson 网格重建输出 GLB。

## 启动

```bash
./.venv/bin/python -m pip install -r requirements.txt
PATH="$HOME/.local/bin:$PATH" ./.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

接口文档启动后位于 `http://localhost:8000/docs`。

## API

上传视频并创建任务：

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "video=@./sample.mp4"
```

查询任务状态：

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

获取处理结果：

```bash
curl -OJ http://localhost:8000/api/v1/jobs/{job_id}/result
```

任务中断后续跑稠密阶段（见「中断与续跑」）：

```bash
curl -X POST http://localhost:8000/api/v1/jobs/{job_id}/resume
```

## 接入静态网页

`video_upload.html` 已经接入本 API。它会把视频作为 `video` 字段上传到 `/api/v1/jobs`，轮询任务状态，并在完成后显示 `model.glb` 下载链接。请先启动后端，再在项目目录运行一个静态文件服务器：

```bash
cd /home/rog/video-to-3d-model
python3 -m http.server 5500
```

浏览器打开 `http://127.0.0.1:5500/video_upload.html`。如果后端不在本机 8000 端口，修改网页中的 `API_BASE` 常量。后端已启用 CORS，静态网页和 API 使用不同端口也可以通信。

## 安装重建工具

Python 依赖只负责 API。还需要在系统中安装并确保命令可执行：

- FFmpeg：用于抽帧
- COLMAP：用于特征提取、匹配、相机定位和稠密点云
- Open3D：通过 Poisson 重建把点云转换为三角网格 GLB

可通过环境变量指定非标准安装路径：

```bash
export MODEL_API_FFMPEG_BINARY=/opt/ffmpeg/bin/ffmpeg
export MODEL_API_COLMAP_BINARY=/opt/colmap/bin/colmap
```

输出结果为 `model.glb`，前端可以直接交给 Three.js、Babylon.js 或 `<model-viewer>` 预览。中间点云保存在任务目录的 `fused.ply`，便于排查重建质量。

真实重建是长时间、GPU/CPU 密集型任务。当前实现使用 FastAPI `BackgroundTasks` 便于本地联调；生产环境应拆成 API 服务、Redis/RabbitMQ 任务队列和独立 GPU worker，以支持重启恢复、并发限制、任务重试和 GPU 调度。

## 拍摄规范（决定成败）

摄影测量的前提是**场景刚性静止、只有相机在动**。软件参数无法弥补错误的拍摄方式，请务必按下面执行：

1. **被摄物必须静止**：放在桌面上，或放在转盘上，不要手持被摄物转动。
2. **相机绕物体走**：手持手机/相机绕物体拍一圈或多圈。若必须用转盘，背景必须随物体一起转动（用大张背景纸把转盘和物体一起包住），否则前景与背景运动矛盾会导致重建坍塌。
3. **覆盖足够视角**：拍 2-3 圈，例如平视一圈、俯视 30° 一圈、俯视 60° 一圈，每圈 40-60 张，相邻两张画面重叠 70% 以上。
4. **让物体占满画面**：物体建议占画面 1/2 以上，距离 30-50cm，不要拍进去大量静止背景。
5. **避免运动模糊**：移动要慢且平稳，光线不足时补光而不是硬拍，必要时用三脚架。
6. **光照均匀**：避免逆光和窗户直射，不要让物体一半亮一半暗、也不要出现强反光。
7. **增加纹理**：白墙、纯色、光滑面、木头长条纹理都是低纹理区域。对薄板或光滑物体，贴几张打印了随机花纹的纸作为标记（不要贴规则网格，规则图案会造成误匹配）。
8. **薄片类物体**：立起来拍比平放好，边缘和厚度需要更多斜视角才能重建出来。

拍摄完成后，可在任务目录里检查 `pipeline.log` 中的重建统计：

```bash
grep '重建统计' storage/<job_id>/pipeline.log
```

- 注册图像占比应尽量高（低于 30% 基本无望）
- 稀疏点应有数万以上；只有几千点说明几何太弱，模型必然粗糙
- 每张图的平均观测数应在上千量级
- 日志若出现「生成了 N 个互不连通的稀疏模型」，说明相机运动过快造成了视角断裂。程序会自动选用注册图像最多的那个模型，但这是**拍摄有问题的强烈信号**，应重新拍摄

查看每个稀疏模型的大小：

```bash
for m in storage/<job_id>/sparse/*/; do echo -n "$m "; colmap model_analyzer --path "$m" | grep -E 'Registered images|Points'; done
```

## 中断与续跑（重要）

### 保活机制：为什么用 `setsid`，光有 `nohup` 不够

**后端进程死掉会连带杀死正在跑的 colmap**（BackgroundTasks 与 uvicorn 同生共死），因此千万不要把后端挂在会被关闭的交互式终端下。

`nohup` 只把 `SIGHUP` 设为忽略，进程**仍留在原终端所属的会话里**，可以用 `ps -o sid=` 验证。VS Code 关闭终端时如果定向发 `SIGTERM`，照样杀得掉 —— 本项目被这样坑过一次（退出码 143，667 视角的稠密进度停在 51 个）。`setsid` 会新建会话、脱离控制终端，才真正安全。

`start.sh` 用的是 `setsid` + `nohup` + `< /dev/null`，验证方法：

```bash
ps -o pid,ppid,pgid,sid,tty -p "$(cat logs/backend.pid)"
# TT 为空、且 SID 等于自身 PID = 已真正脱离终端
# 若 SID 与某个 pts/N 的会话号相同 = 仍挂在终端下，有被杀的风险
```

### 日常检查（不需要打开编辑器）

在任意 WSL 命令行窗口里执行：

```bash
cd ~/video-to-3d-model && ./start.sh status
./start.sh watch      # 每 30 秒自动刷新（Ctrl-C 退出，不影响后台）
```

输出示例：

```
后端    : 运行中 (pid 323312)  {"status":"ok"}
  会话  : SID=323312  TT=?  (TT 为空 = 已脱离终端)
前端    : 运行中 (pid 323314)  http://localhost:5500/video_upload.html
colmap  : 运行中 (pid 445961)
任务    : completed  100%  处理完成（注册 667/1335 张图像）  (e7f9f9e5)
模型1    运行中 (pid 445810)   第1遍/共2遍  产物 69/1,326 (5.2%)   2.9/分钟   剩余 427 分钟 (约 05:22)
```

最后一行是**独立稠密构建**（`tools/build_dense_model.py`，例如为第二个稀疏模型建稠密点云）。
它跑在单独的会话里、不属于后端任务，所以由 `tools/dense_status.py` 单独汇报。
进度直接数 `depth_maps` 里的产物文件，速率用最早/最新产物的 mtime 推算 ——
不解析日志，中途重启也能给出正确速率。

`./start.sh` 全部子命令：

| 命令 | 作用 |
| --- | --- |
| `start` | 启动后端 + 前端（已在跑则跳过） |
| `status` | 前后台状态 + 任务进度 + 独立稠密构建进度 |
| `watch` | 每 30 秒刷新 status |
| `restart` | 重启后端并自动续跑未完成任务 |
| `resume` | 只提交续跑，不重启服务 |
| `stop` | 停止后端与前端，**保留**独立稠密构建 |
| `stop-dense` | 显式停止独立稠密构建（产物保留，可断点续跑） |

不想用脚本的话，原始命令：

```bash
pgrep -af 'uvicorn|http.server 5500|colmap'      # 进程在不在
ss -ltn | grep -E ':(8000|5500)'                 # 端口在不在听
curl -s http://127.0.0.1:8000/health             # 后端活着吗
curl -s http://127.0.0.1:8000/api/v1/jobs | python3 -m json.tool | head -n 20
```

`./start.sh status` 会在检测到**多个 colmap 同时运行**时告警——那意味着它们在抢 GPU 和同一批输出文件。

### 续跑

如果任务还是中断了，产物仍在 `storage/<job_id>/`，**不要重试整个任务**（那会重做几小时的稀疏重建）：

```bash
./start.sh restart     # 重启服务并自动续跑所有未完成任务
# 或手动指定：
curl -X POST http://127.0.0.1:8000/api/v1/jobs/<job_id>/resume
```

续跑之所以可行，是因为 `patch_match_stereo` 会**跳过 `depth_maps` 里已存在的视角**，所以它只会补齐剩下的视角，然后继续跑融合 → 网格 → GLB。实测：667 视角中已完成 51 个，续跑后直接从 `Processing view 52 / 667` 开始。

代码上，稠密阶段被抽成可重复调用的 `run_dense_stage()`，正式流程和续跑共用同一份实现。

### 停止时的孤儿进程

`./start.sh stop` 除了杀后端，还会**显式清理遗留的 colmap 进程**。这一步不能省：uvicorn 优雅退出不会杀死 BackgroundTasks 起的子进程，被遗弃的 colmap 会继续跑，与下次启动的 colmap 抢 GPU 并写同一批文件。

### 检查文件有没有被写坏

被杀进程可能在**正在处理的那个视角**留下截断文件。COLMAP 的 `.bin` 深度图是定长格式（大小由图像尺寸决定），所以比对尺寸即可：

```bash
ls -l storage/<job_id>/dense/stereo/depth_maps/*.bin | awk '{print $5}' | sort -u
ls -l storage/<job_id>/dense/stereo/normal_maps/*.bin | awk '{print $5}' | sort -u
```

各应只输出**一个**尺寸（1080p 下深度图 `8209668`、法线图 `24628980`）。出现多个就删掉异常的那个视角的 `.bin` 再续跑，COLMAP 会重算。

## 网格重建参数（出模型质量的关键）

拿到好点云不等于拿到好模型。这一步曾把项目坑得很惨：`fused.ply` 有 1854 万个点、质量很好，出来的却是一个**没有细节的糊块**。两个原因叠加：

### 1. 包围盒被离群点撑大，导致分辨率白降

Poisson 的叶节点尺寸 = **包围盒最长边 / 2^depth**。实测某任务的点云：

| 范围 | 尺寸 | 含点数 |
| --- | --- | --- |
| 原始包围盒 | 29.87 × 22.14 × 13.94 | 100% |
| 1%~99% 分位 | **2.50 × 2.19 × 1.93** | 94.9% |

被测物体只占约 2.5 单位，少数深度估计失误的飞点把包围盒撑到 29.87。同一个 `depth=9`，叶节点从应有的 0.005 变成 0.058 —— **分辨率掉了 12 倍**，物体被抹平。所以建网格前必须裁离群点（`MODEL_API_POISSON_OUTLIER_PERCENTILE`）。

### 2. 用「体积/点数」估点间距会高估一到两个数量级

摄影测量的点云是**表面**分布，不是填满体积。用 $(V/n)^{1/3}$ 估间距，实测得到 0.0792，而真实最近邻距离中位数只有 **0.0011** —— 高估 72 倍。据此推出的 Poisson 深度自然严重偏低。正确做法是**抽样后用最近邻距离中位数**估计（`estimate_point_spacing`）。

### 还有个隐性坑：内存与耗时不匹配

`transfer_colors` 原本用 Open3D 的 `KDTreeFlann` **逐顶点**查询最近点，百万级网格顶点就是百万次 Python 调用。已改为 `scipy.spatial.cKDTree` 批量查询（`workers=-1` 多线程）。

### 实测对比

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 网格顶点 / 三角面 | 38,550 / 76,895 | **1,016,104 / 2,025,551** |
| GLB 大小 | 1.5 MB | **48.3 MB** |
| 内存峰值 | 21.5 GB（swap 打满、抖动） | **2.9 GB** |
| 耗时 | 50+ 分钟未完成 | **54 秒** |

改了网格参数后想重新出模型，**不必重跑稠密重建**（那是几小时）：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs/<job_id>/remesh
```

它直接用现成的 `fused.ply` 重出 GLB，约一分钟。想调参数就先看日志判断当前取值是否合理：

```bash
grep '\[mesh\]' backend.log | tail -n 12
```

### 3. 主体周围残留的散点/薄片

这些来自两处，处理方式完全不同：

**(a) Poisson 长出的互不连通小碎块** —— 已由 `MODEL_API_MESH_MIN_COMPONENT_RATIO`（默认 `0.02`，即只保留不小于最大分量 2% 的连通分量）去掉。实测某次 Poisson 产生 **702 个分量**，过滤后只留 1 个。

**(b) 桌面/窗户等背景被重建成边缘撕裂的薄片** —— 这个**没有简单的自动方法**，因为它往往与主体**拓扑连通**（物体就放在桌面上），连通分量过滤不掉。两条候选路线都试过并失败：

- **RANSAC 移除平面**（`MODEL_API_POISSON_REMOVE_PLANES`）：对**箱体类物体是危险的**。RANSAC 会选中物体自己的一个大平面 —— 实测检测出的"平面"有 179 万内点，正好是 1580 万点的 1/6，即木块的一个面；移除它等于在物体上挖洞，Poisson 再把洞补成撕裂薄片，反而更糟。**默认关闭，且不要盲目调高。**
- **靠密度区分**：不可行。实测桌面的点间距 `0.00088` 比主体的 `0.00115` **更密**（平面比曲面采样更密）。

可控的修边杠杆是 `MODEL_API_MESH_DENSITY_QUANTILE`（默认 `0.02`）：调高会削掉 Poisson 结果里密度最低的部分，能减轻薄片边缘的锯齿，代价是极少量细节。想彻底去掉背景，现实做法是**重新拍摄**：让被摄物占满画面、背景尽量简洁（例如垫一块纯色布、避开窗户），而不是事后靠滤波分离。

**(c) 按颜色直接提取被摄物 —— 目前最有效的方案**

如果被摄物是**暖色**（棕木、陶土、黄铜）而支撑面/背景是**中性色**（白桌面、水泥、天空），可以用 `MODEL_API_MESH_OBJECT_WARMTH` 按 `R-B` 直接筛出被摄物，把背景整体丢掉。这是本项目里**唯一**可靠分离物块与桌面的手段：

| 手段 | 结果 |
| --- | --- |
| 空间聚类（DBSCAN） | 对整片点云只给出**一个簇**，物块与桌面完全连通 |
| 密度 | 桌面 `0.00088` **比物块 `0.00115` 更密** |
| RANSAC 平面 | 命中的是**物块自己的一个大面**（179 万内点 ≈ 物块点的 1/6） |
| **颜色（R-B）** | ✅ 桌面薄片与散点全部消失，物块完整 |

⚠️ **阈值是场景相关参数，无法可靠自动推导**（实测数据）：

- Otsu 自动值 `0.140` **偏大** —— 物块表面有阴影与高光，颜色方差大，阈值取高就削掉暗部（包围盒从 2.11 缩到 1.65，切掉约 35%）
- 用「中性色 ∩ 落在支撑平面上」的点（594,318 个，确定属于桌面）反推阈值**同样失败**：该集合 warmth 均值 0.051、标准差 0.036，**桌面阴影区本身偏暖**，与物块暗部颜色重叠，推得 0.14~0.18 一样切物块
- 也不要在颜色之外再加「远离平面就保留」的几何保护：中性色点里远离平面的部分是**窗外景物**，加了保护等于把背景又救回来

**本场景（白桌面 + 棕木块）实测 `0.08` 最佳**。用法：

```bash
# 设为 -1（默认）关闭；0 = Otsu 自动（需目视确认）；>0 = 显式阈值
MODEL_API_MESH_OBJECT_WARMTH=0.08 ...
```

实测效果：去掉 177 万桌面点，最大连通簇占 99.6%，尺寸 `2.50×2.19×1.93 → 2.11×2.19×1.87`（物块真实约 `2.0×2.19×1.82`）。

## 可配置项

- `MODEL_API_STORAGE_DIR`：文件存储目录，默认 `./storage`
- `MODEL_API_MAX_UPLOAD_SIZE_MB`：单文件大小上限（MB），默认 `8192`（8 GB）。注意视频越大抽帧越多，稠密重建耗时随帧数线性增长
- `MODEL_API_PIPELINE`：处理器名称，当前为 `colmap`
- `MODEL_API_FRAME_FPS`：关键帧抽取频率，默认 `3`；原视频会先统一转码为 1080p/30fps。**耗时的决定性参数**：稠密重建按「注册视角数 × 2 遍」计费，实测 1080p 下约 19 秒/视角/遍，所以 667 个注册视角的稠密阶段约需 7 小时。缓慢环绕拍摄用 `2-3` 即可，不要盲目调高
- `MODEL_API_NORMALIZE_WIDTH`：预处理宽度，默认 `1920`
- `MODEL_API_NORMALIZE_HEIGHT`：预处理高度，默认 `1080`
- `MODEL_API_NORMALIZE_FPS`：预处理帧率，默认 `30`
- `MODEL_API_COMMAND_TIMEOUT_SECONDS`：单条外部命令超时，默认 `86400`
- `MODEL_API_POISSON_DEPTH`：Open3D Poisson 八叉树深度，默认 `0` = **按点云采样密度自动推导**（推荐）。详见「网格重建参数」。
- `MODEL_API_POISSON_MAX_DEPTH`：自动推导时的深度上限，默认 `9`（24GB 内存下的实际上限）。八叉树叶节点数约 `8^depth`，depth 每 +1 内存约 ×8
- `MODEL_API_POISSON_OUTLIER_PERCENTILE`：建网格前按各轴分位数裁掉离群点的百分比，默认 `1.0`。**不要设为 0**：COLMAP 的点云总有少量飞点把包围盒撑大十几倍，会直接把分辨率拖垮
- `MODEL_API_POISSON_VOXEL_SIZE`：建网格前的体素降采样尺寸，默认 `0`（不降采样）。设为点间距的 1~2 倍可大幅降内存且几乎不损细节
- `MODEL_API_POISSON_ORIENT_MAX_POINTS`：超过该点数就跳过全局法线定向，默认 `5000000`。该步骤为每个点建 30 邻域图，是主要内存开销之一
- `MODEL_API_FEATURE_EXTRACTOR`：特征提取器，默认 `ALIKED_N16ROT`（学习型，低纹理/光滑物体比 SIFT 强很多）；也可选 `SIFT`、`ALIKED_N32`、`LOMA`
- `MODEL_API_MATCHER_TYPE`：匹配器，留空则按特征提取器自动选择（`ALIKED_*` → `ALIKED_BRUTEFORCE`，`SIFT` → `SIFT_BRUTEFORCE`）
- `MODEL_API_LEARNED_EXTRACTOR_GPU`：学习型特征是否走 GPU，默认 `true`。需要 cuDNN，缺失时会自动降级为 CPU 而不是崩溃
- `MODEL_API_CUDNN_LIB_DIR`：cuDNN 动态库目录，留空则自动探测虚拟环境里 pip 安装的 `nvidia/cudnn/lib`
- `MODEL_API_USE_GPU`：是否优先让 COLMAP 的 SIFT 特征提取和匹配使用 GPU，默认 `true`
- `MODEL_API_GPU_INDEX`：使用的 GPU 编号，默认 `0`
- `MODEL_API_PREFER_CUDA`：FFmpeg 是否优先使用 CUDA 硬件解码，默认 `true`
- `MODEL_API_ALLOW_CPU_FALLBACK`：CUDA 不可用时是否自动回退 CPU，默认 `true`
- `MODEL_API_STALE_JOB_TIMEOUT_SECONDS`：任务日志多久未更新则判定为异常退出，默认 `3600`。别设得太小：Poisson 网格等阶段可以长时间没有任何日志输出
- `MODEL_API_MIN_REGISTERED_IMAGES`：最大稀疏模型的注册图像数低于该值时判定质量不合格并终止，默认 `20`
- `MODEL_API_SINGLE_CAMERA`：抽帧来自同一台相机，共用一套内参，默认 `true`。关闭后 COLMAP 会为每张图单独建相机，标定被过度参数化、位姿更不稳定
- `MODEL_API_MAX_NUM_FEATURES`：每张图提取的特征数上限，默认 `16384`（同时作用于 SIFT 与 ALIKED）
- `MODEL_API_SIFT_PEAK_THRESHOLD`：SIFT 峰值阈值，默认 `0.00667`；调低可提取更多弱特征，但也会引入更多噪声
- `MODEL_API_SIFT_ESTIMATE_AFFINE_SHAPE`：估计仿射形状，默认 `false`（实测在低纹理物体上收益有限且明显变慢）
- `MODEL_API_SIFT_DOMAIN_SIZE_POOLING`：域尺寸池化，默认 `false`（同上）
- `MODEL_API_SEQUENTIAL_OVERLAP`：顺序匹配的重叠窗口，默认 `15`
- `MODEL_API_GUIDED_MATCHING`：引导匹配，可显著增加匹配点数量，默认 `true`
- `MODEL_API_STEREO_FUSION_MIN_NUM_PIXELS`：融合时每个点被接受所需的最小像素数，默认 `3`，调低点云更密但噪声更多
- `MODEL_API_PATCH_MATCH_MAX_IMAGE_SIZE`：稠密重建的分辨率上限，默认 `-1`（原始 1920×1080）。**这是稠密阶段的主要调速旋钮**：耗时与像素数近似成正比，设为 `1280` 约快 2.2 倍、`960` 约快 4 倍，代价是模型细节减少
- `MODEL_API_PATCH_MATCH_GEOM_CONSISTENCY`：几何一致性过滤，默认 `true`。**开启后稠密阶段会把全部视角跑两遍**（先算 `.photometric.bin`，再做跨视角一致性过滤写 `.geometric.bin`），耗时直接翻倍；换来的是剔除不一致的深度估计，在低纹理表面上明显减少漂浮噪点。追求速度可设 `false`（耗时减半、质量略降）
- `MODEL_API_STEREO_FUSION_MAX_IMAGE_SIZE`：融合阶段的分辨率上限，默认 `-1`，一般与上一项保持一致
- `MODEL_API_MAPPER_BA_USE_GPU`：让光束法平差走 GPU，默认 `false`。可加速 mapper 阶段，但部分 COLMAP 构建不支持该选项

## 查看任务进度

任务共分 8 个阶段，后端通过 `progress`（0/10/20/30/45/60/75/90/95/100）和 `stage` 文案暴露进度。四个长耗时阶段会额外显示 `当前/总数` 的细粒度计数（由后台线程每 2 秒解析 `pipeline.log` 尾部并回写）：

| 阶段 | progress | 细粒度文案示例 | 日志特征串 |
| --- | --- | --- | --- |
| 特征提取 | 30 → 45 | `COLMAP 提取图像特征（ALIKED_N16ROT）（812/1335）` | `Processed file [k/n]` |
| 估计相机位姿 | 60 → 60 | `COLMAP 估计相机位姿（已注册图像）（已完成 480）` | `num_reg_frames=N` |
| 生成稠密点云 | 75 → 90 | `COLMAP 生成稠密点云（光度+几何一致性两遍）（716/1334）` | — （按深度图文件数计） |
| 融合点云 | 90 → 95 | `COLMAP 融合点云（已融合视角）（210/667）` | `Fusing image [k/n]` |

估计相机位姿阶段总数未知（增量式 SfM 是串行试错，无法预知最终能注册多少张），因此进度值固定在 `60`，只更新已注册帧数，避免出现「卡在 87% 永远不动」的假象。

稠密阶段**不解析日志**，而是直接数 `dense/stereo/depth_maps/*.bin`。原因是开启 `geom_consistency` 后 COLMAP 会把 `Processing view 1..N` 跑两遍（先写 `.photometric.bin`，再写 `.geometric.bin`），日志里的视角编号会从 `N/667` 跳回 `1/667`，用它汇报进度会让进度条**倒退**。按文件计数天然单调递增，总数 = `注册视角数 × 2`（关闭 `geom_consistency` 时 × 1）。

四种查看方式：

```bash
# 1. 网页：8 步时间线 + 进度条 + 阶段文案（每 1 秒刷新一次）
#    http://localhost:5500/video_upload.html

# 2. 接口
curl -fsS http://127.0.0.1:8000/api/v1/jobs/<job_id>

# 3. 原始日志（所有外部命令的完整输出）
tail -f storage/<job_id>/pipeline.log

# 4. 稠密阶段按文件数直接数（单调递增，两遍都算在内；总数 = 注册视角数 × 2）
find storage/<job_id>/dense/stereo/depth_maps -name '*.bin' | wc -l
ls -1 storage/<job_id>/dense/stereo/depth_maps | sed 's/.*\.\([a-z]*\)\.bin/\1.bin/' | sort | uniq -c
```

稀疏重建结束后还会在日志里写入一行统计，用于判断拍摄质量：

```bash
grep '重建统计' storage/<job_id>/pipeline.log
```

## GPU 与学习型特征（ALIKED）

本机使用自编译的 CUDA 版 COLMAP（`~/.local/bin/colmap`），系统自带的 `/usr/bin/colmap` 是 CPU 版，启动时用 `MODEL_API_COLMAP_BINARY` 指定前者。

```bash
export MODEL_API_COLMAP_BINARY="$HOME/.local/bin/colmap"
export MODEL_API_USE_GPU=true
export MODEL_API_GPU_INDEX=0
```

学习型特征（ALIKED/LOMA）由 ONNX Runtime 驱动，**启用 GPU 需要 cuDNN 9**。若缺 cuDNN，CUDA provider 会让进程直接 `abort`（不是回退），因此后端会自动探测并降级为 CPU。安装方式（无需 root）：

```bash
./.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nvidia-cudnn-cu12
```

装好后后端会自动把 `.venv/.../nvidia/cudnn/lib` 注入子进程的 `LD_LIBRARY_PATH`。实测速度：ALIKED 在 GPU 上约 **65 毫秒/图**，CPU 约 **900 毫秒/图**（约 14 倍差距）。

特征提取器可切换，`ALIKED_N16ROT` 在低纹理、光滑表面上明显强于 SIFT：

```bash
export MODEL_API_FEATURE_EXTRACTOR=ALIKED_N16ROT   # 默认
export MODEL_API_FEATURE_EXTRACTOR=SIFT            # 无 ONNX 依赖、更轻量
```

首次使用会从 GitHub 下载 ONNX 模型并缓存到 `~/.cache/colmap/`；若下载失败，后端会记录日志并自动回退到 SIFT。

COLMAP 的 `mapper`、`image_undistorter` 以及 Open3D 的 Poisson 网格重建主要使用 CPU，GPU 主要加速特征提取与匹配。稠密重建（`patch_match_stereo`）耗时与注册帧数近似成正比，约 25-30 秒/帧，是整个流程的主要瓶颈。
