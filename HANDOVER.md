# 迁移与交接说明

> 生成时间：2026-09-12 · 源机器：WSL2 Ubuntu 24.04.5 · RTX 3080 Ti 12GB（驱动 595.97）

## 0. 一页速览

| 项目 | 情况 |
| --- | --- |
| 做什么 | 上传视频 → FFmpeg 抽帧 → COLMAP 特征提取/匹配/SfM/稠密重建 → Open3D Poisson 网格 → 导出 GLB |
| 技术栈 | FastAPI + uvicorn（后端 :8000）、静态 HTML（前端 :5500）、COLMAP 4.3.0.dev0 **CUDA 自编译版**、Open3D 0.19 |
| 代码规模 | `app/main.py`（流程编排 + API）+ `app/convert.py`（网格化 + GLB 写出），共约 170KB |
| 当前状态 | 任务 `e7f9f9e5…` 已完成。最新交付：**`storage/e7f9f9e5…/merge/merged_cube_final.glb`（135 MB，干净的立方体，凹槽 + 贯穿圆孔完整保留）**；流水线自动产出仍是 `model.glb` 36MB |
| 迁移难度 | **中等**。源码可整体拷贝；`.venv` 与 CUDA 版 COLMAP 必须在目标机重建 |
| 迁移体积 | 仅源码 **~250KB** / 带交付产物 **~4.6GB** / 带稠密中间结果 **~47GB** |

**一句话结论**：代码和产物都能搬，唯一有门槛的是 **CUDA 版 COLMAP**。如果目标机器也是 NVIDIA GPU，强烈建议在目标机重编 COLMAP；否则整条流水线的稠密重建会慢 10 倍以上甚至无法运行。

---

## 1. 迁移内容清单

| 路径 | 体积 | 必需性 | 说明 |
| --- | --- | --- | --- |
| `app/` | 168 KB | **必需** | 全部后端逻辑 |
| `video_upload.html` | 32 KB | **必需** | 前端页面 |
| `requirements.txt` | 4 KB | **必需** | Python 依赖 |
| `start.sh` | 8 KB | **必需** | 启动/停止/状态/续跑脚本 |
| `README.md` | 24 KB | 建议 | 完整技术文档（含设计决策与踩坑记录） |
| `HANDOVER.md` | 本文件 | 建议 | 迁移指南 |
| `export_for_migration.sh` | 4 KB | 建议 | 打包迁移文件 |
| `.venv/` | 3.9 GB | **不要拷贝** | 含绝对路径，目标机重建 |
| `logs/` | 1.5 MB | 可选 | 运行日志，出问题时有用 |
| `storage/<job>/model.glb` | 36 MB | **强烈建议** | 最终交付模型 |
| `storage/<job>/fused.ply` | 478 MB | **强烈建议** | 融合点云，`remesh` 只需它，重跑要 7 小时 |
| `storage/<job>/sparse/` | 39 MB | **强烈建议** | 稀疏模型（2 个），重跑要 1 小时 |
| `storage/<job>/database.db` | 687 MB | 建议 | 特征与匹配库，重跑要 1 小时 |
| `storage/<job>/frames/` | 42 MB | 建议 | 抽帧结果（1335 张 jpg） |
| `storage/<job>/input.mp4` | 3.2 GB | 建议 | 原始视频 |
| `storage/<job>/dense/` | **42 GB** | 可选 | 深度图。保留可重跑融合；`remesh` 不需要它 |
| `storage/<job>/fused.ply.vis` | 416 MB | 不要 | COLMAP 可视化用，本项目走 Open3D 用不到 |
| `storage/<job>/normalized_1080p30.mp4` | 99 MB | 不要 | 可由 `input.mp4` 重新生成 |
| `~/.local/bin/colmap` | 56 MB | 机器相关 | 见第 2.2 节 |
| `~/.cache/colmap/*.onnx` | 2.9 MB | 建议 | ALIKED 与匹配器的 ONNX 模型，无网时可预拷 |

打包命令：

```bash
./export_for_migration.sh                # 源码 + 交付产物（约 4.6 GB）
./export_for_migration.sh --with-dense   # 再带上 42GB 深度图
./export_for_migration.sh --source-only  # 只带源码
```

---

## 2. 目标机器环境要求

### 2.1 硬件与系统

| 项 | 要求 | 源机器实测 |
| --- | --- | --- |
| GPU | **NVIDIA，显存 ≥ 8GB**（>12GB 更好；Poisson 网格吃内存但用 RAM 不吃显存） | RTX 3080 Ti 12GB |
| 驱动 | 支持 CUDA 12.x | 595.97 |
| 内存 | **≥ 24GB**。`stereo_fusion` 在 667 视角规模下峰值约 20GB | 24 GB |
| 磁盘 | **≥ 100GB 可用**（单个任务的 dense 目录就 42GB） | 760GB 可用 |
| 系统 | Ubuntu 22.04/24.04（或 WSL2） | Ubuntu 24.04.5 (WSL2) |
| CPU | 8 核以上（mapper、Poisson 主要吃 CPU） | 16 逻辑核 |

### 2.2 机器相关、**不可直接拷贝**的部分

1. **`.venv/`** —— 内嵌绝对路径，必须重建。
2. **CUDA 版 COLMAP** —— 源机器是自编译的 `~/.local/bin/colmap`：
   ```
   COLMAP 4.3.0.dev0 (Commit 8b9936c3 on 2026-09-10 with CUDA)
   ```
   系统 `/usr/bin/colmap` 是 CPU 版（3.9.1），**不要用**，稠密重建会慢 10 倍以上。
   若目标机 GPU 架构与 CUDA 版本一致，可尝试直接拷贝二进制 + 拷贝依赖库；否则重编。
   验证方法：`~/.local/bin/colmap -h 2>&1 | grep -i cuda`，出现 `with CUDA` 才算对。
3. **cuDNN** —— 由 pip 装在 venv 里（`nvidia-cudnn-cu12`），随 venv 重建即可。
4. **ONNX 模型缓存** —— `~/.cache/colmap/`，首次运行需联网下载；无网时预拷这 2.9MB。

---

## 3. 目标机器搭建步骤

```bash
# ---- 1. 解包 ----
mkdir -p ~/video-to-3d-model && tar -xzf migration-*.tar.gz -C ~/video-to-3d-model
cd ~/video-to-3d-model

# ---- 2. 系统依赖 ----
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg git

# ---- 3. Python 虚拟环境（用清华镜像）----
python3 -m venv .venv
./.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
./.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# cuDNN 9：学习型特征（ALIKED）走 GPU 必须要它，缺了会让进程直接 abort
./.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nvidia-cudnn-cu12

# ---- 4. CUDA 版 COLMAP ----
# 方案 A：目标机器 GPU/CUDA 与源机器一致，直接拷贝
mkdir -p ~/.local/bin && cp /path/to/old/colmap ~/.local/bin/colmap && chmod +x ~/.local/bin/colmap
~/.local/bin/colmap -h 2>&1 | grep -i cuda      # 必须输出 "with CUDA"
# 若报缺库，用 ldd 查（例如 ldd ~/.local/bin/colmap | grep 'not found'）

# 方案 B：在目标机器自编译（推荐，最稳）
#   参考 https://colmap.github.io/install.html 的 "Build from source"，
#   CMake 配置务必打开 -DCUDA_ENABLED=ON，并确保能找到 cuDNN。
#   本项目只需用到的子命令：feature_extractor / sequential_matcher / mapper /
#   image_undistorter / patch_match_stereo / stereo_fusion

# ---- 5. 启动 ----
./start.sh              # 后端 :8000 + 前端 :5500，均已脱离终端
./start.sh status       # 查看运行状态与任务进度
# 浏览器打开 http://localhost:5500/video_upload.html
```

### 3.1 启动环境变量

`start.sh` 已内置，无需手动设置。如需调整，参照下表：

```bash
export PATH="$HOME/.local/bin:$PATH"
export MODEL_API_COLMAP_BINARY="$HOME/.local/bin/colmap"
export MODEL_API_USE_GPU=true
export MODEL_API_PREFER_CUDA=true
export MODEL_API_ALLOW_CPU_FALLBACK=false
export MODEL_API_FEATURE_EXTRACTOR=ALIKED_N16ROT
```

---

## 4. 验证清单（一分钟）

```bash
./start.sh status                       # 四项都是「运行中」
curl -s http://127.0.0.1:8000/health    # {"status":"ok"}
~/.local/bin/colmap -h 2>&1 | grep -i cuda          # with CUDA
./.venv/bin/python -c "import open3d, scipy, fastapi, numpy; print('依赖 OK')"
```

若要做端到端冒烟测试，**用一小段视频**（30 秒、缓慢环绕）走一遍全流程，确认能产出 GLB，再上大视频。

---

## 5. 当前任务状态与常用操作

任务 ID：`e7f9f9e5d046490d93b34e4538bc9ef1`（源视频 `1000059185.mp4`）

| 阶段 | 结果 |
| --- | --- |
| 抽帧 | 1335 张 @3FPS |
| 特征 / 匹配 | ALIKED_N16ROT（GPU）+ ALIKED_BRUTEFORCE |
| 相机位姿 | 注册 **1330/1335（99.6%）**，但**分裂成 2 个互不连通的模型** |
| 稠密 | 667 视角 × 2 遍（`geom_consistency=true`），约 7 小时 |
| 网格 | Poisson 自动深度 9，1,465,658 三角面 |
| 交付 | `storage/<job>/model.glb` 36 MB |

常用操作：

```bash
# 改了网格参数后重新出模型（约 1 分钟，不重跑稠密重建）
curl -X POST http://127.0.0.1:8000/api/v1/jobs/e7f9f9e5d046490d93b34e4538bc9ef1/remesh

# 任务中断后续跑稠密阶段（自动跳过已算好的视角）
curl -X POST http://127.0.0.1:8000/api/v1/jobs/e7f9f9e5d046490d93b34e4538bc9ef1/resume

# 看网格各步骤耗时与数量
grep '\[mesh\]' logs/backend.log | tail -n 12
```

---

## 6. 关键设计决策（迁移后不要改错的地方）

1. **Poisson 深度不能硬编码**。叶节点尺寸 = 包围盒最长边 / 2^depth，必须**先裁离群点、再按真实点间距自动定深度**。曾因硬编码 `depth=9` + 包围盒被离群点撑大到 29.87（真实只有 2.50），把 24GB 内存耗尽、swap 打满。
2. **点间距必须用抽样最近邻距离中位数**，不能用 `(体积/点数)^(1/3)` —— 点云是**表面**分布，后者实测高估 72 倍（0.0792 vs 真实 0.0011）。
3. **绝不能用 `sparse/0`**。COLMAP 会生成多个互不连通的模型，代码里用 `select_best_model()` 取注册图像最多的那个。
4. **稠密阶段要按「注册视角数 × 2 遍」估时**。`geom_consistency=true` 时 `patch_match_stereo` 把全部视角跑两遍（先 `.photometric.bin` 再 `.geometric.bin`），只按一遍估会少算一半。
5. **进度上报不要解析 `Processing view k / n`**，因为第二遍编号会从 n 跳回 1，进度条会倒退。代码改为数 `depth_maps/*.bin` 文件个数。
6. **`patch-match.cfg` 是续跑的依据**：`patch_match_stereo` 会跳过 `depth_maps` 里已存在的视角，所以中断后能断点续跑。
7. **不要用 `nohup` 保活，要用 `setsid`**。`nohup` 只忽略 SIGHUP，进程仍在原终端会话里，VS Code 关终端时定向发 SIGTERM 照样杀掉（本项目被坑过一次）。
8. **uvicorn 优雅退出不会杀 BackgroundTasks 的子进程**，colmap 会变孤儿继续跑并与新进程抢 GPU。`start.sh stop` 会显式清理。

---

## 7. 已知未解决的问题

### 7.1 主体周围残留的背景薄片

桌面/窗户被重建成边缘撕裂的薄片，与主体**拓扑连通**（物体就放在桌面），连通分量过滤删不掉。已试过并**失败**的两条路线：

- **RANSAC 移除平面**：对箱体类物体危险。实测检测出的「平面」有 179 万内点 ≈ 木块 1580 万点的 1/6，即**木块自己的一个面**；移除它等于在物体上挖洞。相关开关 `MODEL_API_POISSON_REMOVE_PLANES` 默认关闭。
- **靠密度区分**：不可行。桌面点间距 `0.00088` 比主体 `0.00115` **更密**。

可控杠杆：`MODEL_API_MESH_DENSITY_QUANTILE`（默认 0.02），调高可减轻边缘锯齿。
**根本解法是拍摄时让被摄物占满画面、背景简洁**（纯色布、避开窗户）。

### 7.2 底面缺失（任务分裂）—— 已重建 + 已合并，但存在对称歧义

视频第 670~673 帧之间，木块被手拿起来翻转了一次（第 672 帧手入画、有运动模糊，未注册）。物体相对静止背景换了姿态 ⇒ COLMAP 正确地拆成两段：

| 模型 | 注册帧数 | 帧号范围 | 内容 |
| --- | --- | --- | --- |
| `sparse/0` | 667 | 1 ~ 670 | 朝向 A（已交付） |
| `sparse/1` | 663 | 673 ~ 1335 | 朝向 B，**含底面**（第 1320 帧清晰可见） |

`select_best_model` 只取了 `sparse/0`，因此底面整段被丢弃。

**已完成的工作（2026-09-13）**

1. `sparse/1` 的稠密重建 ✅ —— `tools/build_dense_model.py --model 1`，跑了 8 小时（663 视角 × 2 遍，`geom_consistency=true`）。
   产物 `fused_1.ply`：**20,619,014 点 / 531 MB**。注意该脚本在写完点云后被打断，没写 `finished_at`，点云本身完整（PLY 头声明的点数 × 27 字节 == 文件大小）。
2. 合并工具 ✅ —— `tools/merge_models.py`，把两段点云配准后合并、摆正居中、出 GLB。
3. 产物在 `storage/<job>/merge/`：

| 文件 | 说明 |
| --- | --- |
| `merged_final_c1.glb` | **最终模型（正确解）**，59 MB / 1,291,714 顶点 / 2,437,830 三角面 |
| `merged_final_c1.ply` | 合并后点云 29,865,194 点，底面朝下 + 居中（1453 MB）|
| `merged_final_c2.glb` | 对照用：仅按木纹选出的**错误解**（底面被糊死），可删 |
| `bottom_compare.png` | 从正下方对比 #1 / #2 底面（正确解有圆孔，错误解没有）|
| `candidates_compare.png` | 6 个对称等价候选的对照图（红=A 段 蓝=B 段）|
| `final_c2_natural.png` / `final_c2_closeup.png` | 成品原色环拍预览 |
| `uncovered_dense_1.png` | 配准判据可视化（#1 的无对映区正好连成完整一个面 = A 段缺的底面）|

**⚠️ 已被绕过的坑：木块近似正方体 ⇒ 24 重对称歧义，纯几何分不开**

实测 6 个候选**两两之间的旋转角全是精确的 88.8° / 180.0°**（立方体对称群的特征）：`fitness` 0.721~0.785、细尺度 `rmse` 0.00717~0.00732（点间距 0.0058）、缩放 1.0200~1.0208 —— **全部无区分度**。基于 PCA 的覆盖率、无对映连通分量、平面性等几何判据同样全部失效。

**一度用「木纹颜色一致性」排序，选出了 #2 rand152，但那是错的**：近正方体各面木纹相似，颜色判据会被误导（它给 #2 的 0.1158 反而优于正确解）。**教训：颜色只能当辅助，不能当主判据。**

### 7.3 正确解法：用物理约束一刀切开（已验证）

**关键洞察来自用户：木块在视频里被翻了一面，这是硬物理事实。** 于是正确变换必须满足

$$\mathbf{R}\cdot u_B = -u_A$$

其中 $u_A,u_B$ 是两段各自的**桌面法向**（＝「上」方向，用 RANSAC 拟合物体附近的中性色点得到，内点占比 62~63%）。这一条把姿态从「24 重对称 × 任意三维旋转」压到**只剩 1 个自由度**，直接排除掉 3/4 的对称类。

配合第二个判据 —— **木块底面就贴在桌面上**，所以在桌面平面内、木块投影范围里数点即可：

| 候选 | `R·u_B` 与 `−u_A` 夹角 | 落在桌面平面 + 木块投影范围内的 B 段点数 |
| --- | --- | --- |
| A 段物体单独（基线） | — | 24,498 |
| **#1 rand259（正确）** | **8.0°** ✅ | **135,281** ✅ |
| #2 rand152 | 171.9° ❌ | 44,452 |
| #3 axis0 | 171.9° ❌ | 39,757 |
| #4 rand223 | 172.0° ❌ | 45,249 |
| #5 rand273 | 172.2° ❌ | 46,160 |
| #6 axis9 | 172.2° ❌ | 38,443 |

**#1 把 B 段的底面点正好铺在桌面平面上（13.5 万点，是其余候选 4 万点的 3 倍多）**，并且目视验证：从正下方看，**#1 的底面完整、圆孔清晰；#2 的底面被"糊死"了**（B 的底面被错误地转到了顶面）。

**最终采用 `#1 rand259`。** 产物 `merged_final_c1.glb`（59 MB / 1,291,714 顶点 / 2,437,830 三角面）。

### 7.4 复现步骤与可复用判据

```bash
# 配准（约 6 分钟，结果可复现）
./.venv/bin/python tools/merge_models.py --source dense --random 300 --min-fitness 0.0 --refine-top 6
# 出最终模型（按桌面法向摆正 + 居中 → GLB）
./.venv/bin/python tools/merge_models.py --source dense --random 300 --min-fitness 0.0 --refine-top 6 --final 1
```

**下次遇到「物体被翻面重拍」的通用判据**（按可靠性排序）：

1. **翻转约束** `R·u_B = −u_A`（u = 桌面/支撑面法向）—— 最硬，直接砍掉 3/4 候选
2. **支撑面覆盖**：物体底面贴在支撑面上 → 在该平面上、物体投影范围内数点，正确解会出现一整层点
3. 木纹/颜色一致性 —— 只能当辅助（近正方体会误导）
4. 目视：从支撑面一侧看，正确解的底面完整；错误解该面被别的面"糊死"

**摆正所用的「上」方向**：用**桌面法向**（见上）。不要用相机姿态估重力 —— 手持环绕时相机自身倾斜很大，667 帧的「上」方向与均值夹角中位数达 57°，估出来偏差 45°。

**下次拍摄建议（能彻底消除该问题）**：物体**不要拿起来**，一次连续环绕拍完所有面；或把物体放在一块**带明显纹理/标记的垫板上一起拍**，垫板可作绝对参照。

### 7.5 网格精度：Poisson 深度是唯一瓶颈

**点间距 0.0009，而默认深度 9 的叶节点是 0.0037 —— 粗了 4 倍，白白浪费点云精度。**

同一个 3000 万点合并点云（包围盒 1.78）实测对比：

| 深度 | 叶节点 | 顶点 | 三角面 | GLB | 耗时 | 峰值内存 |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.0037 | 1.29 M | 2.44 M | 59 MB | 82 s | ~3 GB |
| **10** | **0.0017** | **12.2 M** | **24.4 M** | **537 MB** | 268 s | 6.6 GB |

**深度 10 才把木纹的纤维质感和拱形缺口边缘还原出来**（深度 9 只能出"光滑木块"）。24 GB 内存下深度 10 够用；深度 11（叶节点 0.0009 = 恰好等于点间距）预计再 ×4，超出内存。

配套调整：`MODEL_API_MESH_DENSITY_QUANTILE` 从 0.02 降到 **0.005**（原值会削掉细节）；点云已做过连通清理时把 SOR 关掉（`sor_std_ratio=0`）。

⚠️ **别用 matplotlib 散点图判断网格质量** —— 随机采样顶点会把表面画得很毛，看起来像噪声。用浏览器的 `<model-viewer>` 看真实网格。

### 7.6 背景薄片与主体「物理相连」时的清理

残留薄片是 **B 段的桌面**：物体在 B 段是顶面朝下放在桌上，桌面正好接在物体顶面边缘 → 与主体**物理相连**，点云级和网格级的连通分量都切不掉（Poisson 只会把它焊得更牢）。

**有效解法：按被摄物自身的外形裁。** 木块是规整方块，每个面占全部点约 1/6 ≫ 0.5%，所以**逐轴 0.5% / 99.5% 分位就是木块真实的六个面**，超出即背景：

```python
low  = np.percentile(points, 0.5,  axis=0) - spacing * 20
high = np.percentile(points, 99.5, axis=0) + spacing * 20
keep = np.all((points >= low) & (points <= high), axis=1)
```

实测裁掉 0.30 %（88,372 点），包围盒 1.776 → 1.488（薄片把盒撑大了 0.29）。

**清理顺序（跳过任一步都会在成品上留下撕裂薄片）：**

1. 点云级连通分量（体素 + 26 邻域 + `scipy.sparse.csgraph.connected_components`）——清掉真正分离的碎块，实测 16,641 个分量只留 1 个。已封装为 `app.convert.keep_largest_point_component`
2. 按被摄物外形裁剪 —— 清掉与主体相连的背景薄片
3. Poisson 重建

### 7.7 网格收尾：补洞 + 轻度平滑（`tools/finish_mesh.py`）

**问题**：成品上出现撕裂缺口。**但先别急着补洞 —— 先查洞是哪来的。**

### 根因：密度过滤在表面上戳洞

实测同一次 Poisson 输出：

| `density_quantile` | 开口数 | 边界边 |
| --- | --- | --- |
| **0（不过滤）** | **2** | **96** |
| 0.0015 | 10 | 1,353 |
| 0.005（旧默认） | 39 | 3,800 |

**Poisson 原始输出本来就是封闭的**（depth 9 只有 96 条边界边）。是「剔除最低 X% 密度的顶点」这步在表面上戳出了成千上万个小洞。

⚠️ **盲目把洞全补上会把物体的真实特征焊死**：低密度区恰好是**凹槽内壁、圆孔口**这些真实但点少的位置。实测把 520 个开口全补之后，木块的半圆凹槽和圆孔口都被扇形补片堵住了，一眼就能看出来。

**正确顺序**：查洞来源 → 关掉/调小密度过滤（优先）→ 剩下少量小洞再补 → 补完重算法线。

密度过滤大小的权衡（同一个点云实测）：

- `0`：特征完好，但 Poisson 在乱点处会长出**流状薄片**（没有密度过滤清理）
- **`0.0015`：推荐**，只需补 10 处小洞，特征完好
- `0.005`：顶部干净，但补片把凹槽/圆孔口堵住 ❌

### 补洞实现（`tools/finish_mesh.py`）

找只被 1 个三角形使用的边 → 按顶点连通性分成一个个开口 → 每个开口加一个中心顶点、用扇形三角面封上。

- ⚠️ 补的面必须**反向遍历共用边**（原三角形用 a→b，补的面必须用 b→a），否则法线会翻
- ⚠️ **补完必须重算顶点法线**。只动拓扑不算法线会让整块模型明暗全错（实测过一次，整块变深色沙粒）
- `--max-hole-edges N` 可只补小洞；实测大开口留空会露出白色内壁，更难看
- 然后用 Taubin 平滑（默认 3 次）抹掉重建噪声；顶点颜色不参与平滑，木纹颜色纹理不受影响

**实测效果**（30 M 点合并点云，包围盒 1.49，depth 9）：

| 配置 | 顶点 | 三角面 | 开口 | GLB |
| --- | --- | --- | --- | --- |
| `density_quantile=0` | 2.07 M | 4.14 M | 补 2 处 | 101 MB |
| **`density_quantile=0.0015`（推荐）** | **2.06 M** | **4.13 M** | **补 10 处** | **100 MB** |
| `density_quantile=0.005` | 2.05 M | 4.10 M | 补 39 处（堵特征）| 100 MB |

**深度怎么选**：两段扫描之间约 0.0056 的残余错位是数据本身的精度上限。Poisson 叶节点比它还细时（深度 10 → 0.0015）会把两层表面之间的缝"雕"成薄片毛刺。**深度 9（叶节点 0.0029）是更稳的选择**，细节仍远超原始版本，且体积只有 1/10。

**试过但无效/更差的路子**（别再走）：

- **统一法线朝向再重建**：在 441 万点降采样云上跑 `orient_normals_consistent_tangent_plane(k=20)`，再把符号传回全量点云。结果边界边反而更多（7,052 条）、背面出现大片撕裂薄片 → **比原法线更差**
- **只补小洞、大开口留空**：大开口会露出白色内壁，更难看
- **PCA 求箱体朝向做覆盖率判据**：近正方体上主轴不确定，结果全错

**仍有的一处瑕疵**：左上角（凹槽出口）有一块补片呈放射状"星芒"。该处点云本身就是**多层乱结构**（B 段的桌面薄片贴着物体，裁掉外侧后内层还留着），Poisson 无法判断表面在哪。要根治只能重拍。

---

### 7.8 拿到「干净的立方体」：先对正方位，再按外形裁剪（已定稿）

**先搞清楚实物长什么样**（直接看 `storage/<id>/frames/frame_000200.jpg`、`frame_000900.jpg`）：

- 浅色**榉木**立方体，木料自带细斑点纹理 —— 成品上的"砂纸感"有一部分是**真实木纹**，不全是噪声
- 正面有一个**大贯穿圆孔**；顶面那条"凹槽"就是这个孔在顶面的开口（半圆柱）
- 桌面是**浅灰白台面** → 重建里所有奶白色薄片都是它

**坑 1：物体在模型里是歪的 9.6°**

`stand_upright` 只把重力方向摆正，水平方位靠 PCA 长轴 —— 而近正方体的主轴是不确定的，实测偏了 **9.6°**。
按坐标轴分位裁剪 = 斜着切，面会被削掉一条。

正确做法（`最小包围矩形`）：把所有点投到水平面，旋转扫描 0~90°（步长 0.1°），取**包围矩形面积最小**的角度。

```
最优方位 9.60°   水平截面 1.2962 × 1.3022
三向长度 [1.2996 1.2919 1.2758]   长宽高比 [1. 0.982 0.994]   ← 近乎完美立方体
容差 0.0054   保留 29,369,136 / 29,733,382   删掉 1.23%
```

扫描只用 100 万点子集（对 3000 万点做 3600 次分位数要跑几小时），求出角度后再用全量点重算。

**坑 2：验证"裁干净了没有"**

在立方体框架下逐面统计面外的点数：

```
X- 外侧  98,858 点 (0.34%)   范围 -0.6586 ~ -0.6497   （面在 -0.6477，即面外 0.002~0.011）
X+ 外侧  89,821 点 (0.31%)
Y- 外侧  92,026 点 (0.31%)
Y+ 外侧  66,778 点 (0.23%)
Z- 外侧  82,426 点 (0.28%)
Z+ 外侧  83,091 点 (0.28%)
```

**六面外只有 0.23~0.34%，且全部落在容差带内** → 立方体外没有大的离群块，裁剪是成功的。

**坑 3：`density_quantile` 还有个隐藏作用 —— 剔掉"透过圆孔看到的台面"**

B 段是翻转拍摄，**B 的台面正好落在木块顶面高度**；透过圆孔看到的台面被重建成一坨**奶白色块**，在成品上非常显眼。

- 它与主体在点云里是**连通的** → 连通分量删不掉（实测只删得掉 0.80%）
- 用预降噪也不行：`voxel_down_sample(0.003)` 会把它**糊成更大的一整块**，比不降噪更难看
- **`density_quantile=0.005` 恰好把它滤掉** —— 所以密度过滤不只是"磨皮"，还是**去外来几何**的手段

**定稿配方**

```bash
./.venv/bin/python tools/finish_mesh.py \
  --input storage/e7f9f9e5d046490d93b34e4538bc9ef1/merge/merged_final_c1_cube.ply \
  --output storage/e7f9f9e5d046490d93b34e4538bc9ef1/merge/merged_cube_final.glb \
  --depth 9 --density-quantile 0.005 --max-hole-edges 100
```

→ **`merged_cube_final.glb`（135 MB）**：凹槽和贯穿圆孔完整保留、面平棱直、孔里没有台面残留。
边界边 4,869 → 只补 1,644，**故意留 3,225 条大开口不补**（就是凹槽/圆孔的口，本来就该开着）。

对照版本（同目录，可在网页上切换）：
- `merged_final_cube_b.glb`：`--density-quantile 0` → 完全封闭（0 条边界边），但孔内奶白台面残留很明显（网页上的"对照"按钮就是它）
- `merged_final_cube_c.glb`：深度 8，仅 24 MB，细节少一档

**试过但更差、别再走的路子**（生成物已删除，仅保留结论）：
- 体素预降噪（`--pre-voxel 0.003`）+ 密度过滤 0.002 → 异物被糊成更大的整块
- 体素预降噪 + 无密度过滤 + Taubin 20 次 → 同上，且砂纸感没压住
- 点云级连通分量清异物 → 只能删掉 0.80%，异物与主体是连通的

**浏览器对比预览的正确用法**：URL 形如
`http://localhost:5500/video_upload.html#model=<路径>|<标签>&model=<路径>|<标签>`
- 改完 URL **必须再按一次刷新**（只改 `#fragment` 不会重跑脚本）
- 远端开发时端口转发会把 `#` 里的 `=`、`&` 重新编码成 `%3D`、`%26`，前端已做兼容（整串解码后再切分）

---

## 8. 关键参数速查

| 环境变量 | 默认 | 作用 / 调参建议 |
| --- | --- | --- |
| `MODEL_API_FRAME_FPS` | 3 | 抽帧频率。**耗时的决定性参数**，稠密按「注册视角 × 2 遍 × 19 秒」计 |
| `MODEL_API_PATCH_MATCH_GEOM_CONSISTENCY` | true | 开则稠密跑两遍，耗时翻倍换质量 |
| `MODEL_API_PATCH_MATCH_MAX_IMAGE_SIZE` | -1 | 稠密分辨率上限，`960` 约快 4 倍 |
| `MODEL_API_POISSON_DEPTH` | **0（自动）** | 0 = 按点云密度自动推导，**不要硬编码** |
| `MODEL_API_POISSON_MAX_DEPTH` | 9 | 深度上限（内存约束），24GB 内存下 9 是上限 |
| `MODEL_API_POISSON_OUTLIER_PERCENTILE` | 1.0 | 建网格前按分位数裁离群点，**不改会白降分辨率** |
| `MODEL_API_MESH_MIN_COMPONENT_RATIO` | 0.02 | 连通分量过滤碎块（0.02 = 保留不小于最大分量 2% 的） |
| `MODEL_API_MESH_DENSITY_QUANTILE` | 0.02 | 削掉 Poisson 最低密度部分，调高可修边 |
| `MODEL_API_STALE_JOB_TIMEOUT_SECONDS` | 3600 | 无活动多久判定失败。**别设小**，Poisson 阶段可以长时间零日志 |
| `MODEL_API_MAX_UPLOAD_SIZE_MB` | 8192 | 上传上限 |
| `MODEL_API_MIN_REGISTERED_IMAGES` | 20 | 注册数低于此值判定质量不合格 |

完整清单见 `README.md` 的「可配置项」。

---

## 9. 迁移建议

**推荐做法**：只带源码（250KB）+ 交付产物（4.6GB），在目标机按第 3 节重建环境。
不要把 42GB 的 `dense/` 带过去 —— 它只影响「重跑 stereo_fusion」，而 `remesh` 只需要 `fused.ply`（478MB）。

**如果目标机器没有 NVIDIA GPU**：整条稠密重建路径不可行（`patch_match_stereo` 的 CUDA 版是刚需，CPU 版慢 10 倍以上）。此时只能跑稀疏重建，或改走其他 MVS 方案。

**如果目标机器是 Windows 原生环境**：`start.sh` 依赖 bash、`setsid`、`pgrep`，需要改写启动脚本；COLMAP 需要 Windows 版 CUDA 构建。
