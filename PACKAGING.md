# 打包说明 · 最小可复现集

本文件描述这个目录的来历、内容边界与使用方法。**本目录本身就是项目根目录** ——
把它放到目标机器上任意位置，在其中执行 `./bootstrap.sh` 与 `./start.sh` 即可运行。

> 由 `package_minimal.sh` 生成。生成信息与逐项体量见文末「实测清单」。

---

## 1. 这个包里有什么

按用途分成两部分，两者共用同一套源码：

| 部分 | 目的 | 恢复后能做到 |
| --- | --- | --- |
| **网页前后端** | 打开网页、浏览/比对 17 个模型版本、审批基准 | 立即可用（见 §4） |
| **建模过程** | 从原始视频重跑整条重建主链 | 从视频重新建模（见 §5） |

关键点是：**建模链的根输入是 5 个原始视频，而不是任何中间产物。** 因此这一版把
「重跑所需的全部输入」和「展示所需的全部模型」都放进来了，但**不含**稠密重建的
中间态（那是 238 GB，且完全可由视频重算）。

---

## 2. 目录结构

```text
<本目录>/                      ← 项目根目录
├── PACKAGE-README.md          本文件
├── bootstrap.sh               一键重建 Python 环境 + 依赖校验（★ 先跑这个）
├── start.sh                   启动/停止/查看前后端
├── package_minimal.sh         本包的生成脚本（可重跑）
├── requirements.lock.txt      全量锁定依赖（93 条 ==）
├── requirements.txt           原始宽松约束（5 条）
├── app/                       后端：API、任务调度、COLMAP 编排
├── tools/                     离线工具链：稠密重建、配准融合、网格收尾、质检
├── video_upload.html          前端单页
├── docs/                      公开静态展示页（GitHub Pages 用）
├── README.md / DEVELOPMENT.md 项目说明与开发过程记录
└── storage/
    ├── projects.json          项目与产物索引（★ 网页靠它知道有哪些模型）
    └── <job-id>/              每个视频任务一个目录
        ├── job.json           任务状态与阶段描述
        ├── input.mp4          ★ 原始视频（建模链的根输入）
        ├── normalized_1080p30.mp4  归一化视频（见 §3 说明）
        ├── frames/            抽帧结果（特征提取的真正输入）
        └── pipeline.log       ★ 每阶段执行的完整命令行（参数复现靠它）
external/
├── colmap                     ★ 自编译 CUDA 版 COLMAP 4.3.0.dev0
└── cache-colmap/*.onnx        ★ ALIKED_N16ROT + bruteforce-matcher 权重
```

`external/` 里的东西**不在**项目目录内，所以打包时单独收进来了。它们不合规地缺失会
导致完全不同的重建结果，见 §4.2。

---

## 3. 刻意不包含的内容

| 未包含 | 体积 | 为什么可以不要 | 代价 |
| --- | --- | --- | --- |
| `storage/*/dense*/stereo/{depth_maps,normal_maps}` | **238 GB** | 稠密匹配的中间态，纯由视频+参数决定 | 重跑 `patch_match_stereo`，**约 17 小时/任务** |
| `.venv/` | 3.79 GB | `bootstrap.sh` + `requirements.lock.txt` 可精确重建 | 首次需联网装依赖（含约 500 MB 的 cuDNN） |
| `storage/*/database.db` | 3.2 GB | 特征提取+匹配结果，可由 `frames/` 重算 | 约 20 分钟~1 小时/任务 |
| `storage/*/sparse/`、`dense/sparse/` | 362 MB | ⚠️ 见下 | 无法复现同一套相机位姿 |
| `storage/*/fused.ply` | 3.0 GB | 稠密融合输出 | 必须重跑稠密才能再得到 |
| `storage/*/merge/`、`storage/yellow-cube/` 的非产物部分 | 21 GB | 只保留了网页登记的那 17 个产物文件 | 无法复现人工配准结论 |
| `storage/*/dense*/images/` | 1.8 GB | `image_undistorter` 的输出，分钟级重算 | — |
| `.git/` | 1.2 GB | — | 无提交历史 |

### ⚠️ 三个必须知道的边界

1. **`sparse/` 与 `fused.ply` 不在包里，但 `job.json` 仍写着 `completed`。**
   网页能正常显示这 5 个任务及其状态，但后端在任务目录里找不到 `sparse/`、`fused.ply`，
   所以 **`remesh`（用现成 fused.ply 重出 GLB）和 `resume`（稠密断点续跑）都不可用**。
   要恢复这两项能力，追加 `sparse/` + `fused.ply`（约 3.4 GB）。
2. **重跑不会得到逐位相同的结果。** COLMAP 的 `mapper` 是增量式 SfM，本身带随机性；
   同一份视频重跑会得到在数值上等价、但并非同一个模型。要保住当前这套位姿，必须带 `sparse/`。
3. **人工配准结论无法重算。** 近正方体存在 24 重对称歧义（`fitness`、最近邻距离、六面深度图、
   凹槽掩码 IoU 这些自动判据全部无判别力），最终姿态是**目视选定并批准**的。包内保留了
   这些产物文件本身，但产生它们的过程不可再现。

---

## 4. 解包后如何使用

### 4.1 恢复 Python 环境

```bash
cd <本目录>
./bootstrap.sh --check        # 先看现状缺什么（不改动任何文件）
./bootstrap.sh --mirror       # 再实际安装（国内网络建议加 --mirror）
```

`bootstrap.sh` 会校验 9 项并在失败时给出修复命令。**注意 `--check` 会在缺 `.venv` 时
报错**，那是预期行为——去掉 `--check` 即可自动创建。

### 4.2 恢复外部依赖（★ 必须做，很容易被忽略）

这两项不在项目目录里，但缺了会得到**完全不同的重建结果**：

```bash
# ① 自编译 CUDA 版 COLMAP（54 MB）
mkdir -p ~/.local/bin
cp external/colmap ~/.local/bin/colmap && chmod +x ~/.local/bin/colmap

# ② ALIKED 的 ONNX 权重（2.9 MB）
mkdir -p ~/.cache/colmap
cp external/cache-colmap/*.onnx ~/.cache/colmap/

# 验证（应打印 "4.3.0.dev0 ... with CUDA" 且能列出 ALIKED_N16ROT）
~/.local/bin/colmap -h
```

> **不要 `apt install colmap`**：Ubuntu 自带 3.9.1 且不带 CUDA，既不支持 ALIKED，参数名前缀
> 也还是旧的 `SiftExtraction.*`。若没有这两个文件，`colmap` 会退到 CPU（实测慢 14 倍），
> 或因为缺 cuDNN 直接 abort；ONNX 权重缺失且断网时后端会**静默回退 SIFT**，结果差异极大。

`start.sh` 已把 `~/.local/bin` 前置到 `PATH`，并默认 `MODEL_API_COLMAP_BINARY=~/.local/bin/colmap`。
若装到了别处，用环境变量指定即可（`bootstrap.sh` 也认这个变量）。

### 4.3 启动

```bash
./start.sh start
./start.sh status          # 前端/后端进程、会话是否脱离终端、任务进度
```

- 网页：<http://localhost:5500/video_upload.html>
- API 文档：<http://localhost:8000/docs>

### 4.4 验证清单

| 检查 | 期望结果 |
| --- | --- |
| `./bootstrap.sh --check` | 退出码 0，关键项全 ✓（可能有一条 `/usr/bin/colmap` 的警告） |
| `./start.sh status` | 后端「运行中」且 `TT` 为空（= 已脱离终端），前端「运行中」 |
| 打开网页 | 显示 **17 个模型 · 5 个任务**，「已批准基准」自动加载 |
| 模型可旋转 | 能看清凹槽、贯穿圆孔与六个坐标标记（X+/X-/Y+/Y-/Z+） |
| 切换模型版本 | 点右侧任意卡片可加载对应 GLB（无需刷新页面） |

---

## 5. 从视频重新建模（能力边界）

环境与外部依赖就绪后，网页「导入视频」即可提交新的建模任务。**时间预算（RTX 3080 Ti / 24 GB）**：

| 阶段 | 耗时 | 说明 |
| --- | --- | --- |
| 抽帧 + 特征提取 + 匹配 + 稀疏重建 | 约 1 小时 | 按 3 fps 抽帧，本项目素材为 400~1400 帧 |
| **稠密重建 `patch_match_stereo`** | **约 17 小时** | 主要瓶颈。开 `geom_consistency` 会跑两遍，约 23 秒/视角/遍 |
| 融合 `stereo_fusion` | 约 15 分钟 | 已把 `check_num_images` 从默认 50 降到 10（否则会因内存抖动退化到 44 小时） |
| 网格 Poisson + 收尾 | 1~5 分钟 | 可用 `POST /api/v1/jobs/<id>/remesh` 反复调参，不必重跑稠密 |

**抽帧率直接决定总时长**（`总时长 ≈ 帧数 × 2 × 23 秒`），可用环境变量覆盖：

```bash
MODEL_API_FRAME_FPS=1 ./start.sh restart
```

长任务务必用 `start.sh` 启动（它用 `setsid` 新建会话），否则关掉终端会把正在跑的
COLMAP 一起杀掉——本项目曾被这样坑掉一次 3.5 小时的稀疏重建。

> 若希望跳过这 17 小时、直接复用已有密度成果，需要追加
> `storage/*/dense*/stereo/`（238 GB）—— 详见 §6。

---

## 6. 如何升级到更完整的版本

按需追加，越往下越贵也越省事：

| 追加内容 | 体积 | 解锁的能力 |
| --- | --- | --- |
| `storage/*/sparse/` + `dense/sparse/` | 362 MB | 复现同一套相机位姿；`colmap model_analyzer` 质量诊断 |
| `storage/*/fused.ply` + `.vis` | 3.0 GB | `remesh` 重出网格（约 1 分钟）；各类点云级诊断 |
| `storage/*/merge/` + `storage/yellow-cube/` 全部 | 21 GB | 复现全部中间版本、候选与审计数据 |
| `storage/*/dense*/stereo/` | **238 GB** | 从融合阶段续跑（分钟级），不必重跑 17 小时 |
| `storage/*/database.db` + `dense*/images/` | 5.0 GB | 跳过特征提取与匹配；跳过去畸变 |
| `.venv/` | 3.79 GB | 免装依赖（**仅当目标机也是 Ubuntu 24.04 + Python 3.12** 才可靠） |
| `.git/` | 1.2 GB | 提交历史 |

组合参考：**36 GB = 本包 + `sparse` + `fused.ply` + `merge` + 完整 `yellow-cube` + `database.db`**
（后处理阶段全部可复现，仅稠密仍需重跑）；**273 GB = 再加 238 GB 深度图/法线图**（逐阶段完全复现）。

---

## 7. 常见问题

**Q：`./start.sh start` 之后网页打得开，但模型列表是空的、接口全 404。**
后端没起来。`start.sh` 用 `setsid nohup` 启动，错误被重定向进了 `logs/backend.log`，
所以终端看不到报错。直接看日志，再跑 `./bootstrap.sh --check` 定位缺什么。

**Q：网页显示 `17 个模型` 但点开某个是空白/404。**
`storage/projects.json` 里存的是**相对路径**，必须保持本包的目录结构完整；只有当
某个产物文件缺失时才会如此。缺失项会在 `bootstrap.sh` 之外的场景暴露——
可用后端接口 `GET /api/v1/projects/yellow-cube/artifacts/0` 逐个确认。

**Q：`normalized_1080p30.mp4` 有的和 `input.mp4` 一样大，是重复占空间吗？**
不是。源视频已符合 1080p30 规格时，流水线用**硬链接**跳过转码（零拷贝）；包内保留它是
为了避免目标机用不同版本的 ffmpeg 重新转码而引入差异。

**Q：可以只把这套东西拷到 Windows 上用吗？**
不能。COLMAP、Open3D、cuDNN 与本项目全部工具链都依赖 Linux；WSL2 可以。

**Q：`colmap` 一定要自编译吗？**
目前是。本项目的参数依赖 4.3 的参数命名（`--FeatureExtraction.*` / `--FeatureMatching.*`）
与 ALIKED 的 GPU 支持，Ubuntu 仓库里的 3.9.1 两条都不满足。

---

## 实测清单

<!-- MANIFEST -->
