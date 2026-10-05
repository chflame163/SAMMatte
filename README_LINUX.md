# SAMMatte Linux 部署说明（Ubuntu 24.04 + RTX 3090）

本文档只讲 Linux 服务器部署。Windows 侧仍使用 `run_SAMMatte.bat` / `check_runtime.bat`，
并新增了 `setup_windows.bat` / `setup_windows.ps1`（一键创建项目内 `venv\`）；
这两个启动/自检脚本现在会优先使用 `<项目>\venv\Scripts\python.exe`，没有 venv 时回退到 PATH 中的 `python`。

已在内网一台 Ubuntu 24.04.4 服务器（Python 3.12.3，驱动 610.57.04，RTX 3090 24G）上完成部署与验证。

---

## 1. 目录与运行环境

```text
/mnt/nvme0n1p1/video_segment/SAMMatte/
├─ app/                      WebApp 源码（含新增 multipart.py）
├─ models/                   sam3 / sam3.1 / vitmatte / VideoMaMa
├─ tools/ffmpeg_linux/       Linux 静态 ffmpeg + ffprobe（新）
├─ venv/                     项目专用虚拟环境（新）
├─ logs/                     后台运行日志（新）
├─ run_SAMMatte.sh           前台启动（新）
├─ start_sammatte.sh         后台启动（新）
├─ stop_sammatte.sh          停止（新）
├─ check_runtime.sh          环境自检（新）
├─ setup_linux.sh            一键重建环境（新）
└─ requirements.linux.txt    实测版本锁定（新）
```

要点：

- 不需要 root/sudo。venv、ffmpeg、缓存全部在项目目录内。
- Ubuntu 若缺少 `python3-venv`（`ensurepip` 不可用），`setup_linux.sh` 会自动用 `--without-pip` + `get-pip.py` 引导 pip。
- `sam3` 源码通过 `PYTHONPATH` 引入（不 pip 安装），与 Windows 版一致。

## 2. 一键搭建（重装 / 换机器时用）

```bash
cd /mnt/nvme0n1p1/video_segment/SAMMatte
bash setup_linux.sh
```

可选覆盖变量：

```bash
PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
TORCH_VARIANT=cu126 TORCH_VERSION=2.7.1 TORCHVISION_VERSION=0.22.1 \
bash setup_linux.sh
```

脚本会：创建 venv → 装 torch 2.7.1+cu128 / torchvision 0.22.1+cu128 → 装 `requirements.txt` →
把 `diffusers / accelerate / timm / ftfy / opencv-python` 对齐到 Windows 已知可用版本
（VideoMaMa 依赖 `diffusers.pipelines.stable_video_diffusion` 内部符号，版本差异最容易出问题）→
下载静态 ffmpeg 到 `tools/ffmpeg_linux` → 最后自动执行 `check_runtime.sh`。

当前服务器上的实测版本见 `requirements.linux.txt`。

## 3. 启动 / 停止

后台启动（推荐，SSH 断开也不会退出；自动等待健康检查）：

```bash
cd /mnt/nvme0n1p1/video_segment/SAMMatte
bash start_sammatte.sh      # 监听 0.0.0.0:8765
bash stop_sammatte.sh
tail -f logs/sammatte.log
```

前台启动（调试用，Ctrl+C 退出）：

```bash
bash run_SAMMatte.sh
```

访问地址（其他电脑可直接访问）：

```text
http://<服务器IP>:8765
```

换端口 / 只绑本机：

```bash
SAM31_PORT=8790 SAM31_HOST=127.0.0.1 bash start_sammatte.sh
```

端口约定：默认 `8765`，**不要**使用 `8080`（服务器上已有服务占用）或 `8188`。

## 4. 环境自检

```bash
bash check_runtime.sh
# 同时列出占用显存的进程：
SAMMATTE_SHOW_GPU_PROCS=1 bash check_runtime.sh
```

自检会打印：Python/venv 路径、torch 与 CUDA 运行时、各依赖版本、模型文件是否存在、
`sam3` 与 VideoMaMa `pipeline_svd_mask` 能否导入、ffmpeg 位置与版本、以及 `nvidia-smi` 的显存余量。

## 5. ffmpeg

Linux 下 ffmpeg 查找顺序（`app/sam31_webapp/service.py: resolve_ffmpeg_path()`）：

1. 环境变量 `SAM31_FFMPEG` 指定的可执行文件
2. `tools/ffmpeg_linux/ffmpeg`（本项目自带的静态构建，含 `ffprobe`）
3. `tools/ffmpeg/ffmpeg`
4. 系统 `PATH` 中的 `ffmpeg`
5. `imageio-ffmpeg` 包自带的 ffmpeg（兜底）

Windows 的 `tools/ffmpeg/*.exe` 在 Linux 上不会被使用，留着不影响运行。

## 6. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SAM31_HOST` | `0.0.0.0`（脚本）/ `127.0.0.1`（程序） | 绑定地址 |
| `SAM31_PORT` | `8765` | 监听端口 |
| `SAMMATTE_VENV` | `<项目>/venv` | 虚拟环境目录 |
| `SAM31_FFMPEG` | 空 | 手动指定 ffmpeg |
| `SAM31_MAX_INFERENCE_PIXELS` | `1920x1080` | SAM 推理像素上限，显存紧张时改成 `1280x720` |
| `SAM31_VITMATTE_DEVICE` | `gpu` | 可设 `cpu` 省显存 |
| `SAM31_VIDEOMAMA_MAX_RESOLUTION` | `1024` | 显存紧张时降到 `512` |
| `SAM31_VIDEOMAMA_CHUNK_FRAMES` | `0`（自动） | 手动限制 VideoMaMa 每次处理的帧数 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`（脚本设置） | 缓解显存碎片 |

## 7. 显存要求（重要）

服务器上的 3090 目前被其他任务（`/mnt/nvme0n1p1/Strata`，约 22.4 GiB）常驻占用，
剩余仅约 250 MiB，**不足以初始化 CUDA**，此时上传视频会直接返回 `CUDA error: out of memory`。

开始调试前请先释放显存，例如：

```bash
nvidia-smi                                   # 查看占用
SAMMATTE_SHOW_GPU_PROCS=1 bash check_runtime.sh
```

释放显存后再 `bash stop_sammatte.sh && bash start_sammatte.sh`（让进程以干净的 CUDA 状态启动）。

经验值（24G 单卡）：

- SAM 3.1 传播：约 6–10 GiB（与推理像素上限、帧数相关；显存不足时程序会自动把状态 offload 到 CPU）
- ViTMatte 精修：约 1–2 GiB（可选 `cpu`）
- VideoMaMa 精修：约 10–14 GiB（fp16，SVD 底座 + unet）
- 三种后处理模式（binary / vitmatte / videomama）互斥使用，程序会在 VideoMaMa 前释放 SAM 显存

## 8. Linux 移植改了什么

1. `app/sam31_webapp/multipart.py`（新增）：流式 `multipart/form-data` 解析，替换
   `cgi.FieldStorage`（`cgi` 在 Python 3.11 起废弃、3.13 已删除）。大文件走
   `SpooledTemporaryFile` 落盘，不整包读进内存；上传结束统一关闭临时文件。
2. `app/sam31_webapp/service.py`：
   - 新增 `resolve_ffmpeg_path()` / `describe_ffmpeg()`，不再硬编码 `"ffmpeg"`；
   - 新增 `describe_gpu_status()`，用 `nvidia-smi` 报告显存余量（不申请显存、不会干扰其他任务）；
   - `reencode_h264()` 使用解析出来的 ffmpeg 路径。
3. `app/sam31_webapp/app.py`：
   - 上传改用新解析器；
   - `--host/--port` 默认读取 `SAM31_HOST/SAM31_PORT`；
   - 监听 `0.0.0.0` 时打印局域网访问提示；端口被占用时给出明确错误；
   - headless（无 `DISPLAY`/`WAYLAND_DISPLAY`）时跳过自动开浏览器；
   - 启动日志打印 Python/torch/CUDA/GPU/ffmpeg 信息。
4. 新增 `run_SAMMatte.sh`、`start_sammatte.sh`、`stop_sammatte.sh`、`check_runtime.sh`、
   `setup_linux.sh`、`requirements.linux.txt`。
5. 清理了 `app/`、`models/` 下从 Windows 带过来的 `__pycache__`（cpython-311/312 旧字节码）。

模型权重、`models/sam3` 源码、`models/VideoMaMa` 源码本身无需改动（不含任何 Windows 专用代码）。

## 9. 已验证 / 待你在服务器实测

已在 Linux 服务器上验证通过（不依赖显存的部分）：

- venv + 全部依赖安装；`torch 2.7.1+cu128` 识别 CUDA，驱动 610.57.04
- `sam3.model_builder`、VideoMaMa `pipeline_svd_mask`、ViTMatte（CPU 加载）均可导入/加载
- 服务启动、绑定 `0.0.0.0:8765`，`/api/health`、首页、`/static/*`、Range 请求、404/403 路径正常
- 从另一台电脑通过 `http://<服务器IP>:8765` 完成上传：multipart 解析、视频探测、
  缩放生成 SAM 推理视频（`934x986`）全部成功
- ffmpeg 重编码链路：`mp4v` → H.264（`ffprobe` 确认 `codec_name=h264`、帧数一致）
- 显存不足时错误被正确捕获并返回 JSON，服务不崩溃，可继续请求
- `start_sammatte.sh` / `stop_sammatte.sh` 停止再启动流程正常

需要释放显存后由你实测的部分（会真正申请显存）：

- 点选 / 框选 / 文字提示的当前帧预览
- 双向传播与关键帧传播
- `binary` / `vitmatte` / `videomama` 三种遮罩后处理
- 叠加预览与遮罩预览生成、H.264 遮罩视频导出
