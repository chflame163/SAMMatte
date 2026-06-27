# SAMMatte

`SAMMatte` 是一个本地运行的 Web 应用，用于基于 SAM 3.1 的视频目标分割与遮罩传播。

这个仓库只包含应用源码与启动脚本，不包含 Python 运行时、模型权重或 ffmpeg。使用前需要自行准备外置 Python 环境，并下载所需模型文件。附件可以按下面的官方链接分别下载，也可以通过百度网盘一次性下载。

## 1. 环境要求

- Windows 10 或 Windows 11
- NVIDIA GPU
- 正常可用的 CUDA 驱动
- Python 3.12，已按 `requirements.txt` 配置好的外置 Python 环境
- Git
- ffmpeg

说明：

- `requirements.txt` 不包含 `torch` 和 `torchvision`，因为它们需要和你的 CUDA/CPU 环境匹配。
- 如果你不打算使用 `VideoMaMa` 精修，可以只下载 `SAM 3.1` 和 `ViTMatte`；但在开始传播前，需要在页面里把后处理模式切换成 `binary` 或 `ViTMatte`。


## 2. 配置 Python 环境

创建本地python环境。以conda为例：
```powershell
conda create -n sammatte python=3.12
conda activate sammatte
```

安装和本机 CUDA 匹配的 PyTorch。下面是 CUDA 12.8 的示例：

```powershell
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

然后安装其余依赖：

```powershell
pip install -r requirements.txt
```

如果你的 CUDA 版本不同，请使用 PyTorch 官方安装页重新选择命令：

- https://pytorch.org/get-started/locally/

## 3. 下载模型源码与权重

模型和工具文件支持两种获取方式：

1. 通过百度网盘一次性下载完整附件
2. 按官方来源分别下载

如果你希望最快完成部署，推荐直接使用百度网盘方式。下载后请严格按下面的目录结构放置，不要多套一层目录。

### 3.1 百度网盘一次性下载

`tools` 下载链接：

- https://pan.baidu.com/s/1EGBretLiMPv7cKqQmSH-4Q?pwd=tyc5

`models` 下载链接：

- https://pan.baidu.com/s/1vzXiKtrlzjmkeuCc6z2CEg?pwd=9u2x

说明：

- `tools` 压缩包中包含 `ffmpeg` 相关文件。
- `models` 压缩包中包含本项目需要的全部模型文件。
- 下载后请把 `tools` 和 `models` 目录直接放到 `SAMMatte` 根目录下，保持相对路径不变。


放置完成后，目录应类似于：

```text
SAMMatte/
├─ app/
├─ models/
│  ├─ sam3/
│  ├─ sam3.1/
│  │  └─ sam3.1_multiplex.pt
│  ├─ vitmatte-base-composition-1k/
│  └─ VideoMaMa/
│     ├─ model/
│     │  ├─ stable-video-diffusion-img2vid-xt/
│     │  ├─ unet/
│     │  └─ dino_projection_mlp.pth
│     ├─ pipeline_svd_mask.py
│     └─ ...
├─ tools/
│  ├─ ffmpeg/
│  │  ├─ ffmpeg.exe
│  │  ├─ ffplay.exe
│  │  └─ ffprobe.exe
│  └─ ffmpeg_extract/
├─ check_runtime.bat
├─ run_SAMMatte.bat
└─ requirements.txt
```

### 3.2 按官方来源分别下载

#### 3.2.1 SAM 3 源码仓库

下载地址：

- https://github.com/facebookresearch/sam3

#### 3.2.2 SAM 3.1 权重

下载地址：

- https://huggingface.co/facebook/sam3.1


#### 3.2.3 ViTMatte 权重

下载地址：

- https://huggingface.co/hustvl/vitmatte-base-composition-1k


#### 3.2.4 VideoMaMa 源码与权重

源码仓库：

- https://github.com/cvlab-kaist/VideoMaMa

VideoMaMa 权重：

- https://huggingface.co/SammyLim/VideoMaMa

基础模型：

- https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt


## 4. 配置 ffmpeg

推荐两种方式任选其一：

1. 通过上面的百度网盘下载完整 `tools` 目录，并放到 `SAMMatte\` 根目录下
2. 直接把 ffmpeg 安装到系统 `PATH`

如果没有 ffmpeg，应用仍可能启动，但 H.264 预览和导出重编码会失败。

## 5. 启动

建议先检查环境：

```powershell
.\check_runtime.bat
```

然后启动：

```powershell
.\run_SAMMatte.bat
```

默认访问地址：

```text
http://127.0.0.1:8765
```

如果端口冲突：

```powershell
$env:SAM31_PORT=8766
.\run_SAMMatte.bat
```

如果希望让局域网内其他设备访问当前 WebApp：

```powershell
$env:SAM31_HOST=0.0.0.0
$env:SAM31_PORT=8765
.\run_SAMMatte.bat
```

此时请使用运行机器的局域网 IP 访问，例如：

```text
http://192.168.1.10:8765
```

## 6. WebApp 使用流程

### 6.1 上传视频

- 启动后可以把视频文件直接拖入首页，也可以点击“选择视频”按钮。
- 上传完成后，页面会显示视频分辨率、总帧数、帧率，以及当前 SAM 推理分辨率。
- 如果之前已经打开过一个视频，再次上传会自动替换为新视频。

### 6.2 浏览帧和查看画面

- 可以通过“上一帧 / 下一帧”逐帧浏览。
- 可以拖动时间滑块快速跳转到任意帧。
- 也可以直接在“当前帧”输入框中输入帧号跳转。
- 右上角提供“缩小 / 放大 / 适应 / 适应宽度 / 适应高度”。
- 鼠标滚轮可直接缩放画面。
- 当画面被放大后，可用中键拖动画面进行平移查看。

### 6.3 选择提示模式

WebApp 支持三种提示方式：点选、框选、文字。

#### 点选

- 左键在画面上添加当前点类型的点。
- “正向”点表示希望保留的目标区域。
- “负向”点表示希望排除的区域。
- 在点选模式下，右键空白处会直接添加一个负向点。
- 右键已有点会弹出菜单，可以把该点切换为正向/负向，或删除该点。
- 点选模式适合做精细修正，尤其适合边界复杂、局部漏分或误分的情况。

#### 框选

- 在画面上按住左键拖拽，生成一个矩形框。
- 框选适合先快速圈出目标的大致范围，再让 SAM 生成当前帧遮罩。

#### 文字

- 在文本框中输入文字提示后，点击“当前帧预览”。
- 当前实现建议使用英文提示词，例如 `person`、`hair`、`dress`。
- 文字模式适合目标语义清晰、画面内容不复杂的情况。

### 6.4 关键帧模式

- 关键帧开关只在“点选”模式下生效。
- 打开后，不同帧可以分别保存各自独立的点组。
- 在某一帧上放好点后，点击“当前帧预览”，这帧就会成为已确认关键帧之一。
- 可以通过“上个关键帧 / 下个关键帧”在关键帧之间跳转。
- “删除当前”可删除当前帧的本地点组，或删除已经确认的关键帧提示。
- 当视频中目标外观、姿态、遮挡关系变化较大时，推荐使用多个关键帧共同约束传播结果。

### 6.5 当前帧预览

- “当前帧预览”只会在当前帧上生成并确认提示结果，不会开始整段传播。
- 如果当前模式是点选，至少需要一个点。
- 如果当前模式是框选，必须先画出一个框。
- 如果当前模式是文字，必须输入非空文本。
- 预览成功后，页面状态会更新为“已确认”，此时就可以开始传播。

### 6.6 传播

- 点击“开始传播”后，系统会从已确认帧向前后双向传播遮罩。
- 如果启用了关键帧模式，则会综合所有已确认关键帧一起约束传播。
- 传播完成后，页面会自动生成两个预览：
  - 彩色叠加预览：用于检查目标跟踪是否稳定。
  - 遮罩预览：用于检查最终输出的遮罩质量。

### 6.7 预览和导出

- 两个预览视频都可以播放、暂停，并用各自的时间滑块逐帧查看。
- 在“遮罩预览”卡片上右键，可以打开导出菜单。
- 输入导出码率后，点击“导出 H.264 遮罩视频”，浏览器会下载导出结果。
- 导出文件是 H.264 编码的视频文件，适合直接交付或进入后续剪辑流程。

### 6.8 状态区说明

页面底部状态区会显示当前会话信息：

- 目标数：当前已确认提示中的对象数量。
- 帧率：原视频帧率。
- 分辨率：原视频分辨率。
- SAM 推理：SAM 实际使用的推理分辨率。如果视频被缩小推理，这里会显示缩放后的尺寸。
- 锚定帧：当前已确认帧；在关键帧模式下会显示关键帧列表。
- 关键帧：当前会话中已确认关键帧的数量。

## 7. 参数说明

### 7.1 启动参数

#### `SAM31_HOST`

- 默认值：`127.0.0.1`
- 作用：控制 Web 服务绑定地址。
- 常见用法：
  - `127.0.0.1`：仅本机访问。
  - `0.0.0.0`：允许局域网设备访问。

#### `SAM31_PORT`

- 默认值：`8765`
- 作用：控制 Web 服务端口。
- 当默认端口被占用时，可以改成 `8766`、`8780` 等其他未被占用的端口。

### 7.2 传播区参数

#### 视频码率

- 默认值：`10M`
- 作用：控制传播完成后生成的预览视频码率。
- 影响对象：彩色叠加预览和遮罩预览。
- 可用格式：
  - `10M`
  - `8m`
  - `8000k`
  - 纯数字，例如 `10`，程序会自动按 `10M` 处理
- 码率越高，预览越清晰，但生成文件更大，重编码时间也可能更长。

#### 推理像素上限

- 默认值：`1920x1080`
- 作用：限制 SAM 3.1 内部推理时使用的最大总像素数。
- 可用格式：
  - 分辨率形式，如 `1920x1080`
  - 整数形式，如 `2073600`
- 如果原视频像素超过这个上限，系统会先缩小视频再交给 SAM 推理。
- 这个参数越小，显存占用通常越低，速度可能更快，但细节可能下降。
- 这个参数越大，遮罩细节通常更好，但显存和耗时也会增加。
- 当视频较大、显存不足或传播很慢时，可以优先尝试调小这个值。

#### 遮罩后处理

可选项有三种：

- `binary`
  - 直接输出二值遮罩。
  - 速度最快，资源占用最低。
  - 适合不需要半透明边缘的场景。

- `videomama`
  - 使用 VideoMaMa 做时序一致的精修。
  - 通常能得到更平滑、更稳定的边缘和透明过渡。
  - 需要完整的 VideoMaMa 仓库、权重和 CUDA 环境。
  - 显存占用和耗时通常高于另外两种模式。

- `vitmatte`
  - 使用 ViTMatte 对遮罩边缘做单帧级精修。
  - 相比 VideoMaMa 更轻量，通常更容易部署。
  - 适合需要柔和边缘，但又不想使用 VideoMaMa 全套模型的情况。

#### VideoMaMa `max_resolution`

- 默认值：`1024`
- 可调范围：`256` 到 `2048`
- 作用：控制 VideoMaMa 精修时允许使用的最大分辨率。
- 值越大，通常能保留更多边缘细节，但显存占用和耗时会增加。
- 如果 VideoMaMa 模式下显存吃紧，可以优先把它调低到 `768` 或 `512`。

#### ViTMatte device

- 可选值：`GPU`、`CPU`
- `GPU`：
  - 速度更快
  - 占用显存
- `CPU`：
  - 速度更慢
  - 更节省显存，适合显存不足时兜底使用

#### Trimap 腐蚀

- 默认值：`12 px`
- 作用：在 ViTMatte 精修前，先收缩“确定前景”区域。
- 值越大，确定前景区域越保守，边缘附近会留给模型更多判断空间。
- 当边缘容易粘连、前景溢出时，可以适当调大。

#### Trimap 膨胀

- 默认值：`16 px`
- 作用：在 ViTMatte 精修前，扩大“未知区域”范围。
- 值越大，模型参与细化的边缘带越宽。
- 对头发、薄纱、半透明边缘等细节较多的目标，适当增大通常更有帮助。
- 如果边缘发灰过宽或处理太慢，可以适当减小。

### 7.3 导出参数

#### 导出码率

- 默认值：`8M`
- 作用：控制右键导出 H.264 遮罩视频时的码率。
- 格式与“视频码率”相同，支持 `4M`、`8000k`、`12` 这类写法。
- 如果只需要检查遮罩形状，较低码率通常已经足够。
- 如果后续还要进专业后期流程，可以适当提高码率。

## 8. 著作权、许可与致谢

### 8.1 本项目代码的许可范围

- 本项目自有代码部分按 MIT License 发布。
- 这里的“自有代码部分”主要指本仓库中用于本地部署、Web 交互、工程整合和启动运行的内容，例如：
  - `app/sam31_webapp/` 下的 WebApp 代码
  - `app/run_sam31_webapp.py`
  - `run_SAMMatte.bat`
  - `check_runtime.bat`
  - `requirements.txt`
  - 本 README 及与本项目发布相关的辅助文件
- 除非另有说明，本项目作者对上述自有代码保留著作权。

### 8.2 第三方代码、模型和工具不自动转为 MIT

本项目依赖多个第三方开源项目、模型权重和工具。即使本仓库自有代码按 MIT 发布，这些第三方内容仍然分别受其原始许可证、模型使用条款和分发条件约束，不会因为被本项目调用、打包、转存到百度网盘或放入 `models/`、`tools/` 目录而自动转为 MIT。

使用、复制、再分发以下内容时，请分别遵守其原始许可：

- `SAM 3 / SAM 3.1`
  - 来源：Meta
  - 涉及内容：`models/sam3/` 源码、`models/sam3.1/` 权重与相关文件
  - 许可与条款：遵循 Meta 提供的 SAM License 及相应模型访问条款

- `VideoMaMa`
  - 来源：KAIST / Adobe Research / Korea University 等作者团队
  - 涉及内容：`models/VideoMaMa/` 源码与其相关模型文件
  - 代码许可：`CC BY-NC 4.0`
  - 说明：`VideoMaMa` 仓库 README 中还说明其部分模型权重受 Stability AI Community License 约束

- `ViTMatte`
  - 来源：HUST / 原论文作者；本项目使用的是 Hugging Face 上的 `vitmatte-base-composition-1k`
  - 涉及内容：`models/vitmatte-base-composition-1k/`
  - 许可证信息：当前模型卡标注为 `Apache-2.0`

- `FFmpeg`
  - 来源：FFmpeg 项目；当前便携包中的 Windows 构建来自 `gyan.dev`
  - 涉及内容：`tools/ffmpeg/` 及相关二进制文件
  - 当前打包说明文件标注许可证：`GPL v3`

### 8.3 关于百度网盘附件的说明

- README 中提供的百度网盘附件仅用于方便部署，不改变第三方内容的原始著作权归属和许可证。
- 如果你分发 `tools`、`models` 或其任何子内容，建议同时保留原始目录中的 `LICENSE`、`README`、`License.md`、模型卡或其他上游说明文件。
- 对于需要额外访问授权、模型条款确认或账号登录后才能下载的内容，使用和再分发前请自行确认是否满足上游要求。

### 8.4 非官方集成说明

- 本项目是面向本地使用场景的工程化整合与 Web 交互封装。
- 本项目不是 Meta、VideoMaMa 作者团队、ViTMatte 作者团队或 FFmpeg 官方发布版本。
- 本项目中的 Windows 启动脚本、WebUI、参数组织方式和部署说明，属于在上游项目基础上的二次开发与集成。

### 8.5 参考项目与引用

本项目主要建立在以下工作之上：

- SAM 3 / SAM 3.1
  - Project: `https://ai.meta.com/sam3`
  - Repo: `https://github.com/facebookresearch/sam3`
  - Paper: `SAM 3: Segment Anything with Concepts`

- VideoMaMa
  - Repo: `https://github.com/cvlab-kaist/VideoMaMa`
  - Project Page: `https://cvlab-kaist.github.io/VideoMaMa/`
  - Paper: `VideoMaMa: Mask-Guided Video Matting via Generative Prior`

- ViTMatte
  - Repo: `https://github.com/hustvl/ViTMatte`
  - Paper: `ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers`

如果你在论文、技术报告、演示或产品中使用了本项目，并且实际使用到了上述模型或其输出结果，建议同时引用对应上游项目与论文，而不仅仅引用本项目本身。

### 8.6 致谢

感谢以下项目和作者团队的开源工作，为本项目提供了核心能力与基础组件：

- Meta 的 SAM 3 / SAM 3.1 项目与相关研究工作
- VideoMaMa 作者团队提供的视频抠像与时序精修方法
- ViTMatte 作者团队提供的图像抠图模型
- FFmpeg 项目及其 Windows 预编译分发维护者
