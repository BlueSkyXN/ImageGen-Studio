---
title: ImageGen Studio 中文版
emoji: 🖼
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.12.12"
startup_duration_timeout: 1h
short_description: 中文优先的多任务图片生成与编辑工作台
license: gpl-3.0
pinned: true
models:
# This Space supports a wide variety of image generation pipelines. To maintain transparency, credit the original creators, and help users explore the Hugging Face ecosystem, we list and link several types of models in our metadata:
# 1. **Directly Run Models:** Models and checkpoints actively loaded by our pipelines (configured via `yaml/file_list.yaml`).
# 2. **Upstream Base Models:** The original foundation architectures from which our optimized ports, quantized versions, or wrappers are derived.
  # Directly Run Models
  - AiAF/Illustrious-XL-v0.1.safetensors
  - alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
  - black-forest-labs/FLUX.1-Redux-dev
  - black-forest-labs/FLUX.2-dev-NVFP4
  - black-forest-labs/FLUX.2-klein-4b-nvfp4
  - black-forest-labs/FLUX.2-klein-9b-nvfp4
  - black-forest-labs/FLUX.2-klein-9b-kv-fp8
  - black-forest-labs/FLUX.2-klein-base-4b-nvfp4
  - black-forest-labs/FLUX.2-klein-base-9b-nvfp4
  - bluepen5805/4nima_pencil-XL
  - bluepen5805/anima-models
  - bluepen5805/anima_pencil-XL
  - bluepen5805/blue_pencil-XL
  - bluepen5805/illustrious_pencil-XL
  - bluepen5805/mellow_pencil-XL
  - bluepen5805/noob_v_pencil-XL
  - bluepen5805/pony_pencil-XL
  - cagliostrolab/animagine-xl-3.1
  - cagliostrolab/animagine-xl-4.0
  - ChenkinNoob/ChenkinNoob-XL-V0.5
  - circlestone-labs/Anima
  - Clybius/Chroma-fp8-scaled
  - comfyanonymous/ControlNet-v1-1_fp16_safetensors
  - comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI
  - comfyanonymous/flux_text_encoders
  - Comfy-Org/Anima-LLLite
  - Comfy-Org/Boogu-Image
  - Comfy-Org/ERNIE-Image
  - Comfy-Org/FLUX.1-Krea-dev_ComfyUI
  - Comfy-Org/flux2-dev
  - Comfy-Org/HiDream-I1_ComfyUI
  - Comfy-Org/HiDream-O1-Image
  - Comfy-Org/HunyuanImage_2.1_ComfyUI
  - Comfy-Org/Ideogram-4
  - Comfy-Org/Krea-2
  - Comfy-Org/Lens
  - Comfy-Org/LongCat-Image
  - Comfy-Org/Lumina_Image_2.0_Repackaged
  - Comfy-Org/Mage-Flow
  - Comfy-Org/NewBie-image-Exp0.1_repackaged
  - Comfy-Org/Omnigen2_ComfyUI_repackaged
  - Comfy-Org/Ovis-Image
  - Comfy-Org/PixelDiT
  - Comfy-Org/Qwen-Image_ComfyUI
  - Comfy-Org/Qwen-Image-Edit_ComfyUI
  - Comfy-Org/sigclip_vision_384
  - Comfy-Org/stable-diffusion-3.5-fp8
  - Comfy-Org/vae-text-encorder-for-flux-klein-4b
  - Comfy-Org/vae-text-encorder-for-flux-klein-9b
  - Comfy-Org/Wan_2.1_ComfyUI_repackaged
  - Comfy-Org/z_image
  - Comfy-Org/z_image_turbo
  - conradlocke/krea2-identity-edit
  - cyberdelia/CyberRealisticPony
  - diffusionmodels1254ani/hassakuAnima
  - diffusionmodels1254ani/kirazuriAnima_v30AnimaBase1
  - diffusionmodels1254ani/waiANIMA
  - duongve/AnimaYume
  - Eugeoter/noob-sdxl-controlnet-canny
  - Eugeoter/noob-sdxl-controlnet-depth
  - Eugeoter/noob-sdxl-controlnet-lineart_anime
  - Eugeoter/noob-sdxl-controlnet-lineart_realistic
  - Eugeoter/noob-sdxl-controlnet-manga_line
  - Eugeoter/noob-sdxl-controlnet-normal
  - Eugeoter/noob-sdxl-controlnet-softedge_hed
  - Eugeoter/noob-sdxl-controlnet-tile
  - fal/AuraFlow-v0.3
  - frankjoshua/novaAnimeXL_ilV180
  - h94/IP-Adapter
  - h94/IP-Adapter-FaceID
  - InstantX/FLUX.1-dev-IP-Adapter
  - InstantX/Qwen-Image-ControlNet-Inpainting
  - InstantX/Qwen-Image-ControlNet-Union
  - InstantX/SD3.5-Large-IP-Adapter
  - jdopensource/JoyAI-Image-Edit-ComfyUI
  - jdopensource/JoyAI-Image-Edit-Plus-ComfyUI
  - kandinskylab/Kandinsky-5.0-T2I-Lite
  - Kijai/flux-fp8
  - Laxhar/noob_openpose
  - Laxhar/noobai-XL-1.1
  - Laxhar/noobai-XL-Vpred-1.0
  - licyk/sd_control_collection
  - lightx2v/Qwen-Image-2512-Lightning
  - lightx2v/Qwen-Image-Edit-2511-Lightning
  - LyliaEngine/Pony_Diffusion_V6_XL
  - MIC-Lab/illustriousXLv0.1_controlnet
  - MIC-Lab/illustriousXLv1.1_controlnet
  - misri/hassakuXLIllustrious_v30
  - nvidia/Cosmos-Predict2-2B-Text2Image
  - nvidia/Cosmos-Predict2-14B-Text2Image
  - OnomaAIResearch/Illustrious-XL-v1.0
  - OnomaAIResearch/Illustrious-XL-v1.1
  - OnomaAIResearch/Illustrious-XL-v2.0
  - ostris/krea2_turbo_style_reference
  - Patil/Krea-2-depth-controlnet
  - RedRayz/hikari_noob_v-pred_1.2.4
  - Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0
  - silveroxides/Chroma1-Radiance-fp8-scaled
  - stabilityai/sdxl-turbo
  - stabilityai/stable-diffusion-3.5-controlnets
  - stabilityai/stable-diffusion-xl-base-1.0
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - tsukiyomi/krea2_raw-nvfp4
  - Wenaka/NoobAI_XL_Inpainting_ControlNet_Full
  - xinsir/anime-painter
  - xinsir/controlnet-canny-sdxl-1.0
  - xinsir/controlnet-depth-sdxl-1.0
  - xinsir/controlnet-openpose-sdxl-1.0
  - xinsir/controlnet-scribble-sdxl-1.0
  - xinsir/controlnet-tile-sdxl-1.0
  - xinsir/controlnet-union-sdxl-1.0
  - XLabs-AI/flux-controlnet-collections
  - zhenshipo/waiIllustriousSDXL_v170
  # Upstream Base Models
  - AIDC-AI/Ovis-Image-7B
  - Alpha-VLLM/Lumina-Image-2.0
  - baidu/ERNIE-Image
  - baidu/ERNIE-Image-Turbo
  - black-forest-labs/FLUX.1-dev
  - black-forest-labs/FLUX.1-Krea-dev
  - black-forest-labs/FLUX.1-schnell
  - Boogu/Boogu-Image-0.1-Turbo
  - Boogu/Boogu-Image-0.1-Base
  - Boogu/Boogu-Image-0.1-Edit
  - Boogu/Boogu-Image-0.1-Edit-Turbo
  - HiDream-ai/HiDream-I1-Dev
  - HiDream-ai/HiDream-I1-Fast
  - HiDream-ai/HiDream-I1-Full
  - HiDream-ai/HiDream-O1-Image
  - HiDream-ai/HiDream-O1-Image-Dev
  - ideogram-ai/ideogram-4-fp8
  - kohya-ss/Anima-LLLite
  - krea/Krea-2-Raw
  - krea/Krea-2-Turbo
  - lodestones/Chroma1-HD
  - lodestones/Chroma1-Radiance
  - mage-flow-community/Mage-Flow
  - mage-flow-community/Mage-Flow-Base
  - mage-flow-community/Mage-Flow-Edit
  - mage-flow-community/Mage-Flow-Edit-Base
  - mage-flow-community/Mage-Flow-Edit-Turbo
  - mage-flow-community/Mage-Flow-Turbo
  - meituan-longcat/LongCat-Image
  - microsoft/Lens
  - microsoft/Lens-Turbo
  - NewBie-AI/NewBie-image-Exp0.1
  - nvidia/PiD
  - nvidia/PixelDiT-1300M-1024px
  - OmniGen2/OmniGen2
  - Qwen/Qwen-Image
  - Qwen/Qwen-Image-2512
  - Qwen/Qwen-Image-Edit
  - Qwen/Qwen-Image-Edit-2509
  - Qwen/Qwen-Image-Edit-2511
  - stabilityai/stable-diffusion-3.5-large
  - stabilityai/stable-diffusion-3.5-medium
  - tencent/HunyuanImage-2.1
  - Tongyi-MAI/Z-Image
  - Tongyi-MAI/Z-Image-Turbo
---

# ImageGen Studio 中文版

这是 [RioShiina/ImageGen](https://huggingface.co/spaces/RioShiina/ImageGen) 与 [Dekonstruktio/Fluxus](https://huggingface.co/spaces/Dekonstruktio/Fluxus) 的整合重构版，保留 YAML 配方、动态工作流和 92 个模型目录，重做任务切换、并发调度、中文指导与移动端体验。

Fluxus 与 ImageGen 共享同一 Git 历史，Fluxus 当前版本只调整了品牌文案和 MCP 开关，没有第二套推理核心。因此本项目使用一个 ImageGen 引擎，并兼容两边的 API 契约，避免双后端、双状态和重复下载。

## 主要改进

| 项目 | 原版 | 本整合版 |
|---|---|---|
| 任务界面 | 5 个完整 Tab，重复创建全部高级控件 | 1 个共享工作台，动态切换 5 类任务 |
| 任务切换 | 模型、Prompt 和素材分散在各 Tab | 保留模型、Prompt、参数与上传素材 |
| 模型切换 | 会覆盖正负 Prompt 和推荐参数 | 保留用户 Prompt；推荐采样参数可自动跟随、关闭或一键恢复 |
| 复杂度 | 约 4,033 组件 / 506 事件 | 烟测为 881 组件 / 120 事件 |
| GPU 并发 | UI 与 MCP 可绕过彼此并发执行 | 各入口有界，最终汇合到公平的单 GPU 闸门 |
| 异步 MCP | 每次请求创建无上限 daemon 线程 | 有界线程池、队列满错误、任务表上限与锁 |
| 临时文件 | 4 位随机后缀，可能碰撞 | UUID 文件名 |
| 启动依赖 | 运行时拉取最新版并覆盖项目根 | 固定 commit，隔离在 `_vendor/ComfyUI` |
| 中文体验 | 英文界面，无模型语言提示 | 中文任务指导、模型用途说明、可操作错误信息 |
| MCP 兼容 | ImageGen 与 Fluxus 接口名不同 | 同时保留新版接口与两个 Fluxus 旧别名 |

## 支持能力

- 任务：文生图、图生图、局部重绘、扩图、高清修复。
- 模型：由 `yaml/model_list.yaml` 驱动，当前包含 30 类架构、92 个显示模型。
- 扩展：LoRA、ControlNet、IP-Adapter、Embedding、区域提示、多图编辑、VAE、PiD 等。
- 输出：PNG 内写入生成参数和完整 ComfyUI 工作流元数据。
- 中文 Prompt：默认原样传递，不做隐式翻译；模型说明会区分中文自然语言和英文标签型模型。

## 批量、多图与模型 PK

这些能力只增加一个有上限的顺序编排层，不创建第二套推理引擎，也不会让多个大模型同时占用显存：

| 运行方式 | 输入语义 | 输出语义 |
|---|---|---|
| 普通生成 | 当前任务的一组输入 | 1–4 张同组变体 |
| 模型 PK | 同一 Prompt、尺寸、源图与已解析 Seed | 每个模型分别生成；默认最多 2 个模型 |
| 多图独立 | A、B、C 是互不相关的源图 | A→A′、B→B′、C→C′ |
| 多图 × 多模型 | 多张独立源图和两个模型 | 按模型分组顺序执行图片×模型组合 |
| 多图融合 | 多张图共同作为一组参考 | 由兼容的编辑模型融合为 1–4 张结果 |

- 多图独立目前支持图生图、扩图和高清修复；局部重绘需要逐张图片配对蒙版，暂不做批量。
- 多图融合内部固定走“文生图 + 模型专属参考链”，不会与图生图 base latent 混用。Prompt 可按“参考图 1 / 2 / 3”说明各图角色。
- 多图融合是生成式参考，不保证无损拼接、角色逐像素保留或确定性元素替换。
- PK 默认采用各模型推荐采样参数；关闭后才严格复用当前步数、CFG、采样器和调度器。同一 Seed 在不同架构之间只是尽量控制变量，不表示初始噪声数学等价。
- PK V1 会关闭 LoRA、ControlNet、IP-Adapter、参考链、自定义 VAE 和 PiD，只比较所有基础 checkpoint 都具备的 Prompt / 源图能力，避免某个模型偷偷多一层条件。
- 单进程始终一次只执行一个 GPU 任务。模型文件按需缓存在磁盘；只在 PK 的实际模型边界和 GPU 异常后释放 ComfyUI 模型状态。
- 默认上限为 2 个 PK 模型、4 张输入图、4 个顺序任务、8 张预计输出。某一组合失败时保留已成功结果；取消会停止尚未开始的后续组合。

## 部署到 Hugging Face Space

1. 新建 Gradio Space，硬件选择 ZeroGPU；README metadata 已固定 Python 3.12.12，并把冷启动上限设为 1 小时。
2. 上传本仓库内容；Space 会按 `requirements.txt` 安装 Gradio 5.50 和 MCP 额外依赖。
3. 按需要设置 Secret：
   - `HF_TOKEN`：访问 gated/private Hugging Face 模型。
   - `CIVITAI_API_KEY`：下载需要授权的 Civitai 资源。
4. 首次启动会按 `vendor.lock.yaml` 拉取固定版本的 ComfyUI 与 5 个 custom nodes；首次使用某个模型时才下载其权重。

部署边界：

- 模型目录包含 92 个可选项，但这不等于 92 个模型可同时驻留。ZeroGPU 应使用按需磁盘缓存 + 单模型顺序执行。
- Space 默认磁盘是临时盘，重启/重建后缓存可能消失。若需要长期保留模型，可把 Hugging Face Storage Bucket 挂载到 `/home/user/app/models`，并设置 `HF_HOME=/home/user/app/models/.hf-cache`，让权重实体和项目符号链接位于同一持久卷。
- 下载前会读取远端文件大小并保留默认 3 GB 安全余量；空间不足时会在下载前拒绝，而不是写满磁盘。项目不自动删除 Hub cache，避免误删仍被符号链接引用的模型。
- FLUX、SD3.5、Cosmos 等部分资源可能受访问条款限制。`HF_TOKEN` 所属账号必须先在对应模型页接受条款。
- 公共 Space 建议精选 4–8 个常用模型，并设置 `IMAGEGEN_MAX_BATCH_SIZE=2`；完整 92 模型目录更适合挂载持久存储的专用部署。
- ZeroGPU 单次 GPU 申请上限为 120 秒。大模型首次装载、超高分辨率或长采样仍可能超时；PK 会为每个模型分别申请 GPU，而不是占用一个超长租约。

完整的适配状态、推荐变量与上线实测清单见 [`docs/HF_SPACE_DEPLOYMENT.md`](docs/HF_SPACE_DEPLOYMENT.md)。

默认启动命令由 Space metadata 执行：

```bash
python app.py
```

## 本地运行

需要 Python 3.12、Git，以及与目标模型相匹配的 NVIDIA GPU 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

如已有 ComfyUI，可避免再次克隆：

```bash
COMFYUI_PATH=/absolute/path/to/ComfyUI python app.py
```

## 运行参数

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `IMAGEGEN_GPU_CONCURRENCY` | 固定 `1` | 为兼容旧部署保留名称；单进程始终串行，扩容请增加独立副本 |
| `IMAGEGEN_QUEUE_MAX_SIZE` | `24` | Gradio 等待队列上限 |
| `IMAGEGEN_MCP_MAX_PENDING` | `16` | MCP 后台任务上限 |
| `IMAGEGEN_MCP_TASK_RETENTION` | `200` | 内存任务记录上限 |
| `IMAGEGEN_MAX_BATCH_SIZE` | `4` | 单次最大图片数 |
| `IMAGEGEN_MAX_PK_MODELS` | `2` | PK 模型总数上限（含当前模型） |
| `IMAGEGEN_MAX_MULTI_IMAGES` | `4` | 批量输入/参考图的全局上传上限 |
| `IMAGEGEN_MAX_PLAN_JOBS` | `4` | 一次提交可展开的顺序任务上限 |
| `IMAGEGEN_MAX_PLAN_OUTPUTS` | `8` | 一次提交的预计输出总数上限 |
| `IMAGEGEN_MAX_INPUT_MEGAPIXELS` | `4.2` | 输入图片/文生图画布上限 |
| `IMAGEGEN_MAX_REFERENCE_MEGAPIXELS` | `12` | 一组上传图片的累计像素上限 |
| `IMAGEGEN_MAX_REFERENCE_IMAGES` | `10` | 一次任务中全部参考图/控制图的合计上限 |
| `IMAGEGEN_MAX_OUTPUT_MEGAPIXELS` | `16` | 扩图和高清修复预计输出上限 |
| `IMAGEGEN_MIN_FREE_DISK_GB` | `3` | 模型下载后必须保留的磁盘余量 |
| `IMAGEGEN_OUTPUT_RETENTION` | `80` | 本地保留的生成 PNG 数量 |
| `IMAGEGEN_ENABLE_MCP` | `true` | 是否启动 MCP；设为 `false` 即采用 Fluxus 的关闭方式 |
| `IMAGEGEN_STARTUP_GPU_PROBE` | `false` | 是否在启动时申请一次 GPU |
| `IMAGEGEN_USE_SAGE_ATTENTION` | `auto` | `auto` / `true` / `false` |
| `IMAGEGEN_SKIP_CUSTOM_NODES` | `false` | 本地调试时跳过 custom nodes 拉取 |
| `IMAGEGEN_GIT_TIMEOUT_SECONDS` | `180` | 单次 Git clone/fetch 超时 |
| `IMAGEGEN_GIT_ATTEMPTS` | `2` | Git 网络操作尝试次数 |

## MCP / Gradio API

公开 7 个主接口：

- `get_task_list`
- `get_model_architecture_list`
- `get_model_list`
- `get_feature_list`
- `get_model_features`
- `run`
- `get_task_status`

同时保留 Fluxus 客户端兼容别名：

- `run_imagegen(json_params)` → `run(json_params)`
- `get_chain_schema(chain_type)` → 返回单个扩展能力的完整 schema

`get_feature_list()` 空参默认返回 Fluxus 兼容的完整 schema；如需节省 MCP token，传 `compact=true` 获取摘要。也可以传 `feature_name`，或使用 `get_chain_schema(chain_type)` 查询单项完整 schema。

所有 UI 原子事件均隐藏，不作为公共 API 暴露。

## 验证

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

界面烟测会以 stub ComfyUI 构建完整 Gradio 配置，因此不需要下载模型或占用 GPU。

## 目录

```text
app.py                     # Space 入口与有界 Gradio 队列
core/                      # 生成管线、工作流装配、统一调度
ui/shared/studio_ui.py     # 单一动态工作台
ui/events/                 # 切换、模型、扩展与生成事件
mcp_tools/                 # 高层 API、兼容别名、任务状态
yaml/                      # 模型、能力、默认参数与配方注册表
vendor.lock.yaml           # ComfyUI / custom nodes 固定版本
tests/                     # 并发、注册表、MCP 与 UI 烟测
```

## 许可与来源

本项目是 ImageGen 的修改版，整体继续采用 GPL-3.0，并保留上游署名。Fluxus 也是同一 GPL-3.0 代码谱系。模型权重、ComfyUI 与 custom nodes 各自受其上游许可证约束；本仓库不重新分发模型权重。具体版本与修改声明见 `NOTICE.md`。
