# 架构说明

## 为什么只有一个引擎

Fluxus 是 ImageGen 的直接 fork，并共享模型注册表、ComfyUI 执行器、YAML 工作流与 UI。因此“双后端切换”只会复制状态、模型下载和显存占用，不能增加能力。本项目把两者统一为一个本地 ImageGen 引擎，用配置开关和 API alias 处理部署差异。

## 请求路径

```text
统一 Gradio 工作台 / MCP 高层接口
                ↓
          规范化输入字典
                ↓
     模型能力过滤 + 参数校验
                ↓
      有界队列 + GPU 信号量
                ↓
    YAML recipe + chain injector
                ↓
       进程内 ComfyUI 节点执行
                ↓
   PNG 参数 + workflow 元数据输出
```

## 切换语义

- 任务切换只控制可见区域；共用模型、Prompt、种子和采样参数。
- 源图片、重绘蒙版和各任务参数保留在各自组件中，切回时仍可继续。
- 架构/分类筛选时，当前模型若仍有效则保持不变。
- 切模型会更新推荐步数、CFG、Sampler、Scheduler 和兼容扩展，但不会改写正面或负面 Prompt。
- Pipeline 在提交前再次过滤不兼容扩展，避免隐藏旧状态进入工作流。

## 并发边界

- Gradio 生成事件使用共同的 `concurrency_id="gpu_generation"`。
- Pipeline 入口使用 FIFO ticket gate，覆盖 UI、同步 API 和异步 MCP 三条路径。
- MCP 异步任务使用固定大小线程池和待处理信号量，不再创建无限线程。
- 模型/LoRA/ControlNet/IP-Adapter 下载与链接由进程级可重入锁保护。
- 临时文件使用 UUID；任务表读写使用锁，并只保留有限条记录。
- “取消排队”会取消 Gradio 等待任务；内部 GPU 闸门会轮询取消标记并跳过对应票据；模型准备完成后和进入 GPU 前还会再次检查。已开始采样的任务目前不能强制中断。

Gradio 与 MCP 各自保留有界入口，最后在 Pipeline 的公平 GPU 闸门汇合。它们不是同一个可观测队列，因此网页排队位置只表示 Gradio 侧等待情况；FIFO 闸门负责避免已经到达 Pipeline 的任一来源长期饥饿，并保证共享 ComfyUI 运行时一次只执行一个任务。

## 批量与 PK 编排

- `core/execution_plan.py` 只把一次提交展开成若干现有 Pipeline 输入，不复制工作流执行器。
- 独立多图按同模型分组，减少 checkpoint 切换；图片×模型总任务受硬上限约束。
- 模型 PK 共享 Prompt、源图、尺寸和一次解析出的 Seed；默认采用各模型推荐采样参数。
- PK V1 清空高级 injector，仅比较基础 checkpoint 的共同输入能力。
- 多图融合固定为 `txt2img + 模型专属 reference chain`；它与普通 img2img 的 source latent 是两种不同语义。
- 模型边界在 GPU 租约内调用 ComfyUI 的 unload/cleanup；普通同模型连续生成保留热缓存。

## 可复现依赖

`vendor.lock.yaml` 固定 ComfyUI 和 custom nodes 的完整 commit。默认克隆到 `_vendor/ComfyUI` 与 `custom_nodes/`，不再把 ComfyUI 内容覆盖到项目根。设置 `COMFYUI_PATH` 可使用已有 checkout。

升级依赖时应：

1. 在独立分支更新一个 revision；
2. 运行全部测试和至少一个低分辨率 GPU 冒烟任务；
3. 检查节点类名、输入 schema 和 PNG metadata；
4. 再更新 `vendor.lock.yaml` 与 `NOTICE.md`。
