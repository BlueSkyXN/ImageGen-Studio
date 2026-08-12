# Hugging Face Space 部署审查

审查日期：2026-08-08

## 结论

当前仓库达到“可构建、可启动、适合小范围 ZeroGPU 验证”的状态；尚不能承诺 92 个模型都能在默认公共 Space 的磁盘、配额与 120 秒 GPU 时间内稳定运行。92 是可选模型目录，不是显存常驻数量。

| 项目 | 状态 | 处理 |
|---|---|---|
| Gradio / MCP | 通过本地隔离环境烟测 | `gradio[mcp]==5.50.0`，仅 9 个高层公共端点 |
| Python | 已适配 | metadata 固定官方 ZeroGPU 支持的 `3.12.12` |
| ZeroGPU 串行 | 已适配 | 所有入口最终进入单 GPU FIFO 闸门；PK 每模型分别申请 GPU |
| 多模型显存 | 不支持，且不计划支持 | 磁盘按需缓存；GPU 一次只执行一个模型；模型边界释放 Comfy 状态 |
| 磁盘 | 有保护，仍需规划 | 下载前检查远端文件大小并预留 3 GB；不自动误删 Hub cache |
| 冷启动 | 有缓解 | ComfyUI/custom nodes 固定 commit；metadata 启动上限为 1 小时 |
| 持久缓存 | 可配置 | Storage Bucket 挂载到 `/home/user/app/models`，同时设置 `HF_HOME` |
| gated 模型 | 需部署者操作 | 设置 `HF_TOKEN`，且账号须先在各模型页接受条款 |
| PyTorch ABI | 待真实 Space 固化 | ZeroGPU 支持当前范围，但三件套暂沿用上游非固定版本 |
| 真实 GPU 全矩阵 | 未完成 | 需在目标 Space 对精选模型逐个做低分辨率冷/热启动测试 |

## 推荐的公共 ZeroGPU 配置

```text
IMAGEGEN_MAX_BATCH_SIZE=2
IMAGEGEN_MAX_PK_MODELS=2
IMAGEGEN_MAX_MULTI_IMAGES=4
IMAGEGEN_MAX_PLAN_JOBS=4
IMAGEGEN_MAX_PLAN_OUTPUTS=8
IMAGEGEN_MIN_FREE_DISK_GB=3
```

默认不要做模型常驻池，也不要把四个大模型装进一次 GPU 租约。PK 会解析一次随机 Seed，按模型顺序运行；模型切换前释放上一模型，最后一个模型保留为热状态。

## 持久模型缓存

1. 创建 Hugging Face Storage Bucket。
2. 挂载到 `/home/user/app/models`。
3. 设置变量 `HF_HOME=/home/user/app/models/.hf-cache`。
4. 重启 Space，并先用一个小模型做冷启动测试。

代码中的 `models/` 文件大多是指向 Hub cache 的符号链接，因此两者应放在同一持久卷。自动 LRU 暂不启用：删除 Hub revision 可能使仍在使用的符号链接失效，公共部署应先精选模型或人工清理。

## 上线前必须实测

- 默认模型：首次下载、首次 GPU 装载、第二次热运行。
- 两模型 PK：相同 Prompt / Seed，确认模型边界后显存释放。
- 两张独立图片 × 两模型：4 个顺序任务，确认取消能保留已完成结果。
- 编辑模型：1 张参考图→1 张图、3 张参考图→1 张图、参考组→多张变体。
- 磁盘余量不足：确认下载前给出中文错误且不留下破损链接。
- gated 模型：无 Token、未接受条款、已授权三种状态。
- Space 重建：确认持久卷挂载和 `HF_HOME` 生效。

## 官方依据

- [ZeroGPU 规格、Python/PyTorch 版本、配额和模型装载建议](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Space 默认资源与临时磁盘](https://huggingface.co/docs/hub/spaces-overview)
- [README metadata、启动超时与预加载字段](https://huggingface.co/docs/hub/spaces-config-reference)
- [Space 存储与 Storage Buckets](https://huggingface.co/docs/hub/spaces-storage)
