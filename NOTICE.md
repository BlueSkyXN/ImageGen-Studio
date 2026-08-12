# 来源与修改声明

本仓库是下列 GPL-3.0 项目的修改与整合版本：

- RioShiina/ImageGen，核对版本：`3622e14a8b6587699eb3e0616c167d0c35649ca7`
- Dekonstruktio/Fluxus，核对版本：`4aaca1f35ace66be1fb73de9bc46d390501f4ef6`
- 两者共同祖先：`9dbb7e3b34150ab09eaa0b83b92c7ea7e0493860`

Fluxus 在共同祖先之后只改动 README、MCP 启用开关和界面品牌文案，未形成独立推理引擎。本修改版因此以 ImageGen 为主干，并保留 Fluxus 的 MCP 关闭选项以及 `run_imagegen`、`get_chain_schema` 兼容接口。

主要修改包括：

- 将五套重复 Gradio 页面重构为一个动态工作台；
- 新增中文界面、任务说明、模型语言提示与快捷模型方案；
- 调整模型切换语义，避免覆盖用户 Prompt；
- 为 UI、Gradio API 与 MCP 增加共享的有界 GPU 调度；
- 将 MCP 裸线程改为有界执行器，并为任务表增加锁和容量限制；
- 将临时文件改为 UUID，并为下载和共享目录写入增加进程锁；
- 固定 ComfyUI/custom nodes commit，并与应用源码隔离；
- 避免重复执行 ComfyUI SaveImage；
- 增加输入/输出像素、批量和 URL 图片约束；
- 增加测试、中文部署文档和 Fluxus 接口兼容层。

本仓库保留原始 `LICENSE`，整体按 GPL-3.0 分发。这里列出的模型仅在运行时按需下载，其许可证与使用限制由各模型作者决定。ComfyUI、custom nodes、SageAttention 及其他第三方依赖也继续适用各自许可证；本声明不改变这些许可。
