"""Application layout: one shared workspace for all image tasks."""

from __future__ import annotations

import gradio as gr

from core.runtime_config import CONFIG
from .shared import studio_ui
from .theme import APP_CSS


def build_ui(event_handler_function):
    ui_components = {}
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
    )

    with gr.Blocks(
        theme=theme,
        css=APP_CSS,
        title="ImageGen Studio 中文版",
        fill_width=True,
    ) as demo:
        gr.HTML(
            """
            <section id="studio-hero">
              <h1>ImageGen Studio</h1>
              <p>一套界面完成文生图、图生图、局部重绘、扩图和高清修复。模型与任务切换会保留你的提示词和已上传素材。</p>
            </section>
            <section class="studio-steps" aria-label="快速开始">
              <div class="studio-step"><b>① 选任务</b><span>先决定生成、编辑，还是修复图片</span></div>
              <div class="studio-step"><b>② 选模型并描述</b><span>不确定时使用快速方案和模型提示</span></div>
              <div class="studio-step"><b>③ 生成并迭代</b><span>先用默认参数出图，再逐项微调</span></div>
            </section>
            """
        )

        ui_components.update(studio_ui.create_ui())

        with gr.Accordion("使用说明与常见问题", open=False):
            gr.Markdown(
                """
- **切换任务会丢内容吗？** 不会。提示词、模型、参数和上传素材都保留，只切换当前需要的控件。
- **中文提示词是否可用？** Qwen-Image、Z-Image、HunyuanImage 等优先支持中文；SDXL 动漫模型更适合英文标签，模型说明会给出建议。
- **为什么显示排队？** GPU 任务通过同一队列调度，避免 UI 和 MCP 同时抢占显存。等待期间可以修改表单，但同一按钮不会重复提交。
- **第一次为什么较慢？** 首次使用某模型会下载权重；下载过程有互斥保护，避免并发请求重复下载或写坏文件。
- **扩展能力为什么会变化？** LoRA、ControlNet、IP-Adapter 等只在当前模型兼容时显示；不兼容的隐藏状态不会送入生成管线。
                """
            )

        gr.HTML(
            """
            <footer class="studio-footer">
              基于 <a href="https://huggingface.co/spaces/RioShiina/ImageGen" target="_blank" rel="noopener">RioShiina/ImageGen</a>
              重构，并整合 <a href="https://huggingface.co/spaces/Dekonstruktio/Fluxus" target="_blank" rel="noopener">Dekonstruktio/Fluxus</a>
              的部署选择；遵循 GPL-3.0，保留上游署名。修改版并非上游官方发行。
            </footer>
            """
        )

        event_handler_function(ui_components, demo)

        high_level_names = set()
        if CONFIG.enable_mcp:
            try:
                import mcp_tools as mcp

                high_level_names = getattr(mcp, "HIGH_LEVEL_MCP_API_NAMES", set())
                if hasattr(mcp, "register_high_level_mcp_apis"):
                    mcp.register_high_level_mcp_apis(demo)
                    mcp.cleanup_dependencies_api_names(demo)
                elif hasattr(mcp, "MCP_FUNCTIONS") and isinstance(mcp.MCP_FUNCTIONS, list):
                    for func in mcp.MCP_FUNCTIONS:
                        gr.api(func)
            except Exception as exc:
                print(f"⚠️ MCP 接口注册失败：{exc}")

        # Only the intentional high-level MCP surface is public.
        for fn in demo.fns.values():
            if getattr(fn, "api_name", None) not in high_level_names:
                fn.show_api = False

    return demo
