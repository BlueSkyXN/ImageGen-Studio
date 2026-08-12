import gradio as gr
from core.settings import VAE_DIR, LORA_DIR, EMBEDDING_DIR, ARCHITECTURES_CONFIG
from imagegen_utils.app_utils import save_uploaded_file_with_hash
from ui.shared.ui_components import (
    MAX_CONTROLNETS,
    MAX_IPADAPTERS,
    MAX_EMBEDDINGS,
    MAX_CONDITIONINGS,
    MAX_LORAS
)
from .config_loaders import (
    load_controlnet_config,
    load_anima_controlnet_lllite_config,
    load_diffsynth_controlnet_config,
    load_krea2_controlnet_config,
    load_ipadapter_config
)

def on_vae_upload(file_obj):
    if not file_obj:
        return gr.update(), gr.update(), None

    hashed_filename = save_uploaded_file_with_hash(file_obj, VAE_DIR)
    return hashed_filename, "File", file_obj


def on_lora_upload(file_obj):
    if not file_obj:
        return gr.update(), gr.update()

    hashed_filename = save_uploaded_file_with_hash(file_obj, LORA_DIR)
    return hashed_filename, "File"


def on_embedding_upload(file_obj):
    if not file_obj:
        return gr.update(), gr.update(), None

    hashed_filename = save_uploaded_file_with_hash(file_obj, EMBEDDING_DIR)
    return hashed_filename, "File", file_obj


def create_lora_event_handlers(prefix, ui_components):
    lora_rows = ui_components.get(f'lora_rows_{prefix}')
    if not lora_rows: return
    lora_ids = ui_components[f'lora_ids_{prefix}']
    lora_scales = ui_components[f'lora_scales_{prefix}']
    lora_uploads = ui_components[f'lora_uploads_{prefix}']
    lora_sources = ui_components[f'lora_sources_{prefix}']
    count_state = ui_components[f'lora_count_state_{prefix}']
    add_button = ui_components[f'add_lora_button_{prefix}']
    del_button = ui_components[f'delete_lora_button_{prefix}']

    for i in range(MAX_LORAS):
        lora_uploads[i].upload(
            fn=on_lora_upload,
            inputs=[lora_uploads[i]],
            outputs=[lora_ids[i], lora_sources[i]],
            show_progress=True
        )

    def add_lora_row(c):
        updates = {}
        if c < MAX_LORAS:
            c += 1
            updates[lora_rows[c - 1]] = gr.update(visible=True)

        updates[count_state] = c
        updates[add_button] = gr.update(visible=c < MAX_LORAS)
        updates[del_button] = gr.update(visible=c > 1)
        return updates

    def del_lora_row(c):
        updates = {}
        if c > 1:
            updates[lora_rows[c - 1]] = gr.update(visible=False)
            updates[lora_ids[c - 1]] = ""
            updates[lora_scales[c - 1]] = 0.0
            updates[lora_uploads[c - 1]] = None
            c -= 1

        updates[count_state] = c
        updates[add_button] = gr.update(visible=True)
        updates[del_button] = gr.update(visible=c > 1)
        return updates

    add_outputs = [count_state, add_button, del_button] + lora_rows
    del_outputs = [count_state, add_button, del_button] + lora_rows + lora_ids + lora_scales + lora_uploads

    add_button.click(add_lora_row, [count_state], add_outputs, show_progress=False)
    del_button.click(del_lora_row, [count_state], del_outputs, show_progress=False)


def create_controlnet_event_handlers(prefix, ui_components):
    cn_rows = ui_components.get(f'controlnet_rows_{prefix}')
    if not cn_rows: return
    cn_types = ui_components[f'controlnet_types_{prefix}']
    cn_series = ui_components[f'controlnet_series_{prefix}']
    cn_filepaths = ui_components[f'controlnet_filepaths_{prefix}']
    cn_images = ui_components[f'controlnet_images_{prefix}']
    cn_strengths = ui_components[f'controlnet_strengths_{prefix}']

    count_state = ui_components[f'controlnet_count_state_{prefix}']
    add_button = ui_components[f'add_controlnet_button_{prefix}']
    del_button = ui_components[f'delete_controlnet_button_{prefix}']
    accordion = ui_components[f'controlnet_accordion_{prefix}']

    base_model_comp = ui_components.get(f'base_model_{prefix}')
    actual_arch_comp = base_model_comp if base_model_comp else gr.State("SDXL")

    def add_cn_row(c):
        c += 1
        updates = {
            count_state: c,
            cn_rows[c-1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_CONTROLNETS),
            del_button: gr.update(visible=True)
        }
        return updates

    def del_cn_row(c):
        c -= 1
        updates = {
            count_state: c,
            cn_rows[c]: gr.update(visible=False),
            cn_images[c]: None,
            cn_strengths[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0)
        }
        return updates

    add_outputs = [count_state, add_button, del_button] + cn_rows
    del_outputs = [count_state, add_button, del_button] + cn_rows + cn_images + cn_strengths
    add_button.click(fn=add_cn_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_cn_row, inputs=[count_state], outputs=del_outputs, show_progress=False)

    def on_cn_type_change(selected_type, model_name):
        from core.settings import MODEL_TYPE_MAP
        m_type = MODEL_TYPE_MAP.get(model_name, "SDXL") if model_name else "SDXL"
        cn_full_config = load_controlnet_config()

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        cn_config = cn_full_config.get(controlnet_key, [])
        series_choices = []
        if selected_type:
            series_choices = sorted(list(set(
                model.get("Series", "Default") for model in cn_config
                if selected_type in model.get("Type", [])
            )))
        default_series = series_choices[0] if series_choices else None
        filepath = "None"
        if default_series:
            for model in cn_config:
                if model.get("Series") == default_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return gr.update(choices=series_choices, value=default_series), filepath

    def on_cn_series_change(selected_series, selected_type, model_name):
        from core.settings import MODEL_TYPE_MAP
        m_type = MODEL_TYPE_MAP.get(model_name, "SDXL") if model_name else "SDXL"
        cn_full_config = load_controlnet_config()

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        cn_config = cn_full_config.get(controlnet_key, [])
        filepath = "None"
        if selected_series and selected_type:
            for model in cn_config:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    for i in range(MAX_CONTROLNETS):
        cn_types[i].change(
            fn=on_cn_type_change,
            inputs=[cn_types[i], actual_arch_comp],
            outputs=[cn_series[i], cn_filepaths[i]],
            show_progress=False
        )
        cn_series[i].change(
            fn=on_cn_series_change,
            inputs=[cn_series[i], cn_types[i], actual_arch_comp],
            outputs=[cn_filepaths[i]],
            show_progress=False
        )

    def on_accordion_expand(*images):
        return [gr.update() for _ in images]

    accordion.expand(
        fn=on_accordion_expand,
        inputs=cn_images,
        outputs=cn_images,
        show_progress=False
    )


def create_krea2_controlnet_event_handlers(prefix, ui_components):
    cn_rows = ui_components.get(f'krea2_controlnet_rows_{prefix}')
    if not cn_rows: return
    cn_types = ui_components[f'krea2_controlnet_types_{prefix}']
    cn_series = ui_components[f'krea2_controlnet_series_{prefix}']
    cn_filepaths = ui_components[f'krea2_controlnet_filepaths_{prefix}']
    cn_images = ui_components[f'krea2_controlnet_images_{prefix}']
    cn_strengths = ui_components[f'krea2_controlnet_strengths_{prefix}']

    count_state = ui_components[f'krea2_controlnet_count_state_{prefix}']
    add_button = ui_components[f'add_krea2_controlnet_button_{prefix}']
    del_button = ui_components[f'delete_krea2_controlnet_button_{prefix}']
    accordion = ui_components[f'krea2_controlnet_accordion_{prefix}']

    def add_cn_row(c):
        c += 1
        updates = {
            count_state: c,
            cn_rows[c-1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_CONTROLNETS),
            del_button: gr.update(visible=True)
        }
        return updates

    def del_cn_row(c):
        c -= 1
        updates = {
            count_state: c,
            cn_rows[c]: gr.update(visible=False),
            cn_images[c]: None,
            cn_strengths[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0)
        }
        return updates

    add_outputs = [count_state, add_button, del_button] + cn_rows
    del_outputs = [count_state, add_button, del_button] + cn_rows + cn_images + cn_strengths
    add_button.click(fn=add_cn_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_cn_row, inputs=[count_state], outputs=del_outputs, show_progress=False)

    def on_cn_type_change(selected_type):
        cn_config = load_krea2_controlnet_config()
        series_choices = []
        if selected_type:
            series_choices = sorted(list(set(
                model.get("Series", "Default") for model in cn_config
                if selected_type in model.get("Type", [])
            )))
        default_series = series_choices[0] if series_choices else None
        filepath = "None"
        if default_series:
            for model in cn_config:
                if model.get("Series") == default_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return gr.update(choices=series_choices, value=default_series), filepath

    def on_cn_series_change(selected_series, selected_type):
        cn_config = load_krea2_controlnet_config()
        filepath = "None"
        if selected_series and selected_type:
            for model in cn_config:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    for i in range(MAX_CONTROLNETS):
        cn_types[i].change(
            fn=on_cn_type_change,
            inputs=[cn_types[i]],
            outputs=[cn_series[i], cn_filepaths[i]],
            show_progress=False
        )
        cn_series[i].change(
            fn=on_cn_series_change,
            inputs=[cn_series[i], cn_types[i]],
            outputs=[cn_filepaths[i]],
            show_progress=False
        )

    def on_accordion_expand(*images):
        return [gr.update() for _ in images]

    accordion.expand(
        fn=on_accordion_expand,
        inputs=cn_images,
        outputs=cn_images,
        show_progress=False
    )


def create_anima_controlnet_lllite_event_handlers(prefix, ui_components):
    cn_rows = ui_components.get(f'anima_controlnet_lllite_rows_{prefix}')
    if not cn_rows: return
    cn_types = ui_components[f'anima_controlnet_lllite_types_{prefix}']
    cn_series = ui_components[f'anima_controlnet_lllite_series_{prefix}']
    cn_filepaths = ui_components[f'anima_controlnet_lllite_filepaths_{prefix}']
    cn_images = ui_components[f'anima_controlnet_lllite_images_{prefix}']
    cn_strengths = ui_components[f'anima_controlnet_lllite_strengths_{prefix}']

    count_state = ui_components[f'anima_controlnet_lllite_count_state_{prefix}']
    add_button = ui_components[f'add_anima_controlnet_lllite_button_{prefix}']
    del_button = ui_components[f'delete_anima_controlnet_lllite_button_{prefix}']
    accordion = ui_components[f'anima_controlnet_lllite_accordion_{prefix}']

    def add_cn_row(c):
        c += 1
        updates = {
            count_state: c,
            cn_rows[c-1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_CONTROLNETS),
            del_button: gr.update(visible=True)
        }
        return updates

    def del_cn_row(c):
        c -= 1
        updates = {
            count_state: c,
            cn_rows[c]: gr.update(visible=False),
            cn_images[c]: None,
            cn_strengths[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0)
        }
        return updates

    add_outputs = [count_state, add_button, del_button] + cn_rows
    del_outputs = [count_state, add_button, del_button] + cn_rows + cn_images + cn_strengths
    add_button.click(fn=add_cn_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_cn_row, inputs=[count_state], outputs=del_outputs, show_progress=False)

    def on_cn_type_change(selected_type):
        cn_config = load_anima_controlnet_lllite_config()
        series_choices = []
        if selected_type:
            series_choices = sorted(list(set(
                model.get("Series", "Default") for model in cn_config
                if selected_type in model.get("Type", [])
            )))
        default_series = series_choices[0] if series_choices else None
        filepath = "None"
        if default_series:
            for model in cn_config:
                if model.get("Series") == default_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return gr.update(choices=series_choices, value=default_series), filepath

    def on_cn_series_change(selected_series, selected_type):
        cn_config = load_anima_controlnet_lllite_config()
        filepath = "None"
        if selected_series and selected_type:
            for model in cn_config:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    for i in range(MAX_CONTROLNETS):
        cn_types[i].change(
            fn=on_cn_type_change,
            inputs=[cn_types[i]],
            outputs=[cn_series[i], cn_filepaths[i]],
            show_progress=False
        )
        cn_series[i].change(
            fn=on_cn_series_change,
            inputs=[cn_series[i], cn_types[i]],
            outputs=[cn_filepaths[i]],
            show_progress=False
        )

    def on_accordion_expand(*images):
        return [gr.update() for _ in images]

    accordion.expand(
        fn=on_accordion_expand,
        inputs=cn_images,
        outputs=cn_images,
        show_progress=False
    )


def create_diffsynth_controlnet_event_handlers(prefix, ui_components):
    cn_rows = ui_components.get(f'diffsynth_controlnet_rows_{prefix}')
    if not cn_rows: return
    cn_types = ui_components[f'diffsynth_controlnet_types_{prefix}']
    cn_series = ui_components[f'diffsynth_controlnet_series_{prefix}']
    cn_filepaths = ui_components[f'diffsynth_controlnet_filepaths_{prefix}']
    cn_images = ui_components[f'diffsynth_controlnet_images_{prefix}']
    cn_strengths = ui_components[f'diffsynth_controlnet_strengths_{prefix}']

    count_state = ui_components[f'diffsynth_controlnet_count_state_{prefix}']
    add_button = ui_components[f'add_diffsynth_controlnet_button_{prefix}']
    del_button = ui_components[f'delete_diffsynth_controlnet_button_{prefix}']
    accordion = ui_components[f'diffsynth_controlnet_accordion_{prefix}']

    base_model_comp = ui_components.get(f'base_model_{prefix}')
    actual_arch_comp = base_model_comp if base_model_comp else gr.State("SDXL")

    def add_cn_row(c):
        c += 1
        updates = {
            count_state: c,
            cn_rows[c-1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_CONTROLNETS),
            del_button: gr.update(visible=True)
        }
        return updates

    def del_cn_row(c):
        c -= 1
        updates = {
            count_state: c,
            cn_rows[c]: gr.update(visible=False),
            cn_images[c]: None,
            cn_strengths[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0)
        }
        return updates

    add_outputs = [count_state, add_button, del_button] + cn_rows
    del_outputs = [count_state, add_button, del_button] + cn_rows + cn_images + cn_strengths
    add_button.click(fn=add_cn_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_cn_row, inputs=[count_state], outputs=del_outputs, show_progress=False)

    def on_cn_type_change(selected_type, model_name):
        from core.settings import MODEL_TYPE_MAP
        m_type = MODEL_TYPE_MAP.get(model_name, "SDXL") if model_name else "SDXL"
        cn_full_config = load_diffsynth_controlnet_config()

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        cn_config = cn_full_config.get(controlnet_key, [])
        series_choices = []
        if selected_type:
            series_choices = sorted(list(set(
                model.get("Series", "Default") for model in cn_config
                if selected_type in model.get("Type", [])
            )))
        default_series = series_choices[0] if series_choices else None
        filepath = "None"
        if default_series:
            for model in cn_config:
                if model.get("Series") == default_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return gr.update(choices=series_choices, value=default_series), filepath

    def on_cn_series_change(selected_series, selected_type, model_name):
        from core.settings import MODEL_TYPE_MAP
        m_type = MODEL_TYPE_MAP.get(model_name, "SDXL") if model_name else "SDXL"
        cn_full_config = load_diffsynth_controlnet_config()

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        cn_config = cn_full_config.get(controlnet_key, [])
        filepath = "None"
        if selected_series and selected_type:
            for model in cn_config:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    for i in range(MAX_CONTROLNETS):
        cn_types[i].change(
            fn=on_cn_type_change,
            inputs=[cn_types[i], actual_arch_comp],
            outputs=[cn_series[i], cn_filepaths[i]],
            show_progress=False
        )
        cn_series[i].change(
            fn=on_cn_series_change,
            inputs=[cn_series[i], cn_types[i], actual_arch_comp],
            outputs=[cn_filepaths[i]],
            show_progress=False
        )

    def on_accordion_expand(*images):
        return [gr.update() for _ in images]

    accordion.expand(
        fn=on_accordion_expand,
        inputs=cn_images,
        outputs=cn_images,
        show_progress=False
    )


def create_flux1_ipadapter_event_handlers(prefix, ui_components):
    fipa_rows = ui_components.get(f'flux1_ipadapter_rows_{prefix}')
    if not fipa_rows: return
    count_state = ui_components[f'flux1_ipadapter_count_state_{prefix}']
    add_button = ui_components[f'add_flux1_ipadapter_button_{prefix}']
    del_button = ui_components[f'delete_flux1_ipadapter_button_{prefix}']
    images = ui_components[f'flux1_ipadapter_images_{prefix}']
    weights = ui_components[f'flux1_ipadapter_weights_{prefix}']
    start_percents = ui_components[f'flux1_ipadapter_start_percents_{prefix}']
    end_percents = ui_components[f'flux1_ipadapter_end_percents_{prefix}']

    def add_fipa_row(c):
        c += 1
        return {
            count_state: c,
            fipa_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_IPADAPTERS),
            del_button: gr.update(visible=True),
        }

    def del_fipa_row(c):
        c -= 1
        return {
            count_state: c,
            fipa_rows[c]: gr.update(visible=False),
            images[c]: None,
            weights[c]: 0.6,
            start_percents[c]: 0.0,
            end_percents[c]: 0.6,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + fipa_rows
    del_outputs = [count_state, add_button, del_button] + fipa_rows + images + weights + start_percents + end_percents
    add_button.click(fn=add_fipa_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_fipa_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_sd3_ipadapter_event_handlers(prefix, ui_components):
    ipa_rows = ui_components.get(f'sd3_ipadapter_rows_{prefix}')
    if not ipa_rows: return
    count_state = ui_components[f'sd3_ipadapter_count_state_{prefix}']
    add_button = ui_components[f'add_sd3_ipadapter_button_{prefix}']
    del_button = ui_components[f'delete_sd3_ipadapter_button_{prefix}']
    images = ui_components[f'sd3_ipadapter_images_{prefix}']
    weights = ui_components[f'sd3_ipadapter_weights_{prefix}']
    start_percents = ui_components[f'sd3_ipadapter_start_percents_{prefix}']
    end_percents = ui_components[f'sd3_ipadapter_end_percents_{prefix}']

    def add_ipa_row(c):
        c += 1
        return {
            count_state: c,
            ipa_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_IPADAPTERS),
            del_button: gr.update(visible=True),
        }

    def del_ipa_row(c):
        c -= 1
        return {
            count_state: c,
            ipa_rows[c]: gr.update(visible=False),
            images[c]: None,
            weights[c]: 0.5,
            start_percents[c]: 0.0,
            end_percents[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ipa_rows
    del_outputs = [count_state, add_button, del_button] + ipa_rows + images + weights + start_percents + end_percents
    add_button.click(fn=add_ipa_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ipa_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_style_event_handlers(prefix, ui_components):
    style_rows = ui_components.get(f'style_rows_{prefix}')
    if not style_rows: return
    count_state = ui_components[f'style_count_state_{prefix}']
    add_button = ui_components[f'add_style_button_{prefix}']
    del_button = ui_components[f'delete_style_button_{prefix}']
    images = ui_components[f'style_images_{prefix}']
    strengths = ui_components[f'style_strengths_{prefix}']

    def add_style_row(c):
        c += 1
        return {
            count_state: c,
            style_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 5),
            del_button: gr.update(visible=True),
        }

    def del_style_row(c):
        c -= 1
        return {
            count_state: c,
            style_rows[c]: gr.update(visible=False),
            images[c]: None,
            strengths[c]: 1.0,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + style_rows
    del_outputs = [count_state, add_button, del_button] + style_rows + images + strengths
    add_button.click(fn=add_style_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_style_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_ipadapter_event_handlers(prefix, ui_components):
    ipa_rows = ui_components.get(f'ipadapter_rows_{prefix}')
    if not ipa_rows: return
    ipa_lora_strengths = ui_components[f'ipadapter_lora_strengths_{prefix}']
    ipa_final_preset = ui_components[f'ipadapter_final_preset_{prefix}']
    ipa_final_lora_strength = ui_components[f'ipadapter_final_lora_strength_{prefix}']
    count_state = ui_components[f'ipadapter_count_state_{prefix}']
    add_button = ui_components[f'add_ipadapter_button_{prefix}']
    del_button = ui_components[f'delete_ipadapter_button_{prefix}']
    accordion = ui_components[f'ipadapter_accordion_{prefix}']
    images = ui_components[f'ipadapter_images_{prefix}']
    weights = ui_components[f'ipadapter_weights_{prefix}']

    def add_ipa_row(c):
        c += 1
        return {
            count_state: c,
            ipa_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_IPADAPTERS),
            del_button: gr.update(visible=True),
        }

    def del_ipa_row(c):
        c -= 1
        return {
            count_state: c,
            ipa_rows[c]: gr.update(visible=False),
            images[c]: None,
            weights[c]: 1.0,
            ipa_lora_strengths[c]: 0.6,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ipa_rows
    del_outputs = [count_state, add_button, del_button] + ipa_rows + images + weights + ipa_lora_strengths
    add_button.click(fn=add_ipa_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ipa_row, inputs=[count_state], outputs=del_outputs, show_progress=False)

    def on_preset_change(preset_value):
        config = load_ipadapter_config()
        faceid_presets = []
        if config:
            faceid_presets.extend(config.get("IPAdapter_FaceID_presets", {}).get("SDXL", []))
            faceid_presets.extend(config.get("IPAdapter_FaceID_presets", {}).get("SD1.5", []))

        is_visible = preset_value in faceid_presets
        updates = [gr.update(visible=is_visible)] * (MAX_IPADAPTERS + 1)
        return updates

    all_lora_strength_sliders = [ipa_final_lora_strength] + ipa_lora_strengths
    ipa_final_preset.change(fn=on_preset_change, inputs=[ipa_final_preset], outputs=all_lora_strength_sliders, show_progress=False)

    accordion.expand(fn=lambda *imgs: [gr.update() for _ in imgs], inputs=ui_components[f'ipadapter_images_{prefix}'], outputs=ui_components[f'ipadapter_images_{prefix}'], show_progress=False)


def create_reference_latent_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'reference_latent_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'reference_latent_count_state_{prefix}']
    add_button = ui_components[f'add_reference_latent_button_{prefix}']
    del_button = ui_components[f'delete_reference_latent_button_{prefix}']
    images = ui_components[f'reference_latent_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 10),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_hidream_o1_reference_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'hidream_o1_reference_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'hidream_o1_reference_count_state_{prefix}']
    add_button = ui_components[f'add_hidream_o1_reference_button_{prefix}']
    del_button = ui_components[f'delete_hidream_o1_reference_button_{prefix}']
    images = ui_components[f'hidream_o1_reference_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 10),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_joyai_reference_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'joyai_reference_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'joyai_reference_count_state_{prefix}']
    add_button = ui_components[f'add_joyai_reference_button_{prefix}']
    del_button = ui_components[f'delete_joyai_reference_button_{prefix}']
    images = ui_components[f'joyai_reference_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 2),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_reference_image_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'reference_image_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'reference_image_count_state_{prefix}']
    add_button = ui_components[f'add_reference_image_button_{prefix}']
    del_button = ui_components[f'delete_reference_image_button_{prefix}']
    images = ui_components[f'reference_image_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 10),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_embedding_event_handlers(prefix, ui_components):
    rows = ui_components.get(f'embedding_rows_{prefix}')
    if not rows: return
    ids = ui_components[f'embeddings_ids_{prefix}']
    files = ui_components[f'embeddings_files_{prefix}']
    sources = ui_components[f'embeddings_sources_{prefix}']
    upload_buttons = ui_components[f'embeddings_uploads_{prefix}']
    count_state = ui_components[f'embedding_count_state_{prefix}']
    add_button = ui_components[f'add_embedding_button_{prefix}']
    del_button = ui_components[f'delete_embedding_button_{prefix}']

    for i in range(MAX_EMBEDDINGS):
        upload_buttons[i].upload(
            fn=on_embedding_upload,
            inputs=[upload_buttons[i]],
            outputs=[ids[i], sources[i], files[i]],
            show_progress=True
        )

    def add_row(c):
        c += 1
        return {
            count_state: c,
            rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_EMBEDDINGS),
            del_button: gr.update(visible=True)
        }

    def del_row(c):
        c -= 1
        return {
            count_state: c,
            rows[c]: gr.update(visible=False),
            ids[c]: "",
            files[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0)
        }

    add_outputs = [count_state, add_button, del_button] + rows
    del_outputs = [count_state, add_button, del_button] + rows + ids + files
    add_button.click(fn=add_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_conditioning_event_handlers(prefix, ui_components):
    rows = ui_components.get(f'conditioning_rows_{prefix}')
    if not rows: return
    prompts = ui_components[f'conditioning_prompts_{prefix}']
    count_state = ui_components[f'conditioning_count_state_{prefix}']
    add_button = ui_components[f'add_conditioning_button_{prefix}']
    del_button = ui_components[f'delete_conditioning_button_{prefix}']

    def add_row(c):
        c += 1
        return {
            count_state: c,
            rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < MAX_CONDITIONINGS),
            del_button: gr.update(visible=True),
        }

    def del_row(c):
        c -= 1
        return {
            count_state: c,
            rows[c]: gr.update(visible=False),
            prompts[c]: "",
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + rows
    del_outputs = [count_state, add_button, del_button] + rows + prompts
    add_button.click(fn=add_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_krea2_identity_edit_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'krea2_identity_edit_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'krea2_identity_edit_count_state_{prefix}']
    add_button = ui_components[f'add_krea2_identity_edit_button_{prefix}']
    del_button = ui_components[f'delete_krea2_identity_edit_button_{prefix}']
    images = ui_components[f'krea2_identity_edit_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 2),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_qwen_image_edit_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'qwen_image_edit_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'qwen_image_edit_count_state_{prefix}']
    add_button = ui_components[f'add_qwen_image_edit_button_{prefix}']
    del_button = ui_components[f'delete_qwen_image_edit_button_{prefix}']
    images = ui_components[f'qwen_image_edit_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 3),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_krea2_reference_edit_event_handlers(prefix, ui_components):
    ref_rows = ui_components.get(f'krea2_reference_edit_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'krea2_reference_edit_count_state_{prefix}']
    add_button = ui_components[f'add_krea2_reference_edit_button_{prefix}']
    del_button = ui_components[f'delete_krea2_reference_edit_button_{prefix}']
    images = ui_components[f'krea2_reference_edit_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < 3),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 0),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)


def create_boogu_edit_event_handlers(prefix, ui_components, max_units=10):
    ref_rows = ui_components.get(f'boogu_edit_rows_{prefix}')
    if not ref_rows: return
    count_state = ui_components[f'boogu_edit_count_state_{prefix}']
    add_button = ui_components[f'add_boogu_edit_button_{prefix}']
    del_button = ui_components[f'delete_boogu_edit_button_{prefix}']
    images = ui_components[f'boogu_edit_images_{prefix}']

    def add_ref_row(c):
        c += 1
        return {
            count_state: c,
            ref_rows[c - 1]: gr.update(visible=True),
            add_button: gr.update(visible=c < max_units),
            del_button: gr.update(visible=True),
        }

    def del_ref_row(c):
        c -= 1
        return {
            count_state: c,
            ref_rows[c]: gr.update(visible=False),
            images[c]: None,
            add_button: gr.update(visible=True),
            del_button: gr.update(visible=c > 1),
        }

    add_outputs = [count_state, add_button, del_button] + ref_rows
    del_outputs = [count_state, add_button, del_button] + ref_rows + images
    add_button.click(fn=add_ref_row, inputs=[count_state], outputs=add_outputs, show_progress=False)
    del_button.click(fn=del_ref_row, inputs=[count_state], outputs=del_outputs, show_progress=False)
