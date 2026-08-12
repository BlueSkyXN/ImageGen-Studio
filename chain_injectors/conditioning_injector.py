def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')

    target_node_id = None
    target_input_name = None

    if ksampler_name in assembler.node_map:
        ksampler_id = assembler.node_map[ksampler_name]
        if 'positive' in assembler.workflow[ksampler_id]['inputs']:
            target_node_id = ksampler_id
            target_input_name = 'positive'
            print(f"Conditioning injector targeting KSampler node '{ksampler_name}'.")
    else:
        print(f"Warning: KSampler node '{ksampler_name}' for Conditioning chain not found. Skipping.")
        return

    if not target_node_id:
        print("Warning: Conditioning chain could not find a valid injection point (KSampler may be missing 'positive' input). Skipping.")
        return

    clip_source_str = chain_definition.get('clip_source')
    if not clip_source_str:
        print("Warning: 'clip_source' definition missing in the recipe for the Conditioning chain. Skipping.")
        return
    clip_node_name, clip_idx_str = clip_source_str.split(':')
    if clip_node_name not in assembler.node_map:
        print(f"Warning: CLIP source node '{clip_node_name}' for Conditioning chain not found. Skipping.")
        return
    clip_connection = [assembler.node_map[clip_node_name], int(clip_idx_str)]

    original_positive_connection = assembler.workflow[target_node_id]['inputs'][target_input_name]

    area_conditioning_outputs = []

    for item_data in chain_items:
        prompt = item_data.get('prompt', '')
        if not prompt or not prompt.strip():
            continue

        text_encode_id = assembler._get_unique_id()
        text_encode_node = assembler._get_node_template("CLIPTextEncode")
        text_encode_node['inputs']['text'] = prompt
        text_encode_node['inputs']['clip'] = clip_connection
        assembler.workflow[text_encode_id] = text_encode_node

        set_area_id = assembler._get_unique_id()
        set_area_node = assembler._get_node_template("ConditioningSetArea")
        set_area_node['inputs']['width'] = item_data.get('width', 1024)
        set_area_node['inputs']['height'] = item_data.get('height', 1024)
        set_area_node['inputs']['x'] = item_data.get('x', 0)
        set_area_node['inputs']['y'] = item_data.get('y', 0)
        set_area_node['inputs']['strength'] = item_data.get('strength', 1.0)
        set_area_node['inputs']['conditioning'] = [text_encode_id, 0]
        assembler.workflow[set_area_id] = set_area_node

        area_conditioning_outputs.append([set_area_id, 0])

    if not area_conditioning_outputs:
        return

    current_combined_conditioning = area_conditioning_outputs[0]
    if len(area_conditioning_outputs) > 1:
        for i in range(1, len(area_conditioning_outputs)):
            combine_id = assembler._get_unique_id()
            combine_node = assembler._get_node_template("ConditioningCombine")
            combine_node['inputs']['conditioning_1'] = current_combined_conditioning
            combine_node['inputs']['conditioning_2'] = area_conditioning_outputs[i]
            assembler.workflow[combine_id] = combine_node
            current_combined_conditioning = [combine_id, 0]

    final_combine_id = assembler._get_unique_id()
    final_combine_node = assembler._get_node_template("ConditioningCombine")
    final_combine_node['inputs']['conditioning_1'] = original_positive_connection
    final_combine_node['inputs']['conditioning_2'] = current_combined_conditioning
    assembler.workflow[final_combine_id] = final_combine_node

    assembler.workflow[target_node_id]['inputs'][target_input_name] = [final_combine_id, 0]
    print(f"Conditioning injector applied. Redirected '{target_input_name}' input with {len(area_conditioning_outputs)} regional prompts.")
