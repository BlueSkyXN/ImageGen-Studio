import yaml
import os
import importlib
from copy import deepcopy
from comfy_integration.nodes import NODE_CLASS_MAPPINGS
from chain_injectors import discover_injectors, get_registered_features
from core.settings import FEATURES_CONFIG

class WorkflowAssembler:
    def __init__(self, recipe_path, dynamic_values=None):
        self.base_path = os.path.dirname(recipe_path)
        self.dynamic_values = dynamic_values or {}
        self.node_counter = 0
        self.workflow = {}
        self.node_map = {}

        model_type = self.dynamic_values.get('model_type')
        self._load_injector_config(model_type=model_type)

        self.recipe = self._load_and_merge_recipe(os.path.basename(recipe_path), self.dynamic_values)

    def _load_injector_config(self, model_type=None):
        self.global_injectors = discover_injectors()
        registered_features = get_registered_features()

        order = []
        if model_type and model_type in FEATURES_CONFIG:
            enabled_features = FEATURES_CONFIG[model_type].get('enabled_chains', [])
            for feat in enabled_features:
                if feat in registered_features:
                    chain_key = registered_features[feat]['chain_type']
                else:
                    chain_key = f"dynamic_{feat}_chains"
                if chain_key in self.global_injectors and chain_key not in order:
                    order.append(chain_key)

        for chain_key in self.global_injectors.keys():
            if chain_key not in order:
                order.append(chain_key)

        self.injector_order = order

    def _get_unique_id(self):
        self.node_counter += 1
        return str(self.node_counter)

    def _get_node_template(self, class_type):
        if class_type not in NODE_CLASS_MAPPINGS:
            raise ValueError(f"Node class '{class_type}' not found. Ensure it's correctly imported in comfy_integration/nodes.py.")

        node_class = NODE_CLASS_MAPPINGS[class_type]
        input_types = node_class.INPUT_TYPES()

        template = {
            "inputs": {},
            "class_type": class_type,
            "_meta": {"title": node_class.NODE_NAME if hasattr(node_class, 'NODE_NAME') else class_type}
        }

        all_inputs = {**input_types.get('required', {}), **input_types.get('optional', {})}
        for name, details in all_inputs.items():
            config = details[1] if len(details) > 1 and isinstance(details[1], dict) else {}
            template["inputs"][name] = config.get("default")

        return template

    def _load_and_merge_recipe(self, recipe_filename, dynamic_values, search_context_dir=None):
        search_path = search_context_dir or self.base_path
        recipe_path_to_use = os.path.join(search_path, recipe_filename)

        if not os.path.exists(recipe_path_to_use):
            raise FileNotFoundError(f"Recipe file not found: {recipe_path_to_use}")

        with open(recipe_path_to_use, 'r', encoding='utf-8') as f:
            content = f.read()

        for key, value in dynamic_values.items():
            if value is not None:
                content = content.replace(f"{{{{ {key} }}}}", str(value))

        main_recipe = yaml.safe_load(content)

        merged_recipe = {'nodes': {}, 'connections': [], 'ui_map': {}}
        for key in self.injector_order:
            if key.startswith('dynamic_'):
                merged_recipe[key] = {}

        parent_recipe_dir = os.path.dirname(recipe_path_to_use)
        for import_path_template in main_recipe.get('imports', []):
            import_path = import_path_template
            for key, value in dynamic_values.items():
                if value is not None:
                    import_path = import_path.replace(f"{{{{ {key} }}}}", str(value))

            try:
                imported_recipe = self._load_and_merge_recipe(import_path, dynamic_values, search_context_dir=parent_recipe_dir)
                merged_recipe['nodes'].update(imported_recipe.get('nodes', {}))
                merged_recipe['connections'].extend(imported_recipe.get('connections', []))
                merged_recipe['ui_map'].update(imported_recipe.get('ui_map', {}))
                for key in self.injector_order:
                    if key in imported_recipe and key.startswith('dynamic_'):
                        merged_recipe[key].update(imported_recipe.get(key, {}))
            except FileNotFoundError:
                print(f"Warning: Optional recipe partial '{import_path}' not found. Skipping.")

        merged_recipe['nodes'].update(main_recipe.get('nodes', {}))
        merged_recipe['connections'].extend(main_recipe.get('connections', []))
        merged_recipe['ui_map'].update(main_recipe.get('ui_map', {}))
        for key in self.injector_order:
            if key in main_recipe and key.startswith('dynamic_'):
                merged_recipe[key].update(main_recipe.get(key, {}))

        return merged_recipe

    def assemble(self, ui_values):
        self.ui_values = ui_values
        for name, details in self.recipe['nodes'].items():
            class_type = details['class_type']
            template = self._get_node_template(class_type)
            node_data = deepcopy(template)

            unique_id = self._get_unique_id()
            self.node_map[name] = unique_id

            if 'params' in details:
                for param, value in details['params'].items():
                    if param in node_data['inputs']:
                        node_data['inputs'][param] = value

            self.workflow[unique_id] = node_data

        for ui_key, target in self.recipe.get('ui_map', {}).items():
            if ui_key in ui_values and ui_values[ui_key] is not None:
                target_list = target if isinstance(target, list) else [target]
                for t in target_list:
                    target_name, target_param = t.split(':')
                    if target_name in self.node_map:
                        self.workflow[self.node_map[target_name]]['inputs'][target_param] = ui_values[ui_key]

        for conn in self.recipe.get('connections', []):
            from_name, from_output_idx = conn['from'].split(':')
            to_name, to_input_name = conn['to'].split(':')

            from_id = self.node_map.get(from_name)
            to_id = self.node_map.get(to_name)

            if from_id and to_id:
                self.workflow[to_id]['inputs'][to_input_name] = [from_id, int(from_output_idx)]

        print("--- [Assembler] Applying dynamic injectors ---")
        recipe_chain_types = {key for key in self.recipe if key.startswith('dynamic_')}
        processing_order = [key for key in self.injector_order if key in recipe_chain_types]

        for chain_type in processing_order:
            injector_func = self.global_injectors.get(chain_type)
            if injector_func:
                for chain_key, chain_def in self.recipe.get(chain_type, {}).items():
                    if chain_key in ui_values and ui_values[chain_key]:
                        print(f"  -> Injecting '{chain_type}' for '{chain_key}'...")
                        chain_items = ui_values[chain_key]
                        injector_func(self, chain_def, chain_items)

        print("--- [Assembler] Finished applying injectors ---")

        return self.workflow
