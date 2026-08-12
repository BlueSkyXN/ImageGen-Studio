import torch
from collections import defaultdict, deque
from typing import Dict, Any, List
from comfy_integration.nodes import NODE_CLASS_MAPPINGS
from imagegen_utils.app_utils import get_value_at_index

class WorkflowExecutor:
    @staticmethod
    def topological_sort(workflow: Dict[str, Any]) -> List[str]:
        graph = defaultdict(list)
        in_degree = {node_id: 0 for node_id in workflow}

        for node_id, node_info in workflow.items():
            for input_value in node_info.get('inputs', {}).values():
                if isinstance(input_value, list) and len(input_value) == 2 and isinstance(input_value[0], str):
                    source_node_id = input_value[0]
                    if source_node_id in workflow:
                        graph[source_node_id].append(node_id)
                        in_degree[node_id] += 1

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        sorted_nodes = []
        while queue:
            current_node_id = queue.popleft()
            sorted_nodes.append(current_node_id)

            for neighbor_node_id in graph[current_node_id]:
                in_degree[neighbor_node_id] -= 1
                if in_degree[neighbor_node_id] == 0:
                    queue.append(neighbor_node_id)

        if len(sorted_nodes) != len(workflow):
            raise RuntimeError("Workflow contains a cycle and cannot be executed.")

        return sorted_nodes

    @staticmethod
    def execute_workflow(workflow: Dict[str, Any], initial_objects: Dict[str, Any]):
        with torch.no_grad():
            computed_outputs = initial_objects

            try:
                sorted_node_ids = WorkflowExecutor.topological_sort(workflow)

                final_node_id = None
                for node_id in reversed(sorted_node_ids):
                    if workflow[node_id].get('class_type') == 'SaveImage':
                        final_node_id = node_id
                        break

                if final_node_id:
                    required_nodes = set()
                    nodes_to_visit = [final_node_id]
                    while nodes_to_visit:
                        curr_id = nodes_to_visit.pop()
                        if curr_id in required_nodes:
                            continue
                        required_nodes.add(curr_id)
                        curr_info = workflow.get(curr_id, {})
                        for input_val in curr_info.get('inputs', {}).values():
                            if isinstance(input_val, list) and len(input_val) == 2 and isinstance(input_val[0], str):
                                src_id = input_val[0]
                                if src_id in workflow and src_id not in required_nodes:
                                    nodes_to_visit.append(src_id)

                    sorted_node_ids = [nid for nid in sorted_node_ids if nid in required_nodes]

                print(f"--- [Workflow Executor] Execution order: {sorted_node_ids}")
            except RuntimeError as e:
                print("--- [Workflow Executor] ERROR: Failed to sort workflow. Dumping graph details. ---")
                for node_id, node_info in workflow.items():
                    print(f"  Node {node_id} ({node_info['class_type']}):")
                    for input_name, input_value in node_info['inputs'].items():
                         if isinstance(input_value, list) and len(input_value) == 2 and isinstance(input_value[0], str):
                             print(f"    - {input_name} <- [{input_value[0]}, {input_value[1]}]")
                raise e

            for node_id in sorted_node_ids:
                if node_id in computed_outputs:
                    continue

                node_info = workflow[node_id]
                class_type = node_info['class_type']

                # The pipeline writes the final PNG itself so it can attach the
                # canonical parameters and full workflow metadata. Executing
                # ComfyUI's SaveImage here would create an untracked duplicate.
                if class_type == 'SaveImage':
                    continue

                is_loader_with_filename = 'Loader' in class_type and any(key.endswith('_name') for key in node_info['inputs'])
                if node_id in initial_objects and is_loader_with_filename:
                    continue

                node_class = NODE_CLASS_MAPPINGS.get(class_type)
                if node_class is None:
                     raise RuntimeError(f"Could not find node class '{class_type}'. Is it imported in comfy_integration/nodes.py?")

                node_instance = node_class()

                kwargs = {}
                for param_name, param_value in node_info['inputs'].items():
                    if isinstance(param_value, list) and len(param_value) == 2 and isinstance(param_value[0], str):
                        source_node_id, output_index = param_value
                        if source_node_id not in computed_outputs:
                            raise RuntimeError(f"Workflow integrity error: Output of node {source_node_id} needed for {node_id} but not yet computed.")

                        source_output_tuple = computed_outputs[source_node_id]
                        actual_value = get_value_at_index(source_output_tuple, output_index)
                    else:
                        actual_value = param_value

                    if '.' in param_name:
                        parent_key, child_key = param_name.split('.', 1)
                        if parent_key not in kwargs or not isinstance(kwargs[parent_key], dict):
                            kwargs[parent_key] = {}
                        kwargs[parent_key][child_key] = actual_value
                    else:
                        kwargs[param_name] = actual_value

                function_name = getattr(node_class, 'FUNCTION')
                execution_method = getattr(node_instance, function_name)

                result = execution_method(**kwargs)
                computed_outputs[node_id] = result

            final_node_id = None
            for node_id in reversed(sorted_node_ids):
                 if workflow[node_id]['class_type'] == 'SaveImage':
                     final_node_id = node_id
                     break

            if not final_node_id:
                raise RuntimeError("Workflow does not contain a 'SaveImage' node as the output.")

            save_image_inputs = workflow[final_node_id]['inputs']
            image_source_node_id, image_source_index = save_image_inputs['images']

            return get_value_at_index(computed_outputs[image_source_node_id], image_source_index)
