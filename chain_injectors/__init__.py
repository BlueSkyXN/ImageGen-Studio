import os
import importlib
import pkgutil

def discover_injectors():
    injectors = {}
    package_dir = os.path.dirname(__file__)

    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if is_pkg or module_name.startswith('_'):
            continue

        full_module_name = f"chain_injectors.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            if hasattr(module, 'inject') and callable(module.inject):
                feature_name = getattr(module, 'FEATURE_NAME', None)
                if not feature_name:
                    feature_name = module_name[:-9] if module_name.endswith('_injector') else module_name

                chain_type = getattr(module, 'CHAIN_TYPE', None)
                if not chain_type:
                    chain_type = f"dynamic_{feature_name}_chains"

                injectors[chain_type] = module.inject
            else:
                print(f"Warning: Module '{full_module_name}' does not have a callable 'inject' function.")
        except Exception as e:
            print(f"Error importing injector module '{full_module_name}': {e}")

    return injectors

def get_registered_features():
    features = {}
    package_dir = os.path.dirname(__file__)

    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        if is_pkg or module_name.startswith('_'):
            continue

        feature_name = module_name[:-9] if module_name.endswith('_injector') else module_name
        full_module_name = f"chain_injectors.{module_name}"
        chain_type = f"dynamic_{feature_name}_chains"

        features[feature_name] = {
            'module': full_module_name,
            'chain_type': chain_type
        }

    return features
