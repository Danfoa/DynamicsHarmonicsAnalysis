import importlib
import pathlib

import torch.nn


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def print_dict(d: dict, sort=False):
    str = []
    d_sorted = dict(sorted(d.items())) if sort else d
    for k, v in d_sorted.items():
        str.append(f"{k}={v}")
    return "-".join(str)

def append_dictionaries(dict1, dict2, recursive=True):
    import torch
    result = {}
    for k in set(dict1) | set(dict2):
        item1, item2 = dict1.get(k, 0), dict2.get(k, 0)
        if isinstance(item1, list) and (isinstance(item2, int) or isinstance(item2, float)):
            result[k] = item1 + [item2]
        elif isinstance(item1, int) or isinstance(item1, float):
            result[k] = [item1, item2]
        elif isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            result[k] = torch.cat((item1, item2))
        elif isinstance(item1, dict) and isinstance(item2, dict) and recursive:
            result[k] = append_dictionaries(item1, item2)
    return result

def flatten_dict(d: dict, prefix=''):
    a = {}
    for k, v in d.items():
        if isinstance(v, dict):
            a.update(flatten_dict(v, prefix=f"{k}/"))
        else:
            a[f"{prefix}{k}"] = v
    return a

def check_if_resume_experiment(ckpt_call):
    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    terminated = False
    if best_path.exists() and not ckpt_path.exists():
        terminated = True
    elif ckpt_path.exists() and best_path.exists():
        terminated = False

    return terminated, ckpt_path, best_path