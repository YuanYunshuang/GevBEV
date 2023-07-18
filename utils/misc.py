import os
import logging
import time

import numpy as np
import torch
from rich.logging import RichHandler


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


def setup_logger(exp_name, debug):
    from imp import reload

    reload(logging)
    # reload() reloads a previously imported module. This is useful if you have edited the module source file using an
    # external editor and want to try out the new version without leaving the Python interpreter.

    CUDA_TAG = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    EXP_TAG = exp_name

    logger_config = dict(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"{CUDA_TAG}:[{EXP_TAG}] %(message)s",
        handlers=[RichHandler()],
        datefmt="[%X]",
    )
    logging.basicConfig(**logger_config)


def load_from_pl_state_dict(model, pl_state_dict):
    state_dict = {}
    for k, v in pl_state_dict.items():
        state_dict[k[6:]] = v
    model.load_state_dict(state_dict)
    return model


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def pad_list_to_array_np(data):
    """
    Pad list of numpy data to one single numpy array
    :param data: list of np.ndarray
    :return: np.ndarray
    """
    B = len(data)
    cnt = [len(d) for d in data]
    max_cnt = max(cnt)
    out = np.zeros(B, max_cnt, *data[0].shape[1:])
    for b in range(B):
        out[b, :cnt[b]] = data[b]
    return out


def pad_list_to_array_torch(data):
    """
    Pad list of numpy data to one single numpy array
    :param data: list of np.ndarray
    :return: np.ndarray
    """
    B = len(data)
    cnt = [len(d) for d in data]
    max_cnt = max(cnt)
    out = torch.zeros((B, max_cnt,) + tuple(data[0].shape[1:]),
                      device=data[0].device, dtype=data[0].dtype)
    for b in range(B):
        out[b, :cnt[b]] = data[b]
    return out


def print_exec_time(func):
    """decorate function for evaluating runtime of a normal function"""
    runtimes = {}
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime of {func.__name__:30s}: {runtime * 1000:03.6f} ms")
        if func.__name__ not in runtimes:
            runtimes[func.__name__] = [runtime]
        else:
            runtimes[func.__name__] += [runtime]
        return result

    def get_mean_runtime():
        if runtimes:
            mean_runtime = sum(runtimes[func.__name__]) / len(runtimes[func.__name__])
            ss = f"Mean runtime of {func.__name__:30s}: {mean_runtime * 1000:03.6f} ms"
            print("=" * len(ss))
            print(ss)
        else:
            print("No runtimes recorded.")

    wrapper.get_mean_runtime = get_mean_runtime
    return wrapper
