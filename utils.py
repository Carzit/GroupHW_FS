import os
import sys
import ast
import psutil
import inspect
import logging
import logging.handlers
import argparse
from functools import wraps
from typing import Tuple, Literal, Union, Optional, List, Dict, Any

import math
import pickle
import json
import toml
import yaml
import random

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import safetensors
import safetensors.torch
from matplotlib import pyplot as plt
from colorama import Fore, Style, init
from IPython.display import display, HTML


def str2bool(value:Union[bool, str]):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in {'false', '0', 'no', 'n', 'f'}:
            return False
        elif value.lower() in {'true', '1', 'yes', 'y', 't'}:
            return True
    else:
        raise argparse.ArgumentTypeError(f'Boolean value or bool like string expected. Get unexpected value {value}, whose type is {type(value)}')

def str2dict(args_list):
    result_dict = {}
    if args_list is not None and len(args_list) > 0:
        for arg in args_list:
            key, value = arg.split("=", 1)  # 使用 1 限制分割次数，避免错误处理包含 '=' 的值
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError): # 如果 literal_eval 失败，就把 value 当作字符串处理
                pass
            result_dict[key] = value
    return result_dict

def str2dtype(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return torch.float32
    elif dtype == "FP64":
        return torch.float64
    elif dtype == "FP16":
        return torch.float16
    elif dtype == "BF16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
def dtype2str(dtype:torch.dtype) -> str:
    if dtype == torch.float32:
        return "FP32"
    elif dtype == torch.float64:
        return "FP64"
    elif dtype == torch.float16:
        return "FP16"
    elif dtype == torch.bfloat16:
        return "BF16"
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `torch.float32`, `torch.float64`, `torch.float16` or `torch.bfloat16`.")
    
def str2dtype_np(dtype:Literal["FP32", "FP64", "FP16", "BF16"]) -> torch.dtype:
    if dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "FP16":
        return np.float16
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `FP32`, `FP64`, `FP16` or `BF16`.")
    
def dtype2str_np(dtype:torch.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    elif dtype == np.float64:
        return "FP64"
    elif dtype == np.float16:
        return "FP16"
    else:
        raise ValueError(f"Unexpected dtype `{dtype}`. dtype must be `numpy.float32`, `numpy.float64` or `numpy.float16`.")
    
def str2device(device:Literal["auto", "cpu", "cuda"]) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif device.lower() == "cpu":
        return torch.device("cpu")
    elif device.lower() == "cuda":
        return torch.device("cuda")
    else:
        raise argparse.ArgumentTypeError(f"Unexpected device `{device}`. dtype must be `cuda`, `cpu` or `auto`.")

def save_dataframe(df:Union[pd.DataFrame, pd.Series], 
                   path:str, 
                   format:Literal["auto", "csv", "pkl", "parquet", "feather"]="auto",
                   **kwargs) -> None:
    func_map = {
        "csv": df.to_csv,
        "pkl": df.to_pickle,
        "parquet": df.to_parquet,
        "feather": df.to_feather
    }
    if format == "auto":
        format = os.path.basename(path)[os.path.basename(path).find(".")+1:]
    else:
        if not path.endswith(format):
            path += f".{format}"
    func = func_map.get(format)
    if not func:
        raise ValueError(f"Unsupported format: {format}")

    filtered_kwargs, _ = filter_func_kwargs(func, kwargs)
    
    return func(path, **filtered_kwargs)

def load_dataframe(path:str, 
                   format:Literal["auto", "csv", "pkl", "parquet", "feather"]="auto",
                   **kwargs) -> pd.DataFrame:
    func_map = {
        "csv": pd.read_csv,
        "pkl": pd.read_pickle,
        "parquet": pd.read_parquet,
        "feather": pd.read_feather
    }
    if format == "auto":
        format = os.path.basename(path)[os.path.basename(path).find(".")+1:]
    func = func_map.get(format)
    if not func:
        raise ValueError(f"Unsupported format: {format}")

    filtered_kwargs, _ = filter_func_kwargs(func, kwargs)

    if format == "csv" and "index_col" not in filtered_kwargs:
        filtered_kwargs["index_col"] = 0
    
    return func(path, **filtered_kwargs)

def save_checkpoint(model:torch.nn.Module, save_folder:str, save_name:str, save_format:Literal[".pt",".safetensors"]=".pt"):
    save_path = os.path.join(save_folder, save_name+save_format)
    if save_format == ".pt":
        torch.save(model.state_dict(), save_path)
    elif save_format == ".safetensors":
        safetensors.torch.save_file(model.state_dict(), save_path)
    else:
        raise ValueError(f"Unrecognized file format`{save_format}`")

def load_checkpoint(model:torch.nn.Module, checkpoint_path:str):
    if checkpoint_path.endswith(".pt"):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    elif checkpoint_path.endswith(".safetensors"):
        model.load_state_dict(safetensors.torch.load_file(checkpoint_path))
    else:
        raise ValueError(f"Unrecognized model weights file `{checkpoint_path}`")

def load(path:str):
    if path.endswith(".pt"):
        return torch.load(f=path, weights_only=False)
    elif path.endswith(".safetensors"):
        return safetensors.torch.load_file(filename=path)
    elif path.endswith(".pkl"):
        with open(path, "r") as f:
            r = pickle.load(f)
        return r
    else:
        raise ValueError(f"Unrecognized file `{path}`")
    
def save(obj:Dict[str, torch.Tensor], path:str):
    if path.endswith(".pt"):
        torch.save(obj=obj, f=path)
    elif path.endswith(".safetensors"):
        safetensors.torch.save_file(tensors=obj, filename=path)
    elif path.endswith(".pkl"):
        with open(path, "w") as f:
            pickle.dump(f)
    else:
        raise ValueError(f"Unrecognized file `{path}`")

def check(tensor:torch.Tensor):
    return torch.any(torch.isnan(tensor) | torch.isinf(tensor))

def check_attr_dict_match(obj, dic:Dict, names:List[str]):
    for name in names:
        assert hasattr(obj, name) and name in dic
        assert getattr(obj, name) == dic[name]

def read_configs(config_file:str) -> Dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found.")
    file_ext = os.path.splitext(config_file)[1].lower()

    with open(config_file, 'r') as f:
        if file_ext == '.json':
            config_dict = json.load(f)
        elif file_ext == '.toml':
            config_dict = toml.load(f)
        elif file_ext in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    return config_dict

def save_configs(config_file:str, config_dict:Dict):
    file_ext = os.path.splitext(config_file)[1].lower()
    with open(config_file, 'w') as f:
        if file_ext == '.json':
            json.dump(config_dict, f, indent=4)
        elif file_ext == '.toml':
            toml.dump(config_dict, f)
        elif file_ext in ['.yaml', '.yml']:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

def find_common_root(dirs:str):
    if not dirs:
        raise ValueError("The list of directories is empty.")
    try:
        common_root = os.path.commonpath(dirs)
        if not any([os.path.commonprefix([dir_, common_root]) == common_root for dir_ in dirs]):
            raise RuntimeError("No common root directory found.")
        return common_root
    except ValueError:
        raise RuntimeError("No common root directory found.")

def calculate_chunk_size(file_path, avg_row_size=None, memory_fraction=0.5):
    """
    根据系统可用内存和文件大小动态计算 chunk_size。用于分块读取大文件时自动指定 chunk_size 参数。

    :param file_path: 文件路径
    :param avg_row_size: 每行数据平均占用的字节数, 默认为 None, 会尝试从文件中读取部分数据进行估算
    :param memory_fraction: 使用的内存比例, 默认为 50%
    :return: 动态计算得到的 chunk_size
    """
    
    file_size = os.path.getsize(file_path) # 获取文件大小（字节）
    available_memory = psutil.virtual_memory().available # 获取系统可用内存（字节）

    if avg_row_size is None:
        with open(file_path, 'r') as f:
            sample_lines = [next(f) for _ in range(100)]
            avg_row_size = sum(len(line.encode('utf-8')) for line in sample_lines) // len(sample_lines) # 获取每行数据平均占用的字节数

    max_rows_in_memory = (available_memory * memory_fraction) // avg_row_size # 计算可用行数
    estimated_rows = file_size // avg_row_size # 计算文件总行数的估计值
    chunk_size = min(max_rows_in_memory, estimated_rows) # 动态设置 chunk_size，确保不超过可用内存

    return max(chunk_size, 1)     # 确保 chunk_size 至少为 1

class MeanVarianceAccumulator:
    def __init__(self):
        self._n = 0        # 计数器
        self._mean = 0.0   # 均值
        self._m2 = 0.0     # 用于计算方差的中间量

    def accumulate(self, x):
        if not np.isnan(x):
            # 更新计数
            self._n += 1

            # 计算新的均值
            delta = x - self._mean
            self._mean += delta / self._n

            # 更新M2，用于方差计算
            delta2 = x - self._mean
            self._m2 += delta * delta2

    def clear(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0 # 重置计数器和中间量

    @property
    def count(self):
        return self._n
    
    @property
    def sum(self):
        return self._mean * self._n

    def var(self, ddof:int=0):
        if self._n < 2:
            return float('nan')  # 当样本数小于2时，方差无定义
        return self._m2 / (self._n - ddof)  # 使用无偏估计
    
    def std(self, ddof:int=0):
        return math.sqrt(self.var(ddof=ddof))

    @property
    def mean(self):
        return self._mean
    
    def __enter__(self):
        """进入上下文时，初始化/重置计算"""
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开上下文时存储最终的均值和方差"""
        pass

class Plotter:
    def __init__(self) -> None:
        self.logger:logging.Logger = logging

    def set_logger(self, logger:logging.Logger):
        if not logger:
            logger = logging
        self.logger = logger
    
    def plot_score(self, pred_scores:List[float], metric:str):
        plt.figure(figsize=(10, 6))
        plt.plot(pred_scores, marker='', color="b")
        plt.title(f'{metric} Scores')
        plt.xlabel('Date')
        plt.ylabel('Score')
    
    def plot_pred_sample(self, y_true_list:List[float], y_pred_list:List[float], y_hat_list:Optional[List[float]]=None, idx=0):
        y_true_list = [y_true[idx].item() for y_true in y_true_list]
        y_pred_list = [y_pred[idx].item() for y_pred in y_pred_list]

        plt.figure(figsize=(10, 6))
        plt.plot(y_true_list, label='y true', marker='', color="g")
        plt.plot(y_pred_list, label='y pred', marker='', color="r")

        if y_hat_list:
            y_hat_list = [y_hat[idx].item() for y_hat in y_hat_list]
            plt.plot(y_hat_list, label='y hat', marker='', color="b")

        plt.legend()
        plt.title('Comparison of y_true and y_pred')
        plt.xlabel('Date')
        plt.ylabel('Value')

    def save_fig(self, filename:str):
        if not filename.endswith(".png"):
            filename = filename + ".png"
        plt.savefig(filename)
        plt.close()

def is_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except:
        return False

class ColoredFormatter(logging.Formatter):
    """A log formatter that adapts for terminal or Jupyter Notebook with color support."""
    init(autoreset=True)
    LEVEL_COLORS = {
        logging.DEBUG: ('purple', Fore.BLUE),
        logging.INFO: ('navy', Fore.CYAN),
        logging.WARNING: ('orange', Fore.YELLOW),
        logging.ERROR: ('OrangeRed', Fore.RED),
        logging.CRITICAL: ('OrangeRed', Fore.RED + Style.BRIGHT),
    }


    def format(self, record):
        base_message = super().format(record)
        plain_message = record.getMessage()

        html_color, ansi_color = self.LEVEL_COLORS.get(record.levelno, ('black', Fore.WHITE))

        if is_notebook():
            # For Jupyter Notebook, use HTML formatting
            colored_message = f"<span style='color:{html_color};'>{plain_message}</span>"
            return base_message.replace(plain_message, colored_message)
        else:
            # For terminal, use ANSI colors
            return base_message.replace(plain_message, f"{ansi_color}{plain_message}{Style.RESET_ALL}")


class TqdmLoggingHandler(logging.StreamHandler):
    """Use tqdm.write to avoid disrupting progress bars"""
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:  # prevent empty returns in notebook
                if is_notebook():
                    # Use IPython's display function for HTML output
                    display(HTML(msg))
                else:
                    tqdm.write(msg, end=self.terminator)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class LoggerPreparer:
    def __init__(self,
                 name: str = 'logger',
                 console_level=logging.DEBUG,
                 file_level=logging.DEBUG,
                 log_file: str = None,
                 max_bytes: int = 1e6,
                 backup_count: int = 5):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # 清除已有的 handlers
        self.logger.propagate = False  # 避免重复日志

        formatter = ColoredFormatter('%(asctime)s [%(name)s][%(levelname)s]: %(message)s')

        # Console/Notebook handler
        #if is_notebook():
            #stream_handler = logging.StreamHandler(sys.stdout)  # display() 会接管输出
        #else:
            #stream_handler = TqdmLoggingHandler()
        stream_handler = TqdmLoggingHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # File logging
        if log_file:
            log_folder = os.path.dirname(os.path.normpath(log_file))
            if log_folder:
                os.makedirs(log_folder, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, mode='a', maxBytes=int(max_bytes), backupCount=backup_count
            )
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter('%(asctime)s [%(name)s][%(levelname)s]: %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def prepare(self) -> logging.Logger:
        return self.logger
    
def log_exceptions(logger):
    """
    装饰器，用于捕获函数中的异常并通过提供的logger记录它们。
    :param logger: 传入的logger对象，用于记录异常信息
    :return: 返回装饰后的函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # 使用传入的logger来记录异常信息
                logger.error("Exception occurred in function `%s`", func.__name__, exc_info=True)
                # 可选择是否在这里再次抛出异常
                # raise e
        return wrapper
    return decorator

def log_exceptions_inclass(logger_attr:str="logger"):
    """
    装饰器工厂函数，返回一个用于捕获异常并记录日志的装饰器。
    装饰器会在运行时访问类实例的logger属性。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # 在运行时通过self访问logger
                logger = getattr(self, logger_attr, None)
                if logger is not None:
                    logger.error(f"Exception occurred in {func.__name__}", exc_info=True)
                else:
                    print("Logger not found!")
                raise e
        return wrapper
    return decorator

def filter_func_kwargs(func, kwargs):
    """
    根据函数签名过滤关键字参数，并打印出被过滤掉的参数。
    
    :param func: 要检查其签名的函数。
    :param kwargs: 关键字参数字典。
    :return: 返回过滤后的关键字参数字典。
    """
    # 获取 func 的签名
    if kwargs is None:
        kwargs = {}
    sig = inspect.signature(func)
    
    # 初始化用于存储过滤后的参数和被drop的参数
    filtered_kwargs = {}
    dropped_kwargs = {}
    
    # 过滤无关的 kwargs 并找出被drop的参数
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered_kwargs[k] = v
        else:
            dropped_kwargs[k] = v
    
    # 返回过滤后的关键字参数
    return filtered_kwargs, dropped_kwargs

def check_vram(device:torch.device):
    """
    检查显存使用情况并返回当前显存的使用量和缓存量（以MB为单位）。

    返回:
        dict: {"allocated_memory": float, "cached_memory": float}
    """
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    cached_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return {"allocated_memory": allocated_memory, "cached_memory": cached_memory}

def get_memory_usage():
    """
    获取当前进程占用的内存（以MB为单位）及其占系统总内存的比例。
    
    返回:
        dict: {"used_memory_mb": float, "percent_of_total": float}
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss  # 使用中的常驻内存大小（单位为字节）
    total_memory = psutil.virtual_memory().total
    used_memory_mb = mem_info / 1024 / 1024
    percent = mem_info / total_memory * 100
    return {"used_memory_mb": used_memory_mb, "percent_of_total": percent}

def ensure_dir(directory:str):
    """ 确保文件夹存在 """
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
