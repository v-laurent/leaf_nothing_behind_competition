import os
from typing import Callable, Dict

import torch
from yaecs import Configuration


def get_device(device_name: str) -> torch.device:
    if device_name == 'gpu':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_name != 'cpu':
        raise ValueError(f"Invalid name for device : '{device_name}'. Valid choices are 'gpu', 'cpu'.")
    return device_name


class BaselineConfig(Configuration):
    @staticmethod
    def get_default_config_path() -> str:
        return os.path.join(os.path.dirname(__file__), "default.yaml")

    def make_experiment_subdir(self, directory: str) -> str:
        if directory:
            os.makedirs(directory := os.path.join(self.experiment_logs, directory), exist_ok=True)
        return directory

    def make_results_subdir(self, path):
        if path and self.csv_path:
            os.makedirs(path := os.path.join(path, os.path.splitext(os.path.basename(self.csv_path))[0]), exist_ok=True)
        return path

    def parameters_pre_processing(self) -> Dict[str, Callable]:
        return {
            "tracker_config": self.register_as_tracker_config,
        }

    def parameters_post_processing(self) -> Dict[str, Callable]:
        return {
            "experiment_logs": self.register_as_experiment_path,
            "save_weights_under": self.make_experiment_subdir,
            "device": get_device,
            "save_infers_under": self.make_results_subdir,
        }
