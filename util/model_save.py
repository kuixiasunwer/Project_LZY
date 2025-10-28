import os
from pathlib import Path
import torch

from configs import BaseConfig
from .paths import torch_base_path, get_time

def save_model(models, tag, name=None, path=torch_base_path):
    if not os.path.exists(path):
        os.makedirs(path)

    if name is None:
        temp_model = models
        while isinstance(temp_model, list):
            temp_model = models[0]
        name = temp_model.__class__.__name__ + get_time()

    if not os.path.exists(path / name):
        os.mkdir(path / name)
    if not os.path.exists(path / name / tag):
        os.mkdir(path / name / tag)

    if isinstance(models, list):
        for idx_one, model in enumerate(models):
            if isinstance(model, torch.nn.Module):
                # joint
                torch.save(model.state_dict(), path / name / tag / f"{idx_one}.pth")
            elif isinstance(model, list):
                # ensemble joint
                for idx_second, ensemble_model in enumerate(model):
                    torch.save(ensemble_model.state_dict(), path / name / tag / f"{idx_one}_{idx_second}.pth")

def load_model(model_map, tag, name, path=torch_base_path, config=BaseConfig()):
    result_model = []

    for idx_one, model_one_path in enumerate(os.listdir(path / name / tag)):
        if model_one_path.endswith(".pth"):
            # joint
            model = model_map[idx_one]
            model.load_state_dict(torch.load(path / name / tag / f"{idx_one}.pth", map_location=config.device))
            result_model.append(model)
        else:
            # ensemble
            pass

    return result_model