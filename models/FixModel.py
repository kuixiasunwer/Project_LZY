import torch.nn as nn
import torch

from configs import BaseConfig


class WindFixModel(nn.Module):
    def __init__(self, input_size=24, output_size=96, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.model(x)

class LightFixModel(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.input_seq_len * 2, config.hidden_size, dtype=torch.float),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2, dtype=torch.float),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.output_seq_len, dtype=torch.float),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)