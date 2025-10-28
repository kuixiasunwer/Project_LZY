import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import huber_loss

def r2_score(y_pred: np.ndarray, y_true: np.ndarray):
    full_station_score = []

    for station in range(y_pred.shape[0]):
        full_day_score = []

        for day in range(y_true.shape[1]):
            pred = y_pred[station, day, :]
            true = y_true[station, day, :]

            valid_mask = ~np.isnan(true)

            pred = pred[valid_mask]
            true = true[valid_mask]

            day_score = 1 - np.sqrt(np.mean(((pred - true) / true.clip(min=0.2)) ** 2))
            full_day_score.append(day_score)

        station_score = np.mean(full_day_score)
        full_station_score.append(station_score)

    return np.mean(full_station_score)

class TweedieLoss(nn.Module):
    def __init__(self, power: float = 1.5, training=True):
        super(TweedieLoss, self).__init__()
        self.training = training
        self.p = nn.Parameter(torch.tensor(power, dtype=torch.float)) if training else power

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0 + 1e-6, max=2.0 - 1e-6) if self.training else self.p
        a = y_true * torch.pow(y_pred, 1 - p) / (1 - p)
        b = torch.pow(y_pred, 2 - p) / (2 - p)
        loss = - (a - b)
        return torch.mean(loss)

class R2Loss(nn.Module):
    def __init__(self, training=True):
        self.training = training
        super().__init__()

    def forward(self, pred, target):
        device = pred.device
        target = target.to(device)

        if len(pred.shape) != 2:
            pred = pred.shape(-1, 96)
            target = target.shape(-1, 96)
        if not self.training:
            return torch.mean(1 - torch.sqrt(torch.mean(((pred - target) / target.clip(min=0.2)) ** 2, dim=1)))
        return torch.mean(torch.sqrt(torch.mean(((pred - target) / target.clip(min=0.2)) ** 2, dim=1)))

class HuberLoss(nn.Module):
    def __init__(self, training=True):
        self.training = training
        super().__init__()
    def forward(self, pred, target):
        if len(pred.shape) != 2:
            pred = pred.view(-1, 96)
            target = target.view(-1, 96)
        loss = huber_loss(pred / target.clip(min=0.2), target / target.clip(min=0.2), delta=1.0)
        return loss


class R2LossAndSmoothLoss(nn.Module):
    def __init__(self, training=True, lambda_smooth=0.1, order=2):
        super().__init__()
        self.r2loss = R2Loss(training=training)
        self.training = training
        self.lambda_smooth = lambda_smooth
        self.order = order

    def forward(self, pred, target):
        if self.order == 1:
            diff = pred[:, 1:] - pred[:, :-1]
        elif self.order == 2:
            diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        else:
            raise ValueError("Order must be 1 or 2.")
        return torch.mean(diff ** 2) * self.lambda_smooth + self.r2loss(pred, target)


class FunctionR2Loss(nn.Module):
    def __init__(self, training=True, torch_fn=torch.log):
        self.training = training
        super().__init__()

    def forward(self, pred, target):
        if len(pred.shape) != 2:
            pred = pred.shape(-1, 96)
            target = target.shape(-1, 96)
        pred = torch_fn(pred + 1)
        target = torch_fn(target + 1)
        if not self.training:
            return torch.mean(1 - torch.sqrt(torch.mean(((pred - target) / target.clip(min=0.2)) ** 2, dim=1)))
        return torch.mean(torch.sqrt(torch.mean(((pred - target) / target.clip(min=0.2)) ** 2, dim=1)))