import numpy as np
import pandas as pd
import torch
import xarray as xr
import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader, TensorDataset

from configs import BaseConfig
from models.FixModel import LightFixModel
from train.functions import fix_model_train_fun
from util import paths
from util.data_loader import load_sais_label_data

def get_range(a, b):
    return list(range(a, b))

def wind_data_fix(label_dataset, config=BaseConfig()):
    if config.del_all_day_zero_or_nan[0]:
        label_dataset[np.all((label_dataset == 0) | (label_dataset < 0.01) | np.isnan(label_dataset), axis=2)] = np.nan

    if config.wind_fix_function:
        if config.wind_fix_function == "other_label":
            if config.print_training_information:
                print("start loaded fix label")
            label_dataset = mlp_fix(label_dataset, config)


    return label_dataset

def light_data_fix(label_dataset, train_xr_feature: xr.Dataset, config=BaseConfig()):
    def z_score_normal(dataset, data_mean=None, data_std=None):
        if data_mean is None or data_std is None:
            data_mean = np.nanmean(dataset, axis=(0, 1), keepdims=True)
            data_std = np.nanstd(dataset, axis=(0, 1), keepdims=True)
        dataset = (dataset - data_mean) / data_std
        return dataset

    if config.interpolate_axis != -1:
        assert config.interpolate_axis == 1 or config.interpolate_axis == 0

        for idx, station_label in enumerate(label_dataset):
            station_label = pd.DataFrame(station_label)
            station_label = station_label.interpolate(axis=config.interpolate_axis)
            label_dataset[idx] = station_label.to_numpy()

    if config.del_all_day_zero_or_nan[1]:
        label_dataset[np.isnan(label_dataset)] = 0

    if config.del_all_day_zero_or_nan[1]:
        label_dataset[np.all((label_dataset == 0) | np.isnan(label_dataset), axis=2)] = np.nan

    if config.light_fix_function:
        if config.light_fix_function == "poai_ghi":
            for station_index in range(label_dataset.shape[0]):
                target_label = label_dataset[station_index]
                fix_index = np.isnan(target_label).all(axis=1)
                if fix_index.sum() == 0:
                    continue
                no_need_fix_feature = train_xr_feature.sel(
                    sta=station_index + 5 + 1, time=~fix_index, channel=['NWP1_poai', 'NWP1_ghi'], lat=11 // 2, lon=11 // 2
                )['data'].to_numpy()
                no_need_fix_label = target_label[~fix_index]
                need_fix_feature = train_xr_feature.sel(
                    sta=station_index + 5 + 1, time=fix_index, channel=['NWP1_poai', 'NWP1_ghi'], lat=11 // 2, lon=11 // 2
                )['data'].to_numpy()

                need_fix_feature = z_score_normal(
                    need_fix_feature,
                    np.nanmean(no_need_fix_feature, axis=(0, 1), keepdims=True),
                    np.nanstd(no_need_fix_feature, axis=(0, 1), keepdims=True),
                )
                no_need_fix_feature = z_score_normal(
                    no_need_fix_feature,
                )
                need_fix_feature = need_fix_feature.reshape(-1, 48)
                no_need_fix_feature = no_need_fix_feature.reshape(-1, 48)

                random_indices = np.random.permutation(len(no_need_fix_feature))
                no_need_fix_feature = no_need_fix_feature[random_indices]
                no_need_fix_label = no_need_fix_label[random_indices]
                split_train_index = int(len(no_need_fix_feature) * 0.8)
                no_need_fix_feature_train = no_need_fix_feature[:split_train_index]
                no_need_fix_label_train = no_need_fix_label[:split_train_index]
                no_need_fix_feature_validation = no_need_fix_feature[split_train_index:]
                no_need_fix_label_validation = no_need_fix_label[split_train_index:]

                if config.print_training_information:
                    print(f"start fix label: {station_index + 1}")
                fix_config = BaseConfig()
                fix_config.batch_size = 1
                fix_config.train_epochs = 100

                fix_label = fix_model_train_fun(
                    (no_need_fix_feature_train, no_need_fix_label_train),
                    (no_need_fix_feature_validation, no_need_fix_label_validation),
                    need_fix_feature,
                    model=LightFixModel,
                    config=fix_config,
                )
                target_label[fix_index] = fix_label
        elif config.light_fix_function == "other_label":
            if config.print_training_information:
                print("start loaded fix label")
            label_dataset = mlp_fix(label_dataset, config)

    return label_dataset

def fix_label_dataset(label_dataset, train_xr_feature, config):
    # (10, 59 or more, 96)
    # <0 = 0
    for idx, station_label in enumerate(label_dataset):
        label_dataset[idx][station_label < 0] = 0

    label_dataset[:5] = wind_data_fix(label_dataset[:5], config=config)
    label_dataset[5:] = light_data_fix(label_dataset[5:], train_xr_feature, config)

    return label_dataset


def mlp_fix(label_dataset, config: BaseConfig()):
    config = BaseConfig()
    config.train_epochs = 100

    class FixModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(96 * 2, int(96 * 1.75)),
                nn.ReLU(),
                nn.Linear(int(96 * 1.75), 48),
                nn.ReLU(),
                nn.Linear(48, 96),
                nn.ReLU(),
            )

            # self.rnn = nn.LSTM(input_size=2, hidden_size=32, num_layers=2, batch_first=True)
            # self.decoder = nn.Sequential(
            #     nn.Linear(32, 6),
            #     nn.ReLU(),
            #     nn.Linear(6, 1),
            #     nn.Flatten(start_dim=-2, end_dim=-1),
            # )

        def forward(self, x):
            return self.mlp(x)
            # x = x.reshape(x.shape[0], -1, 2)
            # output, _ = self.rnn(x)
            # output = self.decoder(output)
            # return output

    def train_function(train_dataset, validation_dataset):
        torch.manual_seed(config.random_seed)
        target_model = FixModel().to(config.device)
        optimizer = config.optimizer(target_model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        loss_fn = config.loss_fn.to(config.device)
        best_loss = torch.inf
        temp_model_path = paths.torch_base_path / "temp_model.pth"

        if not os.path.exists(paths.torch_base_path):
            os.makedirs(paths.torch_base_path)

        score_fn = config.score_fn.to(config.device)

        def one_epoch(dataset, training=True):
            loss_result = []
            if training:
                target_model.train()
                for feather, label in dataset:
                    # Skip Nan Label
                    nan_label_idx = torch.isnan(label).any(dim=1)
                    feather = feather[~nan_label_idx]
                    label = label[~nan_label_idx]
                    # Skip Nan Label
                    pred = target_model(feather)
                    loss = loss_fn(pred, label)

                    optimizer.zero_grad()
                    loss.backward()
                    # print(loss.item())
                    # torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
                    optimizer.step()

                target_model.eval()

                full_pred = target_model(dataset.dataset[:][0])
                full_label = dataset.dataset[:][1]
                nan_label_idx = torch.isnan(full_label).any(dim=1)

                return score_fn(full_pred[~nan_label_idx], full_label[~nan_label_idx])
            else:
                target_model.eval()
                with torch.no_grad():
                    for feather, label in dataset:
                        # Skip Nan Label
                        nan_label_idx = torch.isnan(label).any(dim=1)
                        feather = feather[~nan_label_idx]
                        label = label[~nan_label_idx]
                        # Skip Nan Label

                        pred = target_model(feather)

                        r2loss = score_fn(pred, label)
                        loss_result.append([r2loss.item() * label.shape[0]])
                    return (np.array(loss_result).sum(0) / len(dataset.dataset))[0]

        for epoch in range(config.train_epochs):
            train_loss = one_epoch(train_dataset, training=True)
            validation_loss = one_epoch(validation_dataset, training=False)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                if config.print_training_information:
                    print("best_r2", 1 - best_loss.item(), "best_epoch", best_epoch, "train_r2", 1 - train_loss.item())
                torch.save(target_model.state_dict(), temp_model_path)

        if best_loss == torch.inf:
            raise Exception("Best loss was infinite.")

        target_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
        return target_model, best_loss

    raw_label_dataset_shape = label_dataset.shape
    label_null_idx_for_any_station_idx = np.isnan(label_dataset).any(axis=(0, 2))
    no_null_for_compare_label = label_dataset[:, ~label_null_idx_for_any_station_idx, :].reshape(
        raw_label_dataset_shape[0], -1)
    most_like_idx = np.argsort(np.corrcoef(no_null_for_compare_label))

    for station_index in range(label_dataset.shape[0]):

        target_label = label_dataset[station_index]
        fix_index = np.isnan(target_label).all(axis=1)

        print("Station:", station_index)
        print("Fix Count:", fix_index.sum())

        if fix_index.sum() == 0:
            continue

        concat_label = np.concatenate([[target_label], label_dataset[most_like_idx[station_index][-3: -1]]], axis=0)

        common_null_idx = np.isnan(concat_label).any(axis=(0, 2))
        need_fix_null_idx = np.isnan(target_label).any(axis=1)

        label_for_model_label = concat_label[:, ~common_null_idx, :][0]
        label_for_model_feature = concat_label[:, ~common_null_idx, :][[1, 2]].transpose(1, 0, 2).reshape(-1,
                                                                                                          96 * 2)

        split_idx = int(len(label_for_model_feature) * 0.8)
        train_idx = np.random.permutation(len(label_for_model_feature))

        label_for_model_label_train = label_for_model_label[train_idx][:split_idx]
        label_for_model_feature_train = label_for_model_feature[train_idx][:split_idx]
        label_for_model_label_validation = label_for_model_label[train_idx][split_idx:]
        label_for_model_feature_validation = label_for_model_feature[train_idx][split_idx:]

        dataset_for_train = DataLoader(
            TensorDataset(torch.tensor(label_for_model_feature_train, device=config.device).float(),
                          torch.tensor(label_for_model_label_train, device=config.device).float()),
            batch_size=8, shuffle=True, generator=config.seed_generator
        )
        dataset_for_valid = DataLoader(
            TensorDataset(torch.tensor(label_for_model_feature_validation, device=config.device).float(),
                          torch.tensor(label_for_model_label_validation, device=config.device).float()),
            batch_size=8, shuffle=False, generator=config.seed_generator
        )

        # need_fix_label = concat_label[:, need_fix_null_idx][0]
        label_for_fix = torch.tensor(
            concat_label[:, need_fix_null_idx][[1, 2]].transpose(1, 0, 2).reshape(-1, 96 * 2),
            device=config.device).float()
        print(f"station_index: {station_index}")
        model, best_score = train_function(dataset_for_train, dataset_for_valid)

        # if best_score < 0.70:
        #     print("No Fix")
        #     continue

        model.eval()
        fix_label = model(label_for_fix)
        label_dataset[station_index][need_fix_null_idx] = fix_label.cpu().detach().numpy()

    return label_dataset


if __name__ == '__main__':
    from draw.functions import cut_plot_prediction, plot_predictions
    os.chdir('..')

    sais_label_dataset = load_sais_label_data()
    fix_dataset = fix_label_dataset(sais_label_dataset)
    fig = cut_plot_prediction(fix_dataset[5], fig_size=(20, 80))
    fig.show()
