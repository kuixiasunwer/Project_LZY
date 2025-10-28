import pandas as pd
import torch
from torch import nn
import zipfile
import shutil
import os

import util.paths as paths
from configs import BaseConfig
from train.tools import *
from util.data_loader import load_sais_feather_data, load_sais_label_data
from train.loss_function import R2Loss


def old_validation_function(dataset, model, score_fn, config):
    loss_result = []
    for feather, sta_index, label in dataset:
        # Skip Nan Label
        nan_label_idx = torch.isnan(label).any(dim=1)
        feather = feather[~nan_label_idx].to(config.device)
        sta_index = sta_index[~nan_label_idx].to(config.device)
        label = label[~nan_label_idx].to(config.device)
        # Skip Nan Label

        pred = model(feather, sta_index)

        r2loss = score_fn(pred, label)
        loss_result.append([r2loss.item() * label.shape[0]])
    return (np.array(loss_result).sum(0) / len(dataset.dataset))[0]

def new_validation_function(dataset, model, score_fn, config):
    full_feature = dataset.dataset[:][0].to(config.device)
    full_sta_index = dataset.dataset[:][1].to(config.device)
    full_pred = model(full_feature, full_sta_index)
    full_label = dataset.dataset[:][2]
    nan_label_idx = torch.isnan(full_label).any(dim=1)
    score_result = score_fn(full_pred[~nan_label_idx], full_label[~nan_label_idx]).cpu().detach().numpy()
    del full_feature, full_sta_index, full_pred, full_label
    return score_result


def base_train_fun(train_dataset, validation_dataset, model, config=BaseConfig()):
    torch.manual_seed(config.random_seed)
    target_model = model(config).to(config.device)
    optimizer = config.optimizer(target_model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = config.loss_fn.to(config.device)
    best_loss = torch.inf
    early_stopping = EarlyStopping(patience=config.early_patience)
    temp_model_path = paths.torch_base_path / "temp_model.pth"
    best_epoch = 0

    if not os.path.exists(paths.torch_base_path):
        os.makedirs(paths.torch_base_path)

    score_fn = config.score_fn.to(config.device)

    def one_epoch(dataset, training=True):
        loss_result = []
        if training:
            target_model.train()
            for feather, sta_index, label in dataset:
                # Skip Nan Label
                nan_label_idx = torch.isnan(label).any(dim=1)
                feather = feather[~nan_label_idx].to(config.device)
                sta_index = sta_index[~nan_label_idx].to(config.device)
                label = label[~nan_label_idx].to(config.device)
                # Skip Nan Label
                if feather.shape[0] <= 0:
                    continue
                pred = target_model(feather, sta_index)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                # print(loss.item())
                # torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
                optimizer.step()
                del feather, sta_index, label, pred
            target_model.eval()
            with torch.no_grad():
                full_feature = dataset.dataset[:][0].to(config.device)
                if full_feature.shape[0] <= 0:
                    return np.inf
                full_sta_index = dataset.dataset[:][1].to(config.device)
                full_pred = target_model(full_feature, full_sta_index)
                full_label = dataset.dataset[:][2]
                nan_label_idx = torch.isnan(full_label).any(dim=1)
                score_result = score_fn(full_pred[~nan_label_idx], full_label[~nan_label_idx])
                del full_feature, full_sta_index, full_pred, full_label
                return score_result
        else:
            target_model.eval()
            with torch.no_grad():
                # return old_validation_function(validation_dataset, target_model, score_fn, config)
                return new_validation_function(validation_dataset, target_model, score_fn, config)

    for epoch in range(config.train_epochs):
        train_loss = one_epoch(train_dataset, training=True)
        validation_loss = one_epoch(validation_dataset, training=False)
        early_stopping(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            if config.print_training_information:
                print("best_r2", 1 - best_loss.item(), "best_epoch", best_epoch, sep='\t')
            torch.save(target_model.state_dict(), temp_model_path)

        if config.print_no_better_r2:
            # print("best_r2", 1 - best_loss.item(), "best_epoch", best_epoch, sep='\t')
            print("cur_val_r2", 1 - validation_loss.item(), "cur_epoch", epoch, "train_r2", 1 - train_loss.item(), sep='\t')

        if early_stopping.early_stop:
            if config.print_training_information:
                print("Early stopping triggered.")
            break

    if best_loss == torch.inf:
        raise Exception("Best loss was infinite.")

    target_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
    return target_model.to(config.base_device), 1 - best_loss


def anti_train_fun(train_dataset, validation_dataset, model=None):
    torch.manual_seed(random_seed - 1)
    if model is None:
        target_model = BaseModel(CFG).to(device)
    else:
        target_model = model(CFG).to(device)
    torch.manual_seed(random_seed)
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=CFG.learning_rate)
    loss_fn = nn.MSELoss().to(device)
    best_loss = np.inf
    early_stopping = EarlyStopping(patience=CFG.early_patience)

    def one_epoch(dataset, training=True):
        loss_result = []
        if training:
            target_model.train()
            for feather, sta_index, label in dataset:
                # print(feather.shape, sta_index.shape, label.shape)
                feather.requires_grad = True
                pred = target_model(feather, sta_index)
                # loss = r2_loss_function(pred, label)
                loss = loss_fn(pred, label)
                loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)

                grad_sign = feather.grad.data.sign()
                x_adv = torch.clamp(feather + 0.3 * grad_sign, 0, 1).detach()
                logits_adv = target_model(x_adv, sta_index)
                loss_adv = loss_fn(logits_adv, label)

                total_loss = 0.5 * loss + 0.5 * loss_adv
                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()
            return r2_loss_function(
                target_model(dataset.dataset[:][0], dataset.dataset[:][1]), dataset.dataset[:][2]
            )
        else:
            target_model.eval()
            with torch.no_grad():
                for feather, sta_index, label in dataset:
                    pred = target_model(feather, sta_index)
                    loss = r2_loss_function(pred, label)
                    loss_result.append([loss.item() * label.shape[0]])
                return (np.array(loss_result).sum(0) / len(dataset.dataset))[0]

    for epoch in range(CFG.train_epochs):
        train_loss = one_epoch(train_dataset, training=True)
        validation_loss = one_epoch(validation_dataset, training=False)
        early_stopping(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            print("best_r2", 1 - best_loss.item(), "best_epoch", best_epoch, "train_r2", 1 - train_loss.item())
            torch.save(target_model.state_dict(), '/tmp/model.pth')

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    target_model.load_state_dict(torch.load('/tmp/model.pth', weights_only=True))
    return target_model, 1 - best_loss


def noisy_train_fun(train_dataset, validation_dataset, model, config=BaseConfig(), feedback=False):
    torch.manual_seed(config.random_seed)
    target_model = model(config).to(config.device)
    optimizer = config.optimizer(target_model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = config.loss_fn.to(config.device)
    best_loss = torch.inf
    early_stopping = EarlyStopping(patience=config.early_patience)
    temp_model_path = paths.torch_base_path / "temp_model.pth"

    if not os.path.exists(paths.torch_base_path):
        os.makedirs(paths.torch_base_path)

    score_fn = config.score_fn.to(config.device)

    def one_epoch(dataset, training=True, epoch_num=0):
        loss_result = []
        if training:
            target_model.train()
            for feather, sta_index, label in dataset:
                # Skip NaN Label
                nan_label_idx = torch.isnan(label).any(dim=1)
                feather = feather[~nan_label_idx]
                sta_index = sta_index[~nan_label_idx]
                label = label[~nan_label_idx]
                # Skip NaN Label

                # 添加噪声到特征（仅在 epoch >= 10 时）
                if epoch_num >= 10:
                    noise_ratio = getattr(config, 'noise_ratio', 0.2)
                    noise_std = getattr(config, 'noise_std', 0.01)
                    batch_size = feather.size(0)
                    noise_mask = torch.rand(batch_size) < noise_ratio
                    noise = torch.randn_like(feather) * noise_std
                    feather = feather.clone()
                    feather[noise_mask] += noise[noise_mask]

                pred = target_model(feather, sta_index)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            target_model.eval()
            full_pred = target_model(dataset.dataset[:][0], dataset.dataset[:][1])
            full_label = dataset.dataset[:][2]
            nan_label_idx = torch.isnan(full_label).any(dim=1)
            return score_fn(full_pred[~nan_label_idx], full_label[~nan_label_idx])
        else:
            target_model.eval()
            with torch.no_grad():
                for feather, sta_index, label in dataset:
                    nan_label_idx = torch.isnan(label).any(dim=1)
                    feather = feather[~nan_label_idx]
                    sta_index = sta_index[~nan_label_idx]
                    label = label[~nan_label_idx]

                    pred = target_model(feather, sta_index)
                    r2loss = score_fn(pred, label)
                    loss_result.append([r2loss.item() * label.shape[0]])
                return (np.array(loss_result).sum(0) / len(dataset.dataset))[0]

    for epoch in range(config.train_epochs):
        train_loss = one_epoch(train_dataset, training=True, epoch_num=epoch)
        validation_loss = one_epoch(validation_dataset, training=False, epoch_num=epoch)
        early_stopping(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            if feedback:
                print("best_r2", 1 - best_loss.item(), "best_epoch", best_epoch, "train_r2", 1 - train_loss.item())
            torch.save(target_model.state_dict(), temp_model_path)

        if early_stopping.early_stop:
            if feedback:
                print("Early stopping triggered.")
            break

    if best_loss == torch.inf:
        raise Exception("Best loss was infinite.")

    target_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
    return target_model, 1 - best_loss

def fix_model_train_fun(train_dataset, validation_dataset, fix_feature, model, config=BaseConfig()):
    torch_train_dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(train_dataset[0], device=config.device, dtype=torch.float),
            torch.tensor(train_dataset[1], device=config.device, dtype=torch.float)
        ),
        shuffle=True,
        batch_size=config.batch_size, generator=config.seed_generator
    )
    torch_validation_dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(validation_dataset[0], device=config.device, dtype=torch.float),
            torch.tensor(validation_dataset[1], device=config.device, dtype=torch.float)
        ),
        shuffle=False,
        batch_size=config.batch_size, generator=config.seed_generator
    )

    torch.manual_seed(config.random_seed)
    target_model = model(config).to(config.device)
    optimizer = config.optimizer(target_model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = config.loss_fn.to(config.device)
    best_loss = torch.inf
    temp_model_path = paths.torch_base_path / "temp_model.pth"

    if not os.path.exists(paths.torch_base_path):
        os.makedirs(paths.torch_base_path)

    score_fn = nn.MSELoss()

    for epoch in range(config.train_epochs):
        target_model.train()
        for feature, label in torch_train_dataset:
            pred = target_model(feature)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        target_model.eval()
        for feature, label in torch_validation_dataset:
            pred = target_model(feature)
            loss = score_fn(pred, label)
            if loss.item() < best_loss:
                best_loss = loss
                print(f"epoch: {epoch}, best_loss: {best_loss}")
                torch.save(target_model.state_dict(), temp_model_path)

    target_model.load_state_dict(torch.load(temp_model_path, weights_only=True))

    fix_feature = torch.tensor(fix_feature, dtype=torch.float).to(config.device)
    target_model.eval()
    fix_label = target_model(fix_feature)

    return fix_label.cpu().detach().numpy()




def get_n_fold_model(model, joint_dataset, config=BaseConfig(), pre_train_model=None):
    if pre_train_model is not None:
        print("!!!!Loaded pre-trained model!!!!")
    result_models = []
    best_r2s_1 = []
    if config.print_training_information:
        print("Starting training...")
    for fold in range(config.fold_len):
        trained_model, best_r2 = base_train_fun(
            joint_dataset[fold][0],
            joint_dataset[fold][1],
            model=model,
            config=config,
        )
        result_models.append(trained_model)
        best_r2s_1.append(best_r2)
        if config.print_training_information:
            print("----", fold, best_r2)
    print("average:", np.mean(best_r2s_1))
    return result_models


def better_n_fold_train(model, feature: np.ndarray, label: np.ndarray, config=BaseConfig(), station_index=None):
    result_models = []
    best_r2s_1 = []
    station_index_dataset = get_station_index(feature, np_format=True) \
        if station_index is None else get_station_index(feature, station_index)

    skleanr_kf = KFold(n_splits=config.fold_len)
    for fold_index, (train_idx, valid_idx) in enumerate(skleanr_kf.split(feature[0])):

        fold_train_feature = feature[:, train_idx]
        fold_train_label = label[:, train_idx]
        fold_train_station_idx = station_index_dataset[:, train_idx]
        fold_valid_feature = feature[:, valid_idx]
        fold_valid_label = label[:, valid_idx]
        fold_valid_station_idx = station_index_dataset[:, valid_idx]

        fold_train_dataset = DataLoader(
            TensorDataset(
                torch.tensor(fold_train_feature.reshape(-1, *fold_train_feature.shape[2:]), device=config.base_device).float(),
                torch.tensor(fold_train_station_idx.reshape(-1, *fold_train_station_idx.shape[2:]), device=config.base_device).long(),
                torch.tensor(fold_train_label.reshape(-1, *fold_train_label.shape[2:]), device=config.base_device).float(),
            ),
            batch_size=config.batch_size,
            shuffle=True, generator=config.seed_generator
        )

        fold_valid_dataset = DataLoader(
            TensorDataset(
                torch.tensor(fold_valid_feature.reshape(-1, *fold_valid_feature.shape[2:]), device=config.base_device).float(),
                torch.tensor(fold_valid_station_idx.reshape(-1, *fold_valid_station_idx.shape[2:]), device=config.base_device).long(),
                torch.tensor(fold_valid_label.reshape(-1, *fold_valid_label.shape[2:]), device=config.base_device).float(),
            ),
            batch_size=config.batch_size,
            shuffle=True, generator=config.seed_generator
        )

        trained_model, best_r2 = base_train_fun(
            fold_train_dataset,
            fold_valid_dataset,
            model=model,
            config=config,
        )
        result_models.append(trained_model)
        best_r2s_1.append(best_r2)
        if config.print_training_information:
            print("----", fold_index, best_r2)
        del (fold_train_feature, fold_train_station_idx, fold_train_label,
             fold_valid_feature, fold_valid_station_idx, fold_valid_label,
             fold_train_dataset, fold_valid_dataset
             )
    print("average:", np.mean(best_r2s_1))
    return result_models

def get_train_test_model(model, train_test_dataset, config=BaseConfig()):
    print(f"train: {model}")
    if config.print_training_information:
        print("Starting training...")


def print_prediction_to_output(
        models=None, test_dataset=None, station_idx_range=None,
        prediction_result=None, method='n_fold', output_dir=paths.output_base_pth):
    test_time_index = pd.Series((pd.date_range(
        start=pd.to_datetime('2025-03-01 00:00:00'),
        end=pd.to_datetime('2025-04-30 23:45:00'),
        freq='15min').strftime("%Y/%-m/%-d %H:%M"))
                                .values)

    if prediction_result is None:
        prediction_result: list = get_prediction_output(models, test_dataset, method)
    assert len(prediction_result) == len(station_idx_range)
    for index, station_index in enumerate(station_idx_range):
        if len(prediction_result[index].shape) > 1 and prediction_result[index].shape[1] == 2:
            prediction_result[index] = prediction_result[index][:, 1]

        test_label = prediction_result[index].reshape(1, -1)
        test_pd = pd.DataFrame({'': test_time_index, '0': pd.Series(test_label[0])})
        # test_pd.loc[test_pd['0'] < 0, '0'] = 0
        test_pd.to_csv(output_dir / f"output{station_index + 1}.csv", index=False)

    return prediction_result

def zip_output_file_and_move(output_dir, output_zip="output.zip", output_dst_dir=paths.output_dst_path):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname="output/" + file_name)

    if not os.path.exists(output_dst_dir):
        os.mkdir(output_dst_dir)

    if os.path.exists(output_dst_dir / output_zip):
        os.remove(output_dst_dir / output_zip)
    shutil.move(src=output_zip ,dst=output_dst_dir)
