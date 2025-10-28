import torch
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from configs import BaseConfig
from train.loss_function import TweedieLoss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.patience == -1:
            return

        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def get_station_index(feather_dataset, station_index = None, np_format=False):
    if station_index is None:
        station_index = range(len(feather_dataset))

    result_joint_station_index = []
    for i in range(len(feather_dataset)):
        result_joint_station_index.append(np.ones((feather_dataset[i].shape[0], 1)) * station_index[i]
                                          )
    if np_format:
        return np.array(result_joint_station_index)
    return result_joint_station_index

def get_joint_fold_dataset(fold_time, feather_dataset, label_dataset, station_index=None):
    skleanr_kf = KFold(n_splits=fold_time)
    station_index_dataset = get_station_index(feather_dataset) \
        if station_index is None else get_station_index(feather_dataset, station_index)

    result_fold_dataset = []
    for i in range(len(feather_dataset)):
        result_fold_dataset.append([])

    for dataset_index in range(len(feather_dataset)):
        for fold_index, (train_idx, valid_idx) in enumerate(
                skleanr_kf.split(feather_dataset[dataset_index], label_dataset[dataset_index])):
            result_fold_dataset[dataset_index].append((
                (feather_dataset[dataset_index][train_idx], station_index_dataset[dataset_index][train_idx],
                 label_dataset[dataset_index][train_idx]),
                (feather_dataset[dataset_index][valid_idx], station_index_dataset[dataset_index][valid_idx],
                 label_dataset[dataset_index][valid_idx])
            ))
    return result_fold_dataset


def get_station_embedding_dataset(feature, label, config=BaseConfig()):
    raw_feature_shape = feature.shape
    raw_label_shape = label.shape
    station_idx = np.arange(raw_feature_shape[0]).reshape(-1, 1).repeat(raw_feature_shape[1], axis=1)

    feature = feature.reshape(-1, raw_feature_shape[2], raw_feature_shape[3], raw_feature_shape[4], raw_feature_shape[5])
    label = label.reshape(-1, raw_label_shape[2])
    station_idx = station_idx.reshape(-1)
    shuffle_index = np.random.permutation(len(feature))
    split_idx = int(config.train_frac * len(feature))
    train_idx, test_idx = shuffle_index[:split_idx], shuffle_index[split_idx:]

    feature_for_train = feature[train_idx]
    label_for_train = label[train_idx]
    station_idx_for_train = station_idx[train_idx]
    feature_for_test = feature[test_idx]
    label_for_test = label[test_idx]
    station_idx_for_test = station_idx[test_idx]

    train_dataset = DataLoader(
        TensorDataset(
            torch.tensor(feature_for_train, device=config.device).float(),
            torch.tensor(station_idx_for_train, device=config.device).long(),
            torch.tensor(label_for_train, device=config.device).float(),
        ), batch_size=config.batch_size, shuffle=True, generator=config.seed_generator
    )

    test_dataset = DataLoader(
        TensorDataset(
            torch.tensor(feature_for_test, device=config.device).float(),
            torch.tensor(station_idx_for_test, device=config.device).long(),
            torch.tensor(label_for_test, device=config.device).float(),
        ), batch_size=config.batch_size, shuffle=False, generator=config.seed_generator
    )

    return train_dataset, test_dataset


def get_fold_tensor_dataset(target_fold_joint_dataset, extend_feature=None, extend_label=None, config=BaseConfig()):
    result_fold_dataset = []
    if extend_feature is not None and extend_label is not None:
        raw_extend_feature_shape = extend_feature.shape
        raw_extend_label_shape = extend_label.shape

        extend_feature = np.array(extend_feature)
        extend_label = np.array(extend_label)
        extend_station_index = get_station_index(extend_feature, range(5, 5 + extend_feature.shape[0]))
        extend_station_index = np.array(extend_station_index)

        extend_feature = extend_feature.reshape(
            -1, raw_extend_feature_shape[2], raw_extend_feature_shape[3], raw_extend_feature_shape[4], raw_extend_feature_shape[5]
        )
        extend_station_index = extend_station_index.reshape(-1, 1)
        extend_label = extend_label.reshape(-1, raw_extend_label_shape[2])
    else:
        extend_station_index = None


    for fold_index in range(config.fold_len):
        raw_fold_dataset = target_fold_joint_dataset
        temp_train_, temp_validation_ = [[], [], []], [[], [], []]
        for dataset_tuple in raw_fold_dataset:
            dataset_train, dataset_validation = dataset_tuple[fold_index]
            temp_train_[0].append(dataset_train[0]), temp_train_[1].append(dataset_train[1]), temp_train_[2].append(
                dataset_train[2])
            temp_validation_[0].append(dataset_validation[0]), temp_validation_[1].append(dataset_validation[1]), \
                temp_validation_[2].append(dataset_validation[2])
        temp_train_[0], temp_train_[1], temp_train_[2] = np.concatenate(temp_train_[0]), np.concatenate(
            temp_train_[1]), np.concatenate(temp_train_[2])

        #
        if extend_feature is not None and extend_label is not None and extend_station_index is not None:

            temp_train_[0] = np.concatenate([temp_train_[0], extend_feature], axis=0)
            temp_train_[1] = np.concatenate([temp_train_[1], extend_station_index], axis=0)
            temp_train_[2] = np.concatenate([temp_train_[2], extend_label], axis=0)

        #

        temp_validation_[0], temp_validation_[1], temp_validation_[2] = np.concatenate(
            temp_validation_[0]), np.concatenate(temp_validation_[1]), np.concatenate(temp_validation_[2])
        result_fold_dataset.append((
            DataLoader(
                TensorDataset(torch.tensor(temp_train_[0], device=config.base_device).float(),
                              torch.tensor(temp_train_[1], device=config.base_device).long(),
                              torch.tensor(temp_train_[2], device=config.base_device).float()),
                batch_size=config.batch_size, shuffle=True, generator=config.seed_generator ),
            DataLoader(
                TensorDataset(torch.tensor(temp_validation_[0], device=config.base_device).float(),
                              torch.tensor(temp_validation_[1], device=config.base_device).long(),
                              torch.tensor(temp_validation_[2], device=config.base_device).float()),
                batch_size=config.batch_size, shuffle=False, generator=config.seed_generator )
        ))
    return result_fold_dataset

def get_prediction_output(models, test_dataset: list, method='n_fold', config=BaseConfig(), np_format=True):
    def base_n_fold_predict(model_list, test_dataset_for_fold, idx_dataset_for_fold):
        with torch.no_grad():
            test_dataset_for_fold, idx_dataset_for_fold = test_dataset_for_fold.to(config.device), idx_dataset_for_fold.to(config.device)
            n_fold_prediction_results = []
            for fold_model in model_list:
                assert isinstance(fold_model, nn.Module)
                fold_model = fold_model.to(config.device)
                fold_model.eval()
                n_fold_prediction_results.append(
                    fold_model(test_dataset_for_fold, idx_dataset_for_fold).cpu().detach().numpy())
            del test_dataset_for_fold, idx_dataset_for_fold
            return np.mean(n_fold_prediction_results, axis=0)

    if method == 'n_fold':
        assert isinstance(models, list)
        full_prediction_result = []
        site_idx_data = get_station_index(test_dataset)
        for site_idx, site_test_dataset in enumerate(test_dataset):
            site_test_dataset = torch.tensor(site_test_dataset).float()
            site_idx_dataset = torch.tensor(site_idx_data[site_idx]).long()
            full_prediction_result.append(base_n_fold_predict(models, site_test_dataset, site_idx_dataset))
        return np.array(full_prediction_result) if np_format else full_prediction_result
    elif method == "train_test":
        assert not isinstance(models, list)
        full_prediction_result = []
        site_idx_data = get_station_index(test_dataset)
        for site_idx, site_test_dataset in enumerate(test_dataset):
            site_test_dataset = torch.tensor(site_test_dataset).float()
            site_idx_dataset = torch.tensor(site_idx_data[site_idx]).long()

            assert isinstance(models, nn.Module)
            device = models.device
            result = models(site_test_dataset.to(device), site_idx_dataset.to(device)).cpu().detach().numpy()
            full_prediction_result.append(result)
        return np.array(full_prediction_result) if np_format else full_prediction_result
    elif method == "no_station_embed":
        assert isinstance(models, list)
        full_prediction_result = []
        site_idx_data = get_station_index(test_dataset)
        for site_idx, site_test_dataset in enumerate(test_dataset):
            site_test_dataset = torch.tensor(site_test_dataset, device=config.device).float()
            site_idx_dataset = torch.tensor(site_idx_data[site_idx], device=config.device).long()
            full_prediction_result.append(base_n_fold_predict(models[site_idx], site_test_dataset, site_idx_dataset))
        return np.array(full_prediction_result) if np_format else full_prediction_result
    return None

def ensemble_train(models: list[nn.Module], configs: list[BaseConfig]):
    assert len(models) == len(configs)
    for index in range(len(configs)):
        pass
