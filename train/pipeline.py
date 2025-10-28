from __future__ import annotations

from typing import Type
import numpy as np
import torch
import os
from datetime import datetime
import random
import xarray as xr
from numpy import shape

from configs import BaseConfig
from draw.functions import load_post_data, plot_predictions, cut_plot_prediction
from pre.feather_engineering import base_feather_engineering
from train.functions import get_n_fold_model, zip_output_file_and_move, print_prediction_to_output, \
    base_train_fun, better_n_fold_train
from train.loss_function import r2_score
from train.tools import get_fold_tensor_dataset, get_joint_fold_dataset, get_prediction_output, \
    get_station_embedding_dataset
from util import paths
from util.data_loader import load_sais_feather_data, load_sais_label_data, get_time_feature_base_pd_time
from util.model_save import save_model

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import os


class BasePipeline:
    def __init__(self, model: Type[torch.nn.Module] | None, config=BaseConfig(), copy_pipeline: BasePipeline = None):
        self.set_seed(config.random_seed)
        self.split_output = False
        self.is_sais_env = BasePipeline.get_is_sais_env()
        self.gpu_is_available = torch.cuda.is_available()

        self.config = config
        self.wind_config = config
        self.light_config = config

        self.no_wandb = False

        self.print_log("Start Pipeline")

        if self.config.use_temp_data is None:
            self.config.use_temp_data = False if self.is_sais_env else True

        if copy_pipeline is not None:
            self.raw_train_feature, self.raw_test_feather = copy_pipeline.raw_train_feature, copy_pipeline.raw_test_feather
        else:
            self.raw_train_feature, self.raw_test_feather = load_sais_feather_data(
                np_format=False, use_temp=config.use_temp_data, use_mf=False
            )

        self.raw_train_label, self.raw_test_label = self.load_raw_label(copy_pipeline)

        if self.config.use_extend_station_data:
            self.raw_extend_feature = xr.open_dataset(paths.extend_feature_path)
            self.extend_label = np.load(paths.extend_label_path)
            self.print_log("Loaded Extend Station Data")
        else:
            self.raw_extend_feature = None
            self.extend_label = None

        self.wind_model = model
        self.light_model = model

        self.train_feature, self.train_label, self.test_feature, self.test_label, self.extend_feature = (
            None, None, None, None, None
        )

        self.past_prediction_result = None
        self.wind_output, self.light_output = None, None

    def feature_engineering(self):
        if self.train_feature is not None and self.train_label is not None and self.test_feature is not None:
            return

        self.train_feature, self.train_label, self.test_feature, self.extend_feature = base_feather_engineering(
            self.raw_train_feature, self.raw_train_label, self.raw_test_feather,
            config=self.config, extend_features=self.raw_extend_feature
        )

        if self.raw_test_label is not None:
            self.test_label = self.raw_test_label[:, -self.test_feature[0].shape[1]:]

        # del self.raw_train_feature, self.raw_train_label, self.raw_test_feather

    def train_wind_model(self):
        self.print_log(f"Wind Model Feather Shape: {self.train_feature[0].shape}")

        if not isinstance(self.wind_config, list):
            if self.wind_config.skip_train:
                return None

        if isinstance(self.wind_model, list) and isinstance(self.wind_config, list):
            full_output = []
            for index in range(len(self.wind_model)):
                self.wind_config[index].input_channel = self.train_feature[0].shape[3]
                trained_models = self.train_function(
                    self.train_feature[0], self.train_label[:5], self.wind_model[index], self.wind_config[index],
                    extend_feature=self.extend_feature[0],
                    extend_label=self.extend_label[[0, 1, 2]] if self.extend_label is not None else None,
                )
                output = get_prediction_output(trained_models, self.test_feature[0],
                                               self.wind_config[index].train_method)

                if self.config.output_focus_range[0] is not None:
                    output[output < self.config.output_focus_range[0]] = \
                        self.config.output_focus_range[
                            0]
                if self.config.output_focus_range[1] is not None:
                    output[output > self.config.output_focus_range[1]] = \
                        self.config.output_focus_range[
                            1]
                full_output.append(output)
                del trained_models
            self.wind_output = np.mean(full_output, axis=0)
            return None
        else:
            self.wind_config.input_channel = self.train_feature[0].shape[3]
            wind_models = self.train_function(
                self.train_feature[0], self.train_label[:5], self.wind_model, self.wind_config,
                extend_feature=self.extend_feature[0],
                extend_label=self.extend_label[[0, 1, 2]] if self.extend_label is not None else None,
            )
            self.wind_output = get_prediction_output(wind_models, self.test_feature[0], self.wind_config.train_method)

            if self.config.output_focus_range[0] is not None:
                self.wind_output[self.wind_output < self.config.output_focus_range[0]] = self.config.output_focus_range[
                    0]
            if self.config.output_focus_range[1] is not None:
                self.wind_output[self.wind_output > self.config.output_focus_range[1]] = self.config.output_focus_range[
                    1]

            return wind_models

    def train_light_model(self):
        self.print_log(f"Light Model Feather Shape: {self.train_feature[1].shape}")

        if not isinstance(self.light_config, list):
            if self.light_config.skip_train:
                return None

        if isinstance(self.light_model, list) and isinstance(self.light_config, list):
            full_output = []
            for index in range(len(self.light_model)):
                self.light_config[index].input_channel = self.train_feature[1].shape[3]
                trained_models = self.train_function(
                    self.train_feature[1], self.train_label[5:], self.light_model[index], self.light_config[index],
                    extend_feature=self.extend_feature[1],
                    extend_label=self.extend_label[[3, 4]] if self.extend_label is not None else None,
                )
                output = get_prediction_output(trained_models, self.test_feature[1],
                                               self.light_config[index].train_method)

                if self.config.output_focus_range[0] is not None:
                    output[output < self.config.output_focus_range[0]] = \
                        self.config.output_focus_range[
                            0]
                if self.config.output_focus_range[1] is not None:
                    output[output > self.config.output_focus_range[1]] = \
                        self.config.output_focus_range[
                            1]
                full_output.append(output)
                del trained_models
            self.light_output = np.mean(full_output, axis=0)
            return None
        else:
            self.light_config.input_channel = self.train_feature[1].shape[3]
            light_models = self.train_function(
                self.train_feature[1], self.train_label[5:], self.light_model, self.light_config,
                extend_feature=self.extend_feature[1],
                extend_label=self.extend_label[[3, 4]] if self.extend_label is not None else None,
            )
            self.light_output = get_prediction_output(light_models, self.test_feature[1], self.wind_config.train_method)

            if self.config.output_focus_range[0] is not None:
                self.light_output[self.light_output < self.config.output_focus_range[0]] = \
                    self.config.output_focus_range[0]
            if self.config.output_focus_range[1] is not None:
                self.light_output[self.light_output > self.config.output_focus_range[1]] = \
                    self.config.output_focus_range[1]

            return light_models

    def print_output_result(self, output=None):
        self.print_log("OUTPUT RESULT")

        if output is None:
            if self.wind_output is not None:
                print_prediction_to_output(station_idx_range=range(5), prediction_result=self.wind_output)
            if self.light_output is not None:
                print_prediction_to_output(station_idx_range=range(5, 10), prediction_result=self.light_output)
        else:
            print_prediction_to_output(station_idx_range=range(10), prediction_result=output)
        output_name = f"output_{paths.get_time()}.zip" if not self.is_sais_env else "output.zip"
        zip_output_file_and_move(paths.output_base_pth, output_zip=output_name)

    def print_log(self, message):
        if self.config.use_log:
            print(f"[{self.__class__.__name__}][{datetime.now().strftime('%d-%H:%M:%S')}]{message}")

    def print_all_config(self, wind=True, light=True):
        self.print_log("Base Config: ")
        self.print_class_fn(self.print_log, self.config)
        if wind and self.wind_model:
            self.print_log("Wind Model: ")
            if isinstance(self.wind_config, list) and isinstance(self.wind_model, list):
                for idx in range(len(self.wind_model)):
                    self.print_log(self.wind_model[idx](self.wind_config[idx]))
            else:
                self.print_log(self.wind_model(self.wind_config))
            self.print_log("Wind Config: ")
            self.print_class_fn(self.print_log, self.wind_config)
        if light and self.light_model:
            self.print_log("Light Model: ")
            if isinstance(self.light_config, list) and isinstance(self.light_model, list):
                for idx in range(len(self.light_model)):
                    self.print_log(self.light_model[idx](self.light_config[idx]))
            else:
                self.print_log(self.light_model(self.light_config))
            self.print_log("Light Config: ")
            self.print_class_fn(self.print_log, self.light_config)

    def get_r2_score(self, wind_output=None, light_output=None, split_output=False):
        if self.test_label is None:
            return 0

        score = []
        if self.wind_output is not None or wind_output is not None:
            wind_output = np.array(self.wind_output) if wind_output is None else wind_output
            if split_output:
                for i in range(5):
                    score.append(r2_score(wind_output[[i]], self.test_label[[i]]))
            else:
                score.append(r2_score(wind_output, self.test_label[:5]))
        else:
            score.append(None)

        if self.light_output is not None or light_output is not None:
            light_output = np.array(self.light_output) if light_output is None else light_output
            if split_output:
                for i in range(5):
                    score.append(r2_score(light_output[[i]], self.test_label[[i + 5]]))
            else:
                score.append(r2_score(light_output, self.test_label[5:]))
        else:
            score.append(None)

        if len(score) == 0:
            return 0
        return score

    def draw_label(self, show=True):
        if self.test_label is None:
            return None, None
        wind_fig, light_fig = None, None
        if self.wind_output is not None:
            other_data = [
                ("past_output", self.past_prediction_result[:5])] if self.past_prediction_result is not None else None
            wind_fig, _ = plot_predictions(self.wind_output, self.test_label[:5], other_data=other_data)
            if show:
                wind_fig.show()
        if self.light_output is not None:
            other_data = [
                ("past_output", self.past_prediction_result[5:])] if self.past_prediction_result is not None else None
            light_fig, _ = plot_predictions(self.light_output, self.test_label[5:], other_data=other_data)
            if show:
                light_fig.show()
        return wind_fig, light_fig

    def upload_data(self, score=None):
        if not self.is_sais_env and self.config.use_log and self.no_wandb is False:
            import swanlab as wandb
            self.draw_label(show=True)
            score = self.get_r2_score(split_output=self.split_output) if score is None else score
            self.print_log(score)
            if not isinstance(self.wind_config, list) and not isinstance(self.light_config, list):
                if not self.split_output:
                    wandb.log({"wind_score": score[0] if score[0] is not None else 0}, step=self.wind_config.step)
                    wandb.log({"light_score": score[1] if score[1] is not None else 0}, step=self.light_config.step)
                    self.light_config.step += 1
                    self.wind_config.step += 1
                else:
                    for i in range(10):
                        wandb.log({f"station_{i + 1}_score": score[i]})
                    self.light_config.step += 1
                    self.wind_config.step += 1
            else:
                wandb.log({"wind_score": score[0] if score[0] is not None else 0}, step=self.config.step)
                wandb.log({"light_score": score[1] if score[1] is not None else 0}, step=self.config.step)
                self.config.step += 1

    def run_pipeline(self, wind=True, light=True, no_print_output=True, save_torch_model=False, del_model_finally=True):
        self.print_log("Pipeline initialized")
        self.print_log(f"gpu: {self.gpu_is_available}")
        self.print_all_config(wind=wind, light=light)

        self.feature_engineering()
        self.print_log("Feature engineering initialized")

        if wind:
            self.train_wind_model()
            self.print_log("Training wind model initialized")

        if light:
            self.train_light_model()
            self.print_log("Training light model initialized")

        if not self.no_wandb:
            self.upload_data()

        if no_print_output:
            self.print_log("No output created")
        else:
            self.print_output_result()
            self.print_log("Output result initialized")

        if save_torch_model:
            if self.wind_model:
                save_model(self.wind_model, f"wind_model_{self.__class__.__name__}")
                self.print_log(f"Saved Wind Model")
            if self.light_model:
                save_model(self.light_model, f"light_model_{self.__class__.__name__}")
                self.print_log(f"Saved Light Model")

        if del_model_finally:
            self.print_log("Deleting model finally")
            del self.train_feature, self.test_feature, self.train_label

            if self.wind_model:
                del self.wind_model
            if self.light_model:
                del self.light_model

    def train_function(self, train_feature, train_label, model, config, extend_feature=None, extend_label=None):
        # train_feature (5, 365, 24, input_channel, 11, 11)

        def base_n_fold_method(fold_train_feature, fold_train_label):
            # return old_n_fold_train(model=model, feature=fold_train_feature, label=fold_train_label, config=config)
            return better_n_fold_train(model=model, feature=fold_train_feature, label=fold_train_label, config=config)

        if config.train_method == 'n_fold':
            return base_n_fold_method(train_feature, train_label)
        elif config.train_method == 'train_test':
            train_dataset, test_dataset = get_station_embedding_dataset(train_feature, train_label, config=config)
            trained_model, best_r2 = base_train_fun(
                train_dataset,
                test_dataset,
                model=model,
                config=config,
            )
            if config.print_training_information:
                self.print_log("Best r2: {}".format(best_r2))
            return trained_model
        elif config.train_method == 'no_station_embed':
            model_list = []
            for _ in range(train_feature.shape[0]):
                self.print_log(f"Start Train Split Model Count: {_ + 1}")
                model_list.append(base_n_fold_method(train_feature[[_]], train_label[[_]]))
            return model_list
        return None

    def get_output_np(self) -> np.ndarray:
        output = []
        if self.wind_output is not None:
            output.append(self.wind_output)
        if self.light_output is not None:
            output.append(self.light_output)
        if len(output) == 0:
            return None

        return np.concatenate(output, axis=0) if len(output) > 0 else None

    @staticmethod
    def print_class_fn(print_fn, class_to_print):
        if isinstance(class_to_print, list):
            for class_to_ in class_to_print:
                BasePipeline.print_class_fn(print_fn, class_to_)
        else:
            print_fn(f"{class_to_print.__class__.__name__}: {class_to_print.__dict__}")

    @staticmethod
    def set_seed(random_seed):
        # torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        import os
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = str(':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 关闭自动优化
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @staticmethod
    def get_is_sais_env():
        return bool(os.getenv("SAIS"))

    @staticmethod
    def load_raw_label(copy_pipeline):
        if copy_pipeline is not None:
            raw_train_label = copy_pipeline.raw_train_label
            raw_test_label = copy_pipeline.raw_test_label
        else:
            raw_train_label = load_sais_label_data(np_format=True, without_first=True)
            raw_test_label = load_sais_label_data(np_format=True, label_path=paths.nc_test_label_path)
        return raw_train_label, raw_test_label


class SpecialTimeSplitPipeline(BasePipeline):
    def __init__(self, model: Type[torch.nn.Module], config, wind_use_split=True, light_use_split=True):
        super().__init__(model, config)
        self.wind_use_split = wind_use_split
        self.light_use_split = light_use_split

    def train_wind_model(self):
        if not self.wind_use_split:
            return super().train_wind_model()

        self.wind_config.input_channel = self.train_feature[0].shape[3]
        special_train_feature, special_train_label = self.get_special_time_split(self.train_feature[0],
                                                                                 self.train_label[:5])
        wind_models_1 = self.train_function(self.train_feature[0], self.train_label[:5], self.wind_model,
                                            self.wind_config)
        wind_models_2 = self.train_function(special_train_feature, special_train_label, self.wind_model,
                                            self.wind_config)
        wind_output_1 = get_prediction_output(wind_models_1, self.test_feature[0], "n_fold")
        wind_output_2 = get_prediction_output(wind_models_2, self.get_special_time_split(self.test_feature[0]),
                                              "n_fold")
        self.wind_output = self.handle_special_output(wind_output_1, wind_output_2)
        if self.config.output_focus_range[0] is not None:
            self.wind_output[self.wind_output < self.config.output_focus_range[0]] = self.config.output_focus_range[0]
        if self.config.output_focus_range[1] is not None:
            self.wind_output[self.wind_output > self.config.output_focus_range[1]] = self.config.output_focus_range[1]
        return wind_models_1, wind_models_2

    def train_light_model(self):
        if not self.light_use_split:
            return super().train_light_model()

        self.light_config.input_channel = self.train_feature[1].shape[3]
        special_train_feature, special_train_label = self.get_special_time_split(self.train_feature[1],
                                                                                 self.train_label[5:])
        light_models_1 = self.train_function(self.train_feature[1], self.train_label[5:], self.light_model,
                                             self.light_config)
        light_models_2 = self.train_function(special_train_feature, special_train_label, self.light_model,
                                             self.light_config)
        light_output_1 = get_prediction_output(light_models_1, self.test_feature[1], "n_fold")
        light_output_2 = get_prediction_output(light_models_2, self.get_special_time_split(self.test_feature[1]),
                                               "n_fold")
        self.light_output = self.handle_special_output(light_output_1, light_output_2)
        if self.config.output_focus_range[0] is not None:
            self.light_output[self.light_output < self.config.output_focus_range[0]] = self.config.output_focus_range[0]
        if self.config.output_focus_range[1] is not None:
            self.light_output[self.light_output > self.config.output_focus_range[1]] = self.config.output_focus_range[1]
        return light_models_1, light_models_2

    @staticmethod
    def handle_special_output(output_data_1, output_data_2):
        # (5, n, 96), (5, n - 1, 96)
        output_data_1 = np.array(output_data_1)
        output_data_2 = np.array(output_data_2)
        # raw_output_1_shape = output_data_1.shape
        # raw_output_2_shape = output_data_2.shape
        # output_data_1 = output_data_1.reshape(
        #     raw_output_1_shape[0], raw_output_1_shape[1] * raw_output_1_shape[2]
        # )
        # output_data_2 = output_data_2.reshape(
        #     raw_output_2_shape[0], raw_output_2_shape[1] * raw_output_2_shape[2]
        # )
        output_data_1[:, 0:-1, 72:] = output_data_2[:, :, 24:48]
        output_data_1[:, 1:, :24] = output_data_2[:, :, 48:72]
        return output_data_1

    @staticmethod
    def get_special_time_split(feature_dataset, label_dataset=None):
        # input: (5, 365, 24, -1, 11, 11); (5, 365, 96)
        raw_feature_dataset_shape = feature_dataset.shape
        feature_dataset = feature_dataset.reshape(
            raw_feature_dataset_shape[0],
            raw_feature_dataset_shape[1] * raw_feature_dataset_shape[2],
            raw_feature_dataset_shape[3],
            raw_feature_dataset_shape[4], raw_feature_dataset_shape[5],
        )[:, 12: -12].reshape(
            raw_feature_dataset_shape[0],
            raw_feature_dataset_shape[1] - 1,
            raw_feature_dataset_shape[2],
            raw_feature_dataset_shape[3],
            raw_feature_dataset_shape[4], raw_feature_dataset_shape[5],
        )
        if label_dataset is None:
            return feature_dataset
        raw_label_dataset_shape = label_dataset.shape
        label_dataset = label_dataset.reshape(
            raw_label_dataset_shape[0],
            raw_label_dataset_shape[1] * raw_label_dataset_shape[2]
        )[:, 48: -48].reshape(
            raw_label_dataset_shape[0],
            raw_label_dataset_shape[1] - 1,
            raw_label_dataset_shape[2],
        )
        return feature_dataset, label_dataset


class NoStationEmbedPipeline(BasePipeline):
    def __init__(self, model: Type[torch.nn.Module] | None, config=BaseConfig(), copy_pipeline: BasePipeline = None):
        super().__init__(model, config, copy_pipeline)

    def train_wind_model(self):
        pass

    def train_light_model(self):
        pass
