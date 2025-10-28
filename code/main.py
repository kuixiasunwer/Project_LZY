import os
import sys
import numpy as np
import torch
from typing import Type

import swanlab

########## IMPORTANT ###########
####### NO IMPORT BEFORE HERE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

if not os.path.exists("run.sh"):
    os.chdir("..")
else:
    os.chdir(".")
sys.path.append(".")
sys.path.append("/app/")
from models.LSTM5_24 import BaseLSTMModel, ResNetSmallConvLSTMModel, ResNetMultiLSTMModel
from configs.base_config import BaseConfig, LSTMConfig, ConvDecoderConfig
from pre.feather_engineering import ExtractFunctions
from train.loss_function import R2Loss, R2LossAndSmoothLoss
from train.pipeline import BasePipeline
import util.paths as paths

from util.data_loader import load_sais_feather_data
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def history_pipeline_0_11() -> BasePipeline:
    base_config = LSTMConfig()
    base_config.feather_engineering_list.append(ExtractFunctions.extract_wind_speed)

    wind_config = LSTMConfig()
    wind_config.hidden_size = 256
    wind_config.rnn_input_feature = 15
    wind_config.train_epochs = 100

    light_config = LSTMConfig()
    light_config.hidden_size = 256
    light_config.rnn_input_feature = 15
    light_config.train_epochs = 50

    base_config.del_all_day_zero_or_nan = (True, True)

    pipeline = BasePipeline(BaseLSTMModel, base_config)

    pipeline.wind_config = wind_config

    pipeline.light_config = light_config

    return pipeline


def lab_1_multi_nwp():
    lab_index = 1
    for use_nwp1 in [False, True]:
        for use_nwp2 in [False, True]:
            for use_nwp3 in [False, True]:
                nwp_source = [
                    "NWP1" if use_nwp1 else "-",
                    "NWP2" if use_nwp2 else "-",
                    "NWP3" if use_nwp3 else "-"
                ]
                if not use_nwp1 and not use_nwp2 and not use_nwp3:
                    continue

                pipeline = history_pipeline_0_11()
                pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
                pipeline.wind_config.hidden_size = 512
                pipeline.light_config.hidden_size = 64

                pipeline.wind_config.resnet_shape_list = [(5, 5), (7, 7), (11, 11)]
                pipeline.light_config.resnet_shape_list = [(11, 11)]
                pipeline.config.nwp_source = nwp_source

                swanlab.init(project="Aug-Project-Lab1-8-8", config=pipeline.config.__dict__)
                pipeline.run_pipeline(light=False)
                swanlab.log({"lab index": lab_index})
                swanlab.finish()


def lab_2_resnet():
    count = 0
    for big in [False, True]:
        for middle2 in [False, True]:
            for middle1 in [False, True]:
                for small in [True, False]:
                    resnet_shape = [
                        (11, 11) if big else (None, None),
                        (7, 7) if middle2 else (None, None),
                        (5, 5) if middle1 else (None, None),
                        (1, 1) if small else (None, None),
                    ]
                    resnet_shape = set(resnet_shape)
                    resnet_shape = list(resnet_shape)

                    count += 1

                    pipeline = history_pipeline_0_11()
                    pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
                    pipeline.wind_config.resnet_shape_list = pipeline.light_config.resnet_shape_list = resnet_shape

                    swanlab.init(project="Aug-Project", config=pipeline.wind_config.__dict__)
                    pipeline.run_pipeline()
                    swanlab.finish()


def lab_2_resnet_nwp2_nwp3_solar():
    for big in [False, True]:
        for middle2 in [False, True]:
            for middle1 in [False, True]:
                for small in [True, False]:
                    resnet_shape = [
                        (11, 11) if big else (None, None),
                        (7, 7) if middle2 else (None, None),
                        (5, 5) if middle1 else (None, None),
                        (1, 1) if small else (None, None),
                    ]
                    resnet_shape = set(resnet_shape)
                    resnet_shape = list(resnet_shape)

                    pipeline = history_pipeline_0_11()
                    pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
                    pipeline.config.nwp_source = ['NWP2', 'NWP3']
                    pipeline.wind_config.resnet_shape_list = pipeline.light_config.resnet_shape_list = resnet_shape

                    swanlab.init(project="Aug-Project", config=pipeline.wind_config.__dict__)
                    pipeline.run_pipeline(wind=False)
                    swanlab.finish()


def lab_3():
    for use_mlp_fix in [True, False]:
        pipeline = history_pipeline_0_11()
        pipeline.config.wind_fix_function = pipeline.config.light_fix_function = "other_label" if use_mlp_fix else None
        pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel

        pipeline.wind_config.resnet_shape_list = [(5, 5), (7, 7), (11, 11)]
        pipeline.light_config.resnet_shape_list = [(11, 11)]
        pipeline.wind_config.train_method = pipeline.light_config.train_method = "no_station_embed"
        pipeline.split_output = True

        swanlab.init(project="Aug-Project-Lab3", config=pipeline.config.__dict__)
        pipeline.run_pipeline()
        swanlab.finish()


def lab_4():
    for batch_size in [16, 32, 48, 64]:
        pipeline = history_pipeline_0_11()
        pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
        pipeline.config.batch_size = pipeline.wind_config.batch_size = pipeline.light_config.batch_size = batch_size


        pipeline.wind_config.resnet_shape_list = [(5, 5), (7, 7), (11, 11)]
        pipeline.light_config.resnet_shape_list = [(11, 11)]

        swanlab.init(project="Aug-Project-HParam-BatchSize", config={"batch size": batch_size})
        pipeline.run_pipeline()
        swanlab.finish()


def lab_5():
    for hidden_size in [32, 48, 640, 812]:
        pipeline = history_pipeline_0_11()

        pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
        pipeline.config.hidden_size = pipeline.wind_config.hidden_size = pipeline.light_config.hidden_size = hidden_size
        pipeline.config.random_seed = pipeline.wind_config.random_seed = pipeline.light_config.random_seed = 999


        pipeline.wind_config.resnet_shape_list = [(5, 5), (7, 7), (11, 11)]
        pipeline.light_config.resnet_shape_list = [(11, 11)]

        swanlab.init(project="Aug-Project-HParam-HiddenSize", config={"hidden size": hidden_size})
        pipeline.run_pipeline()
        swanlab.finish()

def lab_6():
    for train_epoch in [50, 100, 150]:
        pipeline = history_pipeline_0_11()

        pipeline.wind_model = pipeline.light_model = ResNetMultiLSTMModel
        pipeline.config.train_epochs = pipeline.wind_config.train_epochs = pipeline.light_config.train_epochs = train_epoch
        pipeline.config.random_seed = pipeline.wind_config.random_seed = pipeline.light_config.random_seed = 999


        pipeline.wind_config.resnet_shape_list = [(5, 5), (7, 7), (11, 11)]
        pipeline.light_config.resnet_shape_list = [(11, 11)]

        swanlab.init(project="Aug-Project-HParam-TrainEpoch", config={"train epoch": train_epoch})
        pipeline.run_pipeline()
        swanlab.finish()


if __name__ == '__main__':
    # lab_1_multi_nwp()
    # lab_4()
    lab_5()

    # lab_6()

