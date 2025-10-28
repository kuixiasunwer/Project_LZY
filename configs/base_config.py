import torch.nn as nn
import torch
import os
from typing import Literal, Type

from train.loss_function import R2Loss, TweedieLoss


class BaseConfig:
    def __init__(self):
        self.feather_engineering_list = []
        self.input_seq_len = 24
        self.output_seq_len = 96

        self.learning_rate = 1e-3
        self.train_epochs = 50
        self.batch_size = 64
        self.dropout = 0.05
        self.fold_len = 10
        self.early_patience = 50
        self.input_channel = 24
        self.optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
        self.loss_fn: nn.Module = R2Loss(training=True)
        self.score_fn: nn.Module = R2Loss(training=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_device = torch.device('cuda')
        self.train_method: Literal['n_fold', 'train_test', 'no_station_embed'] = 'n_fold'
        self.train_frac = 0.8
        self.random_seed = 777
        self.use_temp_data = None
        self.station_embedding_dim = 8
        self.wind_feather = []
        self.light_feather = []
        self.wind_fix_function: Literal['other_label', 'poai_ghi'] = None
        self.light_fix_function: Literal['other_label', 'poai_ghi'] = None
        self.use_log = True
        self.print_training_information = True,
        self.pre_train_model_name: str = None
        self.del_row_dead_value = (False, False)
        self.del_all_day_zero_or_nan = (True, True)
        self.use_extend_station_data = False
        self.output_focus_range = (0, None)
        self.interpolate_axis = 1
        self.step = 1
        self.print_no_better_r2 = False
        self.skip_train = False
        self.nwp_source=[]
        self.seed_generator=torch.Generator().manual_seed(self.random_seed)


class LSTMConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.hidden_size = 256
        self.rnn_input_feature = 10
        self.resnet_upsample_radio = 0.25

        self.resnet_shape_list = [(11, 11)]

class ConvDecoderConfig(LSTMConfig):
    def __init__(self):
        super().__init__()
        self.resolution = 2
        self.shared_conv_decoder = False
        self.padding = False
        self.has_relu_in_the_end = True

class TweedieLSTMConfig(LSTMConfig):
    def __init__(self):
        super().__init__()
        self.loss_fn = TweedieLoss(power=1.001, training=False)