from configs import BaseConfig
from configs.base_config import LSTMConfig, ConvDecoderConfig
from models.base.attention import MultiHeadAttentionWrapper
from models.base.convs import ResNetConv, SpecialConv, MultiResNetConv, ResNetSmallConv
from models.base.sequential.lstm import LstmModule, LstmModelCustomDecoder, LstmUpsampleModule
import torch
import torch.nn as nn


class BaseLSTMModel(nn.Module):
    def __init__(self, configs: LSTMConfig):
        super(BaseLSTMModel, self).__init__()
        self.station_embedding = nn.Embedding(10, configs.station_embedding_dim)
        self.rnn_input_feature = configs.rnn_input_feature
        self.configs = configs
        self.conv_layers = nn.Sequential(
            nn.Conv2d(configs.input_channel, 8, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(4, 2, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(50, self.rnn_input_feature) if self.rnn_input_feature != 50 else nn.Identity(),
        )
        self.combine_layer = nn.Sequential(
            nn.Linear(self.rnn_input_feature + configs.station_embedding_dim, self.rnn_input_feature),
            nn.GELU(),
        )

        self.rnn = LstmModule(
            input_feature=self.rnn_input_feature, input_seq_len=configs.input_seq_len,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers
        )


    def station_location_handle(self, data):
        data_shape = data.shape
        data = self.conv_layers(data.reshape(-1, *data.shape[2:]))
        data = data.reshape(*data_shape[:2], -1)
        return data


    def station_embedding_data(self, data, station_idx=None):
        if station_idx is None or self.configs.train_method == "no_station_embed":
            return data

        data_shape = data.shape
        station_idx = station_idx.reshape(-1, 1)
        station_idx = self.station_embedding(station_idx)
        data = torch.concat([data, station_idx.reshape(data_shape[0], 1, -1).repeat(1, self.configs.input_seq_len, 1)],
                            dim=-1)
        data = self.combine_layer(data)
        return data

    def sequential_handle(self, data):
        return self.rnn(data)

    def forward(self, data, station_idx=None):
        data = self.station_location_handle(data)
        data = self.station_embedding_data(data, station_idx)
        data = self.sequential_handle(data)

        return data


class ResNetConvLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super(ResNetConvLSTMModel, self).__init__(configs)
        self.conv_layers = ResNetConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature)


class MultiResNetConvLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super(MultiResNetConvLSTMModel, self).__init__(configs)
        self.conv_layers = MultiResNetConv(configs.input_channel, configs.rnn_input_feature)


class ResNetConvLSTMConvDecoderModel(BaseLSTMModel):
    def __init__(self, configs: ConvDecoderConfig):
        super().__init__(configs)
        self.conv_layers = ResNetConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, 11, 11)
        custom_decoder = SpecialConv(
            configs.hidden_size, configs.input_seq_len, configs.output_seq_len,
            resolution=configs.resolution, shared_layer=configs.shared_conv_decoder, padding=configs.padding,
            relu_in_end=configs.has_relu_in_the_end
        )
        self.rnn = LstmModelCustomDecoder(
            input_feature=self.rnn_input_feature, input_seq_len=configs.input_seq_len,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers,
            custom_decoder=custom_decoder
        )


class ResNetSmallConvLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super().__init__(configs)
        self.conv_layers = ResNetSmallConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, 11, 11)


class ResNetMultiLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super().__init__(configs)
        self.conv_layers = MultiResNetConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, configs.resnet_shape_list)



class ResNetSmallConvLSTMConvDecoderModel(ResNetConvLSTMConvDecoderModel):
    def __init__(self, configs: ConvDecoderConfig):
        super().__init__(configs)
        self.conv_layers = ResNetSmallConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, 11, 11)


class MiddleAverageLSTMConvDecoderModel(ResNetConvLSTMConvDecoderModel):
    def __init__(self, configs: ConvDecoderConfig):
        super().__init__(configs)
        self.conv_layers = GridFeatureExtractor(configs.input_channel, configs.rnn_input_feature)

    def station_location_handle(self, data):
        data = self.conv_layers(data)
        return data

class ResNetConvLSTMIterationDecoderModel(BaseLSTMModel):
    def __init__(self, configs: ConvDecoderConfig):
        super().__init__(configs)
        self.conv_layers = ResNetConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, 11, 11)
        self.custom_decoder = IterationRnnDecoder(hidden_dim=configs.hidden_size, dropout=configs.dropout, num_layers=configs.num_layers)
        self.rnn = LstmModelCustomDecoder(
            input_feature=self.rnn_input_feature, input_seq_len=configs.input_seq_len,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers,
            custom_decoder=self.custom_decoder
        )


class ResNetUpsampleConvLSTMModel(ResNetConvLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super().__init__(configs)
        self.upsample = LstmUpsampleModule(configs.rnn_input_feature)
        self.rnn = LstmModule(
            input_feature=self.rnn_input_feature, input_seq_len=configs.input_seq_len * 4,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers
        )
        self.rnn.decoder_linear = nn.Sequential(
            nn.Linear(configs.hidden_size, 1),
            nn.Flatten(),
            nn.ReLU(),
        )

    def sequential_handle(self, data):
        data = self.upsample(data)
        return self.rnn(data)



class ResNetConvLSTMAttentionDecoderModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super().__init__(configs)
        self.conv_layers = ResNetConv(configs.input_channel, int(configs.input_channel * configs.resnet_upsample_radio), configs.rnn_input_feature, 11, 11)
        custom_decoder = nn.Sequential(
            MultiHeadAttentionWrapper(configs.hidden_size, configs.hidden_size // 4, dropout=configs.dropout),
            nn.Flatten(),
            nn.Linear(configs.input_seq_len * configs.hidden_size // 4, configs.hidden_size),
            nn.Dropout(configs.dropout),
            nn.GELU(),
            nn.Linear(configs.hidden_size, configs.output_seq_len),
            nn.ReLU(),
        )
        self.rnn = LstmModelCustomDecoder(
            input_feature=self.rnn_input_feature, input_seq_len=configs.input_seq_len,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers,
            custom_decoder=custom_decoder
        )


class NoConvLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super(NoConvLSTMModel, self).__init__(configs)
        self.conv_layers = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(configs.input_channel * 11 * 11, configs.hidden_size),
            nn.GELU(),
            nn.Linear(configs.hidden_size, configs.hidden_size),
            nn.GELU(),
            nn.Linear(configs.hidden_size, configs.rnn_input_feature),
        )


class TweedieLSTMModel(BaseLSTMModel):
    def __init__(self, configs: LSTMConfig):
        super(TweedieLSTMModel, self).__init__(configs)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.input_seq_len * configs.hidden_size, configs.hidden_size),
            nn.Dropout(configs.dropout),
            nn.GELU(),
            nn.Linear(configs.hidden_size, configs.output_seq_len),
            nn.Softplus(),
        )
        self.rnn = LstmModelCustomDecoder(
            custom_decoder=self.decoder,
            input_feature=configs.rnn_input_feature, input_seq_len=configs.input_seq_len,
            output_seq_len=configs.output_seq_len,
            hidden_size=configs.hidden_size, num_layers=configs.num_layers)
