from configs.base_config import LSTMConfig, ConvDecoderConfig


def _history_docker_0_4():
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

    return base_config, wind_config, light_config


def _history_docker_0_14():
    wind_config = ConvDecoderConfig()
    wind_config.hidden_size = 256
    wind_config.rnn_input_feature = 15
    wind_config.train_epochs = 100

    light_config = ConvDecoderConfig()
    light_config.hidden_size = 256
    light_config.rnn_input_feature = 15
    light_config.shared_conv_decoder = True
    light_config.resolution = 4
    return wind_config, light_config


def _history_docker_0_22():
    wind_config = ConvDecoderConfig()
    wind_config.hidden_size = 256
    wind_config.rnn_input_feature = 300
    wind_config.train_epochs = 100

    light_config = ConvDecoderConfig()
    light_config.hidden_size = 256
    light_config.rnn_input_feature = 300
    light_config.shared_conv_decoder = True
    light_config.resolution = 4
    return wind_config, light_config

# def history_bigger_pipeline(copy_pipeline=None) -> BasePipeline:
#     base_config = LSTMConfig()
#     base_config.feather_engineering_list.append(ExtractFunctions.extract_wind_speed)
#     base_config.feather_engineering_list.append(ExtractFunctions.extract_wind_direction)
#     base_config.wind_feather = [*ExtractFunctions.get_raw_channel_names(),
#                                     ExtractFunctions.extract_wind_speed()]
#
#     base_config.del_row_dead_value = (True, True)
#     base_config.del_all_day_zero_or_nan = (True, True)
#     base_config.use_extend_station_data = False
#
#     wind_config_1, light_config_1 = _history_docker_0_22()
#     wind_config_2, light_config_2 = _history_docker_0_14()
#     wind_config_1.has_relu_in_the_end = wind_config_2.has_relu_in_the_end = False
#     light_config_1.has_relu_in_the_end = light_config_2.has_relu_in_the_end = False
#     wind_config_list = [wind_config_1, wind_config_2]
#     light_config_list = [light_config_1, light_config_2]
#
#     if not base_config.is_sais_env:
#         wind_config_list[0].base_device = light_config_list[0].base_device = 'cuda'
#         wind_config_list[1].base_device = light_config_list[1].base_device = 'cuda'
#
#     pipeline = BasePipeline(ResNetConvLSTMConvDecoderModel, base_config, copy_pipeline=copy_pipeline)
#     pipeline.wind_config = wind_config_list
#     pipeline.wind_model = [ResNetConvLSTMConvDecoderModel, ResNetSmallConvLSTMConvDecoderModel]
#     pipeline.light_config = light_config_list
#     pipeline.light_model = [ResNetConvLSTMConvDecoderModel, ResNetSmallConvLSTMConvDecoderModel]
#
#     return pipeline