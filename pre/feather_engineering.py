import os

import numpy as np
import xarray as xr

from configs import BaseConfig
from pre.cleaner import fix_label_dataset
from util import paths

nwp_list = ['NWP1', 'NWP2', 'NWP3']


def _save_train_data_mean_and_std(dataset, tag):
    data_mean = np.nanmean(dataset, axis=(0, 1, 2, 4, 5), keepdims=True)
    data_std = np.nanstd(dataset, axis=(0, 1, 2, 4, 5), keepdims=True)
    np.save(paths.np_base_path / f"{tag}_data_mean.npy", data_mean)
    np.save(paths.np_base_path / f"{tag}_data_std.npy", data_std)


def z_score_normal(dataset, data_mean=None, data_std=None):
    if data_mean is None or data_std is None:
        data_mean = np.nanmean(dataset, axis=(0, 1, 2, 4, 5), keepdims=True)
        data_std = np.nanstd(dataset, axis=(0, 1, 2, 4, 5), keepdims=True)
    dataset = (dataset - data_mean) / data_std
    return dataset


def get_channel_by_name(names):
    result = []
    if isinstance(names, str):
        names = [names]
    for name in names:
        flag = False
        for nwp in nwp_list:
            if nwp in name:
                result.append(name)
                flag = True
                break
        if flag:
            continue

        for site in nwp_list:
            if name == 'msl' and site != 'NWP2':
                continue
            elif name == 'sp' and site == 'NWP2':
                continue

            result.append(site + '_' + name)
    return result


def rename_xr_dataset(xr_dataset, raw_name, new_name):
    channels = []
    for channel in xr_dataset['channel'].values:
        channels.append(str(channel).replace(raw_name, new_name))
    xr_dataset = xr_dataset.assign_coords(channel=channels)
    return xr_dataset


def convert_xr_to_np(xr_dataset):
    return xr_dataset['data'].to_numpy()


class ExtractFunctions:
    @staticmethod
    def get_raw_channel_names():
        return ['NWP1_ghi', 'NWP1_poai', 'NWP1_sp', 'NWP1_t2m', 'NWP1_tcc', 'NWP1_tp', 'NWP1_u100', 'NWP1_v100',
                'NWP2_ghi',
                'NWP2_msl', 'NWP2_poai', 'NWP2_t2m', 'NWP2_tcc', 'NWP2_tp', 'NWP2_u100', 'NWP2_v100', 'NWP3_ghi',
                'NWP3_poai',
                'NWP3_sp', 'NWP3_t2m', 'NWP3_tcc', 'NWP3_tp', 'NWP3_u100', 'NWP3_v100']

    @staticmethod
    def base_extract(xr_dataset, input_data: list[str], output_data: str, handel_fn):
        input_data_list = []
        for input_data_name in input_data:
            target_data = xr_dataset['data'].sel(channel=get_channel_by_name(input_data_name))
            target_data = rename_xr_dataset(target_data, input_data_name, output_data)
            input_data_list.append(target_data)
        final_data = xr.DataArray(handel_fn(*input_data_list)).to_dataset()
        xr_feather_dataset = xr.concat([xr_dataset, final_data], dim='channel')
        return xr_feather_dataset

    @staticmethod
    def extract_wind_speed(xr_dataset=None):
        """
        Only: {
            wind: 0.6652590713024255
        }
        """
        out_name = 'wind_sped'
        if xr_dataset is None:
            return out_name

        def speed_fn(u100, v100):
            return np.sqrt(u100 ** 2 + v100 ** 2)

        return ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name, speed_fn)

    @staticmethod
    def extract_wind_direction(xr_dataset=None):
        out_name = 'wind_dir'
        if xr_dataset is None:
            return [out_name + '_wdir1', out_name + '_wdir2']

        def wdir1_fn(u100, v100):
            return 180.0 + np.arctan2(u100, v100) * (180.0 / np.pi)

        def wdir2_fn(u100, v100):
            return 270.0 - np.arctan2(v100, u100) * (180.0 / np.pi)

        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wdir1', wdir1_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wdir2', wdir2_fn)
        return xr_dataset

    @staticmethod
    def extract_wind_speed_diff(xr_dataset=None):
        out_name = 'wind_speed_diff'
        if xr_dataset is None:
            return out_name

        def diff_fn(u100, v100):
            sped = np.sqrt(u100 ** 2 + v100 ** 2)
            flatten_data = sped.stack(datetime=('time', 'lead_time'))
            gradient_data = np.gradient(flatten_data)[-1]
            gradient_data = gradient_data.reshape(sped.shape)
            xr_data = xr.DataArray(gradient_data, dims=sped.dims, coords=sped.coords, attrs=sped.attrs, name=sped.name)
            return xr_data

        return ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name, diff_fn)

    @staticmethod
    def extract_wind_pressure_ratio(xr_dataset=None):
        out_name = 'wsped_sp_ratio'
        if xr_dataset is None:
            return out_name

        def ratio_fn(wind, sp):
            return wind / (sp + 1e-6)

        return ExtractFunctions.base_extract(xr_dataset, [ExtractFunctions.extract_wind_speed()
            , 'sp'], out_name, ratio_fn)


    @staticmethod
    def extract_wind_extend_features(xr_dataset=None):
        out_name = 'wind_extend'
        if xr_dataset is None:
            return [out_name + '_wind_speed', out_name + '_wdir1', out_name + '_wdir2', out_name + '_ghi_poai',
                    out_name + '_wind_power_potential', out_name + '_wind_power_potential_msl']

        def speed_fn(u100, v100):
            return np.sqrt(u100 ** 2 + v100 ** 2)

        def wdir1_fn(u100, v100):
            return 180.0 + np.arctan2(u100, v100) * (180.0 / np.pi)

        def wdir2_fn(u100, v100):
            return 270.0 - np.arctan2(v100, u100) * (180.0 / np.pi)

        def ghi_poai_fn(ghi, poai):
            return ghi - poai

        def wind_power_potential(u100, v100, t2m, sp):
            wind_speed = np.sqrt(u100 ** 2 + v100 ** 2)
            air_density = 1.293 * (1 - 0.00367 * (t2m - 273.15)) * sp
            return 0.5 * air_density * (wind_speed ** 3)

        def wind_power_potential_msl(u100, v100, t2m, msl):
            wind_speed = np.sqrt(u100 ** 2 + v100 ** 2)
            air_density = 1.293 * (1 - 0.00367 * (t2m - 273.15)) * msl
            return 0.5 * air_density * (wind_speed ** 3)

        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wind_speed', speed_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wdir1', wdir1_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wdir2', wdir2_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['ghi', 'poai'], out_name + '_ghi_poai', ghi_poai_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100', 't2m', 'sp'],
                                                   out_name + '_wind_power_potential', wind_power_potential)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100', 't2m', 'msl'],
                                                   out_name + '_wind_power_potential_msl', wind_power_potential_msl)
        return xr_dataset

    @staticmethod
    def extract_light_extend_features(xr_dataset=None):
        # Not Better
        out_name = 'light_extend'
        if xr_dataset is None:
            return [out_name + '_wind_speed', out_name + '_t2m_sp', out_name + '_t2m_msl', out_name + '_ghi_poai',
                    out_name + '_ghi', out_name + '_t2m_squared', out_name + '_panel_temperature', out_name + '_panel_temperature',
                    out_name + '_kd']

        def speed_fn(u100, v100):
            return np.sqrt(u100 ** 2 + v100 ** 2)

        def light_fn(t2m, sp):
            return t2m / (sp + 1e-6)

        def light_fn_msl(t2m, msl):
            return t2m / (msl + 1e-6)

        def ghi_poai_fn(ghi, poai):
            return ghi - poai

        def ghi_fn(ghi, poai):
            return ghi / (poai + 1e-6)

        def t2m_squared(t2m):
            return t2m ** 2

        def Panel_Temperature(t2m, poai):
            return (t2m - 273.15) + (poai * 0.1) / 20

        def kd_fn(ghi, poai):
            return (ghi - poai) / (ghi + 1e-6)

        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['u100', 'v100'], out_name + '_wind_speed', speed_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['t2m', 'sp'], out_name + '_t2m_sp', light_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['t2m', 'msl'], out_name + '_t2m_msl', light_fn_msl)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['ghi', 'poai'], out_name + '_ghi_poai', ghi_poai_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['ghi', 'poai'], out_name + '_ghi', ghi_fn)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['t2m'], out_name + '_t2m_squared', t2m_squared)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['t2m', 'poai'], out_name + '_panel_temperature',
                                                   Panel_Temperature)
        xr_dataset = ExtractFunctions.base_extract(xr_dataset, ['ghi', 'poai'], out_name + '_kd', kd_fn)

        return xr_dataset


def extract_feather(xr_feather_dataset: xr.Dataset, function_list, np_format=True):
    for fun in function_list:
        xr_feather_dataset = fun(xr_feather_dataset)
    if np_format:
        return xr_feather_dataset['data'].to_numpy()
    return xr_feather_dataset


def test_feather_engineering(test_feathers: xr.Dataset, use_np_file=False, base_dataset=None):
    if base_dataset is None and use_np_file is False:
        raise ValueError("base_dataset is none and use_np_file is False")
    if use_np_file:
        test_feathers = extract_feather(test_feathers, [])

        wind_test_feathers = test_feathers[:5]
        light_test_feathers = test_feathers[5:]

        wind_test_feathers = z_score_normal(
            wind_test_feathers,
            data_mean=np.load(paths.np_base_path / "wind_data_mean.npy"),
            data_std=np.load(paths.np_base_path / "wind_data_std.npy")
        )
        light_test_feathers = z_score_normal(
            light_test_feathers,
            data_mean=np.load(paths.np_base_path / "light_data_mean.npy"),
            data_std=np.load(paths.np_base_path / "light_data_std.npy")
        )
        test_feathers = np.concatenate([wind_test_feathers, light_test_feathers], axis=0)

        return test_feathers
    return None


def base_feather_engineering(train_feathers: xr.Dataset, train_labels: xr.Dataset, test_feathers: xr.Dataset,
                             extend_features=None, config=BaseConfig()):
    wind_station = list(range(1, 6))
    wind_extend_station = [2, 3, 5]
    light_extend_station = [7, 9]
    light_station = list(range(6, 11))
    train_feathers = extract_feather(train_feathers, config.feather_engineering_list, np_format=False)
    test_feathers = extract_feather(test_feathers, config.feather_engineering_list, np_format=False)

    if extend_features is not None:
        extend_features = extract_feather(extend_features, config.feather_engineering_list, np_format=False)

    wind_train_feathers = train_feathers.sel(sta=wind_station)
    wind_test_feathers = test_feathers.sel(sta=wind_station)
    light_train_feathers = train_feathers.sel(sta=light_station)
    light_test_feathers = test_feathers.sel(sta=light_station)
    if extend_features is not None:
        wind_extend_feathers = extend_features.sel(sta=wind_extend_station)
        light_extend_feathers = extend_features.sel(sta=light_extend_station)
    else:
        wind_extend_feathers = None
        light_extend_feathers = None

    if len(config.wind_feather):
        target_channel = get_channel_by_name(config.wind_feather)
        for channel in target_channel.copy():
            if channel not in wind_train_feathers['channel'].values:
                target_channel.remove(channel)
        wind_train_feathers = wind_train_feathers.sel(channel=target_channel)
        wind_test_feathers = wind_test_feathers.sel(channel=target_channel)
        if wind_extend_feathers is not None:
            wind_extend_feathers = wind_extend_feathers.sel(channel=target_channel)
    if len(config.light_feather):
        target_channel = get_channel_by_name(config.light_feather)
        for channel in target_channel.copy():
            if channel not in light_train_feathers['channel'].values:
                target_channel.remove(channel)
        light_train_feathers = light_train_feathers.sel(channel=target_channel)
        light_test_feathers = light_test_feathers.sel(channel=target_channel)
        if light_extend_feathers is not None:
            light_extend_feathers = light_extend_feathers.sel(channel=target_channel)

    if len(config.nwp_source):
        choose_channel = []
        for channel in wind_train_feathers['channel'].values:
            for nwp in config.nwp_source:
                if nwp in channel:
                    choose_channel.append(channel)
        wind_train_feathers = wind_train_feathers.sel(channel=choose_channel)
        wind_test_feathers = wind_test_feathers.sel(channel=choose_channel)

        choose_channel = []
        for channel in light_train_feathers['channel'].values:
            for nwp in config.nwp_source:
                if nwp in channel:
                    choose_channel.append(channel)
        light_train_feathers = light_train_feathers.sel(channel=choose_channel)
        light_test_feathers = light_test_feathers.sel(channel=choose_channel)

    wind_train_feathers_np = convert_xr_to_np(wind_train_feathers)
    wind_test_feathers_np = convert_xr_to_np(wind_test_feathers)
    light_train_feathers_np = convert_xr_to_np(light_train_feathers)
    light_test_feathers_np = convert_xr_to_np(light_test_feathers)
    if extend_features is not None:
        wind_extend_feathers = convert_xr_to_np(wind_extend_feathers)
        light_extend_feathers = convert_xr_to_np(light_extend_feathers)

    wind_test_feathers_np = z_score_normal(
        wind_test_feathers_np,
        data_mean=np.nanmean(wind_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True),
        data_std=np.nanstd(wind_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True)
    )
    light_test_feathers_np = z_score_normal(
        light_test_feathers_np,
        data_mean=np.nanmean(light_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True),
        data_std=np.nanstd(light_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True)
    )
    if extend_features is not None:
        wind_extend_feathers = z_score_normal(
            wind_extend_feathers,
            data_mean=np.nanmean(wind_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True),
            data_std=np.nanstd(wind_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True)
        )
        light_extend_feathers = z_score_normal(
            light_extend_feathers,
            data_mean=np.nanmean(light_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True),
            data_std=np.nanstd(light_train_feathers_np, axis=(0, 1, 2, 4, 5), keepdims=True)
        )
        extend_features = [wind_extend_feathers, light_extend_feathers]
    else:
        extend_features = [None, None]

    wind_train_feathers_np = z_score_normal(wind_train_feathers_np)
    light_train_feathers_np = z_score_normal(light_train_feathers_np)

    # train_feathers = np.concatenate([wind_train_feathers, light_train_feathers], axis=0)
    # test_feathers = np.concatenate([wind_test_feathers, light_test_feathers], axis=0)
    train_feathers_np = [wind_train_feathers_np, light_train_feathers_np]
    test_feathers_np = [wind_test_feathers_np, light_test_feathers_np]

    # Fix Label
    train_labels = fix_label_dataset(train_labels, train_feathers, config)

    # train_labels[7] = np.load(paths.np_base_path / "fix_station_8.npy")
    # train_labels[9] = np.load(paths.np_base_path / "fix_station_10.npy")
    # Fix Label

    return train_feathers_np, train_labels, test_feathers_np, extend_features


if __name__ == '__main__':
    from util.data_loader import load_sais_feather_data

    os.chdir("..")

    feather_dataset, _ = load_sais_feather_data(np_format=True)
    _save_train_data_mean_and_std(feather_dataset[:5], tag="wind")
    _save_train_data_mean_and_std(feather_dataset[5:], tag="light")
