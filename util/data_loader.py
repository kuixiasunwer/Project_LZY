import pandas
import xarray as xr
import numpy as np
import pandas as pd
from typing import Union
import os

from util.paths import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
is_sais_env = bool(os.getenv("SAIS"))


def get_time_feature_base_pd_time(time_values):
    time_list = []
    for t in time_values.flatten():
        if isinstance(t, np.ndarray):
            # 如果仍然是数组，取第一个元素
            t = t[0] if len(t) > 0 else None
        time_list.append(t)

    # 计算天数
    days = time_values.shape[1] // 96

    # 创建形状为(10, days, 3)的时间戳数组，3个维度分别是月、日、时
    time_stamps = np.zeros((10, days, 3), dtype=np.int32)

    # 对于每一天，提取第一个时间点（假设是当天的00:00）
    for day in range(days):
        # 获取当天的第一个时间点
        day_start_idx = day * 96  # 每天96个时间点
        if day_start_idx < len(time_list):
            time_str = time_list[day_start_idx]
            # 使用安全的方式转换时间戳
            try:
                if isinstance(time_str, (str, bytes)):
                    dt = pd.Timestamp(time_str)
                elif isinstance(time_str, (pd.Timestamp, np.datetime64)):
                    dt = pd.Timestamp(time_str)
                else:
                    print(f"警告: 无法识别的时间格式: {type(time_str)}, 值: {time_str}")
                    # 使用当前时间作为默认值
                    dt = pd.Timestamp.now()

                # 提取月、日、时
                month = dt.month
                day_of_month = dt.day
                hour = dt.hour

                # 为所有站点设置相同的时间戳
                for station in range(10):
                    time_stamps[station, day, 0] = month
                    time_stamps[station, day, 1] = day_of_month
                    time_stamps[station, day, 2] = hour
            except Exception as e:
                print(f"处理时间戳时出错: {e}, 值: {time_str}")
                # 使用占位符
                for station in range(10):
                    time_stamps[station, day, 0] = 1  # 默认1月
                    time_stamps[station, day, 1] = 1  # 默认1日
                    time_stamps[station, day, 2] = 0  # 默认0时
    return time_stamps, time_list


def load_sais_feather_data(np_format=True, use_temp=True, only_test=False, use_mf=False) -> (
        tuple[xr.Dataset, xr.Dataset] | tuple[np.ndarray, np.ndarray]):
    """
    :return: (train_dataset, test_dataset)
    """
    result_data = []
    nwp_list = ['NWP_1', 'NWP_2', 'NWP_3']
    if use_temp and os.path.exists(xr_base_path / 'xr_raw_train_dataset.nc') and os.path.exists(
            xr_base_path / 'xr_raw_test_dataset.nc'):
        result_data.append(xr.open_dataset(xr_base_path / 'xr_raw_train_dataset.nc'))
        result_data.append(xr.open_dataset(xr_base_path / 'xr_raw_test_dataset.nc'))

        if np_format:
            result_data[0] = result_data[0]['data'].to_numpy()
            result_data[1] = result_data[1]['data'].to_numpy()

        return result_data[0], result_data[1]

    for index, data_path in enumerate([Path(nc_base_trainer_path), Path(nc_base_test_path)]):
        # date_list = sorted(set([item.name.split('.nc')[0] for item in list(data_path.glob('*/*/*.nc'))]))
        # data = []
        if only_test and index == 0:
            result_data.append(None)
            continue

        if use_mf:
            stat_data = []
            for item_stat in range(1, 10 + 1):
                nwp_data_list = []
                for item_nwp in nwp_list:
                    target_path = str(data_path / str(item_stat) / str(item_nwp) / "*.nc")
                    nwp_data = xr.open_mfdataset(paths=target_path, parallel=True,
                                                 # engine='h5netcdf'
                                                 )
                    nwp_data_list.append(nwp_data)
                nwp_data_list = xr.concat(
                    [_.assign_coords(source=f"NWP{nwp_index + 1}_") for nwp_index, _ in enumerate(nwp_data_list)],
                    dim='channel')
                nwp_data_list = nwp_data_list.assign_coords(channel=nwp_data_list.source + nwp_data_list.channel)
                nwp_data_list = nwp_data_list.assign_coords(sta=item_stat)
                stat_data.append(nwp_data_list)
            stat_data = xr.concat(stat_data, dim='sta')
            result_data.append(stat_data)
        else:
            date_list = sorted(set([item.name.split('.nc')[0] for item in list(data_path.glob('*/*/*.nc'))]))
            data = []
            if only_test and index == 0:
                result_data.append(None)
                continue

            for item_stat in range(1, 10 + 1):
                stat_data = []
                for item_date in date_list:
                    item_data = xr.concat(
                        [
                            xr.open_dataset(data_path / str(item_stat) / item_nwp / f'{item_date}.nc')
                            .assign_coords(source=f"NWP{item_nwp.split('_')[-1]}_")
                            for item_nwp in ['NWP_1', 'NWP_2', 'NWP_3']
                        ],
                        dim='channel'
                    )
                    item_data = item_data.assign_coords(
                        channel=item_data.source.astype(str) + item_data.channel.astype(str))
                    stat_data.append(item_data)
                stat_data = xr.concat(stat_data, dim='time')
                stat_data = stat_data.assign_coords(sta=item_stat)
                data.append(stat_data)
            data = xr.concat(data, dim='sta')
            result_data.append(data)

    if use_temp and not is_sais_env:
        if not only_test:
            result_data[0].to_netcdf(xr_base_path / 'xr_raw_train_dataset.nc')
        result_data[1].to_netcdf(xr_base_path / 'xr_raw_test_dataset.nc')

    if np_format:
        if not only_test:
            result_data[0] = result_data[0]['data'].to_numpy()
        result_data[1] = result_data[1]['data'].to_numpy()

    return result_data[0], result_data[1]


def load_sais_label_data(without_first=False, np_format=True, label_path=nc_base_label_path) -> Union[
    xr.Dataset, np.ndarray, dict]:
    if label_path is None:
        return None

    if 'train' in str(label_path):
        data_type = 'train'
    else:
        data_type = 'test'

    result_data: list[xr.Dataset] = []
    for station_index in range(10):
        try:
            pd_raw_data = pd.read_csv(label_path / f"{station_index + 1}_normalization_{data_type}.csv",
                                      encoding='utf-8')
        except UnicodeDecodeError:
            try:
                pd_raw_data = pd.read_csv(label_path / f"{station_index + 1}_normalization_{data_type}.csv",
                                          encoding='gbk')
            except UnicodeDecodeError:
                try:
                    pd_raw_data = pd.read_csv(label_path / f"{station_index + 1}_normalization_{data_type}.csv",
                                              encoding='gb2312')
                except UnicodeDecodeError:
                    # 最后尝试latin-1，它可以读取任何字节序列
                    pd_raw_data = pd.read_csv(label_path / f"{station_index + 1}_normalization_{data_type}.csv",
                                              encoding='latin-1')
        if without_first:
            pd_raw_data = pd_raw_data.iloc[96:]
        pd_raw_data['时间'] = pd.to_datetime(pd_raw_data['时间'])
        xr_data = pd_raw_data.to_xarray()
        result_data.append(xr_data)
    result = xr.concat(result_data, dim='sta')

    if np_format:
        # 原有行为：只返回功率数据
        power_data = result['功率(MW)'].to_numpy()
        power_data = power_data.reshape(10, -1, 96)  # (10, 365, 96)
        return power_data
    else:
        # 返回包含功率和时间的字典，均为numpy格式
        power_data = result['功率(MW)'].to_numpy()
        power_data = power_data.reshape(10, -1, 96)  # (10, 365, 96)

        time_values = result['时间'].values

        # print(f"Time values shape: {time_values.shape}")#(10, 40800)

        time_stamps, time_list = get_time_feature_base_pd_time(time_values)

        # 创建一个包含功率和时间的字典
        result_dict = {
            'power': power_data,  # 形状: (10, days, 96)
            'time_stamps': time_stamps,  # 形状: (10, days, 3) - 月、日、时
            'raw_time': time_list,  # 原始时间戳列表
        }
        return result_dict


def get_xr_data_channel(temp=None) -> np.ndarray:
    if temp is None:
        temp = load_sais_feather_data()[0]
    return temp['data']['channel'].to_numpy().reshape(1, -1)


def get_np_label_data(skip_first_day=True):
    xr_fact_data = load_sais_label_data()['功率(MW)']
    np_result = np.array(xr_fact_data)
    np_result = np_result.reshape(10, -1, 96)
    if skip_first_day:
        return np_result[:, 1:, :]
    return np_result


if __name__ == '__main__':
    os.chdir("..")
    xr_dataset = load_sais_feather_data(np_format=False)[0]
    print(xr_dataset)
