import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(
        predictions: list = None,
        truths: list = None,
        other_data: list[tuple[str, list]] = None,
        titles: list = None,
        all_titles: list = None,
        fig_size: tuple = (20, 20),
        cols: int = 1,
        cut: int | tuple | list[int | tuple[int, int]] = -1,
        xtick_unit: int = 96,
        y_lim: tuple=(0, 1),
        save_file: str = None,
        prediction_name: str = "Predicted",
        with_none_sign: bool = True,
):

    def cut_pre_handle(data, index):
        temp_cut = cut
        if isinstance(cut, list):
            temp_cut = cut[index]
        if isinstance(temp_cut, tuple):
            data = data[temp_cut[0]:temp_cut[1]]
        elif temp_cut != -1:
            data = data[:temp_cut]
        return data

    if predictions is None:
        predictions = [None] * len(other_data[0][1])

    if truths is None:
        truths = [None] * len(predictions)

    if len(predictions) != len(truths):
        raise ValueError(f"`predictions_{len(predictions)}` and `truths_{len(truths)}` must have the same length.")

    n = max(len(predictions), 2)
    rows = (n + cols - 1) // cols

    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = axes.flatten()

    # Plot each pair
    for idx, (pred, true) in enumerate(zip(predictions, truths)):
        ax = axes[idx]

        if pred is not None:
            pred = pred.reshape(-1)
            pred = cut_pre_handle(pred, idx)
            ax.plot(pred, label=prediction_name, linestyle='--')

        if true is not None:
            true = true.reshape(-1)
            true = cut_pre_handle(true, idx)
            ax.plot(true, label='True', linestyle='-')

        if other_data is not None:
            for other_idx, other in enumerate(other_data):
                if not isinstance(other, tuple):
                    other = (f"Other_{other_idx}", other)
                other_ax = other[1][idx]
                other_ax = other_ax.reshape(-1)
                other_ax = cut_pre_handle(other_ax, idx)
                ax.plot(other_ax, label=other[0], linestyle='-')

        # Set title
        if titles is not None:
            ax.set_title(titles[idx])
        else:
            ax.set_title(f"Sample {idx + 1}")

        ax.legend()
        ax.set_ylim(*y_lim)
        ax.set_xticks(np.arange(0, len(pred) + 1, xtick_unit))
        xtick_labels = np.arange(0, len(pred) // xtick_unit + 1, 1)
        if isinstance(cut, list) and isinstance(cut[0], tuple):
            xtick_labels = xtick_labels + cut[idx][0] // xtick_unit
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

    # Remove any extra subplots
    for extra in axes[n:]:
        fig.delaxes(extra)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    return fig, axes

def cut_plot_prediction(data, resolution=20, unit_len=96, fig_size=(20, 20), all_title="cut_plot", with_none_sign=True):
    data_len = len(data)
    split_len = data_len // resolution + 1
    copy_data = [data] * split_len

    fig, _ = plot_predictions(
        copy_data,
        cut=[(unit_len * resolution * i, unit_len * resolution * (i + 1)) for i in range(split_len)],
        titles=[str(f"{unit_len} * {resolution * i}, {unit_len} * {resolution * (i + 1)}") for i in range(split_len)],
        fig_size=fig_size,
        with_none_sign=with_none_sign,
    )

    return fig


def upsample_day_hour_to_15min(data: np.ndarray) -> np.ndarray:
    # 校验输入尺寸
    days, hrs = data.shape
    if hrs != 24:
        raise ValueError("输入 data 必须为 (365, 24) 形状")

    # 将数据展平为一维连续序列
    flat_old = data.reshape(-1)  # 长度 = 365*24

    # 原始时刻点索引：0,1,2,...,365*24-1
    x_old = np.arange(flat_old.size)

    # 目标时刻点：从 0 到 (365*24-1) 等间距取 4 倍数量的点
    # np.arange(n*4)/4 生成 [0,0.25,0.5,...,n-0.25]
    x_new = np.arange(flat_old.size * 4) / 4

    # 构造插值函数（线性），bounds_error=False 保证无越界错误
    f = interp1d(x_old, flat_old, kind='linear', bounds_error=False)

    # 计算插值结果
    flat_new = f(x_new)  # 长度 = 365*24*4 = 365*96

    # 恢复为 (365,96)
    flat_new[[-1, -2, -3]] = flat_new[-4]
    return flat_new.reshape(days, -1)


def load_post_data(file_dir, no_time=True, reshape=None) -> np.ndarray | None:
    if not os.path.exists(file_dir):
        return None

    file_list = os.listdir(file_dir)
    if 'output' in file_list:
        file_dir = os.path.join(file_dir, 'output')
        file_list = os.listdir(file_dir)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    file_list = sorted(file_list, key=lambda t: int(t.split(".")[0][6:]))

    result = []
    for file_name in file_list:
        file_path = os.path.join(file_dir, file_name)
        loaded_np = pd.read_csv(file_path).to_numpy()
        if no_time:
            loaded_np = loaded_np[:, 1].astype(np.double)
        if reshape is not None:
            loaded_np = loaded_np.reshape(reshape)
        result.append(loaded_np)
    return result

