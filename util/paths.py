import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

xr_base_path = Path("data/processed/xr_data")
np_base_path = Path("data/processed/np_data")

extend_feature_path = xr_base_path / "extend_feature.nc"
extend_label_path = np_base_path / "extend_label.npy"

if not bool(os.getenv("SAIS")):
    fix_label_path = np_base_path
    nc_base_trainer_path = Path('data/Input-train')
    nc_base_test_path = Path('data/Input-test')
    nc_base_label_path = Path('data/Input-train/fact_data')
    nc_test_label_path = Path('data/Input-test/fact_data')
    output_dst_path = Path('data/log/past_outputs')
else:
    fix_label_path = np_base_path / "docker_label"
    nc_base_trainer_path = Path('/saisdata/train/POWER_TRAIN_ENV')
    nc_base_test_path = Path('/saisdata/test/POWER_TEST_ENV')
    nc_base_label_path = Path('/saisdata/train/power_train')
    nc_test_label_path = None
    output_dst_path = Path('/saisresult')
torch_base_path = Path('data/processed/torch_data')

output_base_pth = Path('output')

output_backup_path = Path('data/log/past_outputs')

logger_path = Path('data/log')

def get_time():
    return datetime.now().strftime("%m-%d-%H-%M")


def get_test_output_time():
    test_time_index = pd.Series((pd.date_range(
        start=pd.to_datetime('2025-03-01 00:00:00'),
        end=pd.to_datetime('2025-04-30 23:45:00'),
        freq='15min').strftime("%Y/%-m/%-d %H:%M"))
                                .values)

    return np.tile(test_time_index.to_numpy().reshape(1, -1), reps=(10, 1))