from functions import *
from train.loss_function import r2_score
from util import paths
from util.data_loader import *



def display_label():
    pass

if __name__ == '__main__':
    os.chdir("..")
    past_data = load_post_data(paths.output_backup_path / "output_fix_ensmble_with_high_3_6989")
    test_fact_data = load_sais_label_data(np_format=True, label_path=paths.nc_test_label_path)
    train_fact_data = load_sais_label_data(np_format=True)
