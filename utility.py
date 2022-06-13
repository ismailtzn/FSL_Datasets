import pandas as pd
import glob


def load_train_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_train_files = glob.glob(data_dir + "/" + "x_meta_train*")
    y_train_files = glob.glob(data_dir + "/" + "y_meta_train*")

    x_train_files.sort()
    y_train_files.sort()

    assert len(x_train_files) > 0
    assert len(y_train_files) > 0

    x_train_dfs = [pd.read_hdf(file, hdf_key) for file in x_train_files]
    y_train_dfs = [pd.read_hdf(file, hdf_key) for file in y_train_files]

    x_train = pd.concat(x_train_dfs)
    y_train = pd.concat(y_train_dfs)

    return x_train, y_train


def load_test_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_test_files = glob.glob(data_dir + "/" + "x_meta_test*")
    y_test_files = glob.glob(data_dir + "/" + "y_meta_test*")

    x_test_files.sort()
    y_test_files.sort()

    x_test_dfs = [pd.read_hdf(file, hdf_key) for file in x_test_files]
    y_test_dfs = [pd.read_hdf(file, hdf_key) for file in y_test_files]

    x_test = pd.concat(x_test_dfs)
    y_test = pd.concat(y_test_dfs)

    return x_test, y_test


def load_datasets(data_dir, hdf_key="cic_ids_2017"):
    return load_train_datasets(data_dir, hdf_key), load_test_datasets(data_dir, hdf_key)


def load_val_datasets(data_dir, hdf_key="cic_ids_2017"):
    x_val_files = glob.glob(data_dir + "/" + "x_meta_val*")
    y_val_files = glob.glob(data_dir + "/" + "y_meta_val*")

    x_val_files.sort()
    y_val_files.sort()

    x_val_dfs = [pd.read_hdf(file, hdf_key) for file in x_val_files]
    y_val_dfs = [pd.read_hdf(file, hdf_key) for file in y_val_files]

    x_val = pd.concat(x_val_dfs)
    y_val = pd.concat(y_val_dfs)

    return x_val, y_val
