import math
import os
from typing import List

import numpy as np
import pandas as pd
import typer
from statsmodels.tsa.stattools import coint

dataset_names = ["train", "valid", "test", "formation"]
x_symbol = " "
y_symbol = " "


def sub(file_name):
    end_point = file_name.index("_")
    dataset_type = file_name[:end_point]
    return dataset_type


def select_file_name(rolling_dataset_path, dataset_type):
    file_name = os.listdir(rolling_dataset_path)
    file_name = list(filter(lambda x: sub(x) == dataset_type, file_name))
    file_name.sort()
    return file_name


def log_func(x):
    if isinstance(x, str):
        x = float(x.replace(",", ""))
    return math.log(x)


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(sum((a - b) ** 2)) / len(a)


def vertify_coint(asset_x, asset_y, p_threshold) -> bool:
    if len(asset_x) != len(asset_y):
        return False
    _, p_value, _ = coint(asset_x, asset_y)
    return p_value < p_threshold


def corr(a, b):
    a = np.array(a)
    b = np.array(b)
    my_rho = np.corrcoef(a, b)
    return my_rho[0][1]


def load_data(path, file_name):
    df = pd.read_csv(
        path + file_name,
        encoding="gbk",
        header=[0, 1],
        index_col=0,
    )
    return df


def write_pair(
    file_path: str,
    rolling_df: pd.DataFrame,
    pair_x: str,
    pair_y: str,
    dataset_type: str,
) -> None:
    """
    Combine data for two stocks of a stock pair into one .csv

    Args:
    file_path: The generated .csv is stored into file_path.
    rolling_df: Get data for two stocks from rolling_df.
    pair_x: The symbol of the first stock
    pair_y: The symbol of the second stock
    num: Represents from which rolling the output .csv is selected

    Returns:
    Nothing to see here.
    """
    dataframe = pd.DataFrame(
        {
            "date": list(rolling_df.index),
            "close_x": rolling_df[pair_x]["close"].tolist(),
            "open_x": rolling_df[pair_x]["open"].tolist(),
            "volume_x": rolling_df[pair_x]["volume"].tolist(),
            "close_y": rolling_df[pair_y]["close"].tolist(),
            "open_y": rolling_df[pair_y]["open"].tolist(),
            "volume_y": rolling_df[pair_y]["volume"].tolist(),
        }
    )
    dataframe.to_csv(
        f"{file_path}/{pair_x}-{pair_y}_{dataset_type}.csv",
        index=False,
        encoding="gbk",
    )


def select_pairs_eucl(*args: List) -> None:
    """
    Pick the stock pair with the smallest Euclidean distance.

    Args:
    args[0]: Contains data for all stocks in a rolling.

    Returns:
    Nothing to see here.
    """
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    dis_min = float("inf")
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            x_close = list(map(lambda x: x / x_close[0], x_close))
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            y_close = list(map(lambda x: x / y_close[0], y_close))
            dis = dist(x_close, y_close)
            if dis < dis_min:
                dis_min = dis
                x_symbol = column_name[i * 3][0]
                y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol


def select_pairs_coin(*args: List) -> None:
    """
    Pick the stock pair with the smallest Euclidean distance
    where the value of P_value is below p_threshold.

    Args:
    args[0]: Contains data for all stocks in a rolling.
    args[1]: The maximum value of p_value

    Returns:
    Nothing to see here.
    """
    df = args[0]
    p_threshold = args[1]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    dis_min = float("inf")
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(log_func)
                .values.tolist()
            )
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(log_func)
                .values.tolist()
            )
            if vertify_coint(x_close, y_close, p_threshold):
                x_close = list(map(lambda x: x / x_close[0], x_close))
                y_close = list(map(lambda x: x / x_close[0], y_close))
                dis = dist(x_close, y_close)
                if dis < dis_min:
                    dis_min = dis
                    x_symbol = column_name[i * 3][0]
                    y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol


def select_pairs_corr(*args: List) -> None:
    """
    Pick the stock pair with the maximum correlation.

    Args:
    args[0]: Contains data for all stocks in a rolling.

    Returns:
    Nothing to see here.
    """
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    corr_max = 0
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            x_close = list(map(lambda x: x / x_close[0], x_close))
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            y_close = list(map(lambda x: x / y_close[0], y_close))
            cor = corr(x_close, y_close)
            if cor > corr_max:
                corr_max = cor
                x_symbol = column_name[i * 3][0]
                y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol


METHODS = {
    "euclidean": select_pairs_eucl,
    "cointegration": select_pairs_coin,
    "correlation": select_pairs_corr,
}


def main(
    rolling_dataset_path: str,
    store_path: str,
    method: str,
    p_threshold: str,
) -> None:
    """
    Pick a pair of stocks according to method_num from rolling_dataset_path,
    and save the data of the pair in store_path in .csv format.

    Args:
    rolling_dataset_path: All rolling datasets are stored in
                          rolling_dataset_path.
    store_path: The final select_pairs.csv files are stored in store_path.
    method: The methods for selecting stock pairs, including 'euclidean',
            'cointegration' and 'correlation'.
    p_threshold: When method is 'cointegration', p_threshold is meaningful,
                 and p_threshold represents the maximum value of p_value.
                 When method isn't 'cointegration', p_threshold has no meaning.

    Returns:
    Nothing to see here.
    """
    train = select_file_name(rolling_dataset_path, "train")
    valid = select_file_name(rolling_dataset_path, "valid")
    test = select_file_name(rolling_dataset_path, "test")
    rolling_total_number = len(train)
    pairs = []
    store_path = f"{store_path}/{method[0:4]}_pairs"
    os.makedirs(store_path, exist_ok=True)
    for rolling_serial in range(rolling_total_number):
        df_train = load_data(rolling_dataset_path, train[rolling_serial])
        df_valid = load_data(rolling_dataset_path, valid[rolling_serial])
        df_test = load_data(rolling_dataset_path, test[rolling_serial])
        df_formation = pd.concat([df_train, df_valid], axis=0)
        df_rolling = [df_train, df_valid, df_test, df_formation]
        df_formation = df_formation.astype(str)
        args = (
            [df_formation]
            if method != "cointegration"
            else [df_formation, float(p_threshold)]
        )
        x_symbol, y_symbol = METHODS[method](*args)
        num = str(rolling_serial + 1)
        tmp = "rolling_" + num + "_" + x_symbol + "-" + y_symbol
        pairs.append(tmp)
        for dataset_type in range(4):
            name = dataset_names[dataset_type]
            df_output = df_rolling[dataset_type].astype(str)
            write_pair(store_path, df_output, x_symbol, y_symbol, name)
    df = pd.DataFrame(pairs, columns=["pairs"])
    df.to_csv(
        f"{store_path}/select_pairs.csv",
        index=False,
        mode="w",
        encoding="gbk",
    )


if __name__ == "__main__":
    typer.run(main)
