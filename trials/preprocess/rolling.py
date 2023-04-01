import os

import arrow
import pandas as pd
import typer

dataset_names = ["train", "valid", "test"]


def sub(file_name):
    start_point = file_name.index("_") + 1
    end_point = file_name.index(".")
    symbol = file_name[start_point:end_point]
    return symbol


def form_all_time_points(
    stock_data_path: str,
    training_month: str,
    validation_month: str,
    testing_month: str,
) -> list:
    """
    Generate all time nodes within a rolling.

    Args:
    stock_data_path: All stocks data are stored in stock_data_path.
    training_month: The number of months included in the training
    validation_month: The number of months included in the validation
    testing_month: The number of months included in the testing

    Returns:
    A list containing all time nodes within a rolling.The elements
    in this list are in XXXX-YY-ZZ format of str type.
    """
    file_name = os.listdir(stock_data_path)
    begin_time = pd.read_csv(
        stock_data_path + file_name[0], encoding="gbk"
    ).iloc[0]["date"]
    training_begin_time = arrow.get(begin_time, "YYYY-MM-DD")
    training_end_time = training_begin_time.shift(months=+int(training_month))
    validation_end_time = training_end_time.shift(
        months=+int(validation_month)
    )
    testing_end_time = validation_end_time.shift(months=+int(testing_month))
    time_points = [
        point.format("YYYY-MM-DD")
        for point in [
            training_begin_time,
            training_end_time,
            validation_end_time,
            testing_end_time,
        ]
    ]
    return time_points


def form_random_symbol(random_symbol_path: str) -> list:
    """
    Read the symbols contained in each rolling.

    Args:
    random_symbol_path: The stock symbols included in each rolling that
                        are randomly generated by random_stocks_selected.py
                        in advance are stored in random_symbol_path.

    Returns:
    A list containing the list of stocks each roll contains.
    """
    random_symbol = os.listdir(random_symbol_path)
    symbol_for_rolling = []
    for file in random_symbol:
        df = pd.read_csv(random_symbol_path + file, encoding="gbk")
        symbol_list = df["Symbol"].values.tolist()
        symbol_for_rolling.append(symbol_list)
    return symbol_for_rolling


def form_data(
    file_name: list,
    stock_data_path: str,
    begin_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    Read data for multiple stocks over a period of time
    and merge them together.

    Args:
    file_name: A list of stocks to read
    stock_data_path: All stocks data are stored in stock_data_path.
    begin_time: Start time to read stock data in XXXX-YY-ZZ format of str type
    end_time: End time to read stock data in XXXX-YY-ZZ format of str type

    Returns:
    The data form of a certain period contained in the final rolling.
    """
    df = pd.read_csv(stock_data_path + file_name[0], encoding="gbk")
    df = df.loc[(df["date"] >= begin_time) & (df["date"] < end_time)]
    df = df.set_index(["date"])
    length = len(file_name)
    for index in range(1, length):
        df0 = pd.read_csv(stock_data_path + file_name[index], encoding="gbk")
        df0 = df0.loc[(df0["date"] >= begin_time) & (df0["date"] < end_time)]
        df0 = df0.set_index(["date"])
        df = pd.concat([df, df0], axis=1)
    file_name = map(sub, file_name)
    df.columns = pd.MultiIndex.from_product(
        [file_name, ["open", "close", "volume"]]
    )
    return df


def write_data(df, store_path, csv_name, num):
    df.to_csv(
        store_path + csv_name + "_rolling_" + num + ".csv", encoding="gbk"
    )


def form_rolling(
    stock_data_path: str,
    store_path: str,
    training_month: str,
    validation_month: str,
    testing_month: str,
    random_symbol_path: str,
) -> None:
    """
    All rollings are formed, and the dataset for each period
    of each rolling is stored in the form of .csv.

    Args:
    stock_data_path: All stocks data are stored in stock_data_path.
    store_path: The final .csv files are stored in store_path.
    training_month: The number of months included in the training
    validation_month: The number of months included in the validation
    testing_month: The number of months included in the testing
    random_symbol_path: The stock symbols included in each rolling that
                        are randomly generated by random_stocks_selected.py
                        in advance are stored in random_symbol_path.

    Returns:
    Nothing to see here.
    """
    time_points = form_all_time_points(
        stock_data_path, training_month, validation_month, testing_month
    )
    symbol_for_rolling = form_random_symbol(random_symbol_path)
    rolling_total_number = len(symbol_for_rolling)
    # According to the serial number of rolling in the first layer,
    # and the type of dataset in the second layer,
    # the required .csv file is generated cyclically
    for rolling_serial_number in range(rolling_total_number):
        for dataset_type in range(3):
            name = dataset_names[dataset_type]
            df = form_data(
                symbol_for_rolling[rolling_serial_number],
                stock_data_path,
                time_points[dataset_type],
                time_points[dataset_type + 1],
            )
            num = str(rolling_serial_number + 1)
            write_data(df, store_path, name, num)


if __name__ == "__main__":
    typer.run(form_rolling)