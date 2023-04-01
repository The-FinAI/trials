import os

import numpy as np
import pandas as pd
import typer


def sub(file_name):
    start_point = file_name.index("_") + 1
    end_point = file_name.index(".")
    symbol = file_name[start_point:end_point]
    return symbol


def select_stock_name(selected_symbol_path: str, stock_data_path: str) -> list:
    """
    Read the stock symbol to be selected from selected_symbol_path,
    check whether there is the stock symbol in stock_data_path,
    if so, save the stock symbol in train.

    Args:
    selected_symbol_path: Stock symbols to be selected
                          are stored in selected_symbol_path.
    stock_data_path: All stocks are stored in stock_data_path.

    Returns:
    A list of all pending stock symbols.
    """
    df1 = pd.read_csv(selected_symbol_path)
    result = df1["Symbol"].values.tolist()
    file = os.listdir(stock_data_path)
    train = list(
        filter(lambda x: (x[-4:] == ".csv" and sub(x) in result), file)
    )
    return train


def select_stock_time(begin_time: str, end_time: str, temp: list) -> list:
    """
    Return the time from begin_time to end_time in temp1.

    Args:
    begin_time: The start time of the stocks to be screened.
               For example: '2000-01-01'
    end_time: The end time of the stocks to be screened.
             For example: '2020-12-31'
    temp: A list of times.

    Returns:
    A list of the time from begin_time to end_time.
    """
    time_in_period = []
    for time in temp:
        if (time >= begin_time) & (time <= end_time):
            time_in_period.append(time)
    return time_in_period


def form_union_time(path_in: str, temp: list) -> list:
    """
    Take the union of the trading days of stocks.

    Args:
    path_in: Stocks to be processed are stored in path_in.
    temp: A list of the symbols of stocks to be processed.

    Returns:
    A list of the union of the trading days of stocks.
    """
    time = []
    union_time = []
    for file in temp:
        tmp = pd.read_csv(path_in + file, encoding="gbk")["时间"].values.tolist()
        time.append(tmp)
    for t in time:
        union_time = list(set(union_time).union(set(t)))
    union_time.sort()
    return union_time


def stocks_output(
    path_in: str,
    store_path: str,
    begin_time: str,
    end_time: str,
    stock_name: list,
    union_time: list,
) -> None:
    """
    Check if the stock data in path_in is missing, if there is
    an opening and closing price below 1 dollar and save it
    to store_path, begin_time and end_time are the time intervals
    of the stock data.

    Args:
    path_in: Stocks to be processed are stored in path_in.
    store_path: stocks that have been processed are stored in store_path.
    begin_time: The start time of the stocks to be screened.
               For example: '2000-01-01'
    end_time: The end time of the stocks to be screened.
             For example: '2020-12-31'
    stock_name: A list of the symbols of stocks to be processed.
    union_time: A list of the union of the trading days of stocks.

    Returns:
    Nothing to see here.
    """
    for file_name in stock_name:
        df = pd.read_csv(path_in + file_name, encoding="gbk")
        stock_time = df["时间"].values.tolist()
        stock_time = select_stock_time(begin_time, end_time, stock_time)
        if (set(stock_time)) == (set(union_time)):
            # Change the column name of the original data of each stock
            # to form a new table
            df.rename(
                columns={
                    "时间": "date",
                    "开盘价(原始币种)": "open",
                    "收盘价(原始币种)": "close",
                    "成交量(股)": "volume",
                },
                inplace=True,
            )
            dataframe = df.loc[
                (df["date"] >= begin_time) & (df["date"] <= end_time),
                ["date", "open", "close", "volume"],
            ]
            # Determine if a stock has missing data
            if np.all(pd.notnull(dataframe)):
                df2 = dataframe.astype(str)
                df2["open"] = df2["open"].apply(lambda x: x.replace(",", ""))
                df2["close"] = df2["close"].apply(lambda x: x.replace(",", ""))
                open_price = df2["open"].values.tolist()
                close_price = df2["close"].values.tolist()
                # Determine if a stock has an opening or closing price
                # less than $1 on a trading day
                if (all(float(i) >= 1 for i in open_price)) & (
                    all(float(i) >= 1 for i in close_price)
                ):
                    dataframe.to_csv(
                        store_path + file_name, index=False, encoding="gbk"
                    )


def data_processing(
    selected_symbol_path: str,
    stock_path: str,
    store_path: str,
    begin_time: str,
    end_time: str,
) -> None:
    """
    Data preprocessing

    Args:
    selected_symbol_path: Stock symbols to be selected
                          are stored in selected_symbol_path.
    stock_path: All stocks are stored in stock_path
    store_path: stocks that have been processed are stored in store_path.
    begin_time: The start time of the stocks to be screened.
               For example: '2000-01-01'
    end_time: The end time of the stocks to be screened.
             For example: '2020-12-31'

    Returns:
    Nothing to see here.
    """
    train = select_stock_name(selected_symbol_path, stock_path)
    union_time = form_union_time(stock_path, train)
    print(len(union_time))
    union_time = select_stock_time(begin_time, end_time, union_time)
    print(len(union_time))
    stocks_output(
        stock_path, store_path, begin_time, end_time, train, union_time
    )


if __name__ == "__main__":
    typer.run(data_processing)
