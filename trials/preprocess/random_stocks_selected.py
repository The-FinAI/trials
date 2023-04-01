import os
import random

import pandas as pd
import typer


def sub(file_name):
    start_point = file_name.index("_") + 1
    end_point = file_name.index(".")
    symbol = file_name[start_point:end_point]
    return symbol


def form_random_symbol(
    symbols_num_each_rolling: str,
    stock_data_path: str,
    random_symbol_path: str,
) -> None:
    """
    Determine the number of stocks contained in each rolling through
    symbols_num_each_rolling, randomly select symbols_num_each_rolling
    stocks from all stocks to form .csv of symbols, and generate as
    many .csv as possible with disjoint symbols. Then rolling.py can
    use these .csv to generate rolling datasets.

    Args:
    symbols_num_each_rolling: The number of stocks contained in each rolling.
    stock_data_path: All stocks data are stored in stock_data_path.
    random_symbol_path: The stock symbols included in each rolling
                        are stored in random_symbol_path.

    Returns:
    Nothing to see here.
    """
    symbols_num_each_rolling = int(symbols_num_each_rolling)
    stock_symbol = os.listdir(stock_data_path)
    random.shuffle(stock_symbol)
    symbol_num_total = len(stock_symbol)
    rolling_num = symbol_num_total / symbols_num_each_rolling
    rolling_serial_num = 1
    start_index = 0
    while rolling_serial_num <= rolling_num:
        end_index = start_index + symbols_num_each_rolling
        random_symbol = stock_symbol[start_index:end_index]
        random_symbol.sort()
        df = pd.DataFrame(random_symbol, columns=["Symbol"])
        df.to_csv(
            random_symbol_path
            + "symbol_set_"
            + str(rolling_serial_num)
            + ".csv",
            index=False,
            mode="w",
            encoding="gbk",
        )
        rolling_serial_num += 1
        start_index += symbols_num_each_rolling


if __name__ == "__main__":
    typer.run(form_random_symbol)
