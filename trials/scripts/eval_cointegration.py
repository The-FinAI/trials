import csv
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from util import (
    Action,
    PositionState,
    get_curr_net_value,
    get_hedge_ratio,
    get_valid_action_indexes,
    plot_assets,
    zscore,
)


class CointegrationStateMachine(object):
    def __init__(
        self,
        date: List[str],
        asset_name: List[str],
        asset_price: List[List[float]],  # [A, B]，价格取log
        spread_serial: List[float],  # 价差序列，A - hedge_ratio * B
        commission_rate: float = 0.001,  # 手续费
        fund_ratio: float = 1.0,  # 资金量比例，B/A
        init_net_value: float = 1.0,  # 初始的净值
        trading_threshold: float = 1.0,
        stop_loss_threshold: float = 1.0,
    ):
        # 画图用
        self.date = date
        self.asset_name = asset_name

        self.commission_rate = commission_rate
        self.spread_serial = spread_serial
        self.asset_price = asset_price
        self.fund_ratio = fund_ratio

        self.trading_threshold = trading_threshold
        self.stop_loss_threshold = stop_loss_threshold

        # state
        self.position = PositionState.bear
        self.curr_idx = -1
        self.last_buy_idx: Optional[int] = None
        self.last_net_value: Optional[float] = None
        self.init_net_value = init_net_value

    def reset(self):
        self.curr_idx = 0
        self.position = PositionState.bear
        observation = {
            "position": self.position,
            "net_value": 1.0,
            "spread": self.spread_serial[self.curr_idx],
            self.asset_name[0]: self.asset_price[0][self.curr_idx],
            self.asset_name[1]: self.asset_price[1][self.curr_idx],
        }
        self.last_buy_idx = None
        self.last_net_value = self.init_net_value
        return observation

    def close_position(self, position, last_net_value, last_buy_idx, end_idx):
        """
        :return: 返回净值变化量
        """
        return (
            get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                position,
                last_net_value,
                last_buy_idx,
                end_idx,
            )
            * (1 - self.commission_rate)
            - last_net_value
        )

    def step(self):
        """
        t = 1, ..., T
        a_t, s_{t + 1}
        """
        # state转换
        curr_spread = self.spread_serial[self.curr_idx]
        action = -1
        if self.curr_idx == len(self.spread_serial) - 1:
            action = Action.close
        else:
            if self.position == PositionState.bear:
                action = Action.close
                if (
                    self.trading_threshold
                    <= curr_spread
                    < self.stop_loss_threshold
                ):
                    self.position = PositionState.short
                    action = Action.short
                    self.last_buy_idx = self.curr_idx
                elif (
                    -self.trading_threshold
                    >= curr_spread
                    > -self.stop_loss_threshold
                ):
                    self.position = PositionState.long
                    action = Action.long
                    self.last_buy_idx = self.curr_idx
            elif self.position == PositionState.long:
                action = Action.long
                if curr_spread >= 0:
                    # close
                    if (
                        curr_spread < self.trading_threshold
                        or curr_spread >= self.stop_loss_threshold
                    ):
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = None
                        self.position = PositionState.bear
                        action = Action.close
                    # short
                    else:
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = self.curr_idx
                        self.position = PositionState.short
                        action = Action.short
                else:
                    # close
                    if curr_spread <= -self.stop_loss_threshold:
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = None
                        self.position = PositionState.bear
                        action = Action.close
            elif self.position == PositionState.short:
                action = Action.short
                if curr_spread <= 0:
                    # close
                    if (
                        curr_spread > -self.trading_threshold
                        or curr_spread <= -self.stop_loss_threshold
                    ):
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = None
                        self.position = PositionState.bear
                        action = Action.close
                    # long
                    else:
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = self.curr_idx
                        self.position = PositionState.long
                        action = Action.long
                else:
                    # close
                    if curr_spread >= self.stop_loss_threshold:
                        net_value_change = self.close_position(
                            self.position,
                            self.last_net_value,
                            self.last_buy_idx,
                            self.curr_idx,
                        )
                        self.last_net_value += net_value_change
                        self.last_buy_idx = None
                        self.position = PositionState.bear
                        action = Action.close

        self.curr_idx += 1
        if self.curr_idx == len(self.spread_serial):
            done = True
        else:
            done = False

        curr_net_value = self.last_net_value
        if (
            self.position == PositionState.short
            or self.position == PositionState.long
        ):
            curr_net_value = get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx - 1,
            )

        if done:
            # dummy observation
            observation = {
                "position": int(self.position),
                "net_value": curr_net_value,  # 返回的是最终的净值
                "spread": -1,
                self.asset_name[0]: -1,
                self.asset_name[1]: -1,
            }
        else:
            observation = {
                "position": int(self.position),
                "net_value": curr_net_value,
                "spread": self.spread_serial[self.curr_idx],
                self.asset_name[0]: self.asset_price[0][self.curr_idx],
                self.asset_name[1]: self.asset_price[1][self.curr_idx],
            }
        assert action != -1

        return action, observation, done

    def plot_trajectory(
        self,
        action_list: List[int],
        net_value_list: List[float],
        figsize: Tuple[int, int] = (15, 5),
        net_value_limit: Tuple[float, float] = (0.9, 1.1),
    ):
        assert len(net_value_list) == len(action_list)
        long_idxs, short_idxs, close_idxs = get_valid_action_indexes(
            action_list
        )
        figure = plot_assets(
            date=np.array(self.date, dtype="datetime64"),
            asset_x=np.array(self.asset_price[0]),
            asset_x_label=self.asset_name[0],
            asset_y=np.array(self.asset_price[1]),
            asset_y_label=self.asset_name[1],
            net_value=np.array(net_value_list),
            long_idxs=np.array(long_idxs),
            short_idxs=np.array(short_idxs),
            close_idxs=np.array(close_idxs),
            figsize=figsize,
            # figsize=(18, 6),
            net_value_limit=net_value_limit,
            spread=np.array(self.spread_serial),
            trading_threshold=self.trading_threshold,
            stop_loss_threshold=self.stop_loss_threshold,
        )

        return figure


def sub_pair_name(file_name: str) -> str:
    """
    Pick the name of pair from the file_name.

    Args:
    file_name: The file generated by select_pairs.py to store
               the data of the two stocks contained in each pair.

    Returns:
    THe name of a pair.
    """
    end_point = file_name.index("_")
    pair_name = file_name[:end_point]
    return pair_name


def sub_type(file_name: str) -> str:
    """
    Pick the type of dataset from the file_name.

    Args:
    file_name: The file generated by select_pairs.py to store
               the data of the two stocks contained in each pair.

    Returns:
    The type of a dataset.
    """
    start_point = file_name.index("_") + 1
    end_point = file_name.index(".")
    dataset_type = file_name[start_point:end_point]
    return dataset_type


def select_file_name(pairs_path: str, dataset_type: str) -> list:
    """
    Find all files of type dataset_type in pairs_path.

    Args:
    pairs_path: The files generated by select_pairs.py to store
               the data of the two stocks contained in each pair
               are stored in pairs_path.
    dataset_type: The type of dataset.

    Returns:
    A list of filenames of type dataset_type in pairs_path.
    """
    file_name = os.listdir(pairs_path)
    file_name = list(filter(lambda x: sub_type(x) == dataset_type, file_name))
    file_name.sort()
    return file_name


def log_func(x):
    if isinstance(x, str):
        x = float(x.replace(",", ""))
    return math.log(x)


def df2log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the values in df to log values.

    Args:
    df: Pending df

    Returns:
    Processed df
    """
    dataframe = pd.DataFrame(
        {
            "date": df["date"].tolist(),
            "close_x": df["close_x"].apply(func=log_func).tolist(),
            "open_x": df["open_x"].apply(func=log_func).tolist(),
            "volume_x": df["volume_x"].apply(func=log_func).tolist(),
            "close_y": df["close_y"].apply(func=log_func).tolist(),
            "open_y": df["open_y"].apply(func=log_func).tolist(),
            "volume_y": df["volume_y"].apply(func=log_func).tolist(),
        }
    )
    return dataframe


def formation_hedge_ratio(pairs_path: str, file_name: str) -> float:
    """
    Calculate the hedge_ratio of the formation.

    Args:
    pairs_path: The files generated by select_pairs.py to store
               the data of the two stocks contained in each pair
               are stored in pairs_path.
    file_name: Calculate the hedge_ratio from the file_name.

    Returns:
    The value of hedge_ratio.
    """
    df = pd.read_csv(pairs_path + file_name, encoding="gbk")
    df = df2log(df)
    hedge_ratio = get_hedge_ratio(df["close_y"], df["close_x"])
    return hedge_ratio


def main(
    pairs_path: str,
    store_path_dir: str,
    trading_threshold: str,
    stop_loss_threshold: str,
) -> None:
    """
    Generate trading behavior.

    Args:
    pairs_path: The files generated by select_pairs.py to store
               the data of the two stocks contained in each pair
               are stored in pairs_path.
    store_path_dir: The generated trading behavior files are
                    stored in store_path_dir.
    trading_threshold: The value of trading_threshold.
    stop_loss_threshold: The value of stop_loss_threshold.
    Returns:
    Noting to see here.
    """
    formation = select_file_name(pairs_path, "formation")
    trading = select_file_name(pairs_path, "test")
    pairs_total = len(formation)
    for pairs_num in range(pairs_total):
        trading_pairs = sub_pair_name(formation[pairs_num])
        store_path = f"{store_path_dir}/test_monitor_{trading_pairs}"
        os.makedirs(store_path, exist_ok=True)
        hedge_ratio = formation_hedge_ratio(pairs_path, formation[pairs_num])
        trading_series = pd.read_csv(
            pairs_path + trading[pairs_num], encoding="gbk"
        )
        trading_series["date"] = pd.to_datetime(
            trading_series["date"], format="%Y-%m-%d"
        )
        trading_series = df2log(trading_series)
        trading_series.insert(
            0,
            "spread",
            zscore(
                trading_series["close_y"],
                trading_series["close_x"],
                hedge_ratio,
            ),
        )
        trading_pairs = trading_pairs.split("-")
        env = CointegrationStateMachine(
            trading_series["date"].tolist(),
            trading_pairs,
            [
                trading_series["close_x"].tolist(),
                trading_series["close_y"].tolist(),
            ],
            trading_series["spread"].tolist(),
            trading_threshold=float(trading_threshold),
            stop_loss_threshold=float(stop_loss_threshold),
        )

        zscore_list = []
        net_value_list = []
        action_list = []
        obs = env.reset()

        zscore_list.append(obs["spread"])
        while True:
            action, obs, done = env.step()
            net_value_list.append(obs["net_value"])
            action_list.append(int(action))
            if done:
                break
            else:
                zscore_list.append(obs["spread"])
        assert len(zscore_list) == len(action_list) == len(net_value_list)
        figure = env.plot_trajectory(
            action_list, net_value_list, figsize=(15, 10)
        )
        figure.savefig(f"{store_path}/figure.pdf")

        field_names = ["actions", "net_values"]
        with open(f"{store_path}/test_trajectory.csv", "wt") as fw:
            writer = csv.DictWriter(fw, fieldnames=field_names)
            writer.writeheader()
            writer.writerow(
                {
                    "actions": str(action_list),
                    "net_values": str(net_value_list),
                }
            )

        pd.DataFrame(
            {
                "date": trading_series["date"],
                "actions": action_list,
                "net_values": net_value_list,
            }
        ).to_csv(f"{store_path}/test_trajectory_1.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
