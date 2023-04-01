import empyrical
import numpy as np
import pandas as pd
import typer
from eval_cointegration import CointegrationStateMachine, df2log
from select_pairs import log_func
from statsmodels.tsa.stattools import coint
from util import get_hedge_ratio, zscore


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(sum((a - b) ** 2)) / len(a)


def main(
    data_path: str,
    pairs_path: str,
    store_path: str,
    trading_threshold: str,
    stop_loss_threshold: str,
) -> None:
    """

    Args:
        data_path: The data of all assets in the rolling
                   where the specified stock is located
                   in the test is stored in trading_data_path.
        pairs_path: The file containing pairs of different
                    methods is stored in pairs_path.
        store_path:
        trading_threshold:
        stop_loss_threshold:

    Returns: Nothing to see here.

    """
    method = [
        "GGR",
        "Cointegration",
        "Correlation",
        "RL",
        "SAS+RL",
        "ALL+SAS+RL",
        "L2R",
    ]
    method_nums = len(method)
    df_pairs = pd.read_csv(pairs_path, encoding="gbk")
    for num in range(method_nums):
        pairs = df_pairs[method[num]].values.tolist()
        pairs_total = len(pairs)
        p_values = []
        euclidean_dist = []
        sharpe_rate = []
        max_drawdown_list = []
        annual_return_list = []
        annual_volatility_list = []

        for pairs_num in range(pairs_total):
            trading_pairs = pairs[pairs_num].split("-")
            asset_x = trading_pairs[0]
            asset_y = trading_pairs[1]
            df = pd.read_csv(
                data_path + "test_rolling_" + str(pairs_num + 1) + ".csv",
                encoding="gbk",
                header=[0, 1],
                index_col=0,
            ).astype(str)
            df_train = pd.read_csv(
                data_path + "train_rolling_" + str(pairs_num + 1) + ".csv",
                encoding="gbk",
                header=[0, 1],
                index_col=0,
            ).astype(str)
            df_valid = pd.read_csv(
                data_path + "valid_rolling_" + str(pairs_num + 1) + ".csv",
                encoding="gbk",
                header=[0, 1],
                index_col=0,
            ).astype(str)
            df_formation = pd.concat([df_train, df_valid], axis=0)
            trading_series = pd.DataFrame(
                {
                    "date": list(df.index),
                    "close_x": df[asset_x]["close"].tolist(),
                    "open_x": df[asset_x]["open"].tolist(),
                    "volume_x": df[asset_x]["volume"].tolist(),
                    "close_y": df[asset_y]["close"].tolist(),
                    "open_y": df[asset_y]["open"].tolist(),
                    "volume_y": df[asset_y]["volume"].tolist(),
                }
            )
            formation_series = pd.DataFrame(
                {
                    "date": list(df_formation.index),
                    "close_x": df_formation[asset_x]["close"].tolist(),
                    "open_x": df_formation[asset_x]["open"].tolist(),
                    "volume_x": df_formation[asset_x]["volume"].tolist(),
                    "close_y": df_formation[asset_y]["close"].tolist(),
                    "open_y": df_formation[asset_y]["open"].tolist(),
                    "volume_y": df_formation[asset_y]["volume"].tolist(),
                }
            )
            # calculate p_value
            x_close = df[asset_x]["close"].apply(log_func).values.tolist()
            y_close = df[asset_y]["close"].apply(log_func).values.tolist()
            _, p_value, _ = coint(x_close, y_close)
            p_values.append(p_value)
            # calculate euclidean_dist
            x_close = (
                df[asset_x]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            y_close = (
                df[asset_y]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            x_close = list(map(lambda x: x / x_close[0], x_close))
            y_close = list(map(lambda x: x / y_close[0], y_close))
            dis = dist(x_close, y_close)
            euclidean_dist.append(dis)
            # calculate sharpe_ratio
            log_df = df2log(formation_series)
            hedge_ratio = get_hedge_ratio(log_df["close_y"], log_df["close_x"])
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
            returns = [0]
            for i in range(1, len(net_value_list)):
                daily_return = (
                    net_value_list[i] - net_value_list[i - 1]
                ) / net_value_list[i - 1]
                returns.append(daily_return)
            returns = pd.Series(returns)
            sharpe = empyrical.sharpe_ratio(
                returns, risk_free=0.000085, period="daily", annualization=252
            )
            sharpe_rate.append(sharpe)
            # calculate max_drawdown
            max_drawdown = empyrical.max_drawdown(returns)
            max_drawdown_list.append(max_drawdown)
            # calculate annual_return
            annual_return = empyrical.annual_return(
                returns, period="daily", annualization=252
            )
            annual_return_list.append(annual_return)
            # calculate annual_volatility
            annual_volatility = empyrical.annual_volatility(
                returns, period="daily", alpha=2.0, annualization=252
            )
            annual_volatility_list.append(annual_volatility)
        pairs.append("mean")
        pairs.append("var")
        mean_euc = np.mean(euclidean_dist)
        var_euc = np.var(euclidean_dist)
        euclidean_dist.append(mean_euc)
        euclidean_dist.append(var_euc)
        mean_shar = np.mean(sharpe_rate)
        var_shar = np.var(sharpe_rate)
        sharpe_rate.append(mean_shar)
        sharpe_rate.append(var_shar)
        mean_p = np.mean(p_values)
        var_p = np.var(p_values)
        p_values.append(mean_p)
        p_values.append(var_p)
        mean_max_drawdown = np.mean(max_drawdown_list)
        var_max_drawdown = np.var(max_drawdown_list)
        max_drawdown_list.append(mean_max_drawdown)
        max_drawdown_list.append(var_max_drawdown)
        mean_annual_return = np.mean(annual_return_list)
        var_annual_return = np.var(annual_return_list)
        annual_return_list.append(mean_annual_return)
        annual_return_list.append(var_annual_return)
        mean_annual_volatility = np.mean(annual_volatility_list)
        var_annual_volatility = np.var(annual_volatility_list)
        annual_volatility_list.append(mean_annual_volatility)
        annual_volatility_list.append(var_annual_volatility)

        dataframe = pd.DataFrame(
            {
                "pair": pairs,
                "sharpe_rate": sharpe_rate,
                "euclidean_dist": euclidean_dist,
                "p_values": p_values,
                "drawdown_list": max_drawdown_list,
                "annual_return_list": annual_return_list,
                "annual_volatility_list": annual_volatility_list,
            }
        )
        dataframe.to_csv(
            store_path + method[num] + "_indicators.csv",
            index=False,
            encoding="gbk",
        )


if __name__ == "__main__":
    typer.run(main)
