import calendar
import datetime
import glob
import linecache
import math
import os
import tracemalloc
from enum import IntEnum
from typing import List, Optional, Pattern, Tuple

import empyrical
import numpy as np
import pandas as pd
import psutil
import pynvml
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


class Action(IntEnum):
    long = 0  # long A, short B
    short = 1  # short A, long B
    close = 2  # close position


class PositionState(IntEnum):
    long = 0  # long A, short B
    short = 1  # long B, short A
    bear = 2  # bear position


def str2datetime(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, "%Y-%m-%d")


def load_data(input_data_dir: str, pair_name: str) -> pd.DataFrame:
    file_path = f"{input_data_dir}/{pair_name}.csv"
    df = pd.read_csv(file_path)
    return df


def get_hedge_ratio(series_y: pd.Series, series_x: pd.Series) -> float:
    model = sm.OLS(series_y, series_x)
    results = model.fit()
    hedge_ratio = results.params[series_x.name]
    return hedge_ratio


def zscore(
    series_y: pd.Series, series_x: pd.Series, hedge_ratio: float
) -> pd.Series:
    series = series_y - hedge_ratio * series_x
    return (series - series.mean()) / np.std(series)


def plot_assets(
    date: np.ndarray,
    asset_x: np.ndarray,
    asset_x_label: str,
    asset_y: np.ndarray,
    asset_y_label: str,
    long_idxs: np.ndarray,
    short_idxs: np.ndarray,
    close_idxs: np.ndarray,
    net_value: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5),
    net_value_limit: Tuple[float, float] = (0.9, 1.1),
    spread: Optional[np.ndarray] = None,
    trading_threshold: Optional[float] = None,
    stop_loss_threshold: Optional[float] = None,
) -> Figure:
    figure = plt.figure(figsize=figsize)
    if spread is not None:
        ax1: Axes = figure.add_subplot(2, 1, 1)
    else:
        ax1: Axes = figure.add_subplot(1, 1, 1)

    asset_x = np.exp(asset_x)
    asset_y = np.exp(asset_y)

    fontsize = 10
    scattersize = 50
    ax1.plot(date, asset_x, color="darkslategrey", label=asset_x_label)
    if long_idxs.size != 0:
        ax1.scatter(
            date[long_idxs],
            asset_x[long_idxs],
            color="red",
            marker="^",
            s=scattersize,
            label="long",
        )
        for a, b in zip(date[long_idxs], asset_x[long_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if short_idxs.size != 0:
        ax1.scatter(
            date[short_idxs],
            asset_x[short_idxs],
            color="green",
            marker="v",
            s=scattersize,
            label="short",
        )
        for a, b in zip(date[short_idxs], asset_x[short_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if close_idxs.size != 0:
        ax1.scatter(
            date[close_idxs],
            asset_x[close_idxs],
            color="navy",
            marker="x",
            s=scattersize,
            label="close",
        )
        for a, b in zip(date[close_idxs], asset_x[close_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    ax1.plot(date, asset_y, color="blue", label=asset_y_label)
    if long_idxs.size != 0:
        ax1.scatter(
            date[long_idxs],
            asset_y[long_idxs],
            color="green",
            marker="v",
            s=scattersize,
        )
        for a, b in zip(date[long_idxs], asset_y[long_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if short_idxs.size != 0:
        ax1.scatter(
            date[short_idxs],
            asset_y[short_idxs],
            color="red",
            marker="^",
            s=scattersize,
        )
        for a, b in zip(date[short_idxs], asset_y[short_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if close_idxs.size != 0:
        ax1.scatter(
            date[close_idxs],
            asset_y[close_idxs],
            color="navy",
            marker="x",
            s=scattersize,
        )
        for a, b in zip(date[close_idxs], asset_y[close_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if net_value is not None:
        ax2 = ax1.twinx()
        ax2.axhline(1, color="red", linestyle="--")
        ax2.plot(date, net_value, color="red", label="net value")
        if long_idxs.size != 0:
            ax2.scatter(
                date[long_idxs],
                net_value[long_idxs],
                color="red",
                marker="^",
                s=scattersize,
            )
            for a, b in zip(date[long_idxs], net_value[long_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )
        if short_idxs.size != 0:
            ax2.scatter(
                date[short_idxs],
                net_value[short_idxs],
                color="green",
                marker="v",
                s=scattersize,
            )
            for a, b in zip(date[short_idxs], net_value[short_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )
        if close_idxs.size != 0:
            ax2.scatter(
                date[close_idxs],
                net_value[close_idxs],
                color="navy",
                marker="x",
                s=scattersize,
            )
            for a, b in zip(date[close_idxs], net_value[close_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )

    ax1.legend(loc="upper left")
    ax1.set_ylabel("price")
    if net_value is not None:
        ax2.legend(loc="upper right")
        ax2.set_ylabel("net value")
        ax2.set_ylim(bottom=net_value_limit[0], top=net_value_limit[1])

    x_major_locator = MultipleLocator(3)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_xlim(
        left=date[0] - np.timedelta64(1, "D"),
        right=date[-1] + np.timedelta64(1, "D"),
    )
    for xtick in ax1.get_xticklabels():
        xtick.set_rotation(-90)

    if spread is not None:
        ax3 = figure.add_subplot(2, 1, 2)
        ax3.plot(date, spread, color="black", label="spread")
        ax3.axhline(trading_threshold, color="darkviolet", linestyle="--")
        ax3.axhline(-trading_threshold, color="darkviolet", linestyle="--")
        ax3.axhline(stop_loss_threshold, color="darkred", linestyle="--")
        ax3.axhline(-stop_loss_threshold, color="darkred", linestyle="--")
        ax3.axhline(0, color="black", linestyle="--")
        if long_idxs.size != 0:
            ax3.scatter(
                date[long_idxs], spread[long_idxs], color="red", marker="^"
            )
            for a, b in zip(date[long_idxs], spread[long_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if short_idxs.size != 0:
            ax3.scatter(
                date[short_idxs], spread[short_idxs], color="green", marker="v"
            )
            for a, b in zip(date[short_idxs], spread[short_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if close_idxs.size != 0:
            ax3.scatter(
                date[close_idxs], spread[close_idxs], color="navy", marker="x"
            )
            for a, b in zip(date[close_idxs], spread[close_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if net_value is not None:
            ax4 = ax3.twinx()
            ax4.plot(date, net_value, color="red", label="net value")
            if long_idxs.size != 0:
                ax4.scatter(
                    date[long_idxs],
                    net_value[long_idxs],
                    color="red",
                    marker="^",
                )
                for a, b in zip(date[long_idxs], net_value[long_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )
            if short_idxs.size != 0:
                ax4.scatter(
                    date[short_idxs],
                    net_value[short_idxs],
                    color="green",
                    marker="v",
                )
                for a, b in zip(date[short_idxs], net_value[short_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )
            if close_idxs.size != 0:
                ax4.scatter(
                    date[close_idxs],
                    net_value[close_idxs],
                    color="navy",
                    marker="x",
                )
                for a, b in zip(date[close_idxs], net_value[close_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )

        ax3.legend(loc="upper left")
        ax3.set_ylabel("spread")
        if net_value is not None:
            ax4.legend(loc="upper right")
            ax4.set_ylabel("net value")
            ax4.set_ylim(bottom=net_value_limit[0], top=net_value_limit[1])

        x_major_locator = MultipleLocator(3)
        ax3.xaxis.set_major_locator(x_major_locator)
        ax3.set_xlim(
            left=date[0] - np.timedelta64(1, "D"),
            right=date[-1] + np.timedelta64(1, "D"),
        )
        ax3.set_ylim(bottom=-3, top=3)
        for xtick in ax3.get_xticklabels():
            xtick.set_rotation(-90)

    figure.tight_layout()

    return figure


def getFirstAndLastDay(
    date: datetime.datetime,
) -> Tuple[datetime.datetime, datetime.datetime]:
    year, month = date.year, date.month
    # 获取当前月的第一天的星期和当月总天数
    weekDay, monthCountDay = calendar.monthrange(year, month)
    # 获取当前月份第一天
    firstDay = datetime.datetime(year, month, day=1)
    # 获取当前月份最后一天
    lastDay = datetime.datetime(year, month, day=monthCountDay)
    # 返回第一天和最后一天
    return firstDay, lastDay


def get_sorted_dirs(root_dir: str, re_dir: Pattern[str]) -> List[str]:
    target_dirs: List[Tuple[int, str]] = []
    for root, dirs, files in os.walk(root_dir):
        if root == root_dir:
            for _dir in dirs:
                re_result = re_dir.match(_dir)
                if re_result:
                    target_dirs.append((int(re_result.group(1)), _dir))
            break

    target_dirs = sorted(target_dirs, key=lambda x: x[0])
    return [_dir[1] for _dir in target_dirs]


def get_done_idxs(
    root_dir: str, re_dir: Pattern[str], unix_path_pattern: str
) -> List[int]:
    """
    get_dir_idxs(
        "saved_model",
        re.compile,
        "dqn_baseline_run_1_*/test_monitor/test_trajectory.csv"
    )
    :return:
    """
    target_dirs: List[int] = []
    for path in glob.glob(f"{root_dir}/{unix_path_pattern}"):
        re_result = re_dir.search(path)
        if re_result:
            target_dirs.append(int(re_result.group(1)))
    target_dirs = sorted(target_dirs)
    return target_dirs


def get_metrics(
    actions: List[int],
    net_values: List[float],
    risk_free: float = 0.000085,  # 每日无风险收益率
):
    assert len(actions) == len(net_values)
    last_buy_idx: Optional[int] = None
    last_buy_net_value: Optional[float] = None
    curr_state = PositionState.bear
    # 每笔交易的持有时间
    hold_times: List[float] = []
    # 每笔交易的收益率
    trade_returns: List[float] = []
    states: List[int] = []
    for idx, (action, net_value) in enumerate(zip(actions, net_values)):
        if action == -1:
            continue
        elif action == Action.short and (
            curr_state == PositionState.bear
            or curr_state == PositionState.long
        ):
            curr_state = PositionState.short
            if last_buy_idx is not None:
                hold_times.append(idx - last_buy_idx)
                trade_returns.append(net_value / last_buy_net_value - 1)
            last_buy_idx = idx
            last_buy_net_value = net_value
        elif action == Action.long and (
            curr_state == PositionState.bear
            or curr_state == PositionState.short
        ):
            curr_state = PositionState.long
            if last_buy_idx is not None:
                hold_times.append(idx - last_buy_idx)
                trade_returns.append(net_value / last_buy_net_value - 1)
            last_buy_idx = idx
            last_buy_net_value = net_value
        elif action == Action.close and (
            curr_state == PositionState.long
            or curr_state == PositionState.short
        ):
            assert last_buy_idx is not None
            curr_state = PositionState.bear
            hold_times.append(idx - last_buy_idx)
            trade_returns.append(net_value / last_buy_net_value - 1)
            last_buy_idx = None
            last_buy_net_value = None
        states.append(curr_state)

    # 平均空仓时间
    # 因为最后一天强制清仓，所以不考虑最终的状态
    states = states[:-1]
    bear_times = []
    i, j = 0, 0
    while i < len(states):
        j = i
        while i < len(states) and states[i] == PositionState.bear:
            i += 1
        if j != i:
            bear_times.append(i - j)
        while i < len(states) and states[i] != PositionState.bear:
            i += 1
    total_trade_times = len(hold_times)
    mean_hold_time = (
        float(np.mean(hold_times)) if total_trade_times != 0 else 0
    )
    total_bear_times = len(bear_times)
    mean_bear_time = float(np.mean(bear_times)) if total_bear_times != 0 else 0
    mean_return = (
        float(np.mean(trade_returns)) if len(trade_returns) != 0 else 0
    )

    noncumulative_returns = (
        np.array(net_values)[1:] / np.array(net_values)[:-1] - 1
    )
    sharpe_ratio = float(
        empyrical.sharpe_ratio(noncumulative_returns, risk_free)
    )
    if math.isinf(sharpe_ratio):
        sharpe_ratio = 0
    annual_return = float(empyrical.annual_return(noncumulative_returns))
    annual_volatility = float(
        empyrical.annual_volatility(noncumulative_returns)
    )
    max_drawdown = float(empyrical.max_drawdown(noncumulative_returns))

    return {
        "mean_hold_time": mean_hold_time,
        "total_trade_times": total_trade_times,
        "mean_bear_time": mean_bear_time,
        "total_bear_times": total_bear_times,
        "mean_return": mean_return,
        "sharpe_ratio": sharpe_ratio,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
    }


def get_valid_action_indexes(
    actions: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    # 过滤掉无用的操作
    long_idxs = []
    short_idxs = []
    close_idxs = []
    state = PositionState.bear
    for idx, action in enumerate(actions):
        if action == -1:
            continue
        elif action == Action.short and (
            state == PositionState.bear or state == PositionState.long
        ):
            state = PositionState.short
            short_idxs.append(idx)
        elif action == Action.long and (
            state == PositionState.bear or state == PositionState.short
        ):
            state = PositionState.long
            long_idxs.append(idx)
        elif action == Action.close and (
            state == PositionState.long or state == PositionState.short
        ):
            state = PositionState.bear
            close_idxs.append(idx)
    return long_idxs, short_idxs, close_idxs


def get_curr_net_value(
    asset_x_price: List[float],
    asset_y_price: List[float],
    fund_ratio: float,
    position: int,
    last_net_value: float,
    last_buy_idx: int,
    curr_idx: int,
) -> float:
    """
    :return: 计算净值
    """
    if position == PositionState.long:
        return last_net_value * (
                math.exp(asset_x_price[curr_idx]) / math.exp(asset_x_price[last_buy_idx]) +
                fund_ratio * (
                        2 - math.exp(asset_y_price[curr_idx]) / math.exp(asset_y_price[last_buy_idx])
                )
        ) / (1 + fund_ratio)
    elif position == PositionState.short:
        return last_net_value * (
                (
                        2 - math.exp(asset_x_price[curr_idx]) / math.exp(asset_x_price[last_buy_idx])
                ) +
                fund_ratio * (
                        math.exp(asset_y_price[curr_idx]) / math.exp(asset_y_price[last_buy_idx])
                )
        ) / (1 + fund_ratio)
    else:
        return last_net_value

def get_idle_gpu_idx(
    request_mem: float, exclude_gpus: Optional[List[int]] = None
):
    device_list = []
    if exclude_gpus is None:
        exclude_gpus = []
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        if i in exclude_gpus:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        device_list.extend([i] * int((meminfo.free / 2**30 / request_mem)))
    pynvml.nvmlShutdown()
    return device_list


def display_top(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def get_curr_process_memory():
    """
    返回当前进程内存占用
    :return: 单位GB
    """
    return psutil.Process(os.getpid()).memory_info().rss / 2**30
