import math
from typing import List, Optional, Tuple

import empyrical
import gym
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from loguru import logger
from stable_baselines3 import A2C
from stable_baselines3.a2c import MultiInputPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import wandb
from trials.networks.callbacks import TradingEvalCallback
from trials.networks.constant import Action, PositionState
from trials.networks.feature_extractor import TRADING_FEATURE_EXTRACTORS
from trials.scripts.eval_cointegration import CointegrationStateMachine
from trials.scripts.select_pairs import dist
from trials.scripts.util import (
    get_curr_net_value,
    get_hedge_ratio,
    get_valid_action_indexes,
    plot_assets,
    zscore,
)

torch.set_num_threads(20)


class TradingEnv(gym.Env):
    def __init__(
        self,
        name: str,
        form_date: List[str],
        trad_date: List[str],
        asset_name: List[str],
        form_asset_features: np.array,
        trad_asset_features: np.array,
        form_asset_log_prices: np.array,
        trad_asset_log_prices: np.array,
        feature_dim: int,
        **kwargs,
    ):
        """
        :param form_date: 形成期的时间
        :param trad_date: 交易期的时间
        :param asset_name: 一个rolling内30支股票的symbol
        :param form_asset_features: N x T x M 形成期特征
        :param trad_asset_features: N x T x M 交易期特征
        :param form_asset_log_prices: N x 1 形成期初价格
        :param trad_asset_log_prices: N x 1 形成期初价格
        :param feature_dim: 输入每个资产特征维度
        :param kwargs: 其他输入关键字, 包括
        :param init_net_value: 初始的净值ues时默认的值
        :param parallel_selection: 是否为MultiCategoricalAction
        :param trading_threshold: CointegrationStateMachine生成net_values时默认的值
        :param stop_loss_threshold: CointegrationStateMachine生成net_val
        """
        super(TradingEnv, self).__init__()
        self.name = name
        self.asset_num = len(asset_name)
        self.serial_selection = kwargs.get("serial_selection", True)
        self.asset_attention = kwargs.get("asset_attention", True)
        self.action_space = (
            gym.spaces.MultiDiscrete([self.asset_num, self.asset_num])
            if self.serial_selection
            else gym.spaces.Discrete(
                self.asset_num * (self.asset_num - 1) // 2
            )
        )
        self.form_len = len(form_date)
        self.trad_len = len(trad_date)
        self.feature_dim = feature_dim
        self.form_flatten_len = self.form_len * self.asset_num * feature_dim
        self.trad_flatten_len = self.trad_len * self.asset_num * feature_dim
        self.observation_space = gym.spaces.Dict(
            {
                "assets": gym.spaces.Box(
                    -5, 5, shape=(self.form_flatten_len,), dtype=np.float32
                ),
            }
        )
        self.form_date = form_date
        self.trad_date = trad_date
        self.asset_name = asset_name
        self.form_asset_features = form_asset_features
        self.trad_asset_features = trad_asset_features
        self.form_asset_log_prices = form_asset_log_prices
        self.trad_asset_log_prices = trad_asset_log_prices
        self.trading_threshold = kwargs.get("trading_threshold", 1.0)
        self.stop_loss_threshold = kwargs.get("stop_loss_threshold", 1.0)
        self.init_net_value = kwargs.get("init_net_value", 1.0)
        self.euclidean_dist = float("inf")
        self.metric = float("-inf")

        self.observation = {
            "assets": form_asset_features.flatten(),
        }

        if not self.serial_selection:
            self.index_map = {}
            for first_asset_index in range(self.asset_num - 1):
                for second_asset_index in range(
                    first_asset_index + 1, self.asset_num
                ):
                    self.index_map[len(self.index_map)] = (
                        first_asset_index,
                        second_asset_index,
                    )
        logger.info(
            f"Initialize env: form_len: {self.form_len} "
            f"asset_num: {self.asset_num} feature_dim: {self.feature_dim}"
        )

    def render(self, mode="human"):
        pass

    def reset(self):
        self.euclidean_dist = float("inf")
        return self.observation

    def get_map_action(self, action):
        """Map action according to selection mode

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        if self.serial_selection:
            return action
        return self.index_map[action]

    def step(self, action):
        """Reward according to action

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        x_index, y_index = self.get_map_action(action)

        x_name = self.asset_name[x_index]
        y_name = self.asset_name[y_index]
        form_x_price = self.form_asset_log_prices[x_index, :]
        form_y_price = self.form_asset_log_prices[y_index, :]
        trad_x_price = self.trad_asset_log_prices[x_index, :]
        trad_y_price = self.trad_asset_log_prices[y_index, :]
        hedge_ratio = get_hedge_ratio(
            pd.Series(form_y_price), pd.Series(form_x_price).rename("x")
        )
        spread_series = list(
            zscore(
                pd.Series(trad_y_price),
                pd.Series(trad_x_price),
                hedge_ratio,
            )
        )
        trading_pairs = [x_name, y_name]
        net_env = CointegrationStateMachine(
            self.trad_date,
            trading_pairs,
            [
                trad_x_price,
                trad_y_price,
            ],
            spread_series,
            trading_threshold=self.trading_threshold,
            stop_loss_threshold=self.stop_loss_threshold,
        )
        zscore_list = []
        net_values = []
        action_list = []
        obs = net_env.reset()

        zscore_list.append(obs["spread"])
        while True:
            act, obs, trade_done = net_env.step()
            net_values.append(obs["net_value"])
            action_list.append(int(act))
            if trade_done:
                break
            else:
                zscore_list.append(obs["spread"])
        returns = [0]
        for i in range(1, len(net_values)):
            daily_return = (net_values[i] - net_values[i - 1]) / net_values[
                i - 1
            ]
            returns.append(daily_return)
        if sum(returns) == 0:
            sharpe = -2.0
            reward = -1.0
            annual_return = -2.0
            annual_volatility = -2.0
            max_drawdown = -2.0
        else:
            returns = pd.Series(returns)
            sharpe = empyrical.sharpe_ratio(
                returns, risk_free=0.000085, period="daily", annualization=252
            )
            annual_return = empyrical.annual_return(
                returns, period="daily", annualization=252
            )
            annual_volatility = empyrical.annual_volatility(
                returns, period="daily", alpha=2.0, annualization=252
            )
            max_drawdown = empyrical.max_drawdown(returns)
        reward = np.log(net_values[-1])
        self.euclidean_dist = dist(trad_x_price, trad_y_price)
        if x_index == y_index:
            return (
                self.observation,
                -20,
                True,
                {
                    "sharpe_ratio": -20,
                    "euclidean_dist": 0,
                    "annual_return": -20,
                    "annual_volatility": 20,
                    "max_drawdown": -20,
                    "returns": net_values,
                    "actions": action_list,
                },
            )
        info = {
            "sharpe_ratio": sharpe,
            "euclidean_dist": self.euclidean_dist,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "returns": net_values,
            "actions": action_list,
        }
        return self.observation, reward, True, info

    def plot_trajectory(
        self,
        trading_dates: List[str],
        asset_names: List[str],
        x_prices: List[float],
        y_prices: List[float],
        action_list: List[int],
        net_value_list: List[float],
        figsize: Tuple[int, int] = (15, 5),
        net_value_limit: Tuple[float, float] = (0.9, 1.1),
    ):
        assert len(net_value_list) == len(action_list)
        long_idxs, short_idxs, close_idxs = get_valid_action_indexes(
            action_list
        )
        net_value_limit = (
            min(net_value_list) - 0.1,
            max(net_value_list) + 0.1,
        )

        start_index = getattr(self, "window_size")
        logger.info(f"start index is {start_index}")
        start_index = start_index or 1
        start_index = start_index - 1
        figure = plot_assets(
            date=np.array(trading_dates[start_index:], dtype="datetime64"),
            asset_x=np.array(x_prices[start_index:]),
            asset_x_label=asset_names[0],
            asset_y=np.array(y_prices[start_index:]),
            asset_y_label=asset_names[1],
            net_value=np.array(net_value_list),
            long_idxs=np.array(long_idxs),
            short_idxs=np.array(short_idxs),
            close_idxs=np.array(close_idxs),
            figsize=figsize,
            net_value_limit=net_value_limit,
        )

        return figure

    def eval(self, model, step):
        obs = self.reset()
        tensor_obs = obs_as_tensor(obs, model.device)
        probs = model.policy.get_distribution(tensor_obs)
        tensor_obs = preprocess_obs(
            tensor_obs,
            model.policy.observation_space,
            normalize_images=model.policy.normalize_images,
        )
        outputs = model.policy.features_extractor(
            tensor_obs, attention_output=True
        )
        pair_probs = np.zeros([self.asset_num, self.asset_num])
        if isinstance(probs.distribution, list):
            prob_distributions = [dis.probs for dis in probs.distribution]
            for first_asset in range(self.asset_num):
                for second_asset in range(first_asset + 1, self.asset_num):
                    pair_probs[first_asset][second_asset] = (
                        prob_distributions[0][0][first_asset]
                        + prob_distributions[1][0][second_asset]
                    )
                    pair_probs[second_asset][first_asset] = (
                        prob_distributions[1][0][first_asset]
                        + prob_distributions[0][0][second_asset]
                    )
        else:
            prob_dist = probs.distribution.probs[0]
            for first_asset in range(self.asset_num):
                for second_asset in range(first_asset + 1, self.asset_num):
                    pair_index = (
                        first_asset * self.asset_num
                        + second_asset
                        - (first_asset + 2) * (first_asset + 1) // 2
                    )
                    pair_probs[first_asset][second_asset] = prob_dist[
                        pair_index
                    ]
                    pair_probs[second_asset][first_asset] = prob_dist[
                        pair_index
                    ]

        action, _ = model.predict(obs, deterministic=True)
        self.is_eval = True
        obs, reward, done, info = self.step(action)
        self.is_eval = False
        x_index, y_index = self.get_map_action(action)
        figure = self.plot_trajectory(
            self.trad_date,
            [
                self.asset_name[asset_index]
                for asset_index in [x_index, y_index]
            ],
            self.trad_asset_log_prices[x_index, :],
            self.trad_asset_log_prices[y_index, :],
            info["actions"],
            info["returns"],
        )
        wandb_dict = {
            f"{self.name}/final_reward": reward,
        }
        wandb_dict[f"{self.name}/final_sharpe_ratio"] = info["sharpe_ratio"]
        wandb_dict[f"{self.name}/final_annual_return"] = info["annual_return"]
        wandb_dict[f"{self.name}/final_annual_volatility"] = info[
            "annual_volatility"
        ]
        wandb_dict[f"{self.name}/final_max_drawdown"] = info["max_drawdown"]
        asset_df = pd.DataFrame(
            outputs[0][0].reshape(self.asset_num, -1).cpu().detach().numpy()
        )
        asset_df.columns = [str(column) for column in asset_df.columns]
        asset_df["asset_name"] = self.asset_name
        wandb_dict[f"{self.name}/asset_representations"] = wandb.Table(
            dataframe=asset_df
        )
        wandb_dict[f"{self.name}/pair_probabilities"] = px.imshow(pair_probs)
        if outputs[1] is not None:
            wandb_dict[f"{self.name}/temporal_attention"] = px.imshow(
                outputs[1][:, 0].cpu().detach()
            )
        if self.asset_attention:
            wandb_dict[f"{self.name}/asset_attention"] = px.imshow(
                outputs[2][0].cpu().detach()
            )
        wandb_dict[f"{self.name}/trajectory_figure"] = wandb.Image(figure)
        wandb.log(
            wandb_dict,
            step=step
        )


class ReinforceTradingEnv(TradingEnv):
    def __init__(
        self,
        name: str,
        form_date: List[str],
        trad_date: List[str],
        asset_name: List[str],
        form_asset_features: np.array,
        trad_asset_features: np.array,
        form_asset_log_prices: np.array,
        trad_asset_log_prices: np.array,
        feature_dim: int,
        **kwargs,
    ):
        """
        :param form_date: 形成期的时间
        :param trad_date: 交易期的时间
        irint (action)
        :param asset_name: 一个rolling内30支股票的symbol
        :param form_asset_features: N x T x M 形成期特征
        :param trad_asset_features: N x T x M 交易期特征
        :param form_asset_log_prices: N x 1 形成期初价格
        :param trad_asset_log_prices: N x 1 形成期初价格
        :param feature_dim: 输入每个资产特征维度
        :param kwargs: 其他输入关键字, 包括
        :param init_net_value: 初始的净值ues时默认的值
        :param parallel_selection: 是否为MultiCategoricalAction
        :param trading_threshold: CointegrationStateMachine生成net_values时默认的值
        :param stop_loss_threshold: CointegrationStateMachine生成net_val
        """
        super(TradingEnv, self).__init__()
        self.name = name
        self.is_eval = False
        self.asset_num = len(asset_name)
        self.serial_selection = kwargs.get("serial_selection", True)
        self.asset_attention = kwargs.get("asset_attention", True)
        self.action_space = (
            gym.spaces.MultiDiscrete([self.asset_num, self.asset_num])
            if self.serial_selection
            else gym.spaces.Discrete(
                self.asset_num * (self.asset_num - 1) // 2
            )
        )
        self.form_len = len(form_date)
        self.trad_len = len(trad_date)
        self.feature_dim = feature_dim
        self.form_flatten_len = self.form_len * self.asset_num * feature_dim
        self.trad_flatten_len = self.trad_len * self.asset_num * feature_dim
        self.observation_space = gym.spaces.Dict(
            {
                "assets": gym.spaces.Box(
                    -5, 5, shape=(self.form_flatten_len,), dtype=np.float32
                ),
            }
        )
        self.form_date = form_date
        self.trad_date = trad_date
        self.asset_name = asset_name
        self.form_asset_features = form_asset_features
        self.trad_asset_features = trad_asset_features
        self.form_asset_log_prices = form_asset_log_prices
        self.trad_asset_log_prices = trad_asset_log_prices
        self.init_net_value = kwargs.get("init_net_value", 1.0)
        self.euclidean_dist = float("inf")
        self.metric = float("-inf")
        self.train_step = kwargs.get("trading_train_steps", 5)
        self.window_size = kwargs.get("window_size", 20)

        self.observation = {
            "assets": form_asset_features.flatten(),
        }
        self.serial_selection = kwargs.get("serial_selection", True)
        self.asset_attention = kwargs.get("asset_attention", True)

        if not self.serial_selection:
            self.index_map = {}
            for first_asset_index in range(self.asset_num - 1):
                for second_asset_index in range(
                    first_asset_index + 1, self.asset_num
                ):
                    self.index_map[len(self.index_map)] = (
                        first_asset_index,
                        second_asset_index,
                    )

        logger.info(f"Start trading training")
        trad_date = list(self.form_date[-self.window_size :])
        trad_date.extend(self.trad_date)
        self.trad_date = trad_date

        self.trad_asset_log_prices = np.concatenate(
            [
                self.form_asset_log_prices[:, -self.window_size :],
                self.trad_asset_log_prices,
            ],
            axis=1,
        )
        self.trad_asset_log_prices = np.exp(self.trad_asset_log_prices)
        self.trad_asset_log_prices = (
            self.trad_asset_log_prices / self.trad_asset_log_prices[:, :1]
        )
        self.trad_asset_log_prices = np.log(self.trad_asset_log_prices)

        self.form_asset_log_prices = np.exp(self.form_asset_log_prices)
        self.form_asset_log_prices = (
            self.form_asset_log_prices / self.form_asset_log_prices[:, :1]
        )
        self.form_asset_log_prices = np.log(self.form_asset_log_prices)

        def initialize_env(name, date, names, prices):
            if name == "train":
                return StepTradingEnv(
                    name=name,
                    date=date,
                    asset_name=names,
                    log_prices=prices,
                    window_size=self.window_size,
                    max_len=self.form_len,
                )
            else:
                return DummyVecEnv(
                    [
                        lambda: StepTradingEnv(
                            name=name,
                            date=date,
                            asset_name=names,
                            log_prices=prices,
                            window_size=self.window_size,
                            max_len=self.form_len,
                        )
                    ]
                )

        if kwargs["trading_num_process"] > 1:

            def make_env(
                date,
                asset_name,
                prices,
            ):
                """
                Utility function for multiprocessed env.
                :return: (Callable)
                """

                def _init() -> gym.Env:
                    pr = np.exp(prices)
                    pr = pr / pr[:, :1]
                    pr = np.log(pr)
                    env = initialize_env(
                        "train",
                        date,
                        asset_name,
                        pr,
                    )
                    return env

                return _init

            self.train_env = SubprocVecEnv(
                [
                    make_env(
                        self.form_date,
                        asset_name,
                        self.form_asset_log_prices,
                    )
                    for _ in range(kwargs["trading_num_process"])
                ]
            )
        else:
            self.train_env = initialize_env(
            "train", self.form_date, asset_name, self.form_asset_log_prices
        )
        
        self.test_env = initialize_env(
            "test", self.trad_date, asset_name, self.trad_asset_log_prices
        )

        policy_kwargs = {
            "features_extractor_class": TRADING_FEATURE_EXTRACTORS[
                kwargs["trading_feature_extractor"]
            ],
            "features_extractor_kwargs": {
                "feature_dim": kwargs["trading_feature_extractor_feature_dim"],
                "num_layers": kwargs["trading_feature_extractor_num_layers"],
                "drouput": kwargs["trading_dropout"],
            },
        }

        if kwargs.get("worker_model") is not None:
            self.worker_model = kwargs.get("worker_model")
        else:
            logger.info("Initialize new worker model")
            self.worker_model = A2C(
                MultiInputPolicy,
                self.train_env,
                n_steps=1, # batch_size: trading_num_process
                learning_rate=kwargs["trading_learning_rate"],
                tensorboard_log=kwargs["trading_log_dir"],
                seed=kwargs["seed"],
                gamma=kwargs["trading_rl_gamma"],
                ent_coef=kwargs["trading_ent_coef"],
                policy_kwargs=policy_kwargs,
                verbose=0,
            )

        # Callback to obtain reward of manager from worker
        self.trad_callback = TradingEvalCallback(
            self.name,
            self.test_env
        )

        logger.info(
            f"Initialize env: form_len: {self.form_len} "
            f"asset_num: {self.asset_num} feature_dim: {self.feature_dim}"
        )

    def render(self, mode="human"):
        pass

    def reset(self):
        self.euclidean_dist = float("inf")
        return self.observation

    def get_map_action(self, action):
        """Map action according to selection mode

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        if self.serial_selection:
            return action
        if isinstance(action, np.ndarray):
            action = action.item()
        return self.index_map[action]

    def set_trading_indexes(self, x_index, y_index):
        """Set trading indexes for all envs

        :param x_index: (int) the selected first asset index
        :param y_index: (int) the selected second asset index
        """
        trading_indexes = [x_index, y_index]
        self.train_env.set_attr("trading_indexes", trading_indexes)
        self.test_env.set_attr("trading_indexes", trading_indexes)

    def step(self, action):
        """Reward according to action

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        x_index, y_index = self.get_map_action(action)
        if x_index == y_index:
            return (
                self.observation,
                -20,
                True,
                {
                    "sharpe_ratio": -20,
                    "euclidean_dist": 0,
                    "annual_return": -20,
                    "annual_volatility": 20,
                    "max_drawdown": -20,
                    "returns": [],
                    "actions": [],
                },
            )
        self.set_trading_indexes(x_index, y_index)

        if self.train_step > 0 and not self.is_eval:
            self.worker_model.learn(
                total_timesteps=self.train_step,
                reset_num_timesteps=False,
            )
        self.trad_callback.model = self.worker_model
        self.trad_callback.on_step()
        info = self.trad_callback.best_metric
        reward = np.log(info["returns"][-1])
        net_values = info["returns"]
        returns = [0]
        for i in range(1, len(net_values)):
            daily_return = (net_values[i] - net_values[i - 1]) / net_values[
                i - 1
            ]
            returns.append(daily_return)
        if sum(returns) == 0:
            annual_return = -2.0
            annual_volatility = -2.0
            max_drawdown = -2.0
        else:
            returns = pd.Series(returns)
            annual_return = empyrical.annual_return(
                returns, period="daily", annualization=252
            )
            annual_volatility = empyrical.annual_volatility(
                returns, period="daily", alpha=2.0, annualization=252
            )
            max_drawdown = empyrical.max_drawdown(returns)
        info["annual_return"] = annual_return
        info["annual_volatility"] = annual_volatility
        info["max_drawdown"] = max_drawdown

        if reward == 0:
            reward = -2.0
        if self.name == "test":
            logger.info(
                f"{self.name} | Present action: {action}, sharpe: {info['sharpe_ratio']}"
            )

        return self.observation, reward, True, info


class StepTradingEnv(gym.Env):
    def __init__(
        self,
        name: str,
        date: List[str],
        asset_name: List[str],
        log_prices: np.array,
        commission_rate: float = 0.001,
        fund_ratio: float = 1.0,
        init_net_value: float = 1.0,
        risk_free: float = 0.000085,
        successive_close_reward: float = 0,
        window_size: int = 1,
        max_len: int = 252,
    ):
        super(StepTradingEnv, self).__init__()
        self.name = name
        self.asset_name = asset_name
        self.log_prices = log_prices
        self.action_space = gym.spaces.Discrete(3)
        self.trading_indexes = [0, 0]
        self.window_size = window_size
        assert window_size + 1 <= max_len - 1
        observation_dict = {
            "asset_x": gym.spaces.Box(
                -5, 5, shape=(max_len,), dtype=np.float32
            ),
            "asset_y": gym.spaces.Box(
                -5, 5, shape=(max_len,), dtype=np.float32
            ),
            "net_value": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.float32
            ),
            "unrealized_net_value": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.float32
            ),
            "sharpe_ratio": gym.spaces.Box(
                -40, 40, shape=(max_len,), dtype=np.float32
            ),
            "position": gym.spaces.Box(0, 2, shape=(max_len,), dtype=np.int),
            "next_end": gym.spaces.Box(
                0, 1, shape=(max_len,), dtype=np.int
            ),  # 下一个step是否会done
            "hold_threshold": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.int
            ),
            "hold_indicator": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.int
            ),
            "action": gym.spaces.Box(-1, 2, shape=(max_len,), dtype=np.int),
            "mask_len": gym.spaces.Box(0, max_len, shape=(1,), dtype=np.int),
        }
        self.observation_space = gym.spaces.Dict(observation_dict)

        # 画图用
        self.date = date

        # Env设置
        self.commission_rate = commission_rate
        self.fund_ratio = fund_ratio
        self.risk_free = risk_free
        self.init_net_value = init_net_value
        self.max_len = min(max_len, len(self.log_prices[0, :]) + 1)
        self.successive_close_reward = successive_close_reward
        assert len(self.log_prices[0, :]) >= window_size + 1

        # state
        self.position: Optional[int] = PositionState.bear
        self.start_idx: Optional[int] = None
        self.curr_idx: Optional[int] = None
        self.last_buy_idx: Optional[int] = None
        self.last_net_value: Optional[float] = None
        # 实际的资金
        self.net_value: np.ndarray = np.array(
            [init_net_value] * max_len, dtype=float
        )
        # 按照当前股票价格计算的净值
        self.unrealized_net_value: np.ndarray = np.array(
            [init_net_value] * max_len, dtype=float
        )
        self.action_list = []

        self.observation = {
            "asset_x": np.array([0] * max_len, dtype=float),
            "asset_y": np.array([0] * max_len, dtype=float),
            "net_value": np.array([init_net_value] * max_len, dtype=float),
            "unrealized_net_value": np.array(
                [init_net_value] * max_len, dtype=float
            ),
            "sharpe_ratio": np.array([0] * max_len, dtype=float),
            "position": np.array(
                [int(PositionState.bear)] * max_len, dtype=int
            ),
            "next_end": np.array([0] * max_len, dtype=int),
            "hold_threshold": np.array([0] * max_len, dtype=int),
            "hold_indicator": np.array([0] * max_len, dtype=int),
            "action": np.array([-1] * max_len, dtype=int),
            "mask_len": np.array([1], dtype=int),
        }

    def render(self, mode="human"):
        pass

    def reset(self):
        self.pair_name = [
            self.asset_name[self.trading_indexes[0]],
            self.asset_name[self.trading_indexes[1]],
        ]
        self.asset_price = [
            self.log_prices[self.trading_indexes[0], :].tolist(),
            self.log_prices[self.trading_indexes[1], :].tolist(),
        ]
        self.curr_idx = self.window_size - 1
        self.start_idx = self.curr_idx
        self.position = PositionState.bear
        self.net_value.fill(self.init_net_value)
        self.unrealized_net_value.fill(self.init_net_value)
        self.action_list.clear()

        self.observation["asset_x"].fill(0)
        self.observation["asset_y"].fill(0)
        self.observation["net_value"].fill(self.init_net_value)
        self.observation["unrealized_net_value"].fill(self.init_net_value)
        self.observation["sharpe_ratio"].fill(0)
        self.observation["position"].fill(int(PositionState.bear))
        self.observation["next_end"].fill(0)
        self.observation["mask_len"].fill(1)
        self.observation["hold_threshold"].fill(0)
        self.observation["hold_indicator"].fill(0)
        self.observation["action"].fill(-1)

        # 填充window
        for i, j in enumerate(
            range(self.curr_idx + 1 - self.window_size, self.curr_idx + 1)
        ):
            self.observation["asset_x"][i] = self.asset_price[0][j]
            self.observation["asset_y"][i] = self.asset_price[1][j]
        self.observation["mask_len"][0] = self.window_size
        self.last_buy_idx = None
        self.last_net_value = self.init_net_value
        return self.observation

    def get_net_value_change(self):
        """
        :return: 返回净值变化量
        """
        return (
            get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx,
            )
            * (1 - self.commission_rate)
            - self.last_net_value
        )

    def step(self, action):
        # take action_0
        reward = 0
        if (
            self.curr_idx - self.start_idx + self.window_size
            == self.max_len - 1
        ):
            action = Action.close

        if action == Action.short:
            if self.position == PositionState.bear:
                self.last_buy_idx = self.curr_idx
            elif self.position == PositionState.long:
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
                self.last_buy_idx = self.curr_idx
            self.position = PositionState.short
        elif action == Action.long:
            if self.position == PositionState.bear:
                self.last_buy_idx = self.curr_idx
            elif self.position == PositionState.short:
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
                self.last_buy_idx = self.curr_idx
            self.position = PositionState.long
        elif action == Action.close:
            if (
                self.position == PositionState.long
                or self.position == PositionState.short
            ):
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
            else:
                reward += self.successive_close_reward
            self.last_buy_idx = None
            self.position = PositionState.bear
        self.action_list.append(int(action))

        # state转换
        self.curr_idx += 1
        # 如果是done
        if self.curr_idx - self.start_idx + self.window_size == self.max_len:
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

        unrealized_net_value = self.last_net_value
        if (
            self.position == PositionState.short
            or self.position == PositionState.long
        ):
            unrealized_net_value = get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx,
            )

        obs_idx = self.curr_idx - self.start_idx + self.window_size - 1
        self.observation["action"][obs_idx - 1] = int(action)
        # sharpe_ratio依赖unrealized_net_value更新
        self.net_value[obs_idx] = curr_net_value
        self.unrealized_net_value[obs_idx] = unrealized_net_value
        sharpe_ratio = self.get_curr_sharpe_ratio()

        self.observation["net_value"][obs_idx] = curr_net_value

        self.observation["unrealized_net_value"][
            obs_idx
        ] = unrealized_net_value

        self.observation["sharpe_ratio"][obs_idx] = sharpe_ratio
        self.observation["position"][obs_idx] = int(self.position)
        self.observation["mask_len"][0] = obs_idx + 1
        self.observation["hold_threshold"][obs_idx] = 0

        if done:
            self.observation["asset_x"][obs_idx] = -1
            self.observation["asset_y"][obs_idx] = -1
            self.observation["next_end"][obs_idx] = 0
        else:
            self.observation["asset_x"][obs_idx] = self.asset_price[0][
                self.curr_idx
            ]
            self.observation["asset_y"][obs_idx] = self.asset_price[1][
                self.curr_idx
            ]
            self.observation["next_end"][obs_idx] = int(
                obs_idx + 1 == self.max_len - 1
            )
        if self.position == PositionState.bear:
            self.observation["hold_indicator"].fill(0)
        elif (
            self.observation["position"][obs_idx]
            == self.observation["position"][obs_idx - 1]
        ):
            self.observation["hold_indicator"][obs_idx] = 1
        else:
            self.observation["hold_indicator"].fill(0)
            self.observation["hold_indicator"][obs_idx] = 1
            self.observation["hold_indicator"][obs_idx - 1] = 1

        info = {
            "curr_idx": self.curr_idx,
            "last_buy_idx": self.last_buy_idx,
            "last_net_value": self.last_net_value,
            "net_value": curr_net_value,
            "ur_net_value": unrealized_net_value,
            "position": self.position,
            "sharpe_ratio": sharpe_ratio,
        }
        return self.observation, reward, done, info

    def get_curr_sharpe_ratio(self):
        net_values = self.unrealized_net_value[
            self.window_size
            - 1 : self.curr_idx
            - self.start_idx
            + self.window_size
        ]
        noncumulative_returns = (
            np.array(net_values)[1:] / np.array(net_values)[:-1] - 1
        )
        sharpe_ratio = float(
            empyrical.sharpe_ratio(noncumulative_returns, self.risk_free)
        )
        if math.isinf(sharpe_ratio) or math.isnan(sharpe_ratio):
            sharpe_ratio = 0
        sharpe_ratio = max(sharpe_ratio, -3.0)
        return sharpe_ratio

    def plot_trajectory(
        self,
        action_list: List[int],
        net_value_list: List[float],
        figsize: Tuple[int, int] = (15, 5),
        net_value_limit: Tuple[float, float] = (0.9, 1.1),
    ):
        assert (
            len(net_value_list)
            == len(action_list)
            == len(self.date[self.window_size - 1 :])
        )
        long_idxs, short_idxs, close_idxs = get_valid_action_indexes(
            action_list
        )
        figure = plot_assets(
            date=np.array(
                self.date[self.window_size - 1 :], dtype="datetime64"
            ),
            asset_x=np.array(self.asset_price[0][self.window_size - 1 :]),
            asset_x_label=self.pair_name[0],
            asset_y=np.array(self.asset_price[1][self.window_size - 1 :]),
            asset_y_label=self.pair_name[1],
            net_value=np.array(net_value_list),
            long_idxs=np.array(long_idxs),
            short_idxs=np.array(short_idxs),
            close_idxs=np.array(close_idxs),
            figsize=figsize,
            net_value_limit=net_value_limit,
        )

        return figure
