import os
from typing import Any, Callable, Dict, List, Optional, Union
import gym
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

import wandb
from trials.networks.constant import Action


class BestDevRewardCallback(BaseCallback):
    """Run trading results on test env"""

    def __init__(
        self,
        test_env,
        verbose: int = 1,
        train_env=None,
        valid_env=None,
    ):
        super(BestDevRewardCallback, self).__init__(verbose)
        self.field_names = [
            "num_timesteps",
            "actions",
            "rewards",
            "euclidean_dist",
        ]
        self.test_env = test_env
        self.valid_env = valid_env
        self.train_env = train_env

    def _on_step(self) -> bool:
        self.test_env.worker_model = self.train_env.worker_model
        self.test_env.eval(self.model, self.num_timesteps)
        if self.valid_env is not None:
            self.valid_env.worker_model = self.train_env.worker_model
            self.valid_env.eval(self.model, self.num_timesteps)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating a manager.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq``
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy``
        (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        train_env: Union[gym.Env, VecEnv],
        training_env_ori: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        patience_steps: int = 1_000_000,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        exclude_names: Optional[List[str]] = None,
        metric_fn: Optional[
            Callable[
                [List[float], List[float], List[float], List[float]], float
            ]
        ] = None,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.parent = self
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.patience_steps = patience_steps

        self.eval_env = eval_env
        self.train_env = train_env
        self.best_model_save_path = best_model_save_path

        self.exclude_names = exclude_names
        self.metric_fn = metric_fn
        self.best_metric = -np.inf
        self.training_env_ori = training_env_ori

    def _init_callback(self) -> None:
        if not isinstance(self.training_env_ori, type(self.eval_env)):
            logger.info(
                "Training and eval env are not of the same type "
                f"{self.training_env_ori} != type({self.eval_env})"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        if (
            self.eval_freq > 0
            and self.n_calls % self.eval_freq == 0
            and self.num_timesteps >= self.patience_steps
        ):
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(
                        self.training_env_ori, self.eval_env
                    )
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way"
                    )

            self.eval_env.worker_model = self.train_env.worker_model

            # Note that eval_env and train_env are instance of ReinforceTradingEnv
            # evaluate_policy() will invoke step() method, which may trigger env.worker_model.learn()
            backup = self.eval_env.is_eval
            self.eval_env.is_eval = True
            (
                episode_eval_rewards, 
                episode_eval_lengths
            ) = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )
            self.eval_env.is_eval = backup

            backup = self.train_env.is_eval
            self.train_env.is_eval = True
            (
                episode_training_rewards,
                episode_training_lengths,
            ) = evaluate_policy(
                self.model,
                self.train_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )
            self.train_env.is_eval = backup

            mean_eval_reward, std_reward = np.mean(
                episode_eval_rewards
            ), np.std(episode_eval_rewards)
            mean_eval_ep_length, std_ep_length = np.mean(
                episode_eval_lengths
            ), np.std(episode_eval_lengths)
            mean_training_reward = np.mean(episode_training_rewards)
            mean_training_ep_length = np.mean(episode_training_lengths)

            if self.verbose > 0:
                logger.info(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_eval_reward:.2f} "
                    f"+/- {std_reward:.2f}"
                )
                logger.info(
                    f"Episode length: {mean_eval_ep_length:.2f} "
                    f"+/- {std_ep_length:.2f}"
                )
            metric = self.metric_fn(
                episode_eval_rewards,
                episode_eval_lengths,
                episode_training_rewards,
                episode_training_lengths,
            )
            wandb.log(
                {
                    "eval/e_mean_reward": mean_eval_reward.item(),
                    "eval/e_mean_ep_length": mean_eval_ep_length.item(),
                    "train/t_mean_reward": mean_training_reward.item(),
                    "train/t_mean_ep_length": mean_training_ep_length.item(),
                    "eval/metric": metric,
                    "time/total_timesteps": self.num_timesteps,
                },
                step=self.num_timesteps
            )
            logger.info(f"Present metric {metric} | SOTA {self.best_metric}")

            if metric > self.best_metric:
                logger.info(
                    f"New best metric of {metric} over {self.best_metric}"
                )
                self.best_metric = metric
                if self.callback_on_new_best is not None:
                    self.callback_on_new_best.on_step()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def eval_reward_metric(
    episode_eval_rewards: List[float],
    episode_eval_lengths: List[float],
    episode_training_rewards: List[float],
    episode_training_lengths: List[float],
) -> float:
    return np.mean(episode_eval_rewards).item()


class TradingEvalCallback(EventCallback):
    """
    Callback for evaluating a trading worker.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.

    :param eval_env: The environment used for initialization
    :param test_env: The test environment used for initialization
    """

    def __init__(
        self,
        name,
        test_env: Union[gym.Env, VecEnv],
    ):
        super().__init__(self, verbose=0)
        self.name = name
        self.test_env = test_env
        self.close_action = Action.close
        self.num_steps = 0

    def _on_step(self) -> bool:
        self.num_steps += 1
        self.log_env(self.test_env)
        return True

    def log_env(
        self,
        env,
    ):
        # 记录测试env
        obs = env.reset()
        action_list = []
        reward_list = []
        net_value_list = []
        sharpe_ratio_list = []
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            info = info[0]
            action = action[0]
            reward = reward[0]
            action_list.append(action)
            reward_list.append(reward)
            net_value_list.append(info["net_value"])
            sharpe_ratio_list.append(info["sharpe_ratio"])

            if done:
                action_list[-1] = int(self.close_action)
                break
        self.best_metric = {
            "num_timesteps": self.parent.num_timesteps,
            "actions": action_list,
            "rewards": reward_list,
            "returns": net_value_list,
            "sharpe_ratios": sharpe_ratio_list,
            "sharpe_ratio": sharpe_ratio_list[-1],
        }
        trading_metric = {
            f"{self.name}/return": net_value_list[-1],
            f"{self.name}/sharpe_ratio": sharpe_ratio_list[-1],
        }
        
        wandb.log(
            trading_metric,
            step=self.num_steps
        )
