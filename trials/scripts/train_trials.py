import os

import gym
import numpy as np
import pandas as pd
import torch as th
import typer
from loguru import logger
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from trials.networks.callbacks import (
    BestDevRewardCallback,
    EvalCallback,
    eval_reward_metric,
)
from trials.networks.env import ReinforceTradingEnv
from trials.networks.feature_extractor import FEATURE_EXTRACTORS
from trials.networks.policy_network import PairSelectionActorCriticPolicy


def load_data(path, file_name):
    df = pd.read_csv(
        path + file_name,
        encoding="gbk",
        header=[0, 1],
        thousands=",",
        index_col=0,
    )
    return df


def sub(file_name):
    end_point = file_name.index("_")
    dataset_type = file_name[:end_point]
    return dataset_type


def select_file_name(rolling_dataset_path, dataset_type):
    file_name = os.listdir(rolling_dataset_path)
    file_name = list(filter(lambda x: sub(x) == dataset_type, file_name))
    file_name.sort()
    return file_name


def build_dataset(train, valid, test, asset_number, feature_dim):
    """Build formation and trading for train, valid, and test"""
    logger.info(f"Start building dataset")
    asset_names = (
        train.columns.get_level_values(0).drop_duplicates().values.tolist()
    )
    logger.info(f"Assets: {asset_names}")
    train_size = train.shape[0]
    valid_size = valid.shape[0]
    test_size = test.shape[0]
    logger.info(
        f"Original dataset size: train {train_size} "
        f"| valid {valid_size} | test {test_size}"
    )
    assert test_size == valid_size
    trading_size = test_size
    formation_size = train_size - trading_size
    logger.info(
        f"Generate dataset size: trading {trading_size} "
        f"| formation {formation_size}"
    )

    # T x N x M
    train_value = train.values.astype(float).reshape(
        train_size, asset_number, feature_dim
    )
    valid_value = valid.values.astype(float).reshape(
        valid_size, asset_number, feature_dim
    )
    test_value = test.values.astype(float).reshape(
        test_size, asset_number, feature_dim
    )

    def log_price(data):
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        return np.log(data[:, :, 1])

    def normalize(data):
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        data[:, :, :2] = np.log(data[:, :, :2] / data[:, :1, :2])
        data[:, :, 2] = (
            data[:, :, 2] - np.mean(data[:, :, 2], axis=1, keepdims=True)
        ) / np.std(data[:, :, 2], axis=1, keepdims=True)
        return data

    train_formation = normalize(np.array(train_value[:formation_size]))
    train_formation_log_price = log_price(
        np.array(train_value[:formation_size])
    )
    train_trading = normalize(np.array(train_value[formation_size:]))
    train_trading_log_price = log_price(np.array(train_value[formation_size:]))
    train_formation_dates = train.index.values[:formation_size].tolist()
    train_trading_dates = train.index.values[formation_size:].tolist()

    valid_formation = normalize(np.array(train_value[trading_size:]))
    valid_formation_log_price = log_price(np.array(train_value[trading_size:]))
    valid_trading = normalize(np.array(valid_value))
    valid_trading_log_price = log_price(np.array(valid_value))
    valid_formation_dates = train.index.values[trading_size:].tolist()
    valid_trading_dates = valid.index.values.tolist()

    logger.info(
        f"{np.array(train_value[(trading_size * 2):]).shape}, "
        f"{np.array(valid_value).shape}"
    )

    test_formation_data = np.concatenate(
        [
            np.array(train_value[(trading_size * 2) :]),
            np.array(valid_value),
        ],
        axis=0,
    )
    test_formation = normalize(np.array(test_formation_data))
    test_formation_log_price = log_price(np.array(test_formation_data))
    test_trading = normalize(np.array(test_value))
    test_trading_log_price = log_price(np.array(test_value))
    test_formation_dates = (
        train.index.values[(trading_size * 2) :].tolist() + valid_trading_dates
    )
    test_trading_dates = test.index.values.tolist()

    return (
        asset_names,
        (
            train_formation,
            train_formation_log_price,
            train_trading,
            train_trading_log_price,
            train_formation_dates,
            train_trading_dates,
        ),
        (
            valid_formation,
            valid_formation_log_price,
            valid_trading,
            valid_trading_log_price,
            valid_formation_dates,
            valid_trading_dates,
        ),
        (
            test_formation,
            test_formation_log_price,
            test_trading,
            test_trading_log_price,
            test_formation_dates,
            test_trading_dates,
        ),
    )


def train(args, run_id):
    """Train method"""
    set_random_seed(args.seed, using_cuda=True)
    logger.info(f"Start training for {args.rolling_serial}")
    logger.info(f"Load data from {args.rolling_dataset_path}")
    train = select_file_name(args.rolling_dataset_path, "train")
    valid = select_file_name(args.rolling_dataset_path, "valid")
    test = select_file_name(args.rolling_dataset_path, "test")
    assert args.rolling_serial < len(train)
    df_train = load_data(args.rolling_dataset_path, train[args.rolling_serial])
    df_valid = load_data(args.rolling_dataset_path, valid[args.rolling_serial])
    df_test = load_data(args.rolling_dataset_path, test[args.rolling_serial])

    asset_names, train_dataset, valid_dataset, test_dataset = build_dataset(
        df_train, df_valid, df_test, args.asset_num, args.feature_dim
    )

    def log_dataset(name, dataset):
        logger.info(
            f"Generated {name} dataset:\n  "
            f"Formation ({dataset[4][0]} - {dataset[4][-1]})\n  "
            f"Data size (N x T x M): {dataset[0].shape}\n  "
            f"Trading ({dataset[5][0]} - {dataset[5][-1]})\n  "
            f"Data size (N x T x M): {dataset[2].shape}"
        )

    dataset_names = ["train", "valid", "test"]
    datasets = [train_dataset, valid_dataset, test_dataset]
    [
        log_dataset(dataset_names[index], dataset)
        for index, dataset in enumerate(datasets)
    ]

    time_step = len(train_dataset[4])
    serial_selection = args.policy in ["simple_serial_selection"]

    def initialize_env(
        name,
        names,
        dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        trading_train_steps,
        worker_model,
    ):
        return Monitor(
            ReinforceTradingEnv(
                name=name,
                form_date=dataset[4],
                trad_date=dataset[5],
                asset_name=names,
                form_asset_features=dataset[0],
                form_asset_log_prices=dataset[1],
                trad_asset_features=dataset[2],
                trad_asset_log_prices=dataset[3],
                feature_dim=feature_dim,
                serial_selection=serial_selection,
                asset_attention=asset_attention,
                trading_feature_extractor=args.trading_feature_extractor,
                trading_feature_extractor_feature_dim=args.trading_feature_extractor_feature_dim,
                trading_feature_extractor_num_layers=args.trading_feature_extractor_num_layers,
                trading_feature_extractor_hidden_dim=args.trading_feature_extractor_hidden_dim,
                trading_feature_extractor_num_heads=args.trading_feature_extractor_num_heads,
                trading_train_steps=trading_train_steps,
                trading_num_process=args.trading_num_process,
                trading_dropout=args.trading_dropout,
                policy=args.policy,
                trading_learning_rate=args.trading_learning_rate,
                trading_log_dir=args.trading_log_dir,
                trading_rl_gamma=args.trading_rl_gamma,
                trading_ent_coef=args.trading_ent_coef,
                seed=args.seed,
                worker_model=worker_model,
            )
        )

    train_env = initialize_env(
        "train",
        asset_names,
        train_dataset,
        args.feature_dim,
        serial_selection,
        args.asset_attention,
        args.trading_train_steps,
        None,
    )
    valid_env = initialize_env(
        "valid",
        asset_names,
        valid_dataset,
        args.feature_dim,
        serial_selection,
        args.asset_attention,
        0,
        train_env.worker_model,
    )
    test_env = initialize_env(
        "test",
        asset_names,
        test_dataset,
        args.feature_dim,
        serial_selection,
        args.asset_attention,
        0,
        train_env.worker_model,
    )

    policy_kwargs = {
        "features_extractor_class": FEATURE_EXTRACTORS[args.feature_extractor],
        "features_extractor_kwargs": {
            "asset_num": args.asset_num,
            "time_step": time_step,
            "input_feature": args.feature_dim,
            "hidden_dim": args.feature_extractor_hidden_dim,
            "num_layers": args.feature_extractor_num_layers,
            "num_heads": args.feature_extractor_num_heads,
            "asset_attention": args.asset_attention,
            "drouput": args.dropout,
        },
        "hidden_dim": args.feature_extractor_hidden_dim,
        "asset_num": args.asset_num,
        "feature_dim": args.feature_dim,
        "policy": args.policy,
        "num_heads": args.feature_extractor_num_heads,
        "latent_pi": (args.asset_num * 2)
        if serial_selection
        else (args.asset_num * (args.asset_num - 1) // 2),
        "latent_vf": args.policy_network_hidden_dim,
    }

    model = A2C(
        PairSelectionActorCriticPolicy,
        train_env,
        learning_rate=args.learning_rate,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        gamma=args.rl_gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=1
    )

    test_callback = BestDevRewardCallback(
        train_env=train_env,
        test_env=test_env,
        valid_env=valid_env,
    )

    eval_callback = EvalCallback(
        valid_env,
        train_env,
        initialize_env(
            "train",
            asset_names,
            train_dataset,
            args.feature_dim,
            serial_selection,
            args.asset_attention,
            0,
            train_env.worker_model,
        ),
        patience_steps=args.patience_steps,
        best_model_save_path=args.saved_model_dir,
        callback_on_new_best=test_callback,
        verbose=0,
        n_eval_episodes=1,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        exclude_names=[],
        metric_fn=eval_reward_metric,
    )

    model.learn(
        total_timesteps=args.train_steps,
        callback=CallbackList(
            [
                WandbCallback(
                    gradient_save_freq=100,
                    verbose=0,
                ),
                eval_callback,
            ]
        ),
    )


def main(
    log_dir: str = "log",
    saved_model_dir: str = "saved_model",
    rolling_dataset_path: str = "trials/data/",
    policy: str = "simple_serial_selection",
    feature_extractor: str = "mlp",
    trading_feature_extractor: str = "lstm",
    asset_attention: bool = False,
    rolling_serial: int = 1,
    asset_num: int = 30,
    feature_dim: int = 3,
    feature_extractor_hidden_dim: int = 64,
    feature_extractor_num_layers: int = 1,
    feature_extractor_num_heads: int = 2,
    policy_network_hidden_dim: int = 64,
    seed: int = 13,
    patience_steps: int = 0,
    eval_freq: int = 32,
    train_steps: int = 1e4,
    learning_rate: float = 1e-4,
    dropout: float = 0.5,
    rl_gamma: float = 1,
    ent_coef: float = 1e-4,
    project: str = "learning_to_pair",
    entity: str = "jimin",
    trading_train_steps: int = 1e3,
    trading_feature_extractor_feature_dim: int = 3,
    trading_feature_extractor_num_layers: int = 1,
    trading_feature_extractor_hidden_dim: int = 64,
    trading_dropout: float = 0.5,
    trading_feature_extractor_num_heads: int = 2,
    trading_learning_rate: float = 1e-4,
    trading_log_dir: str = "trading_log",
    trading_rl_gamma: float = 1,
    trading_ent_coef: float = 1e-4,
    trading_num_process: int = 1
) -> None:
    """
    Train l2r and its ablations

    Args:
    log_dir: the directory to save logs
    saved_model_dir: the directory to save models
    rolling_dataset_path: All rolling datasets are stored in
                          rolling_dataset_path.
    rolling_serial: the rolling to train
    asset_num: the number of assets
    feature_dim: the size of features
    feature_extractor_hidden_dim: the dim of the hidden layer in feature
    extractor
    policy_network_hidden_dim: the dim of the hidden layer in policy network
    seed: the random seed
    patience_steps: the steps before stop running for poor performance
    eval_freq: evaluation per steps
    train_steps: the total training steps


    Returns:
    Nothing to see here.
    """
    args = dict(
        log_dir=log_dir,
        saved_model_dir=saved_model_dir,
        rolling_dataset_path=rolling_dataset_path,
        policy=policy,
        feature_extractor=feature_extractor,
        trading_feature_extractor=trading_feature_extractor,
        asset_attention=asset_attention,
        rolling_serial=rolling_serial,
        asset_num=asset_num,
        feature_dim=feature_dim,
        feature_extractor_hidden_dim=feature_extractor_hidden_dim,
        feature_extractor_num_layers=feature_extractor_num_layers,
        feature_extractor_num_heads=feature_extractor_num_heads,
        policy_network_hidden_dim=policy_network_hidden_dim,
        seed=seed,
        patience_steps=patience_steps,
        eval_freq=eval_freq,
        train_steps=train_steps,
        learning_rate=learning_rate,
        dropout=dropout,
        rl_gamma=rl_gamma,
        ent_coef=ent_coef,
        project=project,
        entity=entity,
        trading_train_steps=trading_train_steps,
        trading_feature_extractor_feature_dim=trading_feature_extractor_feature_dim,
        trading_feature_extractor_num_layers=trading_feature_extractor_num_layers,
        trading_dropout=trading_dropout,
        trading_feature_extractor_hidden_dim=trading_feature_extractor_hidden_dim,
        trading_feature_extractor_num_heads=trading_feature_extractor_num_heads,
        trading_learning_rate=trading_learning_rate,
        trading_log_dir=trading_log_dir,
        trading_rl_gamma=trading_rl_gamma,
        trading_ent_coef=trading_ent_coef,
        trading_num_process=trading_num_process
    )
    run = wandb.init(
        config=args,
        sync_tensorboard=False,
        monitor_gym=False,
        dir="/data/user_name",
    )
    train(wandb.config, run.id)


if __name__ == "__main__":
    typer.run(main)
