# Trials
[![zh](https://img.shields.io/badge/lang-zh-red.svg)](https://github.com/chancefocus/trials/blob/main/README.zh.md)
[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/chancefocus/trials/blob/main/README.md)
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![poetry](https://img.shields.io/badge/Poetry-config-informational?logo=poetry&logoColor=white)](https://python-poetry.org)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/lazaratan/dyn-gfn/blob/main/LICENSE)

自动配对交易


<img
src="assets/trials.jpg"
title="Trials Overview"
style="display: inline-block; margin: 0 auto; max-width: 800px">

## Description
Trials提供了[Select and Trade](https://arxiv.org/abs/2301.10724)的实现，提出了一种使用层次强化学习进行配对交易的新的范式，能够端到端的同时学习配对选择和交易。在中美两国股票数据集的结果证明了提出方法的有效性。


## Maintainer
[张博艺](https://github.com/zbgzbgzbg)


## 预备条件

开发项目的要求。

- Python ^3.8
- Git
- Poetry
- Python IDE (Visual Studio Code is recommended!)

**⚠️在编写任何代码之前，请阅读以下项目的官方手册或文档⚠️**:

- `git`, 代码版本管理工具
- `Poetry`, 用于管理项目依赖项的 python 库
- `towncrier`,变更日志生成器
- `pre-commit`, 规范git提交代码的钩子
- `commitizen`, 语义版本管理器

##  数据预处理

1. 下载目标股票的代号，将包含目标股票代号的.csv文件放入selected_symbol_path；下载处理后的数据，将所有包含股票数据的 .csv 文件放入 stock_path Tiingo. Tiingo stock

2. 可以从下面网站下载美股股票数据：[Tiingo stock marke](https://api.tiingo.com/documentation/iex)

3. 通过运行以下命令选择和处理符合条件的美国标准普尔 500 指数股票（中文股票数据集处理步骤和下面一致）：

```bash
python trials/preprocess/U.S.SP500-selected.py selected_symbol_path stock_path store_path begin_time end_time

```

- `selected_symbol_path`, 要选择的股票代号存储在 selected_symbol_path
- `stock_path`, 所有的股票存储在stock_path
- `store_path`,  经过筛选处理的符合条件的美标普500股票存储在store_path中
- `begin_time`, 待筛选股票的开始时间。 例如：'2000-01-01'
- `end_time`, 待筛选股票的结束时间。 例如：'2020-12-31'

4. 从所有股票中随机筛选股票，形成股票代码的.csv文件，尽可能多地生成股票代码不相交的.csv文件。 然后 rolling.py 可以使用这些 .csv 文件生成多个rolling数据集

```bash
python trials/preprocess/random_stocks_selected.py symbols_num_each_rolling, stock_data_path, random_symbol_path

```

- `symbols_num_each_rolling`, 每个rolling数据集包含的股票数量
- `stock_data_path`,U.S.SP500-selected.py筛选的所有美国标普500股票数据都存储在stock_data_path中
- `random_symbol_path`, 每个rooling数据集的股票代码存储在 random_symbol_path

5. 生成.csv文件形式的rolling数据集:

```bash
python trials/preprocess/rolling.py stock_data_path store_path training_month validation_month testing_month random_symbol_path csv_name1 csv_name2 csv_name3

```

- `stock_data_path`, U.S.SP500-selected.py筛选的所有美国标普500股票数据都存储在stock_data_path中
- `store_path`, 最终的 .csv 文件存储在 store_path 中
- `training_month`, 训练集中的月的数量
- `validation_month`, 验证集中的月的数量
- `testing_month`,测试集中的月的数量
- `random_symbol_path`, 预先通过random_stocks_selected.py随机生成的每个rolling数据集包含的股票代号存储在random_symbol_path中

6. 我们提供了一部分处理好的数据集，请在/trials/data/里查看。

## 如何运行

1. 单次运行

```bash
poetry run trials/scripts/train_trials.py
```

查看所有参数及其默认值 `poetry run trials/scripts/train_trials.py --help`

2. 超参数搜索
   我们推荐通过 wandb sweep 使用超参数搜索。

从模板 `sweep.yaml` 开始定义搜索参数和值，然后运行：

```bash
wandb sweep rl_sweep.yaml
```

在显示之后

```bash
wandb: Run sweep agent with: wandb agent xxx/xxxxxxx/[sweep-id]
```

启动代理 `sweep-id`:

```bash
wandb agent xxx/xxxxxxx/[sweep-id]
```

或者您可以使用脚本启动多个代理：

```bash
bash train_trials.sh [sweep-id] [num-process-per-gpu]
```

## 引用
```
@misc{han2023select,
      title={Select and Trade: Towards Unified Pair Trading with Hierarchical Reinforcement Learning}, 
      author={Weiguang Han and Boyi Zhang and Qianqian Xie and Min Peng and Yanzhao Lai and Jimin Huang},
      year={2023},
      eprint={2301.10724},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

