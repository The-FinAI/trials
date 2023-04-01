# TRIALS
自动配对交易

## 说明

我们提出 BanditPair，第一个基于 RL 的股票对选择框架。BanditPair将股票对选择视为bandit问题，可以直接最大化未来交易利润。有关我们的论文，请参阅[http link](https://arxiv.org/abs/2301.10724)


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

##  TRIALS的数据预处理

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

## TRIALS

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

## 开发

1. 通过运行以下命令安装项目的开发和生产依赖项：

```
poetry install

```

1. 安装 `pre-commit`

```
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg

```

1. 确认 `pre-commit`

```
poetry run pre-commit run --all-files

```

1. 在 IDE 中编写代码（推荐使用 Visual Studio Code！）
2. 在本地开发环境中运行单元测试

```
poetry run nosetests -w tests/unit

```

1. 单元测试通过后，提交更改
2. 检查 `pre-commit` 检查是否通过
3. 将代码更改推送到远程仓库
4. If CI passed and code quality is ok, bump version with `commitizen`

```
poetry run cz bump --files-only

```

1. 用 `towncrier` 写更新日志
    1. 在 `CHANGES/` 中创建一个新的临时片段
    2. 生成变更日志

    ```
    poetry run towncrier
    
    ```

    3. 删除临时片段
2. 提交更改并将更改推送到远程仓库

## 贡献规则

- 克隆这个仓库
- 首先创建`issues`
- 创建本地分支并推送到远程
- **不要** 直接推送到 `main` 分支
- 创建远程分支的 `pull request` 

## License

[MIT](https://choosealicense.com/licenses/mit/)

## 引用我们的论文
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


# TRIALS
Automatic pairs trading

## Discription
We propose BanditPair, the first RL-based framework for pair selection, that can directly maximize the future trading profit, by formulating pair selection as a contextual bandit problem.See this [http link](https://arxiv.org/abs/2301.10724) for the preprint.


## Prerequisite

The requirements to develop the project.

- Python ^3.8
- Git
- Poetry
- Python IDE (Visual Studio Code is recommended!)

**⚠️ Read the official manual or documentation of following projects before write any code ⚠️**:

- `git`, a version control system
- `Poetry`, a python library to manage project dependencies
- `towncrier`, a changelog generater
- `pre-commit`, a git commit hook runner
- `commitizen`, a semantic version manager

## Data Preparation For  TRIALS

1. Download the symbols of target stocks and put the .csv file containing the symbols of the target stocks into selected_symbol_path； Download the processed data put all .csv files containing stock datas into stock_path

2. You can download the data of US stocks from the following website：[Tiingo stock marke](https://api.tiingo.com/documentation/iex)

3. Select and process eligible U.S.S&P 500 stocks by running(The processing steps of the Chinese stock dataset are the same as the following):

```bash
python trials/preprocess/U.S.SP500-selected.py selected_symbol_path stock_path store_path begin_time end_time

```

- `selected_symbol_path`, Stock symbols to be selected are stored in selected_symbol_path
- `stock_path`, All stocks are stored in stock_path
- `store_path`,  Eligible U.S.S&P 500 stocks that have been screened and processed are stored in store_path
- `begin_time`, The start time of the stocks to be screened. For example: '2000-01-01'
- `end_time`, The end time of the stocks to be screened. For example: '2020-12-31'

4. Randomly screen stocks from all stocks to form .csv of symbols, and generate as many .csv as possible with disjoint symbols. Then rolling.py can use these .csv to generate rolling datasets

```bash
python trials/preprocess/random_stocks_selected.py symbols_num_each_rolling, stock_data_path, random_symbol_path

```

- `symbols_num_each_rolling`, The number of stocks contained in each rolling
- `stock_data_path`, All U.S.S&P 500 stocks data screened by U.S.SP500-selected.py are stored in stock_data_path
- `random_symbol_path`, The stocks symbols in each rolling are stored in random_symbol_path

5. Form datasets for each period of each rolling in the form of .csv. by running:

```bash
python trials/preprocess/rolling.py stock_data_path store_path training_month validation_month testing_month random_symbol_path csv_name1 csv_name2 csv_name3

```

- `stock_data_path`, All U.S.S&P 500 stocks data screened by U.S.SP500-selected.py are stored in stock_data_path
- `store_path`, The final .csv files are stored in store_path
- `training_month`, The number of months included in the training
- `validation_month`, The number of months included in the validation
- `testing_month`,The number of months included in the testing
- `random_symbol_path`, The stock symbols included in each rolling that are randomly generated by random_stocks_selected.py in advance are stored in random_symbol_path

6. We provide a subset of the processed dataset, please check it in /trials/data/.


## TRIALS

1. Single Run

```bash
poetry run trials/scripts/train_trials.py
```

See all parameters and their illustrations by `poetry run trials/scripts/train_trials.py --help`

2. Hyperparameter Search
   We recommend to use hyperparamete search by wandb sweep.

Start with a template `sweep.yaml` to define searching parameters and values, then simply run:

```bash
wandb sweep rl_sweep.yaml
```

After showing

```bash
wandb: Run sweep agent with: wandb agent xxx/xxxxxxx/[sweep-id]
```

start the agents with `sweep-id`:

```bash
wandb agent xxx/xxxxxxx/[sweep-id]
```

Or you can start multiple agents with the script:

```bash
bash train_trials.sh [sweep-id] [num-process-per-gpu]
```

## Development

1. Install dev and production dependencies of the project by running:

```
poetry install

```

1. Install `pre-commit`

```
poetry run pre-commit install
poetry run pre-commit install --hook-type commit-msg

```

1. Verify `pre-commit`

```
poetry run pre-commit run --all-files

```

1. Write your code in your IDE (Visual Studio Code is recommended!)
2. Run unittest in your local dev environment

```
poetry run nosetests -w tests/unit

```

1. After unittest passed, commit the changes
2. Check if `pre-commit` check passed
3. Push code changes to remote repository
4. If CI passed and code quality is ok, bump version with `commitizen`

```
poetry run cz bump --files-only

```

1. Write changelog with `towncrier`
    1. create a new temporary fragement in `CHANGES/`
    2. generate changelog

    ```
    poetry run towncrier
    
    ```

    3. remove temporary fragement
2. Commit and push changes to remote repository

## Contribution Rules

- Clone this repository
- Create `issues` first
- Create local branch and push to remote
- **DO NOT** push directly to `main` branch
- Create `pull request` with remote branch

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Feel free to cite our paper
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
