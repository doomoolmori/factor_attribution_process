from data_process import data_read
import pandas as pd


def bulk_backtest_df(strategy_name_list: list, path: str, rebal: str):
    series_list = []
    for strategy_name in strategy_name_list:
        file_name = f'{strategy_name}_backtest_{rebal}.pickle'
        series_list.append(data_read.read_pickle(
            path=path,
            name=file_name))
    result = pd.concat(series_list, 1)
    result.columns = strategy_name_list
    return result


def bulk_exposure_dict(strategy_name_list: list, path: str):
    exposure_dict = {}
    for strategy_name in strategy_name_list:
        file_name = f'{strategy_name}_exposure.pickle'
        exposure_dict[strategy_name] = data_read.read_pickle(
            path=path,
            name=file_name)
    return exposure_dict
