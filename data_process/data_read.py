import polars as pl
import pandas as pd
import pickle
import gzip
import os


def read_raw_data_df(path: str, name: str) -> pl.DataFrame:
    # file = f'{data_path.RAW_DATA_PATH}/{data_path.RAW_DATA_NAME}'
    file_name = f'{path}/{name}'
    raw_data_df = pl.read_csv(
        file_name,
        quote_char="'",
        low_memory=False,
        dtype={'sedol': pl.Utf8})
    return raw_data_df


def universe_filter_df(df: pl.DataFrame, universe: str) -> pl.DataFrame:
    return df.filter((pl.col(universe) == 1)).sort('date_')


def make_dict_of_pandas(df: pl.DataFrame) -> dict:
    result = {}
    for column in df.columns:
        temp_df = df.pivot(
            index='date_',
            columns='infocode',
            values=column).to_pandas()
        result[column] = temp_df.set_index('date_')
        print(column)
    return result


def save_to_pickle(any_: any, path: str, name: str):
    file_name = f'{path}/{name}'
    # save dictionary to pickle file
    with gzip.open(f'{file_name}', 'wb') as file:
        pickle.dump(any_, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: str, name: str) -> dict:
    file_name = f'{path}/{name}'
    with gzip.open(f'{file_name}', "rb") as file:
        result = pickle.load(file)
    return result


def save_to_csv(df: pd.DataFrame, path: str, name: str):
    file_name = f'{path}/{name}'
    df.to_csv(file_name)


def read_csv_(path: str, name: str):
    file_name = f'{path}/{name}'
    return pd.read_csv(file_name)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('error')


def make_path(path):
    createFolder(path)


def bulk_backtest_df(
        strategy_name_list: list,
        raw_path: str,
        save_path: str,
        rebal: str) -> pd.DataFrame:
    try:
        result = read_csv_(path=save_path,
                           name=f'bulk_backtest_{rebal}.csv')
        result = result.set_index('date_')
    except:
        series_list = []
        for strategy_name in strategy_name_list:
            file_name = f'{strategy_name}_backtest_{rebal}.pickle'
            series_list.append(read_pickle(
                path=raw_path,
                name=file_name))
        result = pd.concat(series_list, 1)
        result.columns = strategy_name_list
        save_to_csv(df=result,
                    path=save_path,
                    name=f'bulk_backtest_{rebal}.csv')
    return result


def bulk_exposure_dict(strategy_name_list: list, path: str) -> dict:
    exposure_dict = {}
    for strategy_name in strategy_name_list:
        file_name = f'{strategy_name}_exposure.pickle'
        exposure_dict[strategy_name] = read_pickle(
            path=path,
            name=file_name)
    return exposure_dict
