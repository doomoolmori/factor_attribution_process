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
