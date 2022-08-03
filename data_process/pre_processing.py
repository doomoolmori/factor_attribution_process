import pandas as pd
import polars as pl
import os
from data_process import data_path
import pickle

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

def save_dict_to_pickle(dict_: dict, path: str, name: str):
    file_name = f'{path}/{name}'
    # save dictionary to pickle file
    with open(f'{file_name}', 'wb') as file:
        pickle.dump(dict_, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle_to_dict(path: str, name: str) -> dict:
    file_name = f'{path}/{name}'
    with open(f'{file_name}', "rb") as file:
        result = pickle.load(file)
    return result


print(os.getcwd())
# file = f'{data_path.RAW_DATA_PATH}/{data_path.RAW_DATA_NAME}'

path = data_path.RAW_DATA_PATH
name = data_path.RAW_DATA_NAME
universe = data_path.KOREA_UNIVERSE
raw_data_df = read_raw_data_df(path=path, name=name)

filter_df = universe_filter_df(df=raw_data_df, universe=universe)
dict_= make_dict_of_pandas(df=filter_df)

pickle_path = data_path.DICT_OF_KOREA_DATA_PATH
pickle_name = data_path.DICT_OF_KOREA_DATA_NAME
save_dict_to_pickle(dict_=dict_, path=pickle_path, name=pickle_name)
dict_of_pandas = read_pickle_to_dict(path=pickle_path, name=pickle_name)
