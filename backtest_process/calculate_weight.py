import numpy as np


def make_flatten_null_arr(length: int) -> np.array:
    result = np.empty(length)
    result[:] = np.nan
    return result[:]


def make_equal_weight_arr(
        stock_pick: np.array,
        number_of_columns: int,
        number_of_raws: int) -> np.array:
    length = number_of_columns * number_of_raws
    arr = make_flatten_null_arr(length=length)
    arr[stock_pick] = 1
    arr = arr.reshape((number_of_raws, number_of_columns))
    arr /= np.nansum(arr, 1).reshape((number_of_raws, 1))
    return arr


# TODO
def make_mkt_weight_arr(
        stock_pick: np.array,
        number_of_columns: int,
        number_of_raws: int) -> np.array:
    print('TODO')

