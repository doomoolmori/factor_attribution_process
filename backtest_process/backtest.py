from data_process import data_read
from data_process import calculation_rank
from data_process import calculation_pct
from itertools import *
import numpy as np
import time
import asyncio
from data_process import pre_processing

def strategy_space(numbers):
    all_score_space = product([0, 0.2, 0.4, 0.6, 0.8, 1], repeat=numbers)
    return [x for x in all_score_space if sum(x) == 1]


def _rank(arr: np.array, order='descending') -> np.array:
    """
    https://stackoverflow.com/questions/64101391/how-to-obtain-the-ranks-in-a-2d-numpy-array
    pandas rank와 유사한 함수
    """
    temp = -arr if order == 'descending' else arr
    idx = temp.argsort(1, 'stable')
    m, n = idx.shape
    out = np.empty((m, n), dtype=float)
    np.put_along_axis(out, idx, np.arange(1, n + 1), axis=1)
    return np.where(np.isnan(arr), np.nan, out)


def shape_mapping_array(arr: np.array, score_weight: list) -> np.array:
    return np.broadcast_to(score_weight, arr.T.shape).T


def factor_rank_sum(arr_of_rank_arr: np.array, score_weight_arr: np.array) -> np.array:
    return np.sum(score_weight_arr * arr_of_rank_arr, axis=0, dtype=np.float16)


def final_rank_array(factor_rank_sum_arr: np.array, epsilon_arr: np.array) -> np.array:
    """
    epsilon을 더하는 이유는 동점등수일 때, 우선순위를 주기위함.
    """
    return _rank(factor_rank_sum_arr + epsilon_arr).astype(np.float16)


def stock_picker(final_rank_arr: np.array, n_top: int = 20) -> np.array:
    return (final_rank_arr <= n_top)


def make_picked_stock_arr(
        arr_of_rank_arr: np.array,
        epsilon_arr: np.array,
        score_weight: list,
        n_top: int = 20) -> np.array:
    score_weight_arr = shape_mapping_array(
        arr=arr_of_rank_arr,
        score_weight=score_weight)

    factor_rank_sum_arr = factor_rank_sum(
        arr_of_rank_arr=arr_of_rank_arr,
        score_weight_arr=score_weight_arr)

    final_rank_arr = final_rank_array(
        factor_rank_sum_arr=factor_rank_sum_arr,
        epsilon_arr=epsilon_arr)

    picked_stock_arr = stock_picker(
        final_rank_arr=final_rank_arr,
        n_top=n_top)

    return picked_stock_arr


def loop_pick_stock(
        arr_of_rank_arr: np.array,
        survive_epsilon_arr: np.array,
        space_set: list,
        path: int):
    for score_weight in space_set:
        picked_stock_arr = make_picked_stock_arr(
            arr_of_rank_arr=arr_of_rank_arr,
            epsilon_arr=survive_epsilon_arr,
            score_weight=score_weight)

        picked_stock = picked_stock_arr.flatten()
        picked_stock_idx = np.where(picked_stock == True)
        picked_stock_idx = picked_stock_idx[0].astype(np.int32)
        name = f'{score_weight}.pickle'
        # save dictionary to pickle file
        data_read.save_to_pickle(any_=picked_stock_idx,
                                 path=path,
                                 name=name)


async def async_(space_set, arr_of_rank_arr, survive_epsilon_arr, path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        loop_pick_stock,
        space_set,
        arr_of_rank_arr,
        survive_epsilon_arr, path)


async def main(space_set, arr_of_rank_arr, survive_epsilon_arr, path):
    await asyncio.gather(
        async_(space_set[:500], arr_of_rank_arr, survive_epsilon_arr, path),
        async_(space_set[500:1000], arr_of_rank_arr, survive_epsilon_arr, path),
        async_(space_set[1000:1500], arr_of_rank_arr, survive_epsilon_arr, path),
        async_(space_set[1500:], arr_of_rank_arr, survive_epsilon_arr, path))


if __name__ == "__main__":
    #data = data_read.DataRead(universe='korea')
    #data.dict_of_rank = calculation_rank.rank_dict_add(data)
    #data.adj_ri = calculation_pct.make_adj_ri_df(data.dict_of_pandas['RI'])
    #data.adj_pct = data.adj_ri.pct_change().shift(-1)
    pre_process = pre_processing.PreProcessing(universe='korea')

    space_set = strategy_space(numbers=len(pre_process.dict_of_rank.keys()))
    survive_df = calculation_rank.make_survive_df(pre_process.data.dict_of_pandas['RI'])
    epsilon_df = pre_process.data.dict_of_pandas['Market Capitalization - Current (U.S.$)']
    survive_epsilon_arr = calculation_rank.fill_survive_data_with_min(
        survive_df=survive_df,
        factor_df=epsilon_df,
        min_value=np.nanmin(epsilon_df)).to_numpy(np.float32)
    survive_epsilon_arr /= (np.nanmax(survive_epsilon_arr) * 10)

    arr_of_rank_arr = []
    for rank_arr in (pre_process.data.dict_of_rank.values()):
        arr_of_rank_arr.append(rank_arr.to_numpy())
    arr_of_rank_arr = np.array(arr_of_rank_arr, dtype=np.float16)

    make_picked_stock_arr(
        arr_of_rank_arr=arr_of_rank_arr,
        epsilon_arr=survive_epsilon_arr,
        score_weight=space_set[0],
        n_top=20)
