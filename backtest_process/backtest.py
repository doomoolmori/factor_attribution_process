from data_process import data_read
from data_process import calculation_rank
from data_process import calculation_pct
from itertools import *
import h5py
import numpy as np


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


def shape_mapping_array(arr: np.array, score_list: list) -> np.array:
    return np.broadcast_to(score_list, arr.T.shape).T


def factor_rank_sum(rank_arr: np.array, score_weight_arr: np.array) -> np.array:
    return np.sum(score_weight_arr * rank_arr, axis=0, dtype=np.float16)


def final_rank_array(factor_rank_sum: np.array, epsilon: np.array) -> np.array:
    """
    epsilon을 더하는 이유는 동점등수일 때, 우선순위를 주기위함.
    """
    return _rank(factor_rank_sum + epsilon).astype(np.float16)


def portfolio_weight(final_rank_arr: np.array, n_top: int = 20) -> np.array:
    weight = (final_rank_arr <= n_top) * 1
    return (weight / (weight.sum(1).reshape(len(weight), 1)))




"""
async def download_page(score_set, ri_arr,abs_silon, name):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rank_, score_set, ri_arr,abs_silon, name)

async def main(score_set, ri_arr):
    abs_silon = ri_arr[0]
    await asyncio.gather(
        download_page(score_set[:500], ri_arr,abs_silon, name='dd'),
        download_page(score_set[500:1000], ri_arr,abs_silon, name='dd1'),
        download_page(score_set[1000:1500], ri_arr,abs_silon, name='dd2'),
        download_page(score_set[1500:2000], ri_arr,abs_silon, name='dd3')
    )
"""

if __name__ == "__main__":
    data = data_read.DataRead(universe='us')
    data.dict_of_rank = calculation_rank.rank_dict_add(data)
    data.adj_ri = calculation_pct.make_adj_ri_df(data.dict_of_pandas['RI'])
    data.adj_pct = data.adj_ri.pct_change().shift(-1)

    space_set = strategy_space(numbers=len(data.dict_of_rank.keys()))

    survive_df = calculation_rank.make_survive_df(data.dict_of_pandas['RI'])
    epsilon_df = data.dict_of_pandas['Market Capitalization - Current (U.S.$)']
    survive_epsilon_arr = calculation_rank.fill_survive_data_with_min(
        survive_df=survive_df,
        factor_df=epsilon_df,
        min_value=np.nanmin(epsilon_df)).to_numpy(np.float32)

    survive_epsilon_arr /= (np.nanmax(survive_epsilon_arr) * 10)

    adj_pct_arr = data.adj_pct.to_numpy(np.float16)

    ri_arr = []
    for i, t in enumerate(data.dict_of_rank.values()):
        ri_arr.append(t.to_numpy())
    ri_arr = np.array(ri_arr, dtype=np.float16)

    data.dict_of_rank.keys()
    import time
    start = time.time()
    score_weight_arr = shape_mapping_array(ri_arr, space_set[23])
    factor_rank_sum_arr = factor_rank_sum(ri_arr, score_weight_arr)
    final_rank_arr = final_rank_array(factor_rank_sum_arr, epsilon=survive_epsilon_arr)
    portfolio_weight_arr = portfolio_weight(final_rank_arr)
    t = np.nansum(adj_pct_arr * portfolio_weight_arr, 1)
    print("time :", time.time() - start)

    d = ((portfolio_weight_arr > 0) * 1).flatten()

    import pickle
    import gzip
    dd = np.where(d == 1)
    file_name = f'ds.pickle'
    # save dictionary to pickle file
    with gzip.open(f'{file_name}', 'wb') as file:
        pickle.dump(dd[0].astype(np.int32), file, protocol=pickle.HIGHEST_PROTOCOL)


    with gzip.open(f'{file_name}', 'rb') as file:
       data = pickle.load(file)

    dd = ''
    for st in d:
        dd += str(st)

    # Saving the array in a text file

    file.write(dd)
    file.close()


    d.astype(np.int8)
    type(d[0])

    np.save('d.npy', d.astype(np.int8))

    np.nanmean(adj_pct_arr)
    np.nanstd(adj_pct_arr)

    dt = adj_pct_arr * portfolio_weight_arr

    ((dt > -1.2) & (dt != 0)).sum(1)

    import matplotlib.pyplot as plt
    plt.plot((t + 1).cumprod())

    (t + 1).min()

    """
    import asyncio
    import time

    start = time.time()
    asyncio.run(main(space_set, ri_arr))
    print("time :", time.time() - start)
    """
