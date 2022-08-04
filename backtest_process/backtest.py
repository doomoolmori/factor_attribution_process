from data_process import data_read
from data_process import data_path
import pandas as pd
import time

import numpy as np
import asyncio



def array_rank(a, order='descending'):
    """
    https://stackoverflow.com/questions/64101391/how-to-obtain-the-ranks-in-a-2d-numpy-array
    pandas rank와 유사한 함수
    """
    b = -a if order == 'descending' else a
    idx = b.argsort(1, 'stable')
    m, n = idx.shape
    out = np.empty((m, n), dtype=float)
    np.put_along_axis(out, idx, np.arange(1, n+1), axis=1)
    return np.where(np.isnan(a), np.nan, out)


def rank_(score_list, ri_arr):
    for i in range(0, 200):
        broad_a = np.broadcast_to(score_list, ri_arr.T.shape).T
        result = array_rank(np.sum(broad_a * ri_arr, axis=0))
        #result = result.rank(1).to_numpy()
        #result.rank(axis=1).to_numpy()
        #t = 0
        #for j in range(0, 4):
        #    t += ri
        #ri.rank()
        #print(i)

async def download_page(score_list, ri_arr):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rank_, score_list, ri_arr)

async def main(score_list, ri_arr) :

    await asyncio.gather(
        download_page(score_list, ri_arr),
        download_page(score_list, ri_arr),
        download_page(score_list, ri_arr),
        download_page(score_list, ri_arr)
    )

from multiprocessing import Pool

if __name__ == "__main__":
    data = data_read.DataRead(universe='korea')

    factor_category = pd.read_csv(f'{data_path.FACTOR_CATEGORY_PATH}/{data_path.FACTOR_CATEGORY_NAME}')
    ri = data.dict_of_pandas['RI'].to_numpy(dtype=np.float16)

    info = data.dict_of_pandas['infocode']

    ri_arr = np.array([ri, ri, ri, ri, ri, ri, ri, ri, ri, ri])
    score_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



    start = time.time()
    asyncio.run(main(score_list, ri_arr))

    print("time :", time.time() - start)
