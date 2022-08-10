from data_process import pre_processing
from data_process import data_read
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import stock_picking
import time

rebal = 'q'
cost = 0.003
n_top = 20
universe = 'korea'
if __name__ == "__main__":
    pre_process = pre_processing.PreProcessing(universe=universe)

    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process, n_top=n_top)

    # 계산 완료되면 돌릴 필요 없어요
    calculate_exposure.calculate_series_exposure(pre_process=pre_process)
    # 계산 완료되면 돌릴 필요 없어요
    calculate_series.calculate_series_backtest(pre_process=pre_process, cost=cost, rebal=rebal)


    """
    under sample
    """
    import pandas as pd
    import numpy as np

    factor_name = list(pre_process.dict_of_rank.keys())
    name_list = list(picking_dict.keys())
    sample = name_list[0]
    sample_dict = dict(zip(factor_name,np.array(sample[1:-1].split(','), dtype=float)))

    backtest_series = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                          name=f'{sample}_backtest_{rebal}.pickle')

    exposure_series = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                            name=f'{sample}_exposure.pickle')

    idx_list = []
    for i, time in enumerate(pre_process.adj_ri.index):
        if time in backtest_series.index:
            idx_list.append(i)

    exposure_df = pd.DataFrame(exposure_series[:, idx_list].T)
    exposure_df.columns = sample_dict.keys()
    exposure_df.index = backtest_series.index
    exposure_df['backtest'] = backtest_series