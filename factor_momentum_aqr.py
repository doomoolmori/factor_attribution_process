from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import serial_stats
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import make_bm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
start = time.time()

if __name__ == "__main__":
    rebal = 'q'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'us'

    garbage_list = [False, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    after_return_list = []
    factor_list = []
    for garbage in garbage_list:
        print(garbage)
        pre_process = pre_processing.PreProcessing(universe=universe, n_top=n_top, garbage=garbage)

        # picking_data_load
        try:
            picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)
            bulk_backtest_df = data_read.bulk_backtest_df(
                strategy_name_list=list(picking_dict.keys()),
                raw_path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                save_path=pre_process.path_dict['STRATEGY_STATS_PATH'],
                rebal=rebal)
        except:
            print(garbage)
            next
        if np.nansum(bulk_backtest_df.pct_change()) == 0:
            next
        else:
            temp_df = bulk_backtest_df.iloc[:, :715]
            return_df = temp_df.pct_change()

            j = 1 * 2
            factor_return_sum = 0
            for t in range(0, j):
                factor_return_sum += return_df.shift(j)
            factor_vol = return_df.rolling(12).std() * np.sqrt(4)
            s_df = factor_return_sum/factor_vol
            s_df[s_df < -2] = -2
            s_df[s_df > 2] = 2

            tsfm_f = s_df.shift(1) * return_df
            one_after_return = temp_df.pct_change(1).shift(-1).copy()

            date_ ='2015-01-01'
            e_d = '2018-01-01'
            tsfm_f_temp = tsfm_f.loc[(tsfm_f.index > date_) & (tsfm_f.index < e_d)].copy()
            one_after_return_temp = one_after_return.loc[(one_after_return.index > date_) & (one_after_return.index < e_d)].copy()

            tsfm_arr = tsfm_f_temp[(tsfm_f_temp > -3) & (one_after_return_temp > -1)].copy().to_numpy()
            tsfm_arr = pd.Series(tsfm_arr.flatten())
            return_arr = one_after_return_temp[(tsfm_f_temp > -3) & (one_after_return_temp > -1)].copy().to_numpy()
            return_arr = pd.Series(return_arr.flatten())
            after_return_list.append(return_arr)
            factor_list.append(tsfm_arr)

    plt.scatter(pd.concat(factor_list, 0), pd.concat(after_return_list, 0))
    pd.concat(factor_list, 0).corr(pd.concat(after_return_list, 0))

    bm = make_bm.BM(pre_process)
    bm_series = bm.get_bm_series(cost=cost, rebal=rebal)
    rf_series = bm.get_rf_series(rebal=rebal, index=list(bm_series.index)) / 100

