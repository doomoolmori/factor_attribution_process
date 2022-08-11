# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:51:33 2022
@author: doomoolmori
"""
import numpy as np
import pandas as pd

"""
import pandas as pd
from dataturbo import DataTurbo
import numpy as np
from scipy import stats
api = DataTurbo()
price_df = api.get_adj_price(["005930 KS Equity"], '1990-01-01', '2022-03-31')
price_df = price_df.resample('Q').last()
ret_df = price_df.pct_change().dropna()
bm_df = api.get_adj_price(["KOSPI Index"], '1990-01-01', '2022-03-31')
bm_price_df = bm_df.resample('Q').last()
bm_ret_df = bm_price_df.pct_change().dropna()
"""


class Stats:
    def __init__(self):
        print('class init')

    def set_rebal(self, rebal: str = 'q'):
        self.rebal = rebal
        if rebal == 'q':
            self.freq = 4
        elif rebal == 'm':
            self.freq = 12

    def set_backtest_series(self, backtest_series: pd.Series):
        self.ret_arr = np.array(backtest_series.pct_change().dropna(), dtype=np.float32)
        self.backtest_series = np.array(backtest_series, dtype=np.float32)
        self.up_boolean = self.ret_arr > 0
        self.down_boolean = self.ret_arr < 0
        self.backtest_drawdown = series_drawdown(price_df=self.backtest_series)
        self.backtest_time = backtest_series.dropna().index

    def set_bm_series(self, bm_series: pd.Series):
        self.bm_series = np.array(bm_series, dtype=np.float32)
        self.bm_ret_arr = np.array(bm_series.pct_change().dropna(), dtype=np.float32)
        self.bm_up_boolean = self.bm_ret_arr > 0
        self.bm_down_boolean = self.bm_ret_arr < 0
        self.bm_drawdown = series_drawdown(price_df=self.bm_series)
        self.bm_time = bm_series.dropna().index

    def check(self):
        assert (self.bm_time != self.backtest_time).sum() == 0, \
            'backtest와 bm의 구간이 동일해야 합니다'


def ret(ret_arr: np.array, freq: int) -> float:
    return (ret_arr.mean() + 1) ** freq - 1


# SD
def sd(ret_arr: np.array, freq: int) -> float:
    return ret_arr.std() * np.sqrt(freq)


# Sharpe
def sharpe(ret_arr: np.array, freq: int) -> float:
    return ret(ret_arr=ret_arr, freq=freq) / sd(ret_arr=ret_arr, freq=freq)


# MinReturn
def min_ret(ret_arr: np.array) -> float:
    return ret_arr.min()


# MaxReturn
def max_ret(ret_arr: np.array) -> float:
    return ret_arr.max()


# UpsideFrequency
def upside_frequency(ret_arr: np.array, up_boolean: np.array) -> float:
    return 1 - up_boolean.sum() / len(ret_arr)


# UpCapture
def up_capture(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_up_boolean: np.array) -> float:
    ret_up = ret_arr[bm_up_boolean].sum()
    bm_up = bm_ret_arr[bm_up_boolean].sum()
    return ret_up / bm_up


# DownCapture
def down_capture(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_down_boolean: np.array) -> float:
    ret_down = ret_arr[bm_down_boolean].sum()
    bm_down = bm_ret_arr[bm_down_boolean].sum()
    return ret_down / bm_down


# UpNumber
def up_number(ret_arr: np.array, bm_up_boolean: np.array) -> float:
    ret_up_number = (ret_arr[bm_up_boolean] > 0).sum()
    bm_up_number = bm_up_boolean.sum()
    return ret_up_number / bm_up_number


# DownNumber
def down_number(ret_arr: np.array, bm_down_boolean: np.array) -> float:
    ret_down_number = (ret_arr[bm_down_boolean] < 0).sum()
    bm_down_number = bm_down_boolean.sum()
    return ret_down_number / bm_down_number


# UpPercent
def up_percent(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_up_boolean: np.array) -> float:
    ret_up_win_number = ((ret_arr - bm_ret_arr)[bm_up_boolean] > 0).sum()
    bm_up = bm_up_boolean.sum()
    return ret_up_win_number / bm_up


# DownPercent
def down_percent(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_down_boolean: np.array) -> float:
    ret_down_win_number = ((ret_arr - bm_ret_arr)[bm_down_boolean] > 0).sum()
    bm_down = bm_down_boolean.sum()
    return ret_down_win_number / bm_down


def series_drawdown(price_df: np.array) -> np.array:
    high = np.maximum.accumulate(price_df)
    return price_df / high - 1


# AverageDrawdown
def average_drawdown(drawdown: np.array) -> float:
    drawdown_number = (drawdown < 0).sum()
    return drawdown.sum() / drawdown_number


# maxDrawdown
def max_drawdown(drawdown: np.array) -> float:
    return drawdown.min()


# TrackingError
def tracking_error(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        freq: int) -> float:
    ret_diff = ret_arr - bm_ret_arr
    return ret_diff.std() * np.sqrt(freq)


# PainIndex
def pain_index(drawdown: np.array) -> float:
    return drawdown.mean()


# AverageLength
def average_length(drawdown: np.array) -> float:
    shift_ = drawdown[:-1]
    shift_ = np.append(np.array([np.NAN]), shift_)
    drawdown_count = ((drawdown == 0) & ((drawdown - shift_) != 0)).sum()
    drawdown_periods = (drawdown != 0).sum()
    return drawdown_periods / drawdown_count


# Beta
def beta(ret_arr: np.array, bm_ret_arr: np.array) -> float:
    cov = np.cov([ret_arr, bm_ret_arr])
    return cov[0, 1] / cov[1, 1]


# Beta.Bull
def beta_bull(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_up_boolean: np.array) -> float:
    cov = np.cov([ret_arr[bm_up_boolean],
                  bm_ret_arr[bm_up_boolean]])
    return cov[0, 1] / cov[1, 1]


# Beta.Bear
def beta_bear(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        bm_down_boolean: np.array) -> float:
    cov = np.cov([ret_arr[bm_down_boolean],
                  bm_ret_arr[bm_down_boolean]])
    return cov[0, 1] / cov[1, 1]


# AverageRecovery 없어도 될듯
# TODO turnover


if __name__ == "__main__":
    from data_process import pre_processing
    from data_process import data_read
    from backtest_process import stock_picking
    import time
    import pandas as pd
    import numpy as np

    rebal = 'q'
    cost = 0.003
    n_top = 20
    universe = 'korea'
    pre_process = pre_processing.PreProcessing(universe=universe)
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process, n_top=n_top)

    # 계산 완료되면 돌릴 필요 없어요
    # calculate_exposure.calculate_series_exposure(pre_process=pre_process)
    # 계산 완료되면 돌릴 필요 없어요
    # calculate_series.calculate_series_backtest(pre_process=pre_process, cost=cost, rebal='m')

    """
    under sample
    """
    import pandas as pd
    import numpy as np

    factor_name = list(pre_process.dict_of_rank.keys())
    name_list = list(picking_dict.keys())
    sample = name_list[0]
    sample_dict = dict(zip(factor_name, np.array(sample[1:-1].split(','), dtype=float)))

    start = time.time()
    backtest_series = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                            name=f'{sample}_backtest_{rebal}.pickle')
    print("time :", time.time() - start)
    bm_series = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                      name=f'{name_list[1]}_backtest_{rebal}.pickle')

    """
    stat sample
    """
    sts = Stats()

    sts.set_rebal(rebal=rebal)
    sts.set_bm_series(bm_series=bm_series)

    start = time.time()
    for i in range(0, 1000):
        sts.set_backtest_series(backtest_series=backtest_series)
        sts.check()
        sts.ret()
        sts.sd()
        sts.sharpe()
        sts.min_ret()
        sts.max_ret()
        sts.upside_frequency()
        sts.up_capture()
        sts.down_capture()
        sts.up_number()
        sts.down_number()

        sts.up_percent()
        sts.down_percent()
        sts.tracking_error()
        sts.beta()
        sts.beta_bull()
        sts.beta_bear()

        sts.average_drawdown(drawdown=sts.backtest_drawdown)
        sts.max_drawdown(drawdown=sts.backtest_drawdown)
        sts.pain_index(drawdown=sts.backtest_drawdown)
        sts.average_length(drawdown=sts.backtest_drawdown)
    print("time :", time.time() - start)
