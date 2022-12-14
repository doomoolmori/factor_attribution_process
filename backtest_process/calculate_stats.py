# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:51:33 2022
@author: doomoolmori
"""
import numpy as np
import pandas as pd



class Stats:
    def __init__(self):
        print('class init')

    def set_rebal(self, rebal: str = 'q'):
        self.rebal = rebal
        if rebal == 'q':
            self.freq = 4
        elif rebal == 'm':
            self.freq = 12

    def set_backtest_arr(self, backtest_arr: np.array):
        self.backtest_arr = backtest_arr
        self.ret_arr = get_ret_arr(arr=self.backtest_arr)
        self.up_boolean = self.ret_arr > 0
        self.down_boolean = self.ret_arr < 0
        self.backtest_drawdown = series_drawdown(price_df=self.backtest_arr)

    def set_out_backtest_arr(self, out_backtest_arr: np.array):
        self.out_backtest_arr = out_backtest_arr
        self.out_ret_arr = get_ret_arr(arr=self.out_backtest_arr)
        self.out_up_boolean = self.out_ret_arr > 0
        self.out_down_boolean = self.out_ret_arr < 0
        self.out_backtest_drawdown = series_drawdown(price_df=self.out_backtest_arr)

    def set_bm_arr(self, bm_arr: np.array):
        self.bm_arr = bm_arr
        self.bm_ret_arr = get_ret_arr(arr=self.bm_arr)
        self.bm_up_boolean = self.bm_ret_arr > 0
        self.bm_down_boolean = self.bm_ret_arr < 0
        self.bm_drawdown = series_drawdown(price_df=self.bm_arr)

    def set_out_bm_arr(self, out_bm_arr: np.array):
        self.out_bm_arr = out_bm_arr
        self.out_bm_ret_arr = get_ret_arr(arr=self.out_bm_arr)
        self.out_bm_up_boolean = self.out_bm_ret_arr > 0
        self.out_bm_down_boolean = self.out_bm_ret_arr < 0
        self.out_bm_drawdown = series_drawdown(price_df=self.out_bm_arr)

    def set_rf_arr(self, rf_arr: np.array):
        self.rf_arr = rf_arr[1:]

    def set_out_rf_arr(self, out_rf_arr: np.array):
        self.out_rf_arr = out_rf_arr[1:]

    def check(self):
        assert len(self.ret_arr) == len(self.bm_ret_arr), \
            'backtest와 bm의 구간이 동일해야 합니다'

def get_ret_arr(arr:np.array) -> np.array:
    return (np.diff(arr, axis=0) / arr[:-1])



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


# Alpha
def alpha(
        ret_arr: np.array,
        bm_ret_arr: np.array,
        rf_arr: np.array,
        freq: int) -> float:
    ret_diff = ret_arr - bm_ret_arr - rf_arr
    return (1 + ret_diff.mean()) ** freq - 1


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


# Payoff
def pay_off(ret_arr: np.array,
            up_boolean: np.array,
            down_boolean: np.array):
    return np.abs(ret_arr[up_boolean].mean()/ret_arr[down_boolean].mean())

# AverageRecovery 없어도 될듯
# TODO turnover


def bulk_stats_dict(stats) -> dict:
    stats_dict = {}
    stats_dict['Return'] = ret(
        ret_arr=stats.ret_arr,
        freq=stats.freq)
    stats_dict['SD'] = sd(
        ret_arr=stats.ret_arr, freq=stats.freq)
    stats_dict['Sharpe'] = stats_dict['Return']/stats_dict['SD']
    stats_dict['MinReturn'] = min_ret(
        ret_arr=stats.ret_arr)
    stats_dict['MaxReturn'] = max_ret(
        ret_arr=stats.ret_arr)
    stats_dict['UpsideFrequency'] = upside_frequency(
        ret_arr=stats.ret_arr,
        up_boolean=stats.up_boolean)
    stats_dict['UpCapture'] = up_capture(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_up_boolean=stats.bm_up_boolean)
    stats_dict['DownCapture'] = down_capture(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_down_boolean=stats.bm_down_boolean)
    stats_dict['UpNumber'] = up_number(
        ret_arr=stats.ret_arr,
        bm_up_boolean=stats.bm_up_boolean)
    stats_dict['DownNumber'] = down_number(
        ret_arr=stats.ret_arr,
        bm_down_boolean=stats.bm_down_boolean)
    stats_dict['UpPercent'] = up_percent(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_up_boolean=stats.bm_up_boolean)
    stats_dict['DownPercent'] = down_percent(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_down_boolean=stats.bm_down_boolean)
    stats_dict['AverageDrawdown'] = average_drawdown(
        drawdown=stats.backtest_drawdown)
    stats_dict['MaxDrawdown'] = max_drawdown(
        drawdown=stats.backtest_drawdown)
    stats_dict['PainIndex'] = pain_index(
        drawdown=stats.backtest_drawdown)
    stats_dict['AverageLength'] = average_length(
        drawdown=stats.backtest_drawdown)
    stats_dict['Alpha'] = alpha(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        rf_arr=stats.rf_arr,
        freq=stats.freq)
    stats_dict['TrackingError'] = tracking_error(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        freq=stats.freq)
    stats_dict['In_Info'] = stats_dict['Alpha']/stats_dict['TrackingError']
    stats_dict['Beta'] = beta(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr)
    stats_dict['Beta.Bull'] = beta_bull(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_up_boolean=stats.bm_up_boolean)
    stats_dict['Beta.Bear'] = beta_bear(
        ret_arr=stats.ret_arr,
        bm_ret_arr=stats.bm_ret_arr,
        bm_down_boolean=stats.bm_down_boolean)
    stats_dict['Payoff'] = pay_off(
        ret_arr=stats.ret_arr,
        up_boolean=stats.up_boolean,
        down_boolean=stats.down_boolean)
    stats_dict['Payoff_to_hitrate'] = stats_dict['Payoff'] * stats_dict['UpsideFrequency']
    stats_dict['OutReturn'] = ret(
        ret_arr=stats.out_ret_arr,
        freq=stats.freq)
    stats_dict['OutSD'] = sd(
        ret_arr=stats.out_ret_arr,
        freq=stats.freq)
    stats_dict['OutAlpha'] = alpha(
        ret_arr=stats.out_ret_arr,
        bm_ret_arr=stats.out_bm_ret_arr,
        rf_arr=stats.out_rf_arr,
        freq=stats.freq)
    stats_dict['OutTrackingError'] = tracking_error(
        ret_arr=stats.out_ret_arr,
        bm_ret_arr=stats.out_bm_ret_arr,
        freq=stats.freq)
    stats_dict['Out_Info'] = stats_dict['OutAlpha'] / stats_dict['OutTrackingError']
    return stats_dict

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

