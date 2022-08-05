# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:51:33 2022
@author: doomoolmori
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


# Return
def ret(ret_df):
    return (ret_df.mean() + 1) ** 4 - 1


# SD
def sd(ret_df):
    return ret_df.std() * np.sqrt(4)


# Sharpe
def sharpe(ret_df):
    return ret(ret_df) / sd(ret_df)


# MinReturn
def min_ret(ret_df):
    return ret_df.min()


# MaxReturn
def max_ret(ret_df):
    return ret_df.max()


# UpsideFrequency
def upside_frequency(ret_df):
    up_count = len(ret_df) - ret_df[ret_df > 0].isna().sum()
    return up_count / len(ret_df)


# UpCapture
def up_capture(ret_df, bm_ret_df):
    bm_up_index = bm_ret_df[bm_ret_df > 0].dropna().index
    return ret_df.loc[bm_up_index].sum()[0] / bm_ret_df.loc[bm_up_index].sum()[0]


# DownCapture
def down_capture(ret_df, bm_ret_df):
    bm_down_index = bm_ret_df[bm_ret_df < 0].dropna().index
    return ret_df.loc[bm_down_index].sum()[0] / bm_ret_df.loc[bm_down_index].sum()[0]


# UpNumber
def up_number(ret_df, bm_ret_df):
    bm_up_index = bm_ret_df[bm_ret_df > 0].dropna().index
    ret_df = ret_df.loc[bm_up_index]
    return len(ret_df[ret_df > 0].dropna()) / len(bm_ret_df.loc[bm_up_index])


# DownNumber
def down_number(ret_df, bm_ret_df):
    bm_down_index = bm_ret_df[bm_ret_df < 0].dropna().index
    ret_df = ret_df.loc[bm_down_index]
    return len(ret_df[ret_df < 0].dropna()) / len(bm_ret_df.loc[bm_down_index])


# UpPercent
def up_percent(ret_df, bm_ret_df):
    bm_up_index = bm_ret_df[bm_ret_df > 0].dropna().index
    ret_df = ret_df.loc[bm_up_index]
    bm_ret_df = bm_ret_df.loc[bm_up_index]
    return (ret_df.iloc[:, 0] > bm_ret_df.iloc[:, 0]).sum() / len(bm_ret_df)


# DownPercent
def down_percent(ret_df, bm_ret_df):
    bm_down_index = bm_ret_df[bm_ret_df < 0].dropna().index
    ret_df = ret_df.loc[bm_down_index]
    bm_ret_df = bm_ret_df.loc[bm_down_index]
    return (ret_df.iloc[:, 0] > bm_ret_df.iloc[:, 0]).sum() / len(bm_ret_df)


# AverageDrawdown
def average_drawdown(price_df):
    high = price_df.cummax()
    drawdown = price_df / high - 1
    drawdown_number = len(drawdown[drawdown < 0].dropna())
    return drawdown.sum() / drawdown_number


# maxDrawdown
def max_drawdown(price_df):
    high = price_df.cummax()
    drawdown = price_df / high - 1
    return drawdown.max()


# TrackingError
def tracking_error(ret_df, bm_ret_df):
    ret_diff = ret_df.iloc[:, 0] - bm_ret_df.iloc[:, 0]
    return ret_diff.std() * np.sqrt(4)


# PainIndex
def pain_index(price_df):
    high = price_df.cummax()
    drawdown = price_df / high - 1
    return drawdown.mean()


# AverageLength
def average_length(price_df):
    high = price_df.cummax()
    drawdown_diff = (high - price_df)
    drawdown_count = ((drawdown_diff == 0) & ((drawdown_diff - drawdown_diff.shift()) != 0)).sum()
    drawdown_periods = ((high - price_df) != 0).sum()
    return drawdown_periods / drawdown_count


# AverageRecovery 없어도 될듯
# Beta
def beta(ret_df, bm_ret_df):
    return stats.linregress(bm_ret_df.iloc[:, 0].values, ret_df.iloc[:, 0].values)[0]


# Beta.Bull
def beta_bull(ret_df, bm_ret_df):
    bull_index = bm_ret_df[bm_ret_df > 0].dropna().index
    ret_bull = ret_df.loc[bull_index]
    bm_ret_bull = bm_ret_df.loc[bull_index]
    return stats.linregress(bm_ret_bull.iloc[:, 0].values, ret_bull.iloc[:, 0].values)[0]


# Beta.Bear
def beta_bear(ret_df, bm_ret_df):
    bear_index = bm_ret_df[bm_ret_df < 0].dropna().index
    ret_bear = ret_df.loc[bear_index]
    bm_ret_bear = bm_ret_df.loc[bear_index]
    return stats.linregress(bm_ret_bear.iloc[:, 0].values, ret_bear.iloc[:, 0].values)[0]

# turnover

#TODO turnover

