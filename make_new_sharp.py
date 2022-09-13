# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:38:11 2022

@author: doomoolmori
"""

import polars as pl
import pandas as pd
import os
import pickle
import numpy as np


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


path = 'C:/Users/doomoolmori/factor_attribution_process'
# path = "C:/Users/SUNGHO/Documents\GitHub/factor_attribution_process"
name = "2022-05-27_cosmos-univ-with-factors_with-finval_global_monthly.csv"
raw_data = read_raw_data_df(path, name)

universe_selection = "Univ_S&P500"
rebal_freq = "M"

univ_factor_df = universe_filter_df(raw_data, universe_selection)
factor_date = univ_factor_df["date_"].unique().sort().to_list()

from data_process import data_read
from backtest_process import calculate_weight

picked_stock = data_read.read_pickle(path=path + '/data/us/strategy_weight',
                                     name='0-(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)_picked.pickle')

adj_ri = data_read.read_csv_(path=path + '/data/us',
                             name='adj_ri.csv')
adj_ri = adj_ri.set_index('date_')

weight = calculate_weight.make_equal_weight_arr(
    stock_pick=picked_stock,
    number_of_columns=len(adj_ri.columns),
    number_of_raws=len(adj_ri.index))

weight_df = pd.DataFrame(weight,
                         columns=adj_ri.columns,
                         index=adj_ri.index)

# %%
univ_factor_df = univ_factor_df
topN_weight = weight_df
all_infocode_roc_monthly = adj_ri.pct_change()
trading_cost = 0.003
rebalancing_freq = 'm'
col_list = ['date_', 'infocode', 'ticker', 'RI', 'Market Capitalization - Current (U.S.$)',
            'Price/Book Value Ratio - Current', 'Price/Earnings Ratio - Current',
            'Net Sales Or Revenues (Q)', 'Ebit & Depreciation (Q)']


def get_factor_decomposition(univ_factor_df, topN_weight, all_infocode_roc_monthly, trading_cost, rebalancing_freq='m'):
    univ_factor_df = univ_factor_df[col_list]
    univ_factor_df['short_MA'] = univ_factor_df['RI']
    univ_factor_df = univ_factor_df[~univ_factor_df['short_MA'].is_null()]

    begin_weight_df = topN_weight
    end_weight_df = begin_weight_df.copy() * np.nan
    begin_amount_df = begin_weight_df.copy() * np.nan
    holding_end_amount_df = begin_weight_df.copy() * np.nan
    amount_begin = 100
    for i in topN_weight.index:
        holding_weight_end = begin_weight_df.loc[i] * (all_infocode_roc_monthly.shift(-1).loc[i] + 1)
        end_weight_df.loc[i] = holding_weight_end / holding_weight_end.sum()
        begin_amount_df.loc[i] = begin_weight_df.loc[i] * amount_begin
        amount_begin = amount_begin * holding_weight_end.sum()
        holding_end_amount_df.loc[i] = amount_begin * end_weight_df.loc[i]

    begin_weight_pl = begin_weight_df.reset_index()
    begin_weight_pl = pd.melt(begin_weight_pl, id_vars=['date_'], value_vars=begin_weight_pl.columns.tolist()[1:])
    begin_weight_pl.columns = ['date_', 'infocode', 'value']
    begin_weight_pl.loc[:, 'variable'] = 'begin_weight'
    begin_weight_pl = begin_weight_pl.astype({'value': 'str', 'infocode': 'int64'})
    begin_weight_pl = pl.DataFrame(begin_weight_pl)
    begin_weight_pl = begin_weight_pl['date_', 'infocode', 'variable', 'value']

    begin_amount_pl = begin_amount_df.reset_index()
    begin_amount_pl = pd.melt(begin_amount_pl, id_vars=['date_'], value_vars=begin_amount_pl.columns.tolist()[1:])
    begin_amount_pl.columns = ['date_', 'infocode', 'value']
    begin_amount_pl.loc[:, 'variable'] = 'begin_amount'
    begin_amount_pl = begin_amount_pl.astype({'value': 'str', 'infocode': 'int64'})
    begin_amount_pl = pl.DataFrame(begin_amount_pl)
    begin_amount_pl = begin_amount_pl['date_', 'infocode', 'variable', 'value']

    end_weight_pl = end_weight_df.reset_index()
    end_weight_pl = pd.melt(end_weight_pl, id_vars=['date_'], value_vars=end_weight_pl.columns.tolist()[1:])
    end_weight_pl.columns = ['date_', 'infocode', 'value']
    end_weight_pl.loc[:, 'variable'] = 'end_weight'
    end_weight_pl = end_weight_pl.astype({'value': 'str', 'infocode': 'int64'})
    end_weight_pl = pl.DataFrame(end_weight_pl)
    end_weight_pl = end_weight_pl['date_', 'infocode', 'variable', 'value']

    holding_end_amount_pl = holding_end_amount_df.reset_index()
    holding_end_amount_pl = pd.melt(holding_end_amount_pl, id_vars=['date_'],
                                    value_vars=holding_end_amount_pl.columns.tolist()[1:])
    holding_end_amount_pl.columns = ['date_', 'infocode', 'value']
    holding_end_amount_pl.loc[:, 'variable'] = 'holding_end_amount'
    holding_end_amount_pl = holding_end_amount_pl.astype({'value': 'str', 'infocode': 'int64'})
    holding_end_amount_pl = pl.DataFrame(holding_end_amount_pl)
    holding_end_amount_pl = holding_end_amount_pl['date_', 'infocode', 'variable', 'value']

    univ_factor_df_unpivot = univ_factor_df.melt(id_vars=['date_', 'infocode'], value_vars=univ_factor_df.columns[2:])
    univ_factor_concat = pl.concat(
        [univ_factor_df_unpivot, begin_weight_pl, begin_amount_pl, end_weight_pl, holding_end_amount_pl])
    univ_factor_df = univ_factor_concat.pivot(values='value', index=['date_', 'infocode'], columns='variable')
    univ_factor_df = univ_factor_df.with_columns([pl.col(
        ["RI", "begin_weight", "begin_amount", "end_weight", "holding_end_amount", "Price/Earnings Ratio - Current",
         "short_MA", 'Market Capitalization - Current (U.S.$)']).cast(pl.Float64, strict=False)])

    decomposition_dic = {}
    for i in topN_weight.index:
        univ_factor_df_i = univ_factor_df[univ_factor_df['date_'] == i]
        univ_factor_df_i = pd.DataFrame(univ_factor_df[univ_factor_df['date_'] == i]).T
        univ_factor_df_i.columns = univ_factor_df.columns
        # 결측값 처리안함 <0 만 중앙값
        univ_factor_df_i['Price/Earnings Ratio - Current'][univ_factor_df_i['Price/Earnings Ratio - Current'] < 0] = \
            univ_factor_df_i['Price/Earnings Ratio - Current'].median()
        # univ_factor_df_i = univ_factor_df_i[univ_factor_df_i["infocode"].is_in(picked_infocode)]
        univ_factor_df_i = univ_factor_df_i.astype({'infocode': 'str'})

        begin_weight_infocode = begin_weight_df.loc[i].dropna().index.tolist()
        i_holdings = univ_factor_df_i[univ_factor_df_i['infocode'].isin(begin_weight_infocode)]

        end_weight_infocode = end_weight_df.loc[i].dropna().index.tolist()
        i_rebal_before_holdings = univ_factor_df_i[univ_factor_df_i['infocode'].isin(end_weight_infocode)]

        if i == topN_weight.index[0]:
            i_rebal_before_holdings['end_weight'] = i_rebal_before_holdings['begin_weight']
            i_rebal_before_holdings['holding_end_amount'] = i_rebal_before_holdings['begin_amount']

        if len(i_rebal_before_holdings) > 0:
            i_rebal_before_holdings['Noise'] = i_rebal_before_holdings['RI'] / i_rebal_before_holdings['short_MA']
            i_rebal_before_holdings['shares'] = i_rebal_before_holdings['holding_end_amount'] / i_rebal_before_holdings[
                'RI']
            i_rebal_before_holdings['ma_shares'] = i_rebal_before_holdings['holding_end_amount'] / \
                                                   i_rebal_before_holdings['short_MA']
            i_rebal_before_holdings['EARNINGS'] = i_rebal_before_holdings['Market Capitalization - Current (U.S.$)'] / \
                                                  i_rebal_before_holdings['Price/Earnings Ratio - Current']
            i_rebal_before_holdings['total_shares'] = i_rebal_before_holdings[
                                                          'Market Capitalization - Current (U.S.$)'] / \
                                                      i_rebal_before_holdings['RI']
            i_rebal_before_holdings['EPS'] = i_rebal_before_holdings['RI'] * (
                    1 / i_rebal_before_holdings['Price/Earnings Ratio - Current'])
            i_rebal_before_holdings['MA_EPS'] = i_rebal_before_holdings['short_MA'] * (
                    1 / i_rebal_before_holdings['Price/Earnings Ratio - Current'])
            i_rebal_before_holdings['PER_'] = i_rebal_before_holdings['Market Capitalization - Current (U.S.$)'] * (
                    1 / i_rebal_before_holdings['EARNINGS'])

            holding_i_amount = i_rebal_before_holdings['holding_end_amount'].sum()
            holding_i_EARNINGS = i_rebal_before_holdings['EARNINGS'].sum()
            holding_i_PRICE = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['RI']).sum()
            holding_i_short_MA = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['short_MA']).sum()
            holding_i_EPS = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['EPS']).sum()
            holding_i_MA_EPS = (i_rebal_before_holdings['ma_shares'] * i_rebal_before_holdings['EPS']).sum()

        i_holdings['Noise'] = i_holdings['RI'] / i_holdings['short_MA']
        i_holdings['shares'] = i_holdings['begin_amount'] / i_holdings['RI']  # 수량이 같아야 하는데 왜 다르지?
        i_holdings['ma_shares'] = i_holdings['begin_amount'] / i_holdings['short_MA']
        i_holdings['EARNINGS'] = i_holdings['Market Capitalization - Current (U.S.$)'] / i_holdings[
            'Price/Earnings Ratio - Current']
        i_holdings['total_shares'] = i_holdings['Market Capitalization - Current (U.S.$)'] / i_holdings['RI']
        i_holdings['EPS'] = i_holdings['RI'] * (1 / i_holdings['Price/Earnings Ratio - Current'])
        i_holdings['MA_EPS'] = i_holdings['short_MA'] * (1 / i_holdings['Price/Earnings Ratio - Current'])
        i_holdings['PER_'] = i_holdings['Market Capitalization - Current (U.S.$)'] * (1 / i_holdings['EARNINGS'])

        i_amount = i_holdings['begin_amount'].sum()
        i_PRICE = (i_holdings['shares'] * i_holdings['RI']).sum()
        i_short_MA = (i_holdings['shares'] * i_holdings['short_MA']).sum()
        i_EPS = (i_holdings['shares'] * i_holdings['EPS']).sum()
        i_MA_EPS = (i_holdings['ma_shares'] * i_holdings['EPS']).sum()

        decomposition_dic[i] = pd.Series({'date_': i, 'i_amount': i_amount, 'holding_i_amount': holding_i_amount,
                                          'holding_i_EPS': holding_i_EPS, 'holding_i_MA_EPS': holding_i_MA_EPS,
                                          'holding_i_NOISE': holding_i_PRICE / holding_i_short_MA,
                                          'holding_i_PER': holding_i_PRICE / holding_i_EPS,
                                          'holding_i_MA_PER': holding_i_short_MA / holding_i_EPS,
                                          'i_EPS': i_EPS,
                                          'i_MA_EPS': i_MA_EPS,
                                          'i_NOISE': i_PRICE / i_short_MA,
                                          'PER': i_PRICE / i_EPS,
                                          'MA_PER': i_short_MA / i_EPS})

    decomposition_ts_df = pd.DataFrame.from_dict(decomposition_dic).T
    info_cols = ['i_amount', 'holding_i_amount', 'holding_i_EPS', 'holding_i_MA_EPS',
                 'holding_i_NOISE', 'holding_i_PER', 'holding_i_MA_PER', 'i_EPS', 'i_MA_EPS',
                 'i_NOISE', 'PER', 'MA_PER']
    decomposition_ts_df.loc[decomposition_ts_df['i_EPS'] < 0, info_cols] = np.nan
    decomposition_ts_df.loc[decomposition_ts_df['holding_i_EPS'] < 0, info_cols] = np.nan
    decomposition_ts_df = decomposition_ts_df.ffill()
    decomposition_ts_df['EPS_shift'] = decomposition_ts_df['i_EPS'].shift()
    decomposition_ts_df['holding_NOISE_change'] = decomposition_ts_df['holding_i_NOISE'] / decomposition_ts_df[
        'i_NOISE'].shift() - 1
    decomposition_ts_df['holding_EARNINGS_change'] = decomposition_ts_df['holding_i_EPS'] / decomposition_ts_df[
        'EPS_shift'] - 1
    decomposition_ts_df['holding_MA_EARNINGS_change'] = decomposition_ts_df['holding_i_MA_EPS'] / decomposition_ts_df[
        'i_MA_EPS'].shift() - 1
    decomposition_ts_df['holding_PER_change'] = decomposition_ts_df['holding_i_PER'] / decomposition_ts_df[
        'PER'].shift() - 1
    decomposition_ts_df['holding_MA_PER_change'] = decomposition_ts_df['holding_i_MA_PER'] / decomposition_ts_df[
        'MA_PER'].shift() - 1
    decomposition_ts_df['rebalancing_EARNINGS_change'] = decomposition_ts_df['i_EPS'] / decomposition_ts_df[
        'holding_i_EPS'] - 1
    decomposition_ts_df['rebalancing_MA_EARNINGS_change'] = decomposition_ts_df['i_MA_EPS'] / decomposition_ts_df[
        'holding_i_EPS'] - 1
    decomposition_ts_df['rebalancing_NOISE_change'] = decomposition_ts_df['i_NOISE'] / decomposition_ts_df[
        'holding_i_NOISE'] - 1
    decomposition_ts_df['rebalancing_PER_change'] = decomposition_ts_df['MA_PER'] / decomposition_ts_df[
        'holding_i_MA_PER'] - 1
    decomposition_ts_df['rebalancing_noise_change'] = decomposition_ts_df['rebalancing_EARNINGS_change'] - \
                                                      decomposition_ts_df['rebalancing_MA_EARNINGS_change']

    if rebalancing_freq == 'm':
        freq_scale = 12
    elif rebalancing_freq == 'q':
        freq_scale = 4
    n_freq = len(decomposition_ts_df) - 1

    annual_TR_log = np.log(
        decomposition_ts_df['i_amount'].iloc[-1] / decomposition_ts_df['i_amount'].iloc[0]) / n_freq * freq_scale
    annual_EPS_growth_total_log = np.log(
        decomposition_ts_df['i_EPS'].iloc[-1] / decomposition_ts_df['i_EPS'].iloc[0]) / n_freq * freq_scale
    annual_unrebalanced_PE_multipleExpansion_total_log = np.log(
        decomposition_ts_df['PER'].iloc[-1] / decomposition_ts_df['PER'].iloc[0]) / n_freq * freq_scale
    annual_MA_NOISE_growth_total_log = np.log(
        decomposition_ts_df['i_NOISE'].iloc[-1] / decomposition_ts_df['i_NOISE'].iloc[0]) / n_freq * freq_scale
    annual_MA_unrebalanced_PE_multipleExpansion_total_log = np.log(
        decomposition_ts_df['MA_PER'].iloc[-1] / decomposition_ts_df['MA_PER'].iloc[0]) / n_freq * freq_scale
    annual_holding_EPS_growth_log = np.log(
        (decomposition_ts_df['holding_EARNINGS_change'] + 1).prod()) / n_freq * freq_scale
    annual_rebalancing_EPS_growth_log = np.log(
        (decomposition_ts_df['rebalancing_EARNINGS_change'] + 1).prod()) / n_freq * freq_scale
    annual_rebalancing_MA_EPS_growth_log = np.log(
        (decomposition_ts_df['rebalancing_MA_EARNINGS_change'] + 1).prod()) / n_freq * freq_scale
    annual_rebalancing_NOISE_growth_log = annual_rebalancing_EPS_growth_log - annual_rebalancing_MA_EPS_growth_log

    decomposition_sd = pd.Series(
        {'TR_sd': np.log(decomposition_ts_df['i_amount'] / decomposition_ts_df['i_amount'].shift()).std(),
         'EPS_sd': np.log(decomposition_ts_df['i_EPS'] / decomposition_ts_df['i_EPS'].shift()).std(),
         'unrebal_PER_sd': np.log(decomposition_ts_df['PER'] / decomposition_ts_df['PER'].shift()).std(),
         'holding_EPS_change_sd': np.log(decomposition_ts_df['holding_EARNINGS_change'] + 1).std(),
         'rebalancing_EPS_change_sd': np.log(decomposition_ts_df['rebalancing_EARNINGS_change'] + 1).std(),
         'rebalancing_MA_EPS_change_sd': np.log(decomposition_ts_df['rebalancing_MA_EARNINGS_change'] + 1).std(),
         'rebalancing_NOISE_change_sd': np.log(decomposition_ts_df['rebalancing_noise_change'] + 1).std()
         })

    decompsed_df = pd.Series({'annual_TR_log': annual_TR_log,
                              'annual_MA_NOISE_growth_total_log': annual_MA_NOISE_growth_total_log,
                              'annual_EPS_growth_total_log': annual_EPS_growth_total_log,
                              'annual_unrebalanced_PE_multipleExpansion_total_log': annual_unrebalanced_PE_multipleExpansion_total_log,
                              'annual_holding_EPS_growth_log': annual_holding_EPS_growth_log,
                              'annual_rebalancing_EPS_growth_log': annual_rebalancing_EPS_growth_log,
                              'annual_rebalancing_MA_EPS_growth_log': annual_rebalancing_MA_EPS_growth_log,
                              'annual_rebalancing_NOISE_growth_log': annual_rebalancing_NOISE_growth_log
                              })
    decomposed_df = pd.concat([decompsed_df, decomposition_sd], axis=0)

    result = {'decomposed_df': decomposed_df,
              'decomposition_ts_df': decomposition_ts_df}
    return result


import time

start = time.time()
test = get_factor_decomposition(univ_factor_df, topN_weight, all_infocode_roc_monthly, trading_cost,
                                rebalancing_freq='m')
print("time :", time.time() - start)

### FactorGen_fun.R 833 line 까지
##### R -> factorGen_fun.R 519줄 solution_holding_weight_dt 길이가 univ_factor_dt_monthly 길이보다 길다?? univ_factor_dt_monthly가 더 길어야 하는거 아닌가..?
#### polars 데이터 합치는거 우째하노..\


##########################
from data_process import data_read
import numpy as np
import pandas as pd
from backtest_process import calculate_weight

# univ_factor_df_i['Price/Earnings Ratio - Current'][univ_factor_df_i['Price/Earnings Ratio - Current'] < 0] = \
# univ_factor_df_i['Price/Earnings Ratio - Current'].median()

dict_of_data = data_read.read_pickle(
    path='C:/Users/doomoolmori/factor_attribution_process/data/us',
    name='us_dict_of_data.pickle')

price_df = dict_of_data['RI']
# univ_factor_df = univ_factor_df[~univ_factor_df['short_MA'].is_null()]

price_df = data_read.read_csv_(path=path + '/data/us',
                               name='adj_ri.csv')
price_df.set_index('date_', inplace=True)

"""
data_read.read_csv_(
path='C:/Users/doomoolmori/factor_attribution_process/data/us',
name='adj_ri.csv')
price_df.set_index('date_', inplace=True)

"""
pe_df = dict_of_data['Price/Earnings Ratio - Current']

price_arr = price_df.to_numpy(dtype=np.float32)
median_pe_arr = pe_df.median(1).to_numpy(dtype=np.float32)
pe_arr = pe_df.to_numpy(dtype=np.float32)

minus_pe_arr = (pe_arr < 0) * median_pe_arr.reshape((len(median_pe_arr), 1))
adj_pe_arr = pe_arr * (pe_arr > 0) + minus_pe_arr
eps_arr = price_arr / adj_pe_arr

pct = price_df.pct_change().shift(-1)

stg_name = '0-(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)'
picked_stock = data_read.read_pickle(path=path + '/data/us/strategy_weight',
                                     name=f'{stg_name}_picked.pickle')

series_df = data_read.read_csv_(path=path + '/data/us/strategy_stats',
                                name='bulk_backtest_m.csv')
series_df.set_index('date_', inplace=True)
sample_series = series_df[stg_name] * 100

weight = calculate_weight.make_equal_weight_arr(
    stock_pick=picked_stock,
    number_of_columns=len(adj_ri.columns),
    number_of_raws=len(adj_ri.index))

weight_df = pd.DataFrame(weight,
                         columns=adj_ri.columns,
                         index=adj_ri.index)

sample_series = ((price_df.pct_change().shift(-1) * weight_df).sum(1) + 1).cumprod() * 100

sample_series = sample_series.shift(1)
sample_series[0] = 100


quantity = (sample_series.to_numpy().reshape(234, 1) * weight_df.to_numpy()) / price_arr
after_quantity = quantity * (1 + price_df.pct_change().shift(-1).fillna(method='ffill').to_numpy())

holding_i_EPS = np.nansum(after_quantity[:, :] * eps_arr[:, :], 1)
i_EPS = np.nansum(quantity[:, :] * eps_arr[:, :], 1)
holding_i_amount = sample_series.shift(-1).fillna(method='bfill').to_numpy()
i_amount = sample_series.to_numpy()

holding_i_PRICE = np.nansum(after_quantity * price_arr, 1)
holding_i_PER = (holding_i_PRICE/holding_i_EPS)

i_PRICE = np.nansum(quantity * price_arr, 1)
PER = (i_PRICE/i_EPS)



eps = i_holdings['RI'] * (1 / i_holdings['Price/Earnings Ratio - Current'])

quan_df = weight_df / weight_df

np.nansum((eps_arr * quan_df.to_numpy())[0, :])
np.nansum((price_arr * quan_df.to_numpy())[0, :]) / np.nansum((eps_arr * quan_df.to_numpy())[0, :])

####
if i == topN_weight.index[0]:
    i_rebal_before_holdings['end_weight'] = i_rebal_before_holdings['begin_weight']
    i_rebal_before_holdings['holding_end_amount'] = i_rebal_before_holdings['begin_amount']

if len(i_rebal_before_holdings) > 0:
    i_rebal_before_holdings['Noise'] = i_rebal_before_holdings['RI'] / i_rebal_before_holdings['short_MA']
    i_rebal_before_holdings['shares'] = i_rebal_before_holdings['holding_end_amount'] / i_rebal_before_holdings[
        'RI']
    i_rebal_before_holdings['ma_shares'] = i_rebal_before_holdings['holding_end_amount'] / \
                                           i_rebal_before_holdings['short_MA']
    i_rebal_before_holdings['EARNINGS'] = i_rebal_before_holdings['Market Capitalization - Current (U.S.$)'] / \
                                          i_rebal_before_holdings['Price/Earnings Ratio - Current']
    i_rebal_before_holdings['total_shares'] = i_rebal_before_holdings[
                                                  'Market Capitalization - Current (U.S.$)'] / \
                                              i_rebal_before_holdings['RI']
    i_rebal_before_holdings['EPS'] = i_rebal_before_holdings['RI'] * (
            1 / i_rebal_before_holdings['Price/Earnings Ratio - Current'])
    i_rebal_before_holdings['MA_EPS'] = i_rebal_before_holdings['short_MA'] * (
            1 / i_rebal_before_holdings['Price/Earnings Ratio - Current'])
    i_rebal_before_holdings['PER_'] = i_rebal_before_holdings['Market Capitalization - Current (U.S.$)'] * (
            1 / i_rebal_before_holdings['EARNINGS'])

    holding_i_amount = i_rebal_before_holdings['holding_end_amount'].sum()
    holding_i_EARNINGS = i_rebal_before_holdings['EARNINGS'].sum()
    holding_i_PRICE = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['RI']).sum()
    holding_i_short_MA = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['short_MA']).sum()
    holding_i_EPS = (i_rebal_before_holdings['shares'] * i_rebal_before_holdings['EPS']).sum()
    holding_i_MA_EPS = (i_rebal_before_holdings['ma_shares'] * i_rebal_before_holdings['EPS']).sum()

i_holdings['Noise'] = i_holdings['RI'] / i_holdings['short_MA']
i_holdings['shares'] = i_holdings['begin_amount'] / i_holdings['RI']
i_holdings['ma_shares'] = i_holdings['begin_amount'] / i_holdings['short_MA']
i_holdings['EARNINGS'] = i_holdings['Market Capitalization - Current (U.S.$)'] / i_holdings[
    'Price/Earnings Ratio - Current']
i_holdings['total_shares'] = i_holdings['Market Capitalization - Current (U.S.$)'] / i_holdings['RI']
i_holdings['EPS'] = i_holdings['RI'] * (1 / i_holdings['Price/Earnings Ratio - Current'])
i_holdings['MA_EPS'] = i_holdings['short_MA'] * (1 / i_holdings['Price/Earnings Ratio - Current'])
i_holdings['PER_'] = i_holdings['Market Capitalization - Current (U.S.$)'] * (1 / i_holdings['EARNINGS'])

i_amount = i_holdings['begin_amount'].sum()
i_PRICE = (i_holdings['shares'] * i_holdings['RI']).sum()
i_short_MA = (i_holdings['shares'] * i_holdings['short_MA']).sum()
i_EPS = (i_holdings['shares'] * i_holdings['EPS']).sum()
i_MA_EPS = (i_holdings['ma_shares'] * i_holdings['EPS']).sum()

decomposition_dic[i] = pd.Series({'date_': i, 'i_amount': i_amount, 'holding_i_amount': holding_i_amount,
                                  'holding_i_EPS': holding_i_EPS, 'holding_i_MA_EPS': holding_i_MA_EPS,
                                  'holding_i_NOISE': holding_i_PRICE / holding_i_short_MA,
                                  'holding_i_PER': holding_i_PRICE / holding_i_EPS,
                                  'holding_i_MA_PER': holding_i_short_MA / holding_i_EPS,
                                  'i_EPS': i_EPS,
                                  'i_MA_EPS': i_MA_EPS,
                                  'i_NOISE': i_PRICE / i_short_MA,
                                  'PER': i_PRICE / i_EPS,
                                  'MA_PER': i_short_MA / i_EPS})


## 차이나는 것은 ADJ_RI 와 RI 라 추정 아래 두 코드는 같으나 RI는 다르다는 것을 확인.
# i_holdings['Price/Earnings Ratio - Current'].sum()
# np.nansum((quan_df.to_numpy() * adj_pe_arr)[0, :])


class NewSharpDecompose:

    def __init__(self):
        print('start NewSharpDecompose')
