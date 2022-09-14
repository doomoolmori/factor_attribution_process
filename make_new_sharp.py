# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:38:11 2022

@author: doomoolmori
"""

from data_process import pre_processing
from data_process import data_read
from backtest_process import calculate_weight
import pandas as pd
import numpy as np

"""
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
name = "2022-07-22_cosmos-univ-with-factors_with-finval_global_monthly.csv"
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
col_list = ['date_', 'infocode', 'ticker', 'RI', 'RI_ma_20d','Market Capitalization - Current (U.S.$)',
            'Price/Book Value Ratio - Current', 'Price/Earnings Ratio - Current',
            'Net Sales Or Revenues (Q)', 'Ebit & Depreciation (Q)']


def get_factor_decomposition(univ_factor_df, topN_weight, all_infocode_roc_monthly, trading_cost, rebalancing_freq='m'):
    univ_factor_df = univ_factor_df[col_list]
    univ_factor_df['short_MA'] = univ_factor_df['RI_ma_20d']
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

    decomposition_ts_df.to_csv('simon_.csv')

    return result


import time

start = time.time()
test = get_factor_decomposition(univ_factor_df, topN_weight, all_infocode_roc_monthly, trading_cost,
                                rebalancing_freq='m')
print("time :", time.time() - start)

### FactorGen_fun.R 833 line 까지
##### R -> factorGen_fun.R 519줄 solution_holding_weight_dt 길이가 univ_factor_dt_monthly 길이보다 길다?? univ_factor_dt_monthly가 더 길어야 하는거 아닌가..?
#### polars 데이터 합치는거 우째하노..\
"""


##########################

# TODO 중복되는 함수들 많으나 우선은 편의를 위해 모두 사용
class Calc:
    @staticmethod
    def calculate_i_PRICE(
            quantity_arr: np.array,
            price_arr: np.array
    ) -> np.array:
        return np.nansum(
            quantity_arr * price_arr,
            axis=1)

    @staticmethod
    def calculate_i_short_MA(
            quantity_arr: np.array,
            ma_price_arr: np.array
    ) -> np.array:
        return np.nansum(
            quantity_arr * ma_price_arr,
            axis=1)

    @staticmethod
    def calculate_holding_i_PRICE(
            after_quantity_arr: np.array,
            price_arr: np.array
    ) -> np.array:
        holding_i_PRICE = np.nansum(
            after_quantity_arr * price_arr,
            axis=1)
        holding_i_PRICE[-1] = holding_i_PRICE[-2]
        return holding_i_PRICE

    @staticmethod
    def calculate_holding_i_short_MA(
            after_quantity_arr: np.array,
            ma_price_arr: np.array
    ) -> np.array:
        holding_i_short_MA = np.nansum(
            after_quantity_arr * ma_price_arr,
            axis=1)
        holding_i_short_MA[-1] = holding_i_short_MA[-2]
        return holding_i_short_MA

    @staticmethod
    def calculate_i_amount(
            series_arr: np.array
    ) -> np.array:
        return series_arr.copy()

    @staticmethod
    def calculate_holding_i_amount(
            series_arr: np.array
    ) -> np.array:
        holding_i_amount = shift_minus(
            arr=series_arr,
            num=-1,
            fill_value=np.nan)
        holding_i_amount[-1] = holding_i_amount[-2]
        return holding_i_amount

    @staticmethod
    def calculate_holing_i_EPS(
            after_quantity: np.array,
            eps_arr: np.array
    ) -> np.array:
        holding_i_EPS = np.nansum(
            after_quantity * eps_arr,
            axis=1)
        holding_i_EPS[-1] = holding_i_EPS[-2]
        return holding_i_EPS

    @staticmethod
    def calculate_holding_i_MA_EPS(
            after_ma_quantity: np.array,
            eps_arr: np.array
    ) -> np.array:
        holding_i_MA_EPS = np.nansum(
            after_ma_quantity * eps_arr,
            axis=1)
        holding_i_MA_EPS[-1] = holding_i_MA_EPS[-2]
        return holding_i_MA_EPS

    @staticmethod
    def calculate_holding_i_NOISE(
            holding_i_PRICE: np.array,
            holding_i_short_MA: np.array
    ) -> np.array:
        return holding_i_PRICE / holding_i_short_MA

    @staticmethod
    def calculate_holding_i_PER(
            holding_i_PRICE: np.array,
            holding_i_EPS: np.array
    ) -> np.array:
        return holding_i_PRICE / holding_i_EPS

    @staticmethod
    def calculate_holding_i_MA_PER(
            holding_i_short_MA: np.array,
            holding_i_EPS: np.array
    ) -> np.array:
        return holding_i_short_MA / holding_i_EPS

    @staticmethod
    def calculate_i_EPS(
            quantity_arr: np.array,
            eps_arr: np.array
    ) -> np.array:
        return np.nansum(
            quantity_arr * eps_arr,
            axis=1)

    @staticmethod
    def calculate_i_MA_EPS(
            ma_quantity_arr: np.array,
            eps_arr: np.array
    ) -> np.array:
        return np.nansum(
            ma_quantity_arr * eps_arr,
            axis=1)

    @staticmethod
    def calculate_i_NOISE(
            i_PRICE: np.array,
            i_short_MA: np.array
    ) -> np.array:
        return i_PRICE / i_short_MA

    @staticmethod
    def calculate_PER(
            i_PRICE: np.array,
            i_EPS: np.array
    ) -> np.array:
        return (i_PRICE / i_EPS)

    @staticmethod
    def calculate_MA_PER(
            i_short_MA: np.array,
            i_EPS: np.array
    ) -> np.array:
        return i_short_MA / i_EPS

    @staticmethod
    def calculate_EPS_shift(
            i_EPS: np.array
    ) -> np.array:
        return shift_plus(
            arr=i_EPS,
            num=1,
            fill_value=np.nan)

    @staticmethod
    def calculate_holding_NOISE_change(
            holding_i_NOISE: np.array,
            i_NOISE: np.array
    ) -> np.array:
        _shift = shift_plus(
            arr=i_NOISE,
            num=1,
            fill_value=np.nan)
        return holding_i_NOISE / _shift - 1

    @staticmethod
    def calculate_holding_EARNINGS_change(
            holding_i_EPS: np.array,
            EPS_shift: np.array
    ) -> np.array:
        return holding_i_EPS / EPS_shift - 1

    @staticmethod
    def calculate_holding_MA_EARNINGS_change(
            holding_i_MA_EPS: np.array,
            i_MA_EPS: np.array
    ) -> np.array:
        _shift = shift_plus(
            arr=i_MA_EPS,
            num=1,
            fill_value=np.nan)
        return holding_i_MA_EPS / _shift - 1

    @staticmethod
    def calculate_holding_PER_change(
            holding_i_PER: np.array,
            PER: np.array
    ) -> np.array:
        _shift = shift_plus(
            arr=PER,
            num=1,
            fill_value=np.nan)
        return holding_i_PER / _shift - 1

    @staticmethod
    def calculate_holding_MA_PER_change(
            holding_i_MA_PER: np.array,
            MA_PER: np.array
    ) -> np.array:
        _shift = shift_plus(
            arr=MA_PER,
            num=1,
            fill_value=np.nan)
        return holding_i_MA_PER / _shift - 1

    @staticmethod
    def calculate_rebalancing_EARNINGS_change(
            i_EPS: np.array,
            holding_i_EPS: np.array
    ) -> np.array:
        return i_EPS / holding_i_EPS - 1

    @staticmethod
    def calculate_rebalancing_MA_EARNINGS_change(
            i_MA_EPS: np.array,
            holding_i_EPS: np.array
    ) -> np.array:
        return i_MA_EPS / holding_i_EPS - 1

    @staticmethod
    def calculate_rebalancing_NOISE_change(
            i_NOISE: np.array,
            holding_i_NOISE: np.array
    ) -> np.array:
        return i_NOISE / holding_i_NOISE - 1

    @staticmethod
    def calculate_rebalancing_PER_change(
            MA_PER: np.array,
            holding_i_MA_PER: np.array
    ) -> np.array:
        return MA_PER / holding_i_MA_PER - 1

    @staticmethod
    def calculate_rebalancing_noise_change(
            rebalancing_EARNINGS_change: np.array,
            rebalancing_MA_EARNINGS_change: np.array
    ) -> np.array:
        return rebalancing_EARNINGS_change - \
               rebalancing_MA_EARNINGS_change


class NewSharpDecompose:
    def __init__(self, pre_process):
        print('start NewSharpDecompose')
        self.pre_process = pre_process
        self.default_setting_bulk_df()
        self.default_setting_dict()
        self.default_setting_price()
        self.default_setting_ma_price()
        self.default_setting_pe()
        self.default_setting_pct()

    def update_setting(self, stg_name: str = '0-(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)'):
        self.set_weight_arr(stg_name=stg_name)
        self.set_series_arr(stg_name=stg_name)

        self.get_eps_arr()
        self.get_quantity_arr()
        self.get_after_quantity_arr()
        self.get_ma_quantity_arr()
        self.get_after_ma_quantity_arr()
        print('update_setting')

    def get_eps_arr(self):
        self.eps_arr = self.price_arr / self.adj_pe_arr

    def get_quantity_arr(self):
        self.quantity_arr = (self.sample_series.reshape(236, 1) * self.weight_arr) \
                            / self.price_arr

    def get_after_quantity_arr(self):
        self.after_quantity_arr = self.quantity_arr * (1 + self.after_pct_arr)

    def get_ma_quantity_arr(self):
        self.ma_quantity_arr = (self.sample_series.reshape(236, 1) * self.weight_arr) \
                               / self.ma_price_arr

    def get_after_ma_quantity_arr(self):
        self.after_ma_quantity_arr = self.ma_quantity_arr * (1 + self.after_pct_arr)

    def default_setting_bulk_df(self):
        self.bulk_df = data_read.read_csv_(
            path=self.pre_process.path_dict['STRATEGY_STATS_PATH'],
            name='bulk_backtest_m.csv')
        self.bulk_df.set_index('date_', inplace=True)

    def default_setting_dict(self):
        self.dict_of_data = self.pre_process.dict_of_pandas

    def default_setting_price(self):
        self.price_df = self.pre_process.adj_ri
        self.price_arr = self.price_df.to_numpy(dtype=np.float32)

    def default_setting_ma_price(self):
        self.ma_price_df = self.dict_of_data['RI_ma_20d']
        self.ma_price_arr = self.ma_price_df.to_numpy(dtype=np.float32)

    def default_setting_pe(self):
        pe_df = self.dict_of_data['Price/Earnings Ratio - Current']
        median_pe_arr = pe_df.median(1).to_numpy(dtype=np.float32)
        pe_arr = pe_df.to_numpy(dtype=np.float32)
        minus_pe_arr = (pe_arr < 0) * median_pe_arr.reshape((len(median_pe_arr), 1))
        self.adj_pe_arr = pe_arr * (pe_arr > 0) + minus_pe_arr

    def default_setting_pct(self):
        self.after_pct_arr = self.price_df.pct_change().shift(-1).to_numpy()

    def set_series_arr(self, stg_name: str):
        sample_series = self.bulk_df[stg_name] * 100
        self.sample_series = sample_series.to_numpy()
        """
        sample_series = (np.nansum(self.after_pct_arr * self.weight_arr, 1) + 1).cumprod() * 100
        sample_series = shift_plus(sample_series, 1, np.nan)
        sample_series[0] = 100
        self.sample_series = sample_series
        """

    def set_weight_arr(self, stg_name: str):
        picked_stock = data_read.read_pickle(
            path=self.pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
            name=f'{stg_name}_picked.pickle')
        self.weight_arr = calculate_weight.make_equal_weight_arr(
            stock_pick=picked_stock,
            number_of_columns=len(self.price_df.columns),
            number_of_raws=len(self.price_df.index))


def shift_plus(
        arr: np.array,
        num: int,
        fill_value: any
) -> np.array:
    assert num > 0, print('num error')
    result = np.empty_like(arr)
    result[:num] = fill_value
    result[num:] = arr[:-num]
    return result


def shift_minus(
        arr: np.array,
        num: int,
        fill_value: any
) -> np.array:
    assert num < 0, print('num error')
    result = np.empty_like(arr)
    result[num:] = fill_value
    result[:num] = arr[-num:]
    return result


def get_new_sharp_path(
        garbage,
        path: str
) -> str:
    if garbage == False:
        new_sharp_path = f'{path}/strategy_new_sharp'
    else:
        new_sharp_path = f'{path}/strategy_new_sharp_garbage_{garbage}'
    return new_sharp_path

def get_new_sharp_data(
    new_sharp
) -> pd.DataFrame:
    i_PRICE = Calc.calculate_i_PRICE(
        quantity_arr=new_sharp.quantity_arr,
        price_arr=new_sharp.price_arr)
    i_short_MA = Calc.calculate_i_short_MA(
        quantity_arr=new_sharp.quantity_arr,
        ma_price_arr=new_sharp.ma_price_arr)
    holding_i_PRICE = Calc.calculate_holding_i_PRICE(
        after_quantity_arr=new_sharp.after_quantity_arr,
        price_arr=new_sharp.price_arr)
    holding_i_short_MA = Calc.calculate_holding_i_short_MA(
        after_quantity_arr=new_sharp.after_quantity_arr,
        ma_price_arr=new_sharp.ma_price_arr)
    i_amount = Calc.calculate_i_amount(
        series_arr=new_sharp.sample_series)
    holding_i_amount = Calc.calculate_holding_i_amount(
        series_arr=new_sharp.sample_series)
    holding_i_EPS = Calc.calculate_holing_i_EPS(
        after_quantity=new_sharp.after_quantity_arr,
        eps_arr=new_sharp.eps_arr)
    holding_i_MA_EPS = Calc.calculate_holding_i_MA_EPS(
        after_ma_quantity=new_sharp.after_ma_quantity_arr,
        eps_arr=new_sharp.eps_arr)
    holding_i_NOISE = Calc.calculate_holding_i_NOISE(
        holding_i_PRICE=holding_i_PRICE,
        holding_i_short_MA=holding_i_short_MA)
    holding_i_PER = Calc.calculate_holding_i_PER(
        holding_i_PRICE=holding_i_PRICE,
        holding_i_EPS=holding_i_EPS)
    holding_i_MA_PER = Calc.calculate_holding_i_MA_PER(
        holding_i_short_MA=holding_i_short_MA,
        holding_i_EPS=holding_i_EPS)
    i_EPS = Calc.calculate_i_EPS(
        quantity_arr=new_sharp.quantity_arr,
        eps_arr=new_sharp.eps_arr)
    i_MA_EPS = Calc.calculate_i_MA_EPS(
        ma_quantity_arr=new_sharp.ma_quantity_arr,
        eps_arr=new_sharp.eps_arr)
    i_NOISE = Calc.calculate_i_NOISE(
        i_PRICE=i_PRICE,
        i_short_MA=i_short_MA)
    PER = Calc.calculate_PER(
        i_PRICE=i_PRICE,
        i_EPS=i_EPS)
    MA_PER = Calc.calculate_MA_PER(
        i_short_MA=i_short_MA,
        i_EPS=i_EPS)
    EPS_shift = Calc.calculate_EPS_shift(
        i_EPS=i_EPS)
    holding_NOISE_change = Calc.calculate_holding_NOISE_change(
        holding_i_NOISE=holding_i_NOISE,
        i_NOISE=i_NOISE)
    holding_EARNINGS_change = Calc.calculate_holding_EARNINGS_change(
        holding_i_EPS=holding_i_EPS,
        EPS_shift=EPS_shift)
    holding_MA_EARNINGS_change = Calc.calculate_holding_MA_EARNINGS_change(
        holding_i_MA_EPS=holding_i_MA_EPS,
        i_MA_EPS=i_MA_EPS)
    holding_PER_change = Calc.calculate_holding_PER_change(
        holding_i_PER=holding_i_PER,
        PER=PER)
    holding_MA_PER_change = Calc.calculate_holding_MA_PER_change(
        holding_i_MA_PER=holding_i_MA_PER,
        MA_PER=MA_PER)
    rebalancing_EARNINGS_change = Calc.calculate_rebalancing_EARNINGS_change(
        i_EPS=i_EPS,
        holding_i_EPS=holding_i_EPS)
    rebalancing_MA_EARNINGS_change = Calc.calculate_rebalancing_MA_EARNINGS_change(
        i_MA_EPS=i_MA_EPS,
        holding_i_EPS=holding_i_EPS)
    rebalancing_NOISE_change = Calc.calculate_rebalancing_NOISE_change(
        i_NOISE=i_NOISE,
        holding_i_NOISE=holding_i_NOISE)
    rebalancing_PER_change = Calc.calculate_rebalancing_PER_change(
        MA_PER=MA_PER,
        holding_i_MA_PER=holding_i_MA_PER)
    rebalancing_noise_change = Calc.calculate_rebalancing_noise_change(
        rebalancing_EARNINGS_change=rebalancing_EARNINGS_change,
        rebalancing_MA_EARNINGS_change=rebalancing_MA_EARNINGS_change)
    result_dict = {'i_amount':i_amount,
                   'holding_i_amount':holding_i_amount,
                   'holding_i_EPS':holding_i_EPS,
                   'holding_i_MA_EPS':holding_i_MA_EPS,
                   'holding_i_NOISE':holding_i_NOISE,
                   'holding_i_PER':holding_i_PER,
                   'holding_i_MA_PER':holding_i_MA_PER,
                   'i_EPS':i_EPS,
                   'i_MA_EPS':i_MA_EPS,
                   'i_NOISE':i_NOISE,
                   'PER':PER,
                   'MA_PER':MA_PER,
                   'EPS_shift':EPS_shift,
                   'holding_NOISE_change':holding_NOISE_change,
                   'holding_EARNINGS_change':holding_EARNINGS_change,
                   'holding_MA_EARNINGS_change':holding_MA_EARNINGS_change,
                   'holding_PER_change':holding_PER_change,
                   'holding_MA_PER_change':holding_MA_PER_change,
                   'rebalancing_EARNINGS_change':rebalancing_EARNINGS_change,
                   'rebalancing_MA_EARNINGS_change':rebalancing_MA_EARNINGS_change,
                   'rebalancing_NOISE_change':rebalancing_NOISE_change,
                   'rebalancing_PER_change':rebalancing_PER_change,
                   'rebalancing_noise_change':rebalancing_noise_change}

    result_df = pd.DataFrame(
        result_dict,
        index=new_sharp.price_df.index)
    result_df.loc[result_df['i_EPS'] < 0, result_df.columns] = np.nan
    result_df.loc[result_df['holding_i_EPS'] < 0, result_df.columns] = np.nan
    return result_df.ffill()


if __name__ == "__main__":
    rebal = 'm'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'us'
    pre_process = pre_processing.PreProcessing(universe=universe, n_top=n_top)

    garbage = False

    new_sharp_path = get_new_sharp_path(
        garbage=garbage,
        path=pre_process.path_dict["DATA_PATH"])
    data_read.make_path(new_sharp_path)
    new_sharp = NewSharpDecompose(pre_process)
    bulk_df = new_sharp.bulk_df
    for stg_name in bulk_df.columns[:10]:
        print(stg_name)
        new_sharp.update_setting(stg_name=stg_name)
        new_sharp_df = get_new_sharp_data(new_sharp=new_sharp)
        data_read.save_to_pickle(
            any_=new_sharp_df,
            path=new_sharp_path,
            name=f'{stg_name}.pickle')
