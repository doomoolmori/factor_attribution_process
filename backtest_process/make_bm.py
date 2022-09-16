from data_process import calculation_rank
from backtest_process import calculate_series
import pandas as pd
from dataturbo import DataTurbo
from backtest_process import stock_picking
import numpy as np

class BM:
    def __init__(self, pre_process):
        self.pre_process = pre_process

    def get_bm_series(self, cost: int = 0.003, rebal: str = 'q'):
        mkt_df = self.pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
        mkt_filter_df = calculation_rank.under_mkt_filter(mkt_df=mkt_df)
        under_cut_mkt_df = mkt_df[mkt_filter_df]
        total_mkt_series = under_cut_mkt_df.sum(1)
        mkt_weight = (under_cut_mkt_df / (total_mkt_series.to_numpy().reshape(len(total_mkt_series), 1)))
        mkt_weight = mkt_weight.fillna(0).to_numpy()

        cost_arr = calculate_series.get_cost_arr(
            stock_weight=mkt_weight,
            cost=cost)
        adj_pct_arr = self.pre_process.adj_ri.pct_change().shift(-1).fillna(0).to_numpy()
        return_arr = calculate_series.get_return_arr(
            adj_pct_arr=adj_pct_arr,
            stock_weight=mkt_weight)
        series = calculate_series.get_backtest_series(
            return_arr=return_arr,
            cost_arr=cost_arr)
        month_index = calculate_series.get_month_index(
            time_index=self.pre_process.adj_ri.index,
            rebal=rebal)
        return pd.Series(series[month_index],
                         index=self.pre_process.adj_ri.index[month_index])

    def get_rf_series(self, rebal: str, index: list):
        api = DataTurbo()
        rf = api.get_adj_price(['USGG10YR Index'], '1990-01-01', '2100-10-31')
        if rebal == 'm':
            rf = rf.resample('M').last() / 12
        elif rebal == 'q':
            rf = rf.resample('M').last() / 4
        return rf.loc[index]

    # new sharp 구하는 것으로 하위 필터 제거
    def get_bm_weight(self):
        mkt_df = self.pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
        mkt_df[mkt_df == 0] = np.nan
        total_mkt_series = mkt_df.sum(1)
        mkt_weight = (mkt_df / (total_mkt_series.to_numpy().reshape(len(total_mkt_series), 1)))
        mkt_weight = mkt_weight.fillna(0).to_numpy()
        return mkt_weight

    # new sharp 구하는 것으로 하위 필터 제거
    def get_small_weight(self):
        mkt_df = self.pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
        rank_mkt = mkt_df.rank(1, ascending=True)
        small_weight = stock_picking.stock_picker_q(rank_mkt.to_numpy(), 5, 1)
        small_weight = small_weight / small_weight.sum(1).reshape(len(small_weight), 1)
        return small_weight

    # new sharp 구하는 것으로 하위 필터 제거
    def get_large_weight(self):
        mkt_df = self.pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
        rank_mkt = mkt_df.rank(1, ascending=True)
        large_weight = stock_picking.stock_picker_q(rank_mkt.to_numpy(), 5, 5)
        large_weight = large_weight / large_weight.sum(1).reshape(len(large_weight), 1)
        return large_weight

    # new sharp 구하는 것으로 하위 필터 제거
    def get_growth_weight(self):
        pb_df = self.pre_process.dict_of_pandas['Price/Book Value Ratio - Current']
        rank_pb = pb_df.rank(1, ascending=True)
        growth_weight = stock_picking.stock_picker_q(rank_pb.to_numpy(), 5, 5)
        growth_weight = growth_weight / growth_weight.sum(1).reshape(len(growth_weight), 1)
        return growth_weight

    # new sharp 구하는 것으로 하위 필터 제거
    def get_value_weight(self):
        pb_df = self.pre_process.dict_of_pandas['Price/Book Value Ratio - Current']
        rank_pb = pb_df.rank(1, ascending=True)
        value_weight = stock_picking.stock_picker_q(rank_pb.to_numpy(), 5, 1)
        value_weight = value_weight / value_weight.sum(1).reshape(len(value_weight), 1)
        return value_weight

    def get_series(self,
                   weight: pd.DataFrame,
                   cost: float,
                   rebal: str):
        cost_arr = calculate_series.get_cost_arr(
            stock_weight=weight,
            cost=cost)
        adj_pct_arr = self.pre_process.adj_ri.pct_change().shift(-1).fillna(0).to_numpy()
        return_arr = calculate_series.get_return_arr(
            adj_pct_arr=adj_pct_arr,
            stock_weight=weight)
        series = calculate_series.get_backtest_series(
            return_arr=return_arr,
            cost_arr=cost_arr)
        month_index = calculate_series.get_month_index(
            time_index=self.pre_process.adj_ri.index,
            rebal=rebal)
        return pd.Series(series[month_index],
                         index=self.pre_process.adj_ri.index[month_index])

if __name__ == "__main__":
    from data_process import pre_processing

    pre_process = pre_processing.PreProcessing(universe='us')
    bm = BM(pre_process=pre_process)
    bm_series = bm.get_bm_series()
    bm_weight = bm.get_bm_weight()
    from data_process import data_read
    data_read.save_to_pickle(bm_weight, path='C:/Users/doomoolmori/factor_attribution_process', name='bm.pickle')

    value_weight = bm.get_value_weight()
    value_series = bm.get_series(value_weight, 0.003, 'm')

    growth_weight = bm.get_growth_weight()
    growth_series = bm.get_series(growth_weight, 0.003, 'm')

    small_weight = bm.get_small_weight()
    small_series = bm.get_series(small_weight, 0.003, 'm')

    large_weight = bm.get_large_weight()
    large_series = bm.get_series(large_weight, 0.003, 'm')