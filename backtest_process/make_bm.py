from data_process import calculation_rank
from backtest_process import calculate_series
import pandas as pd
from dataturbo import DataTurbo


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


if __name__ == "__main__":
    from data_process import pre_processing

    pre_process = pre_processing.PreProcessing(universe='korea')
    bm = BM(pre_process=pre_process)
    bm_series = bm.get_bm_series()
