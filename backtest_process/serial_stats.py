import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import calculate_stats as cs


def get_timestamp(date):
    return pd.to_datetime(date)


def get_timedelta_timestamp(date, yearly_timedelta: int):
    return date + relativedelta(years=yearly_timedelta)


def get_sample_boolean_index(index, start_date, end_date):
    return (index >= start_date) & (index <= end_date)


def date_information_dict(
        index: list,
        in_sample_year: int,
        out_sample_year: int) -> dict:
    result = {}
    for i, time in enumerate(index):
        in_sample_start_date = get_timestamp(time)
        in_sample_end_date = get_timedelta_timestamp(
            date=in_sample_start_date,
            yearly_timedelta=in_sample_year)
        out_sample_start_date = in_sample_end_date
        out_sample_end_date = get_timedelta_timestamp(
            date=out_sample_start_date,
            yearly_timedelta=out_sample_year)
        if out_sample_end_date > index[-1]:
            break
        else:
            result[i] = {
                'in_start': in_sample_start_date,
                'in_end': in_sample_end_date,
                'out_start': out_sample_start_date,
                'out_end': out_sample_end_date}
    return result


class StatsSeries:
    def __init__(self, bulk_backtest_df, bm_series):
        self.bulk_backtest_df = bulk_backtest_df
        self.bulk_backtest_arr = bulk_backtest_df.to_numpy(dtype=np.float32)
        self.bm_series = bm_series
        self.bm_arr = bm_series.to_numpy(dtype=np.float32)
        self.index = pd.to_datetime(self.bulk_backtest_df.index)

    def set_optional(self, rebal, in_sample_year, out_sample_year, path):
        self.rebal = rebal
        self.in_sample_year = in_sample_year
        self.out_sample_year = out_sample_year
        self.path = path

    def set_common_information(self):
        self.stats.set_rebal(rebal=self.rebal)
        self.stats.set_backtest_arr(backtest_arr=self.in_sample_arr)
        self.stats.set_out_backtest_arr(out_backtest_arr=self.out_sample_arr)
        self.stats.set_bm_arr(bm_arr=self.in_sample_bm_arr)
        self.stats.set_out_bm_arr(out_bm_arr=self.out_sample_bm_arr)

    def set_in_sample(self, in_sample_start_date, in_sample_end_date):
        in_sample_index = get_sample_boolean_index(
            index=self.index,
            start_date=in_sample_start_date,
            end_date=in_sample_end_date)
        self.in_sample_arr = self.bulk_backtest_arr[in_sample_index, :].copy()
        self.in_sample_bm_arr = self.bm_arr[in_sample_index].copy()

    def set_out_sample(self, out_sample_start_date, out_sample_end_date):
        out_sample_index = get_sample_boolean_index(
            index=self.index,
            start_date=out_sample_start_date,
            end_date=out_sample_end_date)
        self.out_sample_arr = self.bulk_backtest_arr[out_sample_index, :].copy()
        self.out_sample_bm_arr = self.bm_arr[out_sample_index].copy()

    def set_date_information(self):
        self.date_information = date_information_dict(
            index=self.index,
            in_sample_year=self.in_sample_year,
            out_sample_year=self.out_sample_year)

    def loop_make_stats_df(self):
        self.stats = cs.Stats()
        self.set_date_information()

        for i in self.date_information.keys():
            in_sample_start_date = self.date_information[i]['in_start']
            in_sample_end_date = self.date_information[i]['in_end']
            out_sample_start_date = self.date_information[i]['out_start']
            out_sample_end_date = self.date_information[i]['out_end']
            self.set_in_sample(
                in_sample_start_date=in_sample_start_date,
                in_sample_end_date=in_sample_end_date)
            self.set_out_sample(
                out_sample_start_date=out_sample_start_date,
                out_sample_end_date=out_sample_end_date)
            stats_df = self.make_stats_df()
            stats_df['in_start'] = datetime.datetime.strftime(
                in_sample_start_date,
                "%Y-%m-%d")
            stats_df['in_end_out_start'] = datetime.datetime.strftime(
                in_sample_end_date,
                "%Y-%m-%d")
            stats_df['out_end'] = datetime.datetime.strftime(
                out_sample_end_date,
                "%Y-%m-%d")
            data_read.save_to_csv(df=stats_df,
                                  path=self.path,
                                  name=f'{stats_df["in_start"][0]}_{self.rebal}.csv')

    def make_stats_df(self) -> pd.DataFrame:
        self.set_common_information()
        list_ = []
        for i in range(0, len(self.bulk_backtest_arr.T)):
            self.stats.set_backtest_arr(
                backtest_arr=self.in_sample_arr[:, i])
            self.stats.set_out_backtest_arr(
                out_backtest_arr=self.out_sample_arr[:, i])
            self.stats.check()
            list_.append(list(cs.bulk_stats_dict(self.stats).values()))

        result = pd.DataFrame(np.array(list_, dtype=np.float16))
        result.columns = cs.bulk_stats_dict(self.stats).keys()
        result['strategy'] = self.bulk_backtest_df.columns
        return result


if __name__ == "__main__":
    rebal = 'q'
    cost = 0.003
    n_top = 20
    universe = 'korea'
    pre_process = pre_processing.PreProcessing(universe=universe)
    picking_dict = stock_picking.get_stock_picking_dict(
        pre_process=pre_process,
        n_top=n_top,
        asyncio_=True)

    # 계산 완료되면 돌릴 필요 없어요
    # calculate_exposure.calculate_series_exposure(pre_process=pre_process, asyncio_=True)
    # 계산 완료되면 돌릴 필요 없어요
    # calculate_series.calculate_series_backtest(pre_process=pre_process, cost=cost, rebal=rebal, asyncio_=True)

    bulk_backtest_df = data_read.bulk_backtest_df(
        strategy_name_list=list(picking_dict.keys()),
        raw_path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        save_path=pre_process.path_dict['STRATEGY_STATS_PATH'],
        rebal=rebal)

    bm_series = bulk_backtest_df.iloc[:, 0]
    stats_seires = StatsSeries(bulk_backtest_df, bm_series)
    stats_seires.set_optional(
        rebal=rebal,
        in_sample_year=10,
        out_sample_year=1,
        path=pre_process.path_dict['STRATEGY_STATS_PATH'])
    stats_seires.loop_make_stats_df()
