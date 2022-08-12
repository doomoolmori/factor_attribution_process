from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import serial_stats
import time

start = time.time()
if __name__ == "__main__":
    rebal = 'q'
    cost = 0.003
    n_top = 20
    universe = 'us'
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
    stats = serial_stats.StatsSeries(
        bulk_backtest_df=bulk_backtest_df,
        bm_series=bm_series)
    stats.set_optional(
        rebal=rebal,
        in_sample_year=10,
        out_sample_year=1,
        path=pre_process.path_dict['STRATEGY_STATS_PATH'])
    stats.loop_make_stats_df()
print("time :", time.time() - start)
