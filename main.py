from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import serial_stats
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import make_bm
import time

if __name__ == "__main__":
    rebal = 'q'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'korea'
    pre_process = pre_processing.PreProcessing(universe=universe, n_top=20)

    # 계산 완료되면 돌릴 필요 없어요 filter마다 stock_picking
    start = time.time()
    for filter_number in (pre_process.filter_info['number']):
        stock_picking.StockPick(
            pre_process=pre_process,
            filter_number=filter_number,
            asyncio_=True)
    print("time :", time.time() - start)

    # picking_data_load
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)


    start = time.time()
    # 계산 완료되면 돌릴 필요 없어요 (exposure)
    calculate_exposure.calculate_series_exposure(
        pre_process=pre_process,
        picking_dict=picking_dict,
        asyncio_=True)
    print("time :", time.time() - start)

    start = time.time()
    # 계산 완료되면 돌릴 필요 없어요 (back_test)
    calculate_series.calculate_series_backtest(
        pre_process=pre_process,
        picking_dict=picking_dict,
        cost=cost,
        rebal=rebal,
        asyncio_=True)
    print("time :", time.time() - start)

    bulk_backtest_df = data_read.bulk_backtest_df(
        strategy_name_list=list(picking_dict.keys()),
        raw_path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        save_path=pre_process.path_dict['STRATEGY_STATS_PATH'],
        rebal=rebal)

    bm = make_bm.BM(pre_process)
    bm_series = bm.get_bm_series(cost=cost, rebal=rebal)

    start = time.time()
    stats = serial_stats.StatsSeries(
        bulk_backtest_df=bulk_backtest_df,
        bm_series=bm_series)
    stats.set_optional(
        rebal=rebal,
        in_sample_year=10,
        out_sample_year=0,
        path=pre_process.path_dict['STRATEGY_STATS_PATH'])
    stats.loop_make_stats_df()
    print("time :", time.time() - start)
