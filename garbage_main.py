from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import serial_stats
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import make_bm
import time

start = time.time()

if __name__ == "__main__":
    rebal = 'm'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'korea'
    garbage_list = [False, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    """
    pre_process = pre_processing.PreProcessing(universe=universe, n_top=n_top, garbage=False)
    # 전처리 (dict_of_rank.pickle 완료시 돌릴필요없어요)
    for garbage in garbage_list:
        if garbage != False:
            pre_process.garbage = garbage
            pre_process.garbage_setting(garbage=garbage)
            data_read.make_path(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'])
            data_read.make_path(path=pre_process.path_dict['STRATEGY_STATS_PATH'])
            pre_process._rank_data_processing()
    """
    # 백테스팅
    for garbage in garbage_list[:]:
        pre_process = pre_processing.PreProcessing(universe=universe, n_top=n_top, garbage=garbage)

        # 계산 완료되면 돌릴 필요 없어요 filter마다 stock_picking
        for filter_number in list(pre_process.filter_info['number'])[:1]:
            picking = stock_picking.StockPick(
                pre_process=pre_process,
                filter_number=filter_number,
                asyncio_=True)
            picking.do_stock_pick()
            #picking.do_stock_pick_quantile(quantile=5)

        # picking_data_load
        picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)

        # 계산 완료되면 돌릴 필요 없어요 (back_test)
        calculate_series.calculate_series_backtest(
            pre_process=pre_process,
            picking_dict=picking_dict,
            cost=cost,
            rebal=rebal,
            asyncio_=True)

        bulk_backtest_df = data_read.bulk_backtest_df(
            strategy_name_list=list(picking_dict.keys()),
            raw_path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
            save_path=pre_process.path_dict['STRATEGY_STATS_PATH'],
            rebal=rebal)

        bm = make_bm.BM(pre_process)
        bm_series = bm.get_bm_series(cost=cost, rebal=rebal)
        rf_series = bm.get_rf_series(rebal=rebal, index=list(bm_series.index)) / 100

        start = time.time()
        stats = serial_stats.StatsSeries(
            bulk_backtest_df=bulk_backtest_df,
            bm_series=bm_series,
            rf_series=rf_series)
        stats.set_optional(
            rebal=rebal,
            in_sample_year=10,
            out_sample_year=3,
            path=pre_process.path_dict['STRATEGY_STATS_PATH'])
        stats.loop_make_stats_df()
        print("time :", time.time() - start)

        (1 + (stats.stats.ret_arr - stats.stats.bm_ret_arr - stats.stats.rf_arr).mean()) ** 4  - 1

