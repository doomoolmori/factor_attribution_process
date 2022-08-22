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
    rebal = 'q'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'korea'
    pre_process = pre_processing.PreProcessing(universe=universe, n_top=20)


    # 계산 완료되면 돌릴 필요 없어요 filter마다 stock_picking
    for filter_number in list(pre_process.filter_info['number'])[:1]:
        stock_picking.StockPick(
            pre_process=pre_process,
            filter_number=filter_number,
            asyncio_=True)

    # picking_data_load
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)

    # 계산 완료되면 돌릴 필요 없어요 (exposure)
    calculate_exposure.calculate_series_exposure(
        pre_process=pre_process,
        picking_dict=picking_dict,
        asyncio_=True)

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

    start = time.time()
    stats = serial_stats.StatsSeries(
        bulk_backtest_df=bulk_backtest_df,
        bm_series=bm_series)
    stats.set_optional(
        rebal=rebal,
        in_sample_year=10,
        out_sample_year=3,
        path=pre_process.path_dict['STRATEGY_STATS_PATH'])
    stats.loop_make_stats_df()
    print("time :", time.time() - start)


    """
    이하는 sample
    """
    """
    ## 전략의 exposure
    import pandas as pd
    exposure = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                     name='0-(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)_exposure.pickle')
    exposure_df = pd.DataFrame(exposure.T,
                               columns=pre_process.dict_of_rank[0].keys(),
                               index=pre_process.adj_ri.index)

    ## 전략의 picking stock
    from backtest_process import calculate_weight
    import pandas as pd
    pick = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                 name='0-(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)_picked.pickle')
    weight = calculate_weight.make_equal_weight_arr(
        stock_pick=pick,
        number_of_columns=len(pre_process.adj_ri.columns),
        number_of_raws=len(pre_process.adj_ri.index))
    weight_df = pd.DataFrame(weight,
                             columns=pre_process.adj_ri.columns,
                             index=pre_process.adj_ri.index)
    """


