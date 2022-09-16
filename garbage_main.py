from data_process import pre_processing
from data_process import data_read
from backtest_process import stock_picking
from backtest_process import serial_stats
from backtest_process import calculate_weight
from backtest_process import calculate_series
from backtest_process import make_bm
import make_new_sharp as ns

import time
start = time.time()

if __name__ == "__main__":
    rebal = 'm'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'us'
    garbage_list = [False, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    pre_process = pre_processing.PreProcessing(universe=universe, n_top=n_top, garbage=False)
    # 전처리 (dict_of_rank.pickle 완료시 돌릴필요없어요)
    """
    for garbage in garbage_list[:]:
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

        #
        new_sharp_path = ns.get_new_sharp_path(
            garbage=garbage,
            path=pre_process.path_dict["DATA_PATH"])
        data_read.make_path(new_sharp_path)
        new_sharp = ns.NewSharpDecompose(pre_process)
        bulk_df = new_sharp.bulk_df
        for stg_name in bulk_df.columns[:]:
            picked_stock = data_read.read_pickle(
                path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                name=f'{stg_name}_picked.pickle')
            weight_arr = calculate_weight.make_equal_weight_arr(
                stock_pick=picked_stock,
                number_of_columns=len(pre_process.adj_ri.columns),
                number_of_raws=len(pre_process.adj_ri.index))
            sample_series = bulk_df[stg_name] * 100
            sample_series = sample_series.to_numpy()
            print(stg_name)
            new_sharp.update_setting(weight_arr=weight_arr, sample_series=sample_series)
            new_sharp_df = ns.get_new_sharp_data(new_sharp=new_sharp)
            data_read.save_to_pickle(
                any_=new_sharp_df,
                path=new_sharp_path,
                name=f'{stg_name}.pickle')

        bm = make_bm.BM(pre_process)
        bm_series = bm.get_bm_series(cost=cost, rebal=rebal)

        bm_weight = bm.get_bm_weight()
        bm_series = bm.get_series(bm_weight, 0.003, 'm')
        new_sharp.update_setting(weight_arr=bm_series,
                                 sample_series=bm_series.to_numpy())
        value_sharp_df = ns.get_new_sharp_data(new_sharp=new_sharp)

        value_weight = bm.get_value_weight()
        value_series = bm.get_series(value_weight, 0.003, 'm')
        new_sharp.update_setting(weight_arr=value_weight,
                                 sample_series=value_series.to_numpy())
        value_sharp_df = ns.get_new_sharp_data(new_sharp=new_sharp)

        growth_weight = bm.get_growth_weight()
        growth_series = bm.get_series(growth_weight, 0.003, 'm')
        new_sharp.update_setting(weight_arr=growth_weight,
                                 sample_series=growth_series.to_numpy())
        growth_sharp_df = ns.get_new_sharp_data(new_sharp=new_sharp)

        small_weight = bm.get_small_weight()
        small_series = bm.get_series(small_weight, 0.003, 'm')
        new_sharp.update_setting(weight_arr=small_weight,
                                 sample_series=small_series.to_numpy())
        small_sharp_df = ns.get_new_sharp_data(new_sharp=new_sharp)

        large_weight = bm.get_large_weight()
        large_series = bm.get_series(large_weight, 0.003, 'm')
        new_sharp.update_setting(weight_arr=large_weight,
                                 sample_series=large_series.to_numpy())
        large_sharp_df =ns.get_new_sharp_data(new_sharp=new_sharp)

        # new_sharp_df.to_csv('ff.csv')
        calculation_sharp(new_sharp_df)


