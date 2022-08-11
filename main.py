from data_process import pre_processing
from data_process import data_read
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import stock_picking
import time
import bulk_upload

rebal = 'q'
cost = 0.003
n_top = 20
universe = 'korea'
if __name__ == "__main__":
    pre_process = pre_processing.PreProcessing(universe=universe)
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process, n_top=n_top, asyncio_=True)

    # 계산 완료되면 돌릴 필요 없어요
    #calculate_exposure.calculate_series_exposure(pre_process=pre_process, asyncio_=True)
    # 계산 완료되면 돌릴 필요 없어요
    #calculate_series.calculate_series_backtest(pre_process=pre_process, cost=cost, rebal=rebal, asyncio_=True)


    bulk_backtest_df = bulk_upload.bulk_backtest_df(
        strategy_name_list=list(picking_dict.keys()),
        path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        rebal=rebal)

    bulk_exposure_dict = bulk_upload.bulk_exposure_dict(
        strategy_name_list=list(picking_dict.keys()),
        path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'])

    import numpy as np
    """
    stat sample
    """
    from backtest_process import calculate_stats as cs
    sts = cs.Stats()
    sts.set_rebal(rebal=rebal)
    sts.set_bm_series(bm_series=bulk_backtest_df.iloc[:, 0])
    start = time.time()
    total_list = []
    for i in range(0, len(bulk_backtest_df.columns)):
        sts.set_backtest_series(backtest_series=bulk_backtest_df.iloc[:, i])
        sts.check()
        dt = [cs.ret(ret_arr=sts.ret_arr, freq=sts.freq),
              cs.sd(ret_arr=sts.ret_arr, freq=sts.freq),
              cs.sharpe(ret_arr=sts.ret_arr, freq=sts.freq),
              cs.min_ret(ret_arr=sts.ret_arr),
              cs.max_ret(ret_arr=sts.ret_arr),
              cs.upside_frequency(
                  ret_arr=sts.ret_arr,
                  up_boolean=sts.up_boolean),
              cs.up_capture(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_up_boolean=sts.bm_up_boolean),
              cs.down_capture(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_down_boolean=sts.bm_down_boolean),
              cs.up_number(
                  ret_arr=sts.ret_arr,
                  bm_up_boolean=sts.bm_up_boolean),
              cs.down_number(
                  ret_arr=sts.ret_arr,
                  bm_down_boolean=sts.bm_down_boolean),
              cs.up_percent(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_up_boolean=sts.bm_up_boolean),
              cs.down_percent(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_down_boolean=sts.bm_down_boolean),
              cs.tracking_error(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  freq=sts.freq),
              cs.beta(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr),
              cs.beta_bull(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_up_boolean=sts.bm_up_boolean),
              cs.beta_bear(
                  ret_arr=sts.ret_arr,
                  bm_ret_arr=sts.bm_ret_arr,
                  bm_down_boolean=sts.bm_down_boolean),
              cs.average_drawdown(drawdown=sts.backtest_drawdown),
              cs.max_drawdown(drawdown=sts.backtest_drawdown),
              cs.pain_index(drawdown=sts.backtest_drawdown),
              cs.average_length(drawdown=sts.backtest_drawdown)]
        total_list.append(np.array(dt, dtype=np.float16))
    print("time :", time.time() - start)

    """
    exposure sample
    """
    exposure_series = data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                                            name=f'{sample}_exposure.pickle')
    idx_list = []
    for i, time in enumerate(pre_process.adj_ri.index):
        if time in backtest_series.index:
            idx_list.append(i)

    exposure_df = pd.DataFrame(exposure_series[:, idx_list].T)
    exposure_df.columns = sample_dict.keys()
    exposure_df.index = backtest_series.index
    exposure_df['backtest'] = backtest_series