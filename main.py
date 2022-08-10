from data_process import pre_processing
from data_process import data_read
from backtest_process import calculate_exposure
from backtest_process import calculate_series
from backtest_process import stock_picking
import time


rebal = 'q'
cost = 0.003
n_top = 20
universe = 'us'
if __name__ == "__main__":
    pre_process = pre_processing.PreProcessing(universe=universe)

    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process, n_top=n_top)
    calculate_exposure.calculate_series_exposure(pre_process=pre_process)
    calculate_series.calculate_series_backtest(pre_process=pre_process, cost=cost, rebal=rebal)

    name_list = list(picking_dict.keys())
    sample = name_list[9]

    data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                          name=f'{sample}_backtest_{rebal}.pickle')
