import numpy as np
import pandas as pd
from data_process import data_read
from backtest_process import asyncio_function
from backtest_process import calculate_weight
import asyncio
from backtest_process import stock_picking

class CalculateBacktest:
    def __init__(self, pre_process, cost=0.003, rebal='q', asyncio_=True):

        adj_ri = pre_process.adj_ri
        month_index = get_month_index(time_index=adj_ri.index, rebal=rebal)
        new_datetime = adj_ri.index[month_index]
        adj_pct_arr = get_adj_pct_arr_by_index(adj_ri=adj_ri, month_index=month_index)

        input_dict = {}
        input_dict['picking_dict'] = stock_picking.get_stock_picking_dict(pre_process=pre_process)
        input_dict['number_of_columns'] = len(pre_process.adj_ri.columns)
        input_dict['number_of_raws'] = len(pre_process.adj_ri)
        input_dict['rebal'] = rebal
        input_dict['month_index'] = month_index
        input_dict['new_datetime'] = new_datetime
        input_dict['adj_pct_arr'] = adj_pct_arr
        input_dict['path'] = pre_process.path_dict['STRATEGY_WEIGHT_PATH']
        input_dict['cost'] = cost
        if asyncio_ == True:
            asyncio_dict = {}
            asyncio_dict['function'] = loop_series_backtest_
            asyncio_dict['input'] = input_dict
            asyncio.run(asyncio_function.async_main(**asyncio_dict))
        else:
            input_dict['number'] = 'no_asyncio'
            loop_series_backtest_(kwargs=input_dict)


def get_month_index(time_index: list, rebal: str = 'q') -> list:
    month_list = pd.to_datetime(time_index).month
    if rebal == 'q':
        checking_month = [1, 4, 7, 10]
    else:
        checking_month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    month_index = []
    for i, month in enumerate(month_list):
        if month in checking_month or i + 1 == len(month_list):
            month_index.append(i)
    return month_index


def get_cost_arr(stock_weight: np.array, cost: float) -> np.array:
    diff_weight_abs = np.abs(stock_weight[1:, :] - stock_weight[:-1, :])
    after_cost_arr = np.sum(diff_weight_abs, 1) * cost
    initial_cost_arr = np.array([cost])
    return np.append(initial_cost_arr, after_cost_arr)


def get_return_arr(adj_pct_arr: np.array, stock_weight: np.array) -> np.array:
    after_return_arr = np.sum(adj_pct_arr * stock_weight, 1)[:-1]
    initial_return_arr = np.array([0])
    return np.append(initial_return_arr, after_return_arr)


def get_stock_weight_by_index(
        month_index: list,
        stock_pick: np.array,
        number_of_columns: int,
        number_of_raws: int) -> np.array:
    stock_weight = calculate_weight.make_equal_weight_arr(
        stock_pick=stock_pick,
        number_of_columns=number_of_columns,
        number_of_raws=number_of_raws)
    stock_weight = stock_weight[month_index, :]
    stock_weight[np.isnan(stock_weight)] = 0
    return stock_weight


def get_adj_pct_arr_by_index(adj_ri: pd.DataFrame, month_index: list):
    adj_ri = adj_ri.loc[adj_ri.index[month_index]]
    adj_pct_arr = adj_ri.pct_change().shift(-1).to_numpy(dtype=np.float32)
    adj_pct_arr[np.isnan(adj_pct_arr)] = 0
    return adj_pct_arr


def get_backtest_series(return_arr: np.array, cost_arr: np.array) -> np.array:
    return (return_arr - cost_arr + 1).cumprod()


def loop_series_backtest_(kwargs):
    if type(kwargs['number']) == int:
        start = int(len(kwargs['picking_dict']) / 4) * (kwargs['number'] - 1)
        end = int(len(kwargs['picking_dict']) / 4) * (kwargs['number'])
        if kwargs['number'] == -1:
            end = len(kwargs['picking_dict'])
    else:
        start = 0
        end = len(kwargs['picking_dict'])

    for key_ in list(kwargs['picking_dict'].keys())[start:end]:
        stock_pick = kwargs['picking_dict'][key_]
        stock_weight = get_stock_weight_by_index(
            month_index=kwargs['month_index'],
            stock_pick=stock_pick,
            number_of_columns=kwargs['number_of_columns'],
            number_of_raws=kwargs['number_of_raws'])
        cost_arr = get_cost_arr(stock_weight=stock_weight,
                                cost=kwargs['cost'])
        return_arr = get_return_arr(adj_pct_arr=kwargs['adj_pct_arr'],
                                    stock_weight=stock_weight)
        result = get_backtest_series(return_arr=return_arr, cost_arr=cost_arr)
        result = pd.Series(result, index=kwargs['new_datetime'])
        name = f'{key_}_backtest_{kwargs["rebal"]}.pickle'
        # save dictionary to pickle file
        data_read.save_to_pickle(any_=result,
                                 path=kwargs['path'],
                                 name=name)


def calculate_series_backtest(
        pre_process,
        cost: float = 0.003,
        rebal: str = 'q',
        asyncio_: str = True):
    CalculateBacktest(pre_process=pre_process, cost=cost, rebal=rebal,asyncio_=asyncio_)


if __name__ == '__main__':
    from data_process import pre_processing
    from backtest_process import stock_picking

    pre_process = pre_processing.PreProcessing(universe='korea')
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)

    data_read.read_pickle(path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                          name=f'{list(picking_dict.keys())[365]}_backtest.pickle')

    number_of_columns = len(pre_process.adj_ri.columns)
    number_of_raws = len(pre_process.adj_ri)

    adj_ri = pre_process.adj_ri  # .to_numpy(dtype=np.float32)#.pct_change().shift(-1).to_numpy(dtype=np.float32)
    month_index = get_month_index(time_index=adj_ri.index, rebal='q')
    new_datetime = adj_ri.index[month_index]
    adj_pct_arr = get_adj_pct_arr_by_index(adj_ri=adj_ri, month_index=month_index)

    import time

    start = time.time()
    stock_pick = picking_dict[list(picking_dict.keys())[365]]
    stock_weight = get_stock_weight_by_index(
        month_index=month_index,
        stock_pick=stock_pick,
        number_of_columns=number_of_columns,
        number_of_raws=number_of_raws)
    cost_arr = get_cost_arr(stock_weight=stock_weight, cost=0.003)
    return_arr = get_return_arr(adj_pct_arr=adj_pct_arr, stock_weight=stock_weight)
    temp = get_backtest_series(return_arr=return_arr, cost_arr=cost_arr)
    pd.Series(temp, index=new_datetime)

    print("time :", time.time() - start)
