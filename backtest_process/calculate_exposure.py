import numpy as np
import pandas as pd
from data_process import data_read
from backtest_process import asyncio_function
from backtest_process import calculate_weight
import asyncio
from backtest_process import stock_picking

class CalculateExposure:
    def __init__(self, pre_process, asyncio_=True):
        """
        z_score value sum => sum(stock_weight * (factor)z_score_weight * z_score)
        선형결합이므로 (factor)z_score_weight * z_score를 먼저 구함.
        """
        weighted_sum_z_score_list = []
        for category in pre_process.factor_info['category'].unique():
            total_factor_value_array = 0
            temp = pre_process.factor_info[pre_process.factor_info['category'] == category]
            for factor, weight in zip(temp['factor'],
                                      temp['z_score_weight']):
                total_factor_value_array += pre_process.z_score_factor_dict[factor].flatten() * weight
            weighted_sum_z_score_list.append(total_factor_value_array.copy())
        weighted_sum_z_score_arr = np.array(weighted_sum_z_score_list, dtype=np.float32)

        input_dict = {}
        input_dict['picking_dict'] = stock_picking.get_stock_picking_dict(pre_process=pre_process)
        input_dict['number_of_columns'] = len(pre_process.adj_ri.columns)
        input_dict['number_of_raws'] = len(pre_process.adj_ri)
        input_dict['_shape'] = tuple([len(pre_process.dict_of_rank.keys())]) + pre_process.adj_ri.shape
        input_dict['weighted_sum_z_score_arr'] = weighted_sum_z_score_arr
        input_dict['path'] = pre_process.path_dict['STRATEGY_WEIGHT_PATH']
        if asyncio_ == True:
            asyncio_dict = {}
            asyncio_dict['function'] = loop_series_exposure_
            asyncio_dict['input'] = input_dict
            asyncio.run(asyncio_function.async_main(**asyncio_dict))
        else:
            input_dict['number'] = 'no_asyncio'
            loop_series_exposure_(kwargs=input_dict)


def series_exposure(
        weighted_sum_z_score_arr: np.array,
        stock_pick: np.array,
        stock_weight: np.array,
        _shape: tuple) -> dict:
    temp_arr = np.empty((_shape[0], _shape[1] * _shape[2]))
    result = weighted_sum_z_score_arr[:, stock_pick] * stock_weight[stock_pick]
    temp_arr[:, stock_pick] = result
    temp_arr = temp_arr.reshape(_shape)
    return np.sum(temp_arr, axis=2, dtype=np.float32)


def loop_series_exposure_(kwargs):
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
        stock_weight = calculate_weight.make_equal_weight_arr(
            stock_pick=stock_pick,
            number_of_columns=kwargs['number_of_columns'],
            number_of_raws=kwargs['number_of_raws']).flatten()
        series_exposure_arr = series_exposure(
            weighted_sum_z_score_arr=kwargs['weighted_sum_z_score_arr'],
            stock_pick=stock_pick,
            stock_weight=stock_weight,
            _shape=kwargs['_shape'])
        name = f'{key_}_exposure.pickle'
        # save dictionary to pickle file
        data_read.save_to_pickle(any_=series_exposure_arr,
                                 path=kwargs['path'],
                                 name=name)


def calculate_series_exposure(pre_process, asyncio_=True):
    CalculateExposure(pre_process=pre_process, asyncio_=asyncio_)


if __name__ == '__main__':
    from data_process import pre_processing
    from backtest_process import stock_picking

    pre_process = pre_processing.PreProcessing(universe='korea')
    calculate_series_exposure(pre_process, asyncio_=True)

    sample = data_read.read_pickle(
        path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        name='(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)_exposure.pickle')
    np.mean(sample, 1)