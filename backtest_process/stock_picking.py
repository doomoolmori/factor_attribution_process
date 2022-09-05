import numpy as np
from backtest_process import asyncio_function
import asyncio
from data_process import data_read
from data_process import calculation_rank
from data_process import pre_processing


class StockPick:
    def __init__(self, pre_process, filter_number=0, asyncio_=True):
        self.pre_process = pre_process
        self.filter_number = filter_number
        self.asyncio_ = asyncio_
        self.space_set = pre_process.all_space_set
        survive_df = calculation_rank.make_survive_df(pre_process.dict_of_pandas['RI'])
        epsilon_df = pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
        self.survive_epsilon_arr = calculation_rank.fill_survive_data_with_min(
            survive_df=survive_df,
            factor_df=epsilon_df,
            min_value=np.nanmin(epsilon_df)).to_numpy(np.float32)
        self.survive_epsilon_arr /= (np.nanmax(self.survive_epsilon_arr) * 10)

        arr_of_rank_arr = []
        for rank_arr in (pre_process.dict_of_rank[filter_number].values()):
            arr_of_rank_arr.append(rank_arr.to_numpy())
        self.arr_of_rank_arr = np.array(arr_of_rank_arr, dtype=np.float16)

        input_dict = {}
        input_dict['arr_of_rank_arr'] = self.arr_of_rank_arr
        input_dict['epsilon_arr'] = self.survive_epsilon_arr
        input_dict['space_set'] = self.space_set
        input_dict['path'] = self.pre_process.path_dict['STRATEGY_WEIGHT_PATH']
        input_dict['n_top'] = int(self.pre_process.n_top)
        input_dict['filter_number'] = self.filter_number
        self.input_dict = input_dict

    def do_stock_pick(self):
        if self.asyncio_ == True:
            asyncio_dict = {}
            asyncio_dict['function'] = loop_pick_stock_
            asyncio_dict['input'] = self.input_dict
            asyncio.run(asyncio_function.async_main(**asyncio_dict))
        else:
            self.input_dict['number'] = 'no_asyncio'
            loop_pick_stock_(kwargs=self.input_dict)

    def do_stock_pick_quantile(self, quantile: int = 5):
        self.input_dict['total_q'] = quantile
        for q in [x for x in range(1, quantile + 1)]:
            self.input_dict['current_q'] = q
            if self.asyncio_ == True:
                asyncio_dict = {}
                asyncio_dict['function'] = loop_pick_stock_q
                asyncio_dict['input'] = self.input_dict
                asyncio.run(asyncio_function.async_main(**asyncio_dict))
            else:
                self.input_dict['number'] = 'no_asyncio'
                loop_pick_stock_q(kwargs=self.input_dict)


def _rank(arr: np.array, order='descending') -> np.array:
    """
    https://stackoverflow.com/questions/64101391/how-to-obtain-the-ranks-in-a-2d-numpy-array
    pandas rank와 유사한 함수
    """
    temp = -arr if order == 'descending' else arr
    idx = temp.argsort(1, 'stable')
    m, n = idx.shape
    out = np.empty((m, n), dtype=float)
    np.put_along_axis(out, idx, np.arange(1, n + 1), axis=1)
    return np.where(np.isnan(arr), np.nan, out)


def shape_mapping_array(arr: np.array, score_weight: list) -> np.array:
    return np.broadcast_to(score_weight, arr.T.shape).T


def factor_rank_sum(arr_of_rank_arr: np.array, score_weight_arr: np.array) -> np.array:
    return np.sum(score_weight_arr * arr_of_rank_arr, axis=0, dtype=np.float16)


def final_rank_array(factor_rank_sum_arr: np.array, epsilon_arr: np.array) -> np.array:
    """
    epsilon을 더하는 이유는 동점등수일 때, 우선순위를 주기위함.
    epsilon이 높을 수록 우선순위
    """
    return _rank(factor_rank_sum_arr + epsilon_arr).astype(np.float16)

def _get_final_rank_arr(
        arr_of_rank_arr: np.array,
        epsilon_arr: np.array,
        score_weight: list) -> np.array:
    score_weight_arr = shape_mapping_array(
        arr=arr_of_rank_arr,
        score_weight=score_weight)
    factor_rank_sum_arr = factor_rank_sum(
        arr_of_rank_arr=arr_of_rank_arr,
        score_weight_arr=score_weight_arr)
    final_rank_arr = final_rank_array(
        factor_rank_sum_arr=factor_rank_sum_arr,
        epsilon_arr=epsilon_arr)
    return final_rank_arr


def stock_picker(final_rank_arr: np.array, n_top: int = 20) -> np.array:
    return (final_rank_arr <= n_top)


def stock_picker_q(final_rank_arr: np.array, total_q: int, current_q: int) -> np.array:
    max_rank_arr = np.nanmax(final_rank_arr, 1)
    upper = (current_q/total_q) * max_rank_arr
    under = ((current_q - 1)/total_q) * max_rank_arr
    upper = upper.reshape((len(final_rank_arr), 1))
    under = under.reshape((len(final_rank_arr), 1))
    return (final_rank_arr >= under) & (final_rank_arr <= upper)


def loop_pick_stock_(kwargs):
    point = asyncio_function.async_start_end(**kwargs)

    for score_weight in kwargs['space_set'][point['start']:point['end']]:
        final_rank_arr = _get_final_rank_arr(
            arr_of_rank_arr=kwargs['arr_of_rank_arr'],
            epsilon_arr=kwargs['epsilon_arr'],
            score_weight=score_weight)
        picked_stock_arr = stock_picker(
            final_rank_arr=final_rank_arr,
            n_top=kwargs['n_top']
        )
        picked_stock = picked_stock_arr.flatten()
        picked_stock_idx = np.where(picked_stock == True)
        picked_stock_idx = picked_stock_idx[0].astype(np.int32)
        name = f'{kwargs["filter_number"]}-{score_weight}_picked.pickle'
        # save dictionary to pickle file
        data_read.save_to_pickle(any_=picked_stock_idx,
                                 path=kwargs['path'],
                                 name=name)


def loop_pick_stock_q(kwargs):
    point = asyncio_function.async_start_end(**kwargs)

    for score_weight in kwargs['space_set'][point['start']:point['end']]:
        final_rank_arr = _get_final_rank_arr(
            arr_of_rank_arr=kwargs['arr_of_rank_arr'],
            epsilon_arr=kwargs['epsilon_arr'],
            score_weight=score_weight)
        picked_stock_arr = stock_picker_q(
            final_rank_arr=final_rank_arr,
            total_q=kwargs['total_q'],
            current_q=kwargs['current_q']
        )
        picked_stock = picked_stock_arr.flatten()
        picked_stock_idx = np.where(picked_stock == True)
        picked_stock_idx = picked_stock_idx[0].astype(np.int32)
        name = f'{kwargs["current_q"]}q_{kwargs["filter_number"]}-{score_weight}_picked.pickle'
        # save dictionary to pickle file
        data_read.save_to_pickle(any_=picked_stock_idx,
                                 path=kwargs['path'],
                                 name=name)


def _read_stock_picking_dict(pre_process):
    import os
    file_list = os.listdir(pre_process.path_dict['STRATEGY_WEIGHT_PATH'])
    list_ = []
    name_list = []
    for file_name in file_list:
        if 'picked' in file_name:
            # print(f'read_{file_name}')
            weight_arr = data_read.read_pickle(
                path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                name=file_name).copy()
            list_.append(weight_arr)
            name_list.append(file_name.split('_picked.pickle')[0])
    return dict(zip(name_list, list_))


def get_stock_picking_dict(pre_process):
    try:
        result = _read_stock_picking_dict(pre_process=pre_process)
    except:
        pass
    if len(result) == 0:
        # StockPick(pre_process=pre_process, filter_number=filter_number, asyncio_=asyncio_)
        result = _read_stock_picking_dict(pre_process=pre_process)
    return result


if __name__ == "__main__":
    # TODO 현재 TOP20개로 뽑고있는데, 분위별로 뽑는 작업도 필요함.
    pre_process = pre_processing.PreProcessing(universe='us')
    # no asyncio
    # StockPick(pre_process=pre_process, n_top=20, asyncio_=False)
    # asyncio
    picking = StockPick(pre_process=pre_process, filter_number=0, asyncio_=True)
    #picking.do_stock_pick()
    picking.do_stock_pick_quantile(quantile=5)
    """
    Checking 
    from data_process import calculation_rank
    import numpy as np
    check_ = data_read.read_pickle(
        path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        name=f'{pre_process.all_space_set[77]}_picked.pickle')

    space_set = pre_process.all_space_set
    survive_df = calculation_rank.make_survive_df(pre_process.dict_of_pandas['RI'])
    epsilon_df = pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']
    survive_epsilon_arr = calculation_rank.fill_survive_data_with_min(
        survive_df=survive_df,
        factor_df=epsilon_df,
        min_value=np.nanmin(epsilon_df)).to_numpy(np.float32)
    survive_epsilon_arr /= (np.nanmax(survive_epsilon_arr) * 10)

    arr_of_rank_arr = []
    for rank_arr in (pre_process.dict_of_rank.values()):
        arr_of_rank_arr.append(rank_arr.to_numpy())
    arr_of_rank_arr = np.array(arr_of_rank_arr, dtype=np.float16)

    temp = make_picked_stock_arr(
        arr_of_rank_arr=arr_of_rank_arr,
        epsilon_arr=survive_epsilon_arr,
        score_weight=pre_process.all_space_set[77],
        n_top=20)

    temp_arr = np.zeros(len(temp) * len(temp.T))
    temp_arr[check_] = 1
    temp_arr = temp_arr.reshape(temp.shape)

    # 0 나와야함.
    (temp_arr - temp).sum(1)

    
    Checking 
    """

