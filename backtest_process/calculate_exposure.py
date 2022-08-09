from data_process import calculation_rank
import numpy as np
import pandas as pd
from data_process import data_read


class CalculateExposure:
    def __init__(self, pre_process):
        self.pre_process = pre_process
        try:
            self.z_score_factor_dict = data_read.read_pickle(
                path=pre_process.path_dict['DATA_PATH'],
                name='z_scored_factor.pickle')
            print('already calculation z_score_factor')
        except:
            z_score_factor_dict = make_z_scored_factor_dict(
                data=pre_process.dict_of_pandas,
                factor_info=pre_process.factor_info
            )
            data_read.save_to_pickle(
                any_=z_score_factor_dict,
                path=pre_process.path_dict['DATA_PATH'],
                name='z_scored_factor.pickle')
            self.z_score_factor_dict = data_read.read_pickle(
                path=pre_process.path_dict['DATA_PATH'],
                name='z_scored_factor.pickle')
            print('calculation z_score_factor')

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
                total_factor_value_array += self.z_score_factor_dict[factor].flatten() * weight
            weighted_sum_z_score_list.append(total_factor_value_array.copy())
        self.weighted_sum_z_score_arr = np.array(weighted_sum_z_score_list, dtype=np.float32)

    def mean_exposure(self, stock_weight: np.array, _shape: tuple) -> dict:
        result = np.array(self.weighted_sum_z_score_arr) * stock_weight
        result = result.reshape(_shape)
        return np.nansum(result, 2).mean(1)


def normalization_series(series: pd.Series):
    return (series - (series).mean()) / (series.std())


def make_z_scored_factor_dict(data: dict, factor_info: pd.DataFrame) -> pd.DataFrame:
    """
    1. 팩터 방향 설정 (반대 방향의 경우 - 처리) #direction_control
    2. 해당일에 ri값이 존재하지않는 경우 제거 #na_fill_direction_factor_df
    3. ri값은 존재하나 factor 값은 존재하지 않는 경우 0 z_score 부여 #na_fill_direction_factor_df
    """
    survive_df = calculation_rank.make_survive_df(df=data['RI'])
    z_score_factor_dict = {}
    for direction, factor in zip(factor_info['direction'],
                                 factor_info['factor']):
        direction_factor_df = calculation_rank.direction_control(
            _any=(data[factor].copy()).astype(float),
            direction=direction)
        z_score_df = direction_factor_df.apply(lambda x: normalization_series(x), axis=1)
        na_fill_z_score_df = calculation_rank.fill_survive_data_with_min(
            survive_df=survive_df,
            factor_df=z_score_df,
            min_value=0)
        z_score_factor_dict[factor] = na_fill_z_score_df.to_numpy(
            dtype=np.float32)
    return z_score_factor_dict


if __name__ == '__main__':
    from data_process import pre_processing
    from data_process import data_read
    import os

    pre_process = pre_processing.PreProcessing(universe='korea')
    calculate_exposure = CalculateExposure(pre_process=pre_process)

    x = os.listdir(pre_process.path_dict['STRATEGY_WEIGHT_PATH'])

    _shape = tuple([len(pre_process.dict_of_rank.keys())]) + pre_process.adj_pct.shape
    weight_arr = data_read.read_pickle(
                    path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
                     name=x[0]).copy()
    temp_arr = np.empty(len(pre_process.adj_pct) * len(pre_process.adj_pct.columns))
    temp_arr[:] = np.nan
    temp_arr[weight_arr] = 0.2

    result = []
    for file_name in x[:]:
        import time
        start = time.time()
        result.append(calculate_exposure.mean_exposure(stock_weight=temp_arr, _shape=_shape))
        print("time :", time.time() - start)