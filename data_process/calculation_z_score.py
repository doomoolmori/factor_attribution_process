import pandas as pd
import numpy as np
from data_process import calculation_rank


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
