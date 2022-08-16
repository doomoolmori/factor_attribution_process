from data_process import data_read
from data_process import data_path
import pandas as pd
import numpy as np

# TODO 만약 필터조건이 있으면 여기서 걸러줘야함 ex) 1_filter == (mkt > 300m)

def rank_dict_add(data: dict, factor_info: pd.DataFrame) -> dict:
    """
    1. 팩터 방향 설정 (반대 방향의 경우 - 처리) #direction_control
    2. 해당일에 ri값이 존재하지않는 경우 제거 #na_fill_direction_factor_df
    3. ri값은 존재하나 factor 값은 존재하지 않는 경우 가장 낮은 value 부여 #na_fill_direction_factor_df
       (날짜 축으로 rank 시 해당 날짜에서 가장 낮은 등수를 부여받음)
    4. 동점일시 동점평균등수 부여 #rank_ascending
    5. 랭크 썸에도 4번 규정을 적용
    """
    survive_df = make_survive_df(df=data['RI'])
    rank_sum_factor_list = factor_info['category'].unique()
    rank_sum_factor_init = [0 for x in range(len(rank_sum_factor_list))]
    rank_sum_dict = dict(zip(rank_sum_factor_list, rank_sum_factor_init))

    for direction, factor, category in zip(factor_info['direction'],
                                           factor_info['factor'],
                                           factor_info['category']):
        direction_factor_df = direction_control(
            _any=(data[factor].copy()).astype(float),
            direction=direction)
        na_fill_direction_factor_df = fill_survive_data_with_min(
            survive_df=survive_df,
            factor_df=direction_factor_df,
            min_value=np.nanmin(direction_factor_df))
        ##TODO filter 넣으면 될듯
        rank_df = rank_ascending(na_fill_direction_factor_df)
        # rank_sum
        rank_sum_dict[category] += rank_df
    # rank of rank_sum
    for rank_sum_factor in rank_sum_factor_list:
        temp = rank_sum_dict[rank_sum_factor]
        rank_sum_dict[rank_sum_factor] = rank_ascending(temp)
    print('factor_rank_sum_calculation')
    return rank_sum_dict


def make_survive_df(df: pd.DataFrame) -> pd.DataFrame:
    return df >= 0


def direction_control(_any: any, direction: bool) -> any:
    return _any if direction == True else -_any


def fill_survive_data_with_min(
        survive_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        min_value: float) -> pd.DataFrame:
    factor_df[survive_df & factor_df.isna()] = min_value
    return factor_df


def rank_ascending(df: pd.DataFrame):
    return df.rank(axis=1, method='average', ascending=True)


if __name__ == "__main__":
    from data_process import pre_processing
    pre_process = pre_processing.PreProcessing(universe='korea')


    tt = pre_process.dict_of_pandas['Market Capitalization - Current (U.S.$)']

    tr = tt/(tt.sum(1).to_numpy()).reshape((234, 1))

    result = rank_dict_add(pre_process.dict_of_pandas, pre_process.factor_info)
