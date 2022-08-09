from data_process import data_read
from data_process import data_path
import pandas as pd


def make_adj_ri_df(ri_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. 최초 거래일 이전주가 처리
    (상장 전 주가는 최초주가로 back fill)
    2. 거래정지 이후 가격이 안들어오는 경우에는 상폐로 처리
    (마지막 주가0 처리, 보수적인 백테스팅)
    3. 거래정지 이후 가겨이 들어오는 경우 재상장, 분할 등
    (마지막 거래일 주가로 forward fill, 갭락이 존재하는 경우 공격적 백테스팅 갭등이 존재하는 경우 보수적 백테스팅)
    """
    adj_series_list = []
    for ticker in ri_df:
        temp_series = ri_df[ticker].copy()
        pure_series = temp_series.dropna()
        initial_idx = pure_series.index[0]
        final_idx = pure_series.index[-1]
        temp_series.loc[temp_series.index > final_idx] = 0  # 2. 처리
        temp_series.loc[temp_series.index <= initial_idx] = temp_series.loc[initial_idx]  # 1. 처리
        adj_series_list.append(temp_series.fillna(method='ffill'))  # 3. 처리
    adj_ri_df = pd.concat(adj_series_list, 1).dropna()
    print('adf_ri_calculation')
    return adj_ri_df


if __name__ == "__main__":
    from data_process import pre_processing
    pre_process = pre_processing.PreProcessing(universe='korea')
    ri_df = pre_process.dict_of_pandas['RI']
    adj_ri = make_adj_ri_df(ri_df)
    adj_pct = adj_ri.pct_change().shift(-1)