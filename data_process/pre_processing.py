from data_process import data_read
from data_process import calculation_rank
from data_process import calculation_z_score
from data_process import calculation_pct
from data_process import data_path
from itertools import *
import pandas as pd


def strategy_space(numbers):
    all_score_space = product([0, 0.25, 0.5, 0.75, 1], repeat=numbers)
    return [x for x in all_score_space if (sum(x) == 1)]


class PreProcessing:
    def __init__(self, universe, n_top: int = 20, garbage=False):
        self.universe = universe
        self.garbage = garbage
        self.n_top = n_top
        self.factor_info = pd.read_csv(
            f'{data_path.FACTOR_CATEGORY_PATH}/'
            f'{data_path.FACTOR_CATEGORY_NAME}')
        self.filter_info = pd.read_csv(
            f'{data_path.FACTOR_FILTER_PATH}/'
            f'{data_path.FACTOR_FILTER_NAME}')

        self.raw_data_path = data_path.RAW_DATA_PATH
        self.raw_data_name = data_path.RAW_DATA_NAME
        if self.universe == 'korea':
            self.universe_name = data_path.KOREA_UNIVERSE
            self.path_dict = data_path.KOREA_PATH_DICT.copy()
            self.name_dict = data_path.KOREA_NAME_DICT.copy()
        elif self.universe == 'us':
            self.universe_name = data_path.US_UNIVERSE
            self.path_dict = data_path.US_PATH_DICT.copy()
            self.name_dict = data_path.US_NAME_DICT.copy()
        if garbage != False:
            self.garbage_setting(garbage=self.garbage)
        data_read.make_path(path=self.path_dict['STRATEGY_WEIGHT_PATH'])
        data_read.make_path(path=self.path_dict['STRATEGY_STATS_PATH'])

        try:  # raw data
            self.dict_of_pandas = data_read.read_pickle(
                path=self.path_dict['DATA_PATH'],
                name=self.name_dict['DATA_NAME'])
            print('already exist dict_of_pandas')
        except:
            self._raw_data_processing()

        try:  # rank data
            self.dict_of_rank = data_read.read_pickle(
                path=self.path_dict['DATA_PATH'],
                name=self.name_dict['RANK_NAME'])
            print('already calculation rank')
        except:
            self._rank_data_processing()

        try:  # adj ri data
            self.adj_ri = data_read.read_csv_(
                path=self.path_dict['DATA_PATH'],
                name=self.name_dict['RI_NAME'])
            print('already calculation adj ri')
            self.adj_ri = self.adj_ri.set_index('date_')
        except:
            self._ri_data_processing()

        try:  # all space set
            self.all_space_set = data_read.read_pickle(
                path=self.path_dict['DATA_PATH'],
                name='all_space_set.pickle')
            print('already calculation all space_set')
        except:
            self._all_space_set_processing()

        try:  # z score dict
            self.z_score_factor_dict = data_read.read_pickle(
                path=self.path_dict['DATA_PATH'],
                name='z_scored_factor.pickle')
            print('already calculation z_score_factor')
        except:
            self._z_score_processing()

    def _raw_data_processing(self):
        raw_data_df = data_read.read_raw_data_df(
            path=self.raw_data_path,
            name=self.raw_data_name)
        filter_df = data_read.universe_filter_df(
            df=raw_data_df,
            universe=self.name_dict['UNIVERSE'])
        dict_ = data_read.make_dict_of_pandas(
            df=filter_df)
        data_read.save_to_pickle(
            any_=dict_,
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['DATA_NAME'])
        self.dict_of_pandas = data_read.read_pickle(
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['DATA_NAME'])
        print('calculation dict_of_pandas')

    def _rank_data_processing(self):
        dict_of_rank = calculation_rank.filtered_rank_dict(
            data=self.dict_of_pandas,
            factor_info=self.factor_info,
            filter_info=self.filter_info,
            n_top=self.n_top)
        data_read.save_to_pickle(
            any_=dict_of_rank,
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['RANK_NAME'])
        self.dict_of_rank = data_read.read_pickle(
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['RANK_NAME'])

    def _ri_data_processing(self):
        adj_ri = calculation_pct.make_adj_ri_df(
            ri_df=self.dict_of_pandas['RI'])
        data_read.save_to_csv(
            df=adj_ri,
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['RI_NAME'])
        self.adj_ri = data_read.read_csv_(
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['RI_NAME'])
        self.adj_ri = self.adj_ri.set_index('date_')
        print('calculation adj ri')

    def _pct_data_processing(self):
        adj_pct = self.adj_ri.pct_change().shift(-1)
        data_read.save_to_csv(
            df=adj_pct,
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['PCT_NAME'])
        self.adj_pct = data_read.read_csv_(
            path=self.path_dict['DATA_PATH'],
            name=self.name_dict['PCT_NAME'])
        self.adj_pct = self.adj_pct.set_index('date_')
        print('calculation adj pct')

    def _all_space_set_processing(self):
        all_space_set = strategy_space(
            numbers=len(self.dict_of_rank[0].keys()))
        data_read.save_to_pickle(
            any_=all_space_set,
            path=self.path_dict['DATA_PATH'],
            name='all_space_set.pickle')
        self.all_space_set = data_read.read_pickle(
            path=self.path_dict['DATA_PATH'],
            name='all_space_set.pickle')
        print('calculation all space')

    def _z_score_processing(self):
        z_score_factor_dict = calculation_z_score.make_z_scored_factor_dict(
            data=self.dict_of_pandas,
            factor_info=self.factor_info
        )
        data_read.save_to_pickle(
            any_=z_score_factor_dict,
            path=self.path_dict['DATA_PATH'],
            name='z_scored_factor.pickle')
        self.z_score_factor_dict = data_read.read_pickle(
            path=self.path_dict['DATA_PATH'],
            name='z_scored_factor.pickle')
        print('calculation z_score_factor')

    def garbage_setting(self, garbage):
        self.factor_info = pd.read_csv(
            f'{data_path.FACTOR_CATEGORY_PATH}/garbage_file/'
            f'{data_path.FACTOR_CATEGORY_NAME.split(".csv")[0]}_garbage_{garbage}.csv')
        if self.universe == 'korea':
            self.path_dict['STRATEGY_WEIGHT_PATH'] = \
                f'{data_path.KOREA_PATH_DICT["STRATEGY_WEIGHT_PATH"]}_garbage_{garbage}'
            self.path_dict['STRATEGY_STATS_PATH'] = \
                f'{data_path.KOREA_PATH_DICT["STRATEGY_STATS_PATH"]}_garbage_{garbage}'
            self.name_dict['RANK_NAME'] = \
                f'{data_path.KOREA_NAME_DICT["RANK_NAME"].split(".")[0]}_garbage_{garbage}.pickle'
        elif self.universe == 'us':
            self.path_dict['STRATEGY_WEIGHT_PATH'] = \
                f'{data_path.US_PATH_DICT["STRATEGY_WEIGHT_PATH"]}_garbage_{garbage}'
            self.path_dict['STRATEGY_STATS_PATH'] = \
                f'{data_path.US_PATH_DICT["STRATEGY_STATS_PATH"]}_garbage_{garbage}'
            self.name_dict['RANK_NAME'] = \
                f'{data_path.US_NAME_DICT["RANK_NAME"].split(".")[0]}_garbage_{garbage}.pickle'


if __name__ == "__main__":
    garbage = 0
    pre_process = PreProcessing(universe='korea', garbage=garbage)

    pre_process.dict_of_rank[1]['Earning Momentum'].sum(1)
