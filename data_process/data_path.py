import os

KOREA_UNIVERSE = 'Univ_KOSPI&KOSDAQ'
US_UNIVERSE = 'Univ_S&P500'

FACTOR_CATEGORY_PATH = f'{os.getcwd()}/'
FACTOR_CATEGORY_NAME = 'QT_factor_category_compress_final.csv'

FACTOR_FILTER_PATH = f'{os.getcwd()}/'
FACTOR_FILTER_NAME = 'QT_filter.csv'


RAW_DATA_PATH = f'{os.getcwd()}/'
RAW_DATA_NAME = '2022-05-27_cosmos-univ-with-factors_with-finval_global_monthly.csv'

KOREA_PATH_DICT = {
    'DATA_PATH': f'{os.getcwd()}/data/korea',
    'STRATEGY_WEIGHT_PATH': f'{os.getcwd()}/data/korea/strategy_weight',
    'STRATEGY_STATS_PATH': f'{os.getcwd()}/data/korea/strategy_stats',
    'BM_PATH': f'{os.getcwd()}/data/korea/bm'}

KOREA_NAME_DICT = {
    'UNIVERSE': 'Univ_KOSPI&KOSDAQ',
    'DATA_NAME': 'korea_dict_of_data.pickle',
    'RANK_NAME': 'korea_dict_of_rank.pickle',
    'RI_NAME': 'adj_ri.csv',
    'PCT_NAME': 'adj_pct.csv'}

US_PATH_DICT = {
    'DATA_PATH': f'{os.getcwd()}/data/us',
    'STRATEGY_WEIGHT_PATH': f'{os.getcwd()}/data/us/strategy_weight',
    'STRATEGY_STATS_PATH': f'{os.getcwd()}/data/us/strategy_stats',
    'BM_PATH': f'{os.getcwd()}/data/us/bm'}

US_NAME_DICT = {
    'UNIVERSE': 'Univ_S&P500',
    'DATA_NAME': 'us_dict_of_data.pickle',
    'RANK_NAME': 'us_dict_of_rank.pickle',
    'RI_NAME': 'adj_ri.csv',
    'PCT_NAME': 'adj_pct.csv'}
