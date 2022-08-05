import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

def make_path():
    createFolder(US_STRATEGY_WEIGHT_PATH)
    createFolder(KOREA_STRATEGY_WEIGHT_PATH)


KOREA_UNIVERSE = 'Univ_KOSPI&KOSDAQ'
US_UNIVERSE = 'Univ_S&P500'

FACTOR_CATEGORY_PATH = f'{os.getcwd()}/'
FACTOR_CATEGORY_NAME = 'QT_factor_category_compress_final.csv'

RAW_DATA_PATH = f'{os.getcwd()}/'
RAW_DATA_NAME = '2022-05-27_cosmos-univ-with-factors_with-finval_global_monthly.csv'

DICT_OF_KOREA_DATA_PATH = f'{os.getcwd()}/data/korea'
DICT_OF_KOREA_DATA_NAME = 'korea_dict_of_data.pickle'
DICT_OF_KOREA_RANK_NAME = 'korea_dict_of_rank.pickle'
DICT_OF_KOREA_PCT_NAME = 'korea_dict_of_pct.pickle'
KOREA_STRATEGY_WEIGHT_PATH = f'{DICT_OF_KOREA_DATA_PATH}/strategy_weight'

DICT_OF_US_DATA_PATH = f'{os.getcwd()}/data/us'
DICT_OF_US_DATA_NAME = 'us_dict_of_data.pickle'
DICT_OF_US_RANK_NAME = 'us_dict_of_rank.pickle'
DICT_OF_US_PCT_NAME = 'us_dict_of_pct.pickle'
US_STRATEGY_WEIGHT_PATH = f'{DICT_OF_US_DATA_PATH}/strategy_weight'
