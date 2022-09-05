import pandas as pd
import getFamaFrenchFactors as gff
from data_process import data_read
from backtest_process import serial_stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

us_garbage_list = [False, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   11, 13, 14, 15, 16, 17, 18, 20,
                   21, 22, 23, 25, 26, 27, 28, 29, 30]

korea_garbage_list = [False, 2, 3, 4, 5, 6, 7, 8, 9,
                      11, 13, 14, 15, 16, 17, 18, 20,
                      21, 22, 23, 25, 26, 27, 28, 29, 30]


def get_ff3_data(rebal: str) -> pd.DataFrame:
    ff3_monthly = gff.famaFrench3Factor(frequency='m')
    ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
    ff3_monthly.set_index('Date', inplace=True)
    # TODO cumprod로 바꿔야함.
    if rebal == 'q':
        ff3_monthly = ff3_monthly.rolling(3).sum()
    elif rebal == 'm':
        ff3_monthly = ff3_monthly
    return ff3_monthly


def get_bulk_data(garbage, folder_name):
    path = 'C:\\Users\\doomoolmori\\factor_attribution_process/data/'
    path = path + folder_name + f'/strategy_stats'
    if garbage != False:
        path = path + f'_garbage_{garbage}'
    bulk_backtest_df = data_read.read_csv_(
        path=path,
        name=f'bulk_backtest_{rebal}.csv')
    bulk_backtest_df.set_index('date_', inplace=True)
    """
    pre_process = pre_processing.PreProcessing(
        universe=universe,
        n_top=n_top,
        garbage=garbage)
    # picking_data_load
    picking_dict = stock_picking.get_stock_picking_dict(pre_process=pre_process)
    bulk_backtest_df = data_read.bulk_backtest_df(
        strategy_name_list=list(picking_dict.keys()),
        raw_path=pre_process.path_dict['STRATEGY_WEIGHT_PATH'],
        save_path=pre_process.path_dict['STRATEGY_STATS_PATH'],
        rebal=rebal)
    """
    return bulk_backtest_df


def scatter_plot(x, y):
    return plt.scatter(x, y, c='blue')


if __name__ == "__main__":
    rebal = 'm'  # or 'm'
    cost = 0.003
    n_top = 20
    universe = 'us'

    if universe == 'us':
        garbage_list = us_garbage_list
    elif universe == 'korea':
        garbage_list = korea_garbage_list

    ff3_monthly = get_ff3_data(rebal=rebal)

    total_alpha_list = []
    total_return_list = []
    total_sd_list = []

    total_alpha_list_out = []
    total_return_list_out = []
    total_sd_list_out = []

    for garbage in garbage_list[:]:
        print(garbage)
        bulk_backtest_df = get_bulk_data(garbage=garbage, folder_name=f'{universe}_10_3_{rebal}')
        bulk_pct = bulk_backtest_df.iloc[:, :715].pct_change().dropna()
        ff3_data = ff3_monthly.loc[list(bulk_pct.index)]
        Y = (bulk_pct - ff3_data['RF'].values.reshape((len(bulk_pct), 1))).values
        X = (ff3_data[['Mkt-RF', 'SMB', 'HML']]).values

        date_dict = serial_stats.date_information_dict(
            index=pd.to_datetime(bulk_pct.index),
            in_sample_year=10,
            out_sample_year=3)

        alpha_list = []
        return_list = []
        sd_list = []

        alpha_list_out = []
        return_list_out = []
        sd_list_out = []
        for key in date_dict.keys():
            in_idx = serial_stats.get_sample_boolean_index(
                index=pd.to_datetime(bulk_pct.index),
                start_date=date_dict[key]['in_start'],
                end_date=date_dict[key]['in_end'])

            out_idx = serial_stats.get_sample_boolean_index(
                index=pd.to_datetime(bulk_pct.index),
                start_date=date_dict[key]['out_start'],
                end_date=date_dict[key]['out_end'])

            ff_model_in = sm.OLS(Y[in_idx, :], sm.add_constant(X[in_idx, :])).fit()
            alpha_list.append(ff_model_in.params[0, :])
            return_list.append(bulk_pct.values[in_idx, :].mean(0))
            sd_list.append(bulk_pct.values[in_idx, :].std(0))

            ff_model_out = sm.OLS(Y[out_idx, :][1:], sm.add_constant(X[out_idx, :][1:])).fit()
            alpha_list_out.append(ff_model_out.params[0, :])
            return_list_out.append(bulk_pct.values[out_idx, :][1:].mean(0))
            sd_list_out.append(bulk_pct.values[out_idx, :][1:].std(0))

        total_alpha_list.append(alpha_list)
        total_return_list.append(return_list)
        total_sd_list.append(sd_list)

        total_alpha_list_out.append(alpha_list_out)
        total_return_list_out.append(return_list_out)
        total_sd_list_out.append(sd_list_out)

    x = np.array(total_alpha_list)[1:, 61:, :].flatten().copy()
    y = np.array(total_alpha_list_out)[1:, 61:, :].flatten().copy()

    ax = scatter_plot(x, y)
    ax.remove()

    pd.Series(x).corr(pd.Series(y))

    plt.scatter(np.array(total_return_list).flatten(),
                np.array(total_alpha_list).flatten())
    pd.Series(np.array(total_return_list).flatten() / np.array(total_sd_list).flatten()).corr(
        pd.Series(np.array(total_alpha_list_out).flatten()))

    ## filter 1 ##
    model_para = sm.OLS(np.array(total_alpha_list)[:1, :, :].flatten(),
                        sm.add_constant(np.array(total_alpha_list_out)[:1, :, :].flatten())).fit()
    garbage_para = sm.OLS(np.array(total_alpha_list)[1:, :, :].flatten(),
                          sm.add_constant(np.array(total_alpha_list_out)[1:, :, :].flatten())).fit()

    ina_f = []
    outa_f = []

    ina_uf = []
    outa_uf = []

    in_tag = []
    for j in range(0, len(garbage_list[:1])):
        fit_model = np.array(total_alpha_list)[j, :, :] * model_para.params[1] + model_para.params[0]
        fit_garbage = np.array(total_alpha_list)[j, :, :] * garbage_para.params[1] + garbage_para.params[0]
        for i in range(0, 715):
            temp_in = np.array(total_alpha_list)[j, :, i]
            temp_out = np.array(total_alpha_list_out)[j, :, i]
            mf = np.sqrt(sum((fit_model[:, i] - temp_out) ** 2))
            gf = np.sqrt(sum((fit_garbage[:, i] - temp_out) ** 2))
            if mf < gf:
                print(i)
                ina_f.append(temp_in.mean())
                outa_f.append(temp_out.mean())
                in_tag.append(i)
            else:
                ina_uf.append(temp_in.mean())
                outa_uf.append(temp_out.mean())

    np.sqrt(sum((np.array(ina_f) - np.array(outa_f)) ** 2))
    (np.array(ina_f) - np.array(outa_f)).std()
    np.sqrt(sum((np.array(ina_uf) - np.array(outa_uf)) ** 2))
    np.array(outa_f).mean() - np.array(outa_uf).mean()

    plt.plot(ina_f)
    plt.plot(outa_f)

    plt.plot(ina_uf)
    plt.plot(outa_uf)

    up = np.array(outa_f).mean() - np.array(outa_uf).mean()
    down = np.sqrt((((len(outa_f) - 1) * (np.array(outa_f).std() ** 2) + (len(outa_uf) - 1) * (
                np.array(outa_uf).std() ** 2)) / (len(outa_f) + len(outa_uf) - 2)) * (
                               1 / len(outa_f) + 1 / len(outa_uf)))

    np.array(outa_f)[np.array(ina_f) > 0.00].mean()



    ## filter 2 ##
    ## Rank IC 를 하려면, 모든 종목의 factor value를 구해야함.
    from scipy import stats

    garbage_list = False

    from backtest_process import stock_picking

    correlation = []
    for garbage in garbage_list[1:]:
        print(garbage)
        bulk_backtest_df = get_bulk_data(garbage=garbage, folder_name=f'{universe}_10_3_{rebal}')

        dict_of_rank = data_read.read_pickle(path='C:/Users/doomoolmori/factor_attribution_process/data/us_10_3_m',
                                             name=f'us_dict_of_rank_garbage_{garbage}.pickle')
        filter_number = 0
        arr_of_rank_arr = []
        for rank_arr in (dict_of_rank[filter_number].values()):
            arr_of_rank_arr.append(rank_arr.to_numpy())
        arr_of_rank_arr = np.array(arr_of_rank_arr)

        adj_ri = data_read.read_csv_(path='C:/Users/doomoolmori/factor_attribution_process/data/us_10_3_m',
                                     name='adj_ri.csv')

        adj_ri.set_index('date_', inplace=True)
        adj_after_pct = adj_ri.pct_change(1).shift(-1).to_numpy()#.rank(1).to_numpy()

        for i in range(len(bulk_backtest_df.columns)):
        #for i in in_tag:
            temp = bulk_pct.columns[i]
            score_weight = [float(x) for x in temp.split('-')[-1][1:-1].split(',')]
            after_pct = adj_after_pct.copy()

            score_weight_arr = stock_picking.shape_mapping_array(
                arr=arr_of_rank_arr,
                score_weight=score_weight)
            factor_rank_sum_arr = stock_picking.factor_rank_sum(
                arr_of_rank_arr=arr_of_rank_arr,
                score_weight_arr=score_weight_arr)

            joint = (~np.isnan(factor_rank_sum_arr)) & (~np.isnan(adj_after_pct))
            factor_rank_sum_arr[~joint] = np.nan
            after_pct[~joint] = np.nan

            a = stock_picking._rank(factor_rank_sum_arr, order='ascending')
            b = stock_picking._rank(after_pct, order='ascending')

            result = stats.spearmanr(pd.Series(a.flatten()).dropna(), pd.Series(b.flatten()).dropna())
            if 1:#i in in_tag:
                correlation.append(result.correlation)
            #else:
            #    unpicked.append(result.correlation)
            print(f'{score_weight}_{result}')

    plt.plot(correlation)

    np.mean(correlation)

