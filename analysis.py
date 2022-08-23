import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


# garbage 10, 12, 19, 24 제거

char_list = ['Return', 'SD', 'Sharpe', 'MinReturn', 'MaxReturn', 'UpsideFrequency',
             'UpCapture', 'DownCapture', 'UpNumber', 'DownNumber', 'UpPercent',
             'DownPercent', 'AverageDrawdown', 'MaxDrawdown', 'TrackingError',
             'PainIndex', 'AverageLength', 'Alpha', 'Beta', 'Beta.Bull', 'Beta.Bear',
             'strategy', 'in_end_out_start']
outcome_list = ['OutReturn', 'OutSD']
total_list = char_list + outcome_list

def garbage_bulk_stats(folder_name):
    df_list = []
    for i in range(1, 31):
        print(i)
        if i not in [10, 12, 19, 24]:
            path = f'data/{folder_name}/strategy_stats_garbage_{i}'
            file_list = os.listdir(path)[:-1]
            for file in file_list:
                df_list.append(pd.read_csv(f'{path}/{file}', index_col=0)[total_list])
    return pd.concat(df_list, 0)

def picked_bulk_stats(folder_name):
    df_list = []
    path = f'data/{folder_name}/strategy_stats'
    file_list = os.listdir(path)[:-1]
    for file in file_list:
        df_list.append(pd.read_csv(f'{path}/{file}', index_col=0)[total_list])
    return pd.concat(df_list, 0)


df_5y = garbage_bulk_stats(folder_name='us_5_3')
df_5y = df_5y.set_index('strategy')
df_10y = garbage_bulk_stats(folder_name='us_10_3')
df_10y = df_10y.set_index('strategy')
df_5y['sharp'] = df_5y['Return']/df_5y['SD']
df_5y['Outsharp'] = df_5y['OutReturn']/df_5y['OutSD']
df_10y['sharp'] = df_10y['Return']/df_10y['SD']
df_10y['Outsharp'] = df_10y['OutReturn']/df_10y['OutSD']


joint_date = list(set(df_5y['in_end_out_start']).intersection(set(df_10y['in_end_out_start'])))
joint_date.sort()
date = joint_date[-1]

dated_5y = df_5y[df_5y['in_end_out_start'] == date]
dated_10y = df_10y[df_10y['in_end_out_start'] == date]

idx = (dated_5y['sharp'] < dated_5y['sharp'].mean()) & (dated_10y['sharp'] > dated_10y['sharp'].mean())


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(dated_5y['sharp'], dated_10y['sharp'], dated_10y['Outsharp'], alpha=0.1)
ax.scatter(dated_5y[idx]['sharp'], dated_10y[idx]['sharp'], dated_10y[idx]['Outsharp'], color='red')
ax.set_xlabel('5yR')
ax.set_ylabel('10yR')
ax.set_zlabel('3yOutR')
plt.show()
