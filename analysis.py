import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = f'data/us/strategy_stats'
raw_df = pd.read_csv(f'{path}/2003-01-31_q.csv', index_col=0)

total_q = 5
for sg in raw_df['strategy']:
    for i in range(1, total_q + 1):
        print(i)

raw_df['strategy']


"""
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


df_5y = garbage_bulk_stats(folder_name='us_10_1')
df_5y = df_5y.set_index('strategy')
df_10y = garbage_bulk_stats(folder_name='us_10_3')
df_10y = df_10y.set_index('strategy')
df_5y['sharp'] = df_5y['Return']/df_5y['SD']
df_5y['Outsharp'] = df_5y['OutReturn']/df_5y['OutSD']
df_10y['sharp'] = df_10y['Return']/df_10y['SD']
df_10y['Outsharp'] = df_10y['OutReturn']/df_10y['OutSD']

picked_10y = picked_bulk_stats(folder_name='us_10_3')
picked_10y['sharp'] = picked_10y['Return']/picked_10y['SD']
picked_10y['Outsharp'] = picked_10y['OutReturn']/picked_10y['OutSD']


date_list = (df_5y['in_end_out_start']).unique().copy()

corr_list = []
for date in date_list:
    temp_df = df_5y[df_5y['in_end_out_start'] == date].query('Outsharp < 3 & Outsharp > -3')
    corr_list.append(df_5y["Alpha"].corr(temp_df["Outsharp"]))
    print(f'{date}:{df_5y["Alpha"].corr(temp_df["Outsharp"])}')
plt.plot(pd.Series(corr_list, index=date_list))


for name in df_5y.columns:
    print(f'{name}:{df_5y["Outsharp"].corr(df_5y[name])}')




joint_date = list(set(df_5y['in_end_out_start']).intersection(set(df_10y['in_end_out_start'])))
joint_date.sort()

date = joint_date[10]

fig = plt.figure()
dated_5y = df_5y[df_5y['in_end_out_start'] == date]
dated_10y = df_10y[df_10y['in_end_out_start'] == date]

plt.scatter(dated_5y['sharp'], dated_5y['Outsharp'])
dated_5y['Outsharp'].corr(dated_5y['sharp'])


idx = (dated_5y['Outsharp'] < dated_5y['Outsharp'].mean()) & (dated_10y['Outsharp'] > dated_10y['Outsharp'].mean())


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(dated_5y['Outsharp'], dated_10y['Outsharp'], dated_10y['sharp'], alpha=0.1)
#ax.scatter(dated_5y[idx]['Outsharp'], dated_10y[idx]['Outsharp'], dated_10y[idx]['sharp'], color='red')
ax.set_xlabel('5yR')
ax.set_ylabel('10yR')
ax.set_zlabel('3yOutR')
plt.show()

dated_5y['Outsharp'].corr(dated_10y['sharp'])
"""
