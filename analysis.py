import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def make_q_stat_dict(df:pd.DataFrame, total_q:int):
    stats_dict = {}
    count_q = 0
    count_nor = 0
    for strategy in df['strategy']:
        if 'q' in strategy:
            count_q += 1
        else:
            count_nor += 1
    numbers = (count_q)/total_q
    assert int(numbers) == numbers, "전략수 애러"
    numbers = int(numbers)
    stats_dict['normal'] = df.iloc[:count_nor, :].copy()
    for q in range(1, total_q + 1):
        start = count_nor + numbers * (q - 1)
        end = count_nor + numbers * q
        stats_dict[f'{q}q'] = df.iloc[start:end, :].copy()
    return stats_dict

#_10-3-q/strategy_stats
path = f'data/us/strategy_stats_garbage_30'
file_list = os.listdir(path)
result_list = []
for file in file_list[:-1]:
    raw_df = pd.read_csv(f'{path}/{file}', index_col=0)
    q_stats_dict = make_q_stat_dict(df=raw_df, total_q=5)
    normal_df = q_stats_dict['normal']

    q_list = [q_stats_dict['1q']['Return'].values,
             q_stats_dict['2q']['Return'].values,
             q_stats_dict['3q']['Return'].values,
             q_stats_dict['4q']['Return'].values,
             q_stats_dict['5q']['Return'].values]

    q_list = np.array(q_list).T
    normal_df['1q-5q'] = np.mean(q_list[:, :2], 1) - np.mean(q_list[:, -2:], 1)
    normal_df['1q-5q'] = (normal_df['1q-5q'] - normal_df['1q-5q'].mean()) / normal_df['1q-5q'].std()
    result_list.append(normal_df.copy())
    print(f'{(normal_df[normal_df["1q-5q"] > 1.0]["Alpha"]).mean()}_{(normal_df[normal_df["1q-5q"] > 1.0]["SD"]).mean()}')


result_df = pd.concat(result_list, 0)

#result_df['1q-5q'][(result_df['1q-5q']) > 1] = np.NAN
#result_df['1q-5q'][(result_df['1q-5q']) < -1] = np.NAN

ax = plt.figure()
plt.scatter((result_df['1q-5q']), (result_df['OutReturn']))
(result_df['1q-5q']).corr((result_df['OutReturn']))

#ax = plt.figure()
hurdle = 1.0
plt.scatter((result_df['1q-5q'])[result_df['1q-5q'] > hurdle], (result_df['OutReturn'])[result_df['1q-5q'] > hurdle])
(result_df['1q-5q'])[result_df['1q-5q'] > hurdle].corr((result_df['OutReturn'])[result_df['1q-5q'] > hurdle])

plt.scatter((result_df['1q-5q'])[result_df['1q-5q'] < -hurdle], (result_df['OutReturn'])[result_df['1q-5q'] < -hurdle])
(result_df['1q-5q'])[result_df['1q-5q'] < -hurdle].corr((result_df['OutReturn'])[result_df['1q-5q'] < -hurdle])


print((result_df['OutReturn'])[result_df['1q-5q'] < -hurdle].mean())
print((result_df['OutReturn'])[(result_df['1q-5q'] >= -hurdle) & (result_df['1q-5q'] <= hurdle)].mean())
print((result_df['OutReturn'])[result_df['1q-5q'] > hurdle].mean())


#filtered_df = result_df[result_df['1q-5q'] > 1]
#ax = plt.figure()
#plt.scatter((filtered_df['SD']), (filtered_df['OutReturn']))
#filtered_df['Payoff_to_hitrate'].corr(filtered_df['OutReturn'])
#filtered_df[filtered_df['OutReturn'] < 0].mean()


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
