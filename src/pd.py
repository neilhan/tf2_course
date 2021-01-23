import os
import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
my_list = [1, 2, 3]
a = np.array(my_list)

d = {'a': 1,
     'b': 2,
     'c': 3}

# pd Series
pd_series = pd.Series(data=my_list)

print('my_list pd:\n', pd_series)

pd_series = pd.Series(data=my_list, index=labels)
print('my_list with labels pd:\n', pd_series)

print('pd_series index by label:', pd_series['b'])
print('pd_series index by number:', pd_series[1])

labels_2 = ['b', 'c', 'd']
pd_series_2 = pd.Series(data=[1, 2, 3], index=labels_2)

print('pd_series + pd_series_2:\n', pd_series + pd_series_2)

# pd DataFrame
# --------------------------------------
columns = ['a', 'b', 'c', 'd']
index = ['r1', 'r2', 'r3', 'r4', 'r5']
data = np.random.randint(10, 20, (5, 4))
print('random data:\n', data)

df = pd.DataFrame(data=data, index=index, columns=columns)
print('dataframe:\n', df)
print('df[a]:\n', df['a'])
print('df[[a,b]]:\n', df[['a', 'b']])

print('Adding new column')
df['a+b'] = df['a'] + df['b']
print('df a+b:\n', df)

print('drop a+b:')
print('df.drop(a+b, axis=1):\n', df.drop('a+b', axis=1))

print('df.loc[r1, r3]:\n', df.loc[['r1', 'r3']])
print('df.iloc[[2,4]]:\n', df.iloc[[2, 4]])
print('df.iloc[0:3]:\n', df.iloc[0:3])

print('df.loc["r1","a"]:\n', df.loc['r1', 'a'])
print('df.loc[["r1", "r2"],["a","b"]]:\n', df.loc[['r1', 'r2'], ['a', 'b']])

print('filter ----------')
print('df rows a>13:\n', df[df['a'] > 13])

print('df a>13 & b>13:\n', df[(df['a'] > 13) & (df['b'] > 13)])
print('df a>13 | b>13:\n', df[(df['a'] > 13) | (df['b'] > 13)])

df = df.reset_index()
print('df.reset_index():\n', df)
df['r'] = index
df = df.set_index('r')
print('df.set_index("r"):\n', df)
print('df.describe():\n', df.describe())
print('df.info():\n', df.info())

print('missing data----------')
df = pd.DataFrame({'a': [1, 2, np.nan, 4],
                   'b': [5, np.nan, np.nan, 8],
                   'c': [10, 20, 30, 40]})
df2 = df.copy()
print('df with nan:\n', df)

print('df drop nan/by column:\n', df.dropna(axis=1, thresh=4))
print('df drop nan/by row:\n', df.dropna(axis=0, thresh=2))
print('df fill nan:\n', df.fillna('Fill'))
df['b'] = df['b'].fillna(value=df['b'].mean())
print('df b filled with mean:\n', df)
print('df fillna with each column mean:\n', df2.fillna(df.mean()))

print('GroupBy aggregation funcitons -------------------------------')
# import csv file
df = pd.read_csv(os.path.join('../TF_2_Notebooks_and_Data/01-Pandas-Crash-Course', 'Universities.csv'))
print('loaded Universities.csv:\n', df)
print('head Universities.csv:\n', df.head())
print('groupby year, do mean:\n', df.groupby('Year').mean())
print('groupby year & sector, do mean:\n', df.groupby(['Year', 'Sector']).mean())
