
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics

import tensorflow as tf


df = pd.read_csv('../TF_2_Notebooks_and_Data/DATA/fake_reg.csv')
print(df.head())

X = df[['feature1', 'feature2']].values  # numpy
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# get the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1 ))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X_train, y=y_train, epochs=250)

# to evaluate
print('now evalute the model:')
model.evaluate(X_test, y_test, verbose=2)

test_predictions = model.predict(X_test)
df_test = pd.DataFrame(pd.concat([pd.DataFrame(test_predictions), pd.DataFrame(y_test)], axis=1))


mae = metrics.mean_absolute_error(test_predictions, y_test)
print('mae:', mae)
exit()

df_test.columns = ['a','b']
print('df_test:', df_test)
sns.scatterplot(x='a', y='b',data=df_test)
plt.show() 



# loss_df = pd.DataFrame(model.history.history)
# print('loss_df:', loss_df)
# plt.plot(loss_df)
# plt.show() 


print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)
print('X_train: \n{0}'.format(X_train))
print('X: \n{0}'.format(X))

sns.pairplot(df)
plt.show()
