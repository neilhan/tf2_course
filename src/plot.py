import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn
iris = pd.read_csv('../TF_2_Notebooks_and_Data/DATA/iris.csv')
sns.pairplot(iris, hue='species') # hue gets you kernl density estimation
plt.show()
exit()

df = pd.read_csv('../TF_2_Notebooks_and_Data/DATA/heart.csv')
print('df head:')
print(df.head())

sns.scatterplot(x='chol', y='trestbps', data=df,
                hue='sex', palette='Dark2', size='age')
plt.show()

sns.boxplot(x='target', y='age', data=df)
plt.show()

sns.boxplot(x='target', y='thalach', data=df, hue='sex')
plt.show()

sns.countplot(x='cp', data=df, hue='sex')
plt.show()

sns.displot(df['age'], kde=True, bins=100)
# sns.histplot(df['age'], kde=True, bins=100)
plt.show()

sns.countplot(x='sex', data=df)
plt.show()

sns.countplot(x='target', data=df)
plt.show()

# matplotlib.pyplot
x = [0, 1, 2]
y = [10, 20, 30]
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='red', marker='o', markersize=10, linestyle='dotted')
plt.xlim(-1.0, 2.5)
plt.ylim(-1.0, 35.5)
plt.title('ha, title')
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()

housing = pd.DataFrame({'rooms': [1, 1, 2, 2, 2, 3, 3, 3],
                        'price': [100, 120, 190, 200, 230, 310, 330, 305]})

plt.scatter(housing['rooms'], housing['price'], color='#bc00c0', marker='x')
plt.show()
