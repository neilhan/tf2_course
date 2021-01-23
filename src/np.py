import numpy as np

a = np.array([1, 2])
print(a)


print(np.arange(10))

np.random.seed(0)
print(
    'randome (10, 5):\n',
    np.random.rand(10, 5)
)

print(
    np.random.randint(1, 100, (2, 3))
)
print('25 reshape:\n',
      np.arange(30).reshape(6, 5)
      )
rand_10x10 = np.random.randint(1, 100, (10, 10))
print('Max: {0} \n argmax: {1}\n {2}'.format(rand_10x10.max(),
                                             rand_10x10.argmax(),
                                             rand_10x10))

ten = np.arange(10)
ten[:5] = 10
print('broadcast 5:', ten)
ten_copy = ten.copy()

a_2d = np.array([[1,2,3],[3,4,5],[6,7,8]])
print('a_2d:\n', a_2d)

bool_array = a_2d > 3
print('bool_array:\n', bool_array)

print('Filter a_2d:\n', a_2d[bool_array])
a = np.arange(10)
b = np.arange(10)[::-1]

print('a+b:\n', a+b)
print('a-b:\n', a-b)
print('sqrt(a):\n', np.sqrt(a))
print('std:', a.std())
print('sum:', a.sum())

a = np.arange(4*5).reshape(4,5)
print('a:\n', a)
print('sum by column:', a.sum(axis=0))
print('sum by row:', a.sum(axis=1))
