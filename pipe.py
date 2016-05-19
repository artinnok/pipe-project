import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

START = 'C'
END = 'H'
FLOW = 'A'
PUMP_ROWS = ['B', 'D']
PRESSURE_ROWS = ['E', 'F']


def mvregress(x, y):
    xt = np.transpose(x)
    a = np.dot(xt, x)
    if np.linalg.det(a) == 0:
        print('singular matrix X')
    else:
        ainv = np.linalg.inv(a)
        b = np.dot(ainv, xt)
        b = np.dot(b, y)
        return b

start = Data('F').get_pressure()
end = Data('H').get_pressure()
ones = np.ones(len(start))

flow = Data('A').get_flow()

X = np.array([ones, flow])
X = np.transpose(X)
Y = np.array([start, end])
Y = np.transpose(Y)
B = mvregress(X, Y)

start_predict = 71.38 - 27.81 * flow
end_predict = 87.43 - 103 * flow
plt.scatter(flow, start_predict, c='r')
plt.scatter(flow, start)
plt.title('Зависимость расход - давление на выходе ПНПС Ухта')
plt.figure(2)
plt.scatter(flow, end_predict, c='r')
plt.scatter(flow, end)
plt.title('Зависимость расход - давление на входе ПНПС Синдор')
plt.figure(3)
plt.scatter(flow, end_predict - start_predict, c='r')
plt.scatter(flow, end - start)
plt.title('ЛУ Ухта - Синдор')
plt.show()
print(B)
