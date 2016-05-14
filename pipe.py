import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

START = 'C'
END = 'K'
FLOW = 'A'
PUMP_ROWS = ['B', 'D', 'G']
PRESSURE_ROWS = ['E', 'F', 'H', 'I']


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

start = Data(START).get_pressure()[:16]
end = Data(END).get_pressure()[:16]
ones = np.ones(len(start))

flow = Data(FLOW).get_flow()[:16]
pumps = Data('M').get_pump()[:16]
p1 = Data('O').get_pressure()[:16]

dp = end - start
X = np.array([ones, dp])
X = np.transpose(X)

Y = np.array([p1, flow])
Y = np.transpose(Y)

B = mvregress(X, Y)
print(B)
