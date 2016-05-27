import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from collections import defaultdict

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

START = 'C'
END = 'H'
FLOW = 'A'
PUMP_ROWS = ['D', 'G']
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


def prepare_pumps(data):
    res = []
    for item in data:
        out = '-'
        res.append(out.join(item))
    return res


def join(modes, pressure):
    d = defaultdict(list)
    for index, item in enumerate(modes):
        d[item].append(pressure[index])
    for key, value in d.items():
        d[key] = np.array(value)
    return d

pumps = np.array([Data(item).get_values() for item in PUMP_ROWS])
pumps = np.transpose(pumps)
pressure = np.array([Data(item).get_pressure() for item in PRESSURE_ROWS])
pressure = np.transpose(pressure)

modes = prepare_pumps(pumps)
test = join(modes, pressure)

inp = Data('E').get_pressure()
out = Data('F').get_pressure()
flow = Data('A').get_flow()

plt.scatter(flow, inp)
plt.figure(2)
plt.scatter(flow, out)
plt.figure()
plt.scatter(flow, out - inp)
plt.show()