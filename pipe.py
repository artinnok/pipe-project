import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from collections import defaultdict

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

P1 = 2 * 10 ** 5
PN = 10 ** 5
B1 = 10 ** (-3)
B0 = 0
N = 4

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def get_b(n):
    b0 = np.ones(n) * B0
    b1 = np.ones(n) * B1
    b = np.array([b0, b1])
    b = np.transpose(b)
    return b


def get_b1(n):
    b1 = np.ones(n - 1) * B1
    return b1


def get_b0(n):
    b0 = np.ones(n - 1) * B0
    return b0


def get_a2(n):
    a2 = []
    for i in range(n - 1):
        y = []
        for j in range(n):
            if j == i:
                y.append(-1)
            elif j == i + 1:
                y.append(1)
            else:
                y.append(0)
        a2.append(y)
    return np.array(a2)


def get_a1(n):
    first = np.zeros(n)
    first[0] = 1
    last = np.zeros(n)
    last[-1] = 1
    a1 = np.array([first, last])
    return a1


def crutch(b, a):
    out = []
    for index, item in enumerate(b):
        out.append(a[index] * item)
    return np.array(out)


class Solver:
    def __init__(self, b):
        self.b = b
        self.b0 = b[0]
        self.b1 = b[1]
        self.N = len(b) + 1

    def solve(self):
        x = np.array([P1, PN])
        x = x.reshape(len(x), 1)

        a2t = get_a2(self.N)
        # костыль
        a2t = crutch(b1, a2t)

        a1t = get_a1(N)

        zeros = np.array([0, 0])
        zeros = zeros.reshape(len(zeros), 1)

        # костыль
        ones = np.ones(len(self.b1))
        ones = ones.reshape(len(ones), 1)

        # должно быть b1 вместо ones
        top = np.append(ones, a2t, 1)
        bottom = np.append(zeros, a1t, 1)

        D = np.append(top, bottom, 0)
        g = np.append(b0, x, 0)

        Dinv = np.linalg.inv(D)
        w = np.dot(Dinv, g)

b1 = get_b1(N)
b1 = b1.reshape(len(b1), 1)

b0 = get_b0(N)
b0 = b0.reshape((len(b0), 1))

x = np.array([P1, PN])
x = x.reshape(len(x), 1)

a2t = get_a2(N)
# костыль
a2t = crutch(b1, a2t)

a1t = get_a1(N)

zeros = np.array([0, 0])
zeros = zeros.reshape(len(zeros), 1)

# костыль
ones = np.ones(len(b1))
ones = ones.reshape(len(ones), 1)

# должно быть b1 вместо ones
top = np.append(ones, a2t, 1)
bottom = np.append(zeros, a1t, 1)

D = np.append(top, bottom, 0)
g = np.append(b0, x, 0)

Dinv = np.linalg.inv(D)
w = np.dot(Dinv, g)
print(w)
