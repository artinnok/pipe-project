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
H = 10 ** (-6)

np.set_printoptions(precision=6, suppress=True, linewidth=150)


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
    def get_w(self, b):
        l = len(b) / 2
        b0 = b[:l].reshape(l, 1)
        b1 = b[l:].reshape(l, 1)
        N = int(l + 1)
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
        d = np.append(top, bottom, 0)
        g = np.append(b0, x, 0)

        dinv = np.linalg.inv(d)
        w = np.dot(dinv, g)
        return w

    def get_jacobian(self, b):
        out = []
        for index__i, item_i in enumerate(b):
            copy_b = np.array(b, copy=True)
            copy_b[index__i] += H
            w_mod = self.get_w(copy_b)
            w = self.get_w(b)
            e = w_mod - w
            e /= H
            out.append(e)
        return np.array(out)


b = np.array([B0, B0, B0, B1, B1, B1])
s = Solver()
jacobian = s.get_jacobian(b)
jacobian = np.transpose(jacobian)
a, b, c = jacobian.shape
print(jacobian.reshape(b, c))
