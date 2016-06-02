import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from collections import defaultdict

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

P11 = 2 * 10 ** 5
PN1 = 10 ** 5
P12 = 3 * 10 ** 5
PN2 = 2 * 10 ** 5
X = np.array([[P11, PN1], [P12, PN2]])
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


def normalize(data):
    max_value = max(data)
    min_value = min(data)
    out = []
    for item in data:
        foo = item - min_value
        foo /= max_value - min_value
        out.append(foo)
    return out


class Solver:
    def get_w(self, b, x):
        l = len(b) / 2
        b0 = b[:l].reshape(l, 1)
        b1 = b[l:].reshape(l, 1)
        N = int(l + 1)

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

    def get_jacobian(self, b, x):
        out = []

        for index, item in enumerate(b):
            copy_b = np.array(b, copy=True)
            copy_b[index] += H
            w_mod = self.get_w(copy_b, x)
            w = self.get_w(b, x)
            e = w_mod - w
            e /= H
            out.append(e)

        return np.array(out)

    def get_ws_jacobians(self, x):
        ws, jacobians = [], []

        for item in x:
            item = item.reshape(len(item), 1)
            ws.append(self.get_w(b, item))
            jacobians.append((self.get_jacobian(b, item)))

        ws = np.array(ws)
        jacobians = np.array(jacobians)
        return ws, jacobians

    def get_delta_b(self, x):
        ws, jacobians = self.get_ws_jacobians(x)
        mod = np.array(ws, copy=True)
        print(jacobians)
        out = []

        for item in mod:
            data = np.squeeze(item).T
            data[0] += 50

        ws = np.squeeze(ws).T
        mod = np.squeeze(mod).T
        for index, item in enumerate(jacobians):
            data = np.squeeze(item).T
            a = np.dot(data.T, data)
            ainv = np.linalg.inv(a)
            test = np.dot(ainv, data.T)
            e = mod[:, index] - ws[:, index]
            out.append(np.dot(test, e))
        return np.array(out)


b = np.array([B0, B1])
s = Solver()
print(b + s.get_delta_b(X))
