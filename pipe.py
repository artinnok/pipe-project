import matplotlib.pyplot as plt
import numpy as np
import helper
from math import sqrt

# поддержка кириллицы
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

# установка формата вывода
np.set_printoptions(precision=6, suppress=True, linewidth=150)

P11 = 2 * 10 ** 5
PN1 = 10 ** 5
P12 = 3 * 10 ** 5
PN2 = 2 * 10 ** 5
X = np.array([[[P11], [PN1]], [[P12], [PN2]]])

B1 = 10 ** (-3)
B0 = 0
B = np.array([[B0], [B1]])

H = 10 ** (-6)
E = 0.05


class Solver:
    def get_w(self, b, x):
        """
        Вернет вектор прогноза
        :param b:
        :param x:
        :return:
        """
        l = len(b) / 2
        b0 = b[:l].reshape(l, 1)
        b1 = b[l:].reshape(l, 1)
        n = int(l + 1)

        a2t = helper.get_a2(n)
        # костыль
        a2t = helper.crutch(b1, a2t)

        a1t = helper.get_a1(n)

        zeros = np.zeros((2, 1))
        # костыль
        ones = np.ones((len(b1), 1))

        # должно быть b1 вместо ones
        top = np.append(ones, a2t, 1)
        bottom = np.append(zeros, a1t, 1)
        d = np.append(top, bottom, 0)

        g = np.append(b0, x, 0)
        dinv = np.linalg.inv(d)
        w = np.dot(dinv, g)
        return w

    def get_e(self, b, index, x):
        """
        Вернет одно значение якобиана
        :param b:
        :param index:
        :param x:
        :return:
        """
        copy_b = np.array(b, copy=True)
        copy_b[index] += H
        w_mod = self.get_w(copy_b, x)
        w = self.get_w(b, x)
        e = w_mod - w
        e /= H
        return e

    def get_jacobian(self, b, x):
        """
        Вернет якобиан
        :param b:
        :param x:
        :return:
        """
        out = np.concatenate(
            [self.get_e(b, index, x) for index, item in enumerate(b)],
            axis=1
        )
        return out

    def get_ws_jacobians(self, b, x):
        """
        Вернет вектор всех прогнозов и вектор из якобианов
        :param b:
        :param x:
        :return:
        """
        ws = np.concatenate([self.get_w(b, item) for item in x], axis=0)
        jacobians = np.concatenate(
            [self.get_jacobian(b, item) for item in x], axis=0
        )
        return ws, jacobians

    def get_delta_b(self, y, b, x):
        """
        Вернет оценку дельта b
        :param b:
        :param x:
        :return:
        """
        ws, jacobians = self.get_ws_jacobians(b, x)
        a = np.dot(jacobians.T, jacobians)
        ainv = np.linalg.inv(a)
        delta_b = np.dot(ainv, jacobians.T)
        e = y - ws
        delta_b = np.dot(delta_b, e)
        return delta_b

    def get_some_value(self, delta):
        squares = map(lambda x: x * x, delta)
        squares_sum = sum(list(squares))
        return sqrt(squares_sum) / len(delta)


s = Solver()
y, yy = s.get_ws_jacobians(B, X)
y[0, 0] += 50
delta_B = s.get_delta_b(y, B, X)
delta = delta_B
B = B + delta_B
some_value = s.get_some_value(delta)

while some_value > E:
    delta_B = s.get_delta_b(y, B, X)
    delta = np.append(delta, delta_B, axis=0)
    B = B + delta_B
    some_value = s.get_some_value(delta)

print(B)
