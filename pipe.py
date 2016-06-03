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
PN1 = 1 * 10 ** 5
P12 = 3 * 10 ** 5
PN2 = 1 * 10 ** 5
X = np.array([[[P11], [PN1]], [[P12], [PN2]]])
# X = np.array([[P11], [PN1]])
B1 = 10 ** (-3)
B0 = 0
THETA = np.array([[B0], [B1]])

H = 10 ** (-6)
E = 0.05


class Solver:
    def single_solve(self, theta, x):
        """
        Вернет вектор прогноза
        :param theta:
        :param x:
        :return:
        """
        l = len(theta) / 2
        b0 = theta[:l].reshape(l, 1)
        b1 = theta[l:].reshape(l, 1)
        n = int(l + 1)

        a2t = helper.get_a2(n)
        a2t = helper.crutch(-b1, a2t)  # костыль
        ones = -np.ones((len(b1), 1))  # костыль
        top = np.append(ones, a2t, 1)

        a1t = helper.get_a1(n)
        zeros = np.zeros((2, 1))
        bottom = np.append(zeros, a1t, 1)

        d = np.append(top, bottom, 0)
        dinv = np.linalg.inv(d)
        u = np.append(-b0, x, 0)
        w = np.dot(dinv, u)
        return w

    def solve(self, theta, x):
        """
        Вернет вектор всех прогнозов
        :param theta:
        :param x:
        :return:
        """
        out = np.concatenate(
            [self.single_solve(theta, item) for item in x], axis=0
        )
        return out

    def column_of_single_jacobian(self, theta, index, x):
        """
        Вернет одну колонку якобиана
        :param theta:
        :param index:
        :param x:
        :return:
        """
        copy_theta = np.array(theta, copy=True)
        copy_theta[index] += H
        f_mod = self.single_solve(copy_theta, x)
        f = self.single_solve(theta, x)
        e = f_mod - f
        e /= H
        return e

    def singe_jacobian(self, theta, x):
        """
        Вернет один якобиан
        :param theta:
        :param x:
        :return:
        """
        out = np.concatenate(
            [self.column_of_single_jacobian(theta, index, x)
             for index, item in enumerate(theta)],
            axis=1
        )
        return out

    def jacobian(self, theta, x):
        """
        Вернет матрицу якобиана
        :param theta:
        :param x:
        :return:
        """
        out = np.concatenate(
            [self.singe_jacobian(theta, item) for item in x], axis=0
        )
        return out

    def delta_theta(self, y, theta, x):
        """
        Вернет оценку дельта тета
        :param theta:
        :param y:
        :param x:
        :return:
        """
        ws = self.solve(theta, x)
        jacobian = self.jacobian(theta, x)
        a = np.dot(jacobian.T, jacobian)
        ainv = np.linalg.inv(a)
        delta_theta = np.dot(ainv, jacobian.T)
        e = y - ws
        delta_theta = np.dot(delta_theta, e)
        return delta_theta

    def get_some_value(self, data):
        squares = map(lambda x: x * x, data)
        squares_sum = sum(list(squares))
        return sqrt(squares_sum) / len(data)


s = Solver()
print(s.solve(THETA, X))

"""
y, yy = s.get_ws_jacobians(B, X)
print(y)
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
    """
