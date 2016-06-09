import matplotlib.pyplot as plt
import numpy as np
import helper as hr
from math import sqrt
import matplotlib.mlab as mlab
import scipy.stats as stats

# поддержка кириллицы
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

# установка формата вывода
np.set_printoptions(precision=10, suppress=True, linewidth=150)

P11 = 2 * 10 ** 5
PN1 = 1 * 10 ** 5
P12 = 3 * 10 ** 5
PN2 = 1 * 10 ** 5
P13 = 1.5 * 10 ** 5
PN3 = 1 * 10 ** 5

# краевые условия
X = np.array([
    [
        [P11],
        [PN1]
    ],
    [
        [P12],
        [PN2]
    ]
])

# начальные приближения коэффициентов, также они задают конфигурацию ТУ
B1 = 10 ** (-3)
B0 = 0
THETA = np.array([
    [B0],
    [B0],
    [B1],
    [B1]
])

L = int(len(THETA) / 2 + 2)

STEP = 10 ** (-6)
SETPOINT = 0.5


class Solver:
    def single_solve(self, theta, x_column):
        """
        Вернет вектор прогноза
        :param theta:
        :param x_column:
        :return:
        """
        l = int(len(theta) / 2)
        beta0 = theta[:l].reshape(l, 1)
        beta1 = theta[l:].reshape(l, 1)
        n = int(l + 1)

        A2T = hr.get_a2(n)
        A2T = hr.crutch(-beta1, A2T)  # костыль
        ones = -np.ones((len(beta1), 1))  # костыль
        top = np.append(ones, A2T, 1)

        A1T = hr.get_a1(n)
        zeros = np.zeros((2, 1))
        bottom = np.append(zeros, A1T, 1)

        D = np.append(top, bottom, 0)
        Dinv = np.linalg.inv(D)
        g = np.append(-beta0, x_column, 0)
        f = np.dot(Dinv, g)
        return f

    def solve(self, theta, x):
        """
        Вернет вектор всех прогнозов
        :param theta:
        :param x:
        :return:
        """
        out = np.concatenate(
            [self.single_solve(theta, col) for col in x], axis=0
        )
        return out

    def column_of_single_jacobian(self, theta, index, x_column):
        """
        Вернет одну колонку якобиана
        :param theta:
        :param index:
        :param x_column:
        :return:
        """
        copy_theta = np.array(theta, copy=True)
        copy_theta[index] += STEP
        f_mod = self.single_solve(copy_theta, x_column)
        f = self.single_solve(theta, x_column)
        e = f_mod - f
        e /= STEP
        return e

    def singe_jacobian(self, theta, x_column):
        """
        Вернет один якобиан
        :param theta:
        :param x_column:
        :return:
        """
        out = np.concatenate(
            [self.column_of_single_jacobian(theta, index, x_column)
             for index, th in enumerate(theta)],
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
            [self.singe_jacobian(theta, col) for col in x], axis=0
        )
        return out

    def lsm_theta(self, Y, theta, x):
        """
        МНК
        :param theta:
        :param Y:
        :param x:
        :return:
        """
        F = self.solve(theta, x)
        H = self.jacobian(theta, x)
        A = np.dot(H.T, H)
        Ainv = np.linalg.inv(A)
        E = Y - F
        dTHETA = hr.prod(Ainv, H.T, E)
        return dTHETA, F

    def glsm_theta(self, Y, theta, x):
        """
        ОМНК
        :param theta:
        :param x:
        :param Y:
        :return:
        """
        F = self.solve(theta, x)
        H = self.jacobian(theta, x)
        W = self.weight(theta, x)
        Q = np.dot(W.T, W)
        A = hr.prod(H.T, Q, H)
        Ainv = np.linalg.inv(A)
        E = Y - F
        dTHETA = hr.prod(Ainv, H.T, Q, E)
        return dTHETA, F

    def wrapper(self, func, y, x):
        """
        Обертка для МНК и ОМНК
        :param func: lsm_theta или glsm_theta
        :param y: эталонные измерения
        :param x: краевые условия
        :return:
        """
        dTHETA, v = getattr(self, func)(y, THETA, x)
        delta = dTHETA
        theta = THETA + dTHETA
        some_value = self.some_value(delta)

        while some_value > SETPOINT:
            dTHETA, F = getattr(self, func)(y, theta, x)
            delta = np.append(delta, dTHETA, axis=0)
            theta = theta + dTHETA
            some_value = self.some_value(delta)
        return theta, F

    @staticmethod
    def weight(theta, x):
        """
        Вернет матрицу весов для ОМНК
        :param theta:
        :param x:
        :return:
        """
        l = len(theta) / 2 + 2
        w = np.ones(l)
        w[0] = 1000

        n = len(x)
        W = np.concatenate([w for item in range(n)])
        W = np.diag(W)
        return W

    @staticmethod
    def some_value(data):
        """
        Вернет критерий расстояния для МНК и ОМНК
        :param data:
        :return:
        """
        squares = map(lambda x: x * x, data)
        squares_sum = sum(list(squares))
        return sqrt(squares_sum) / len(data)

    def get_k_epsilon(self, l, q_sigma, p_sigma):
        """
        Считает K_{epsilon}
        :param l:
        :param q_sigma:
        :param p_sigma:
        :return:
        """
        k_epsilon = np.zeros(l)
        k_epsilon[::L] = q_sigma ** 2
        for i in range(2, L - 1):
            k_epsilon[i::L] = p_sigma ** 2
        k_epsilon = np.diag(k_epsilon)
        return k_epsilon

    def get_k_theta(self, theta, x, q_sigma, p_sigma):
        """
        Считает K_{theta}
        :param theta:
        :param x:
        :param q_sigma:
        :param p_sigma:
        :return:
        """
        H = self.jacobian(theta, x)
        W = self.weight(theta, x)
        Q = np.dot(W.T, W)
        A = hr.prod(H.T, Q, H)
        Ainv = np.linalg.inv(A)
        k_epsilon = self.get_k_epsilon(len(H), q_sigma, p_sigma)
        k_theta = hr.prod(Ainv, H.T, Q, k_epsilon, Q, H, Ainv)
        return k_theta

    def get_k_y(self, theta, x, q_sigma, p_sigma):
        """
        Считает K_{y}
        :param theta:
        :param x:
        :param q_sigma:
        :param p_sigma:
        :return:
        """
        k_theta = self.get_k_theta(theta, x, q_sigma, p_sigma)
        h = s.singe_jacobian(THETA, X[0])
        k_y = hr.prod(h, k_theta, h.T)
        return k_y

s = Solver()
