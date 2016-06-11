from math import sqrt

import helper as hr
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np

# поддержка кириллицы
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

# установка формата вывода
np.set_printoptions(precision=10, suppress=True, linewidth=150)

# краевые условия
X = np.array([
    [[2 * 10 ** 5], [1 * 10 ** 5]],
    [[3 * 10 ** 5], [1.5 * 10 ** 5]],
    [[1.5 * 10 ** 5], [1 * 10 ** 5]],
    [[4 * 10 ** 5], [2 * 10 ** 5]],
    [[3.5 * 10 ** 5], [1.9 * 10 ** 5]]
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

# количество объектов ТУ
OBJ = int(len(THETA) / 2)

# количество переменных в гидравлическом уравнении
L = OBJ + 2

# необходимо для вычисления колонки якобиана
STEP = 10 ** (-6)

# уставка для МНК и ОМНК
SETPOINT = 0.5

N = 10
K = 4


class Solver:
    def single_solve(self, theta, x_column):
        """
        Вернет вектор прогноза
        :param theta:
        :param x_column:
        :return:
        """
        beta0 = theta[:OBJ].reshape(OBJ, 1)
        beta1 = theta[OBJ:].reshape(OBJ, 1)
        n = OBJ + 1

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
             for index, th in enumerate(theta)], axis=1
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
        W = self.weight(x)
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
        dTHETA, F = getattr(self, func)(y, THETA, x)
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
    def weight(x):
        """
        Вернет матрицу весов для ОМНК
        :param x:
        :return:
        """
        w = np.ones(OBJ + 2)
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

    def single_k_epsilon(self, Y, theta, x_column):
        """
        Вернет одно значение K_{epsilon}
        :param Y: соответствующий колонке Х зашумленный расчет
        :param theta:
        :param x_column:
        :return:
        """
        F = self.single_solve(theta, x_column)
        E = Y - F
        k_epsilon = E * E.T
        return k_epsilon

    def k_epsilon(self, Y, theta, x):
        """
        Считает K_{epsilon}
        :param Y:
        :param theta:
        :param x:
        :return:
        """
        l = OBJ + 2
        out = sum([
            self.single_k_epsilon(Y[index * l: (index + 1) * l], theta, col)
            for index, col in enumerate(x)
        ])
        out /= N - K
        return out

    def k_epsilon_block(self, Y, theta, x):
        k_epsilon = self.k_epsilon(Y, theta, x)
        k_epsilon = np.array([k_epsilon for i in range(N)])
        block = sc.linalg.block_diag(*k_epsilon)
        return block

    def k_theta(self, Y, theta, x):
        """
        Считает K_{theta}
        :param Y:
        :param theta:
        :param x:
        :return:
        """
        H = self.jacobian(theta, x)
        W = self.weight(x)
        Q = np.dot(W.T, W)
        A = hr.prod(H.T, Q, H)
        Ainv = np.linalg.inv(A)
        k_epsilon = self.k_epsilon_block(Y, theta, x)
        k_theta = hr.prod(Ainv, H.T, Q, k_epsilon, Q, H, Ainv)
        return k_theta

    def k_y(self, Y, theta, x):
        """
        Считает K_{y}
        :param Y:
        :param theta:
        :param x:
        :return:
        """
        k_theta = self.k_theta(Y, theta, x)
        h = s.singe_jacobian(THETA, X[0])
        k_y = hr.prod(h, k_theta, h.T)
        return k_y

s = Solver()
X = hr.repeat(2, X, 0)
F = s.solve(THETA, X)

out = []
for item in range(1000):
    Y = hr.add_noise(F, OBJ)
    result_THETA, calc_F = s.wrapper('glsm_theta', Y, X)
    if item == 0:
        out = result_THETA
        out_e = Y - calc_F
        out_F = calc_F
    else:
        out = np.append(out, result_THETA, axis=1)
        out_e = np.append(out_e, Y - calc_F, axis=1)
        out_F = np.append(out_F, calc_F, axis=1)

b0 = np.concatenate((out[0], out[1]))
b1 = np.concatenate((out[2], out[3]))

q_e = np.concatenate(out_e[::L]).flatten()
p_e = np.concatenate([out_e[i::L] for i in range(2, L - 1)]).flatten()

q = out_F[0]
p = out_F[2]

K_Y = s.k_y(Y, THETA, X)
K_THETA = s.k_theta(Y, THETA, X)

hr.plot_norm(q, F[0, 0], sqrt(K_Y[0, 0]), 'q')
hr.plot_norm(p, F[2, 0], sqrt(K_Y[2, 2]), 'p')

hr.plot_norm(b0, B0, sqrt(K_THETA[0, 0]), 'beta 0')
hr.plot_norm(b1, B1, sqrt(K_THETA[2, 2]), 'beta 1')

hr.plot_norm(q_e, 0, sqrt(K_Y[0, 0]), 'q_e')
hr.plot_norm(p_e, 0, sqrt(K_Y[2, 2]), 'p_e')

t = (p - F[2, 0]) / sqrt(K_Y[2, 2])

hr.plot_t(t, N - K, 't')

plt.show()
