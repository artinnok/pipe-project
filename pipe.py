import matplotlib.pyplot as plt
import numpy as np
import helper
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

h = 10 ** (-6)
E = 0.5


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

        A2T = helper.get_a2(n)
        A2T = helper.crutch(-beta1, A2T)  # костыль
        ones = -np.ones((len(beta1), 1))  # костыль
        top = np.append(ones, A2T, 1)

        A1T = helper.get_a1(n)
        zeros = np.zeros((2, 1))
        bottom = np.append(zeros, A1T, 1)

        D = np.append(top, bottom, 0)
        Dinv = np.linalg.inv(D)
        u = np.append(-beta0, x_column, 0)
        v = np.dot(Dinv, u)
        return v

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
        copy_theta[index] += h
        v_mod = self.single_solve(copy_theta, x_column)
        v = self.single_solve(theta, x_column)
        e = v_mod - v
        e /= h
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
            [self.singe_jacobian(theta, col) for col in x], axis=0
        )
        return out

    def get_ls_theta(self, y, theta, x):
        """
        МНК
        :param theta:
        :param y:
        :param x:
        :return:
        """
        v = self.solve(theta, x)
        H = self.jacobian(theta, x)
        A = np.dot(H.T, H)
        Ainv = np.linalg.inv(A)
        delta_theta = np.dot(Ainv, H.T)
        e = y - v
        delta_theta = np.dot(delta_theta, e)
        return delta_theta, v

    def get_wls_theta(self, y, theta, x):
        """
        ОМНК
        :param theta:
        :param x:
        :param y:
        :return:
        """
        v = self.solve(theta, x)
        H = self.jacobian(theta, x)
        W = self.weight(theta, x)
        Q = np.dot(W.T, W)
        A = np.dot(H.T, Q)
        A = np.dot(A, H)
        Ainv = np.linalg.inv(A)
        delta_theta = np.dot(Ainv, H.T)
        delta_theta = np.dot(delta_theta, Q)
        e = y - v
        delta_theta = np.dot(delta_theta, e)
        return delta_theta, v

    def wrapper(self, func, y, x):
        """
        Обертка для МНК и ОМНК
        :param func: get_ls_theta или get_wls_theta
        :param y: эталонные измерения
        :param x: краевые условия
        :return:
        """
        delta_theta, v = getattr(self, func)(y, THETA, x)
        delta = delta_theta
        theta = THETA + delta_theta
        some_value = self.some_value(delta)

        while some_value > E:
            delta_theta, v = getattr(self, func)(y, theta, x)
            delta = np.append(delta, delta_theta, axis=0)
            theta = theta + delta_theta
            some_value = self.some_value(delta)

        return theta, v

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
        A = np.dot(H.T, Q)
        A = np.dot(A, H)
        k_epsilon = self.get_k_epsilon(len(H), q_sigma, p_sigma)

        Ainv = np.linalg.inv(A)
        k_theta = np.dot(Ainv, H.T)
        k_theta = np.dot(k_theta, Q)
        k_theta = np.dot(k_theta, k_epsilon)
        k_theta = np.dot(k_theta, Q)
        k_theta = np.dot(k_theta, H)
        k_theta = np.dot(k_theta, Ainv)

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
        k_y = np.dot(h, k_theta)
        k_y = np.dot(k_y, h.T)

        return k_y

s = Solver()
F = s.solve(THETA, X)
V = s.single_solve(THETA, X[0])
bound = helper.repeat(5, X)

for item in range(1000):
    modes = helper.mass_generate(5, F, THETA)
    result, calc = s.wrapper(s.get_wls_theta.__name__, modes, bound)
    e = modes - calc
    q = e[::L]
    p = [e[i::L] for i in range(2, L - 1)]
    if item == 0:
        out = result
        q_std = np.std(q)
        p_std = np.std(p)
        k_y = s.get_k_y(result, X, q_std, p_std)
        d = sqrt(k_y[2, 2])
        out_calc = calc
    else:
        out = np.append(out, result, axis=1)
        q_std = np.append(q_std, np.std(q))
        p_std = np.append(p_std, np.std(p))
        out_calc = np.append(out_calc, calc, axis=1)
        k_y = np.append(k_y, s.get_k_y(result, X, np.std(q), np.std(p)), axis=0)
        d = np.append(d, sqrt(s.get_k_y(result, X, np.std(q), np.std(p))[2, 2]))

b0 = np.concatenate((out[0], out[1]))
b1 = np.concatenate((out[2], out[3]))
q = out_calc[0]
p = out_calc[2]

out = np.array([(pr - V[2, 0]) / di for pr, di in zip(p, d)])
plt.figure()
for item in range(6, 11):
    plt.plot(np.sort(out), stats.t.pdf(np.sort(out), item))
plt.hist(out, normed=True)

plt.show()
