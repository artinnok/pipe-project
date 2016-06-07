import matplotlib.pyplot as plt
import numpy as np
import helper
from math import sqrt
import matplotlib.mlab as mlab

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
    [[P11], [PN1]],
    [[P12], [PN2]]
])

B1 = 10 ** (-3)
B0 = 0
THETA = np.array([
    [B0],
    [B0],
    [B1],
    [B1]
])

h = 10 ** (-6)
E = 0.5


class Solver:
    def single_solve(self, theta, x):
        """
        Вернет вектор прогноза
        :param theta:
        :param x:
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
        u = np.append(-beta0, x, 0)
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
        copy_theta[index] += h
        v_mod = self.single_solve(copy_theta, x)
        v = self.single_solve(theta, x)
        e = v_mod - v
        e /= h
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

    def weight(self, theta, x):
        """
        Вернет вес
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

    def get_wls_theta(self, y, theta, x):
        """
        Взвешенный МНК
        :param theta:
        :param x:
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

    def get_some_value(self, data):
        squares = map(lambda x: x * x, data)
        squares_sum = sum(list(squares))
        return sqrt(squares_sum) / len(data)

    def wrapper(self, func, measure, boundary):
        delta_theta, v = getattr(self, func)(measure, THETA, boundary)
        delta = delta_theta
        theta = THETA + delta_theta
        some_value = self.get_some_value(delta)

        while some_value > E:
            delta_theta, v = getattr(self, func)(measure, theta, boundary)
            delta = np.append(delta, delta_theta, axis=0)
            theta = theta + delta_theta
            some_value = self.get_some_value(delta)

        return theta, v


s = Solver()
F = s.solve(THETA, X)
bound = helper.repeat(5, X)
l = int(len(THETA) / 2 + 2)
Y = np.array(F, copy=True)

for item in range(1000):
    modes = helper.mass_generate(5, Y, THETA)
    result, calc = s.wrapper(s.get_wls_theta.__name__, modes, bound)
    e = modes - calc
    q = e[::l]
    p = [e[i::l] for i in range(2, l - 1)]
    if item == 0:
        out = result
        out_calc = calc
        q_std = np.std(q)
        p_std = np.std(p)
    else:
        out = np.append(out, result, axis=1)
        q_std = np.append(q_std, np.std(q))
        p_std = np.append(p_std, np.std(p))
        out_calc = np.append(out_calc, calc, axis=1)

b0 = np.concatenate((out[0], out[1]))
b1 = np.concatenate((out[2], out[3]))

H = s.jacobian(THETA, bound)
W = s.weight(THETA, bound)
Q = np.dot(W.T, W)

k_e = np.zeros(len(H))
k_e[::l] = helper.Q_SIGMA ** 2
for i in range(2, l-1):
    k_e[i::l] = helper.P_SIGMA ** 2

k_e = np.diag(k_e)

A = np.dot(H.T, Q)
A = np.dot(A, H)
Ainv = np.linalg.inv(A)
k_theta = np.dot(Ainv, H.T)
k_theta = np.dot(k_theta, Q)
k_theta = np.dot(k_theta, k_e)
k_theta = np.dot(k_theta, Q)
k_theta = np.dot(k_theta, H)
k_theta = np.dot(k_theta, Ainv)

theta = result
hh = s.singe_jacobian(THETA, X[0])
k_y = np.dot(hh, k_theta)
k_y = np.dot(k_y, hh.T)
v = s.single_solve(THETA, X[0])

q = out_calc[0]
p = out_calc[2]

plt.figure()
plt.plot(np.sort(q), mlab.normpdf(np.sort(q), v[0, 0], sqrt(k_y[0, 0])))
plt.hist(q, normed=True)

plt.figure()
plt.plot(np.sort(p), mlab.normpdf(np.sort(p), v[2, 0], sqrt(k_y[2, 2])))
plt.hist(p, normed=True)

plt.show()



"""
plt.figure()
plt.plot(np.sort(b0), mlab.normpdf(np.sort(b0), B0, sqrt(k_theta[0, 0])))
plt.hist(b0, normed=True)
plt.title('B0')

plt.figure()
plt.plot(np.sort(b1), mlab.normpdf(np.sort(b1), B1, sqrt(k_theta[2, 2])))
plt.hist(b1, normed=True)
plt.title('B1')

plt.show()
"""