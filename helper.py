import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression, RANSACRegressor
from utils import replace_comma, split_comma, split_slash, get_mean, get_rows

WB = load_workbook(filename='input.xlsx', read_only=True)
WS = WB.active

RANGE = range(3, 49)
ROWS = get_rows(RANGE, WS)

P_SIGMA = 3000
Q_SIGMA = 5


class Data:
    def __init__(self, col: str):
        self.col = col
        self.row = ROWS
        self.list = [col + str(item) for item in self.row]

    def get_values(self):
        out = [WS[item].value for item in self.list]
        return out

    def get_pressure(self):
        out = self.get_values()
        out = [replace_comma(item) for item in out]
        out = [split_slash(item) for item in out]
        out = [get_mean(item) for item in out]
        # переводим в м
        out = [(item * 9.8 * 10000 + 101350) / (850 * 9.8) for item in out]
        return np.array(out)

    def get_pump(self):
        out = self.get_values()
        out = [split_comma(item) for item in out]
        return out

    def get_flow(self):
        out = self.get_values()
        # переводим в м3/сек
        out = [(item * 1000 * 1000) / (850 * 2 * 60 * 60) for item in out]
        return np.array(out)


class Regression:
    def linear_predict(self, x, y):  # линейная регрессия
        regr = LinearRegression()
        regr.fit(x, y)
        return regr.predict(x)

    def get_ransac(self, x, y):  # RANSAC регрессор
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=5)
        ransac.fit(x, y)
        return ransac

    def ransac_predict(self, x, y):
        ransac = self.get_ransac(x, y)
        return ransac.predict(x)

    def ransac_mask(self, x, y):
        ransac = self.get_ransac(x, y)
        in_mask = ransac.inlier_mask_
        out_mask = np.logical_not(in_mask)
        return in_mask, out_mask

    def get_inliers(self, x, y):
        in_mask, out_mask = self.ransac_mask(x, y)
        return x[in_mask], y[in_mask]

    def get_outliers(self, x, y):
        in_mask, out_mask = self.ransac_mask(x, y)
        return x[out_mask], y[out_mask]


# матрицы
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


def add_noise(data, l):
    l += 2
    copy = np.array(data, copy=True)
    copy[::l] += np.random.normal(0, Q_SIGMA, copy.shape)[::l]
    for item in range(2, l - 1):
        copy[item::l] += np.random.normal(0, P_SIGMA, copy.shape)[item::l]
    return copy


def repeat(count, data, axis):
    out = data
    for item in range(count - 1):
        out = np.append(out, data, axis=axis)
    return out


def prod(*args):
    out = args[0]
    for item in args[1:]:
        out = np.dot(out, item)
    return out


def plot_norm(x, mu, sigma, title=None):
    plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.plot(np.sort(x), stats.norm.pdf(np.sort(x), mu, sigma), linewidth=5)
    plt.hist(x, normed=True)
    plt.title(title)


def plot_t(x, df, title=None):
    plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.plot(np.sort(x), stats.t.pdf(np.sort(x), df), linewidth=5)
    plt.hist(x, normed=True)
    plt.title(title)
