import numpy as np
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression, RANSACRegressor

from utils import replace_comma, split_comma, split_slash, get_mean, get_rows


WB = load_workbook(filename='data.xlsx', read_only=True)
WS = WB.active

RANGE = range(9, 101)
ROWS = get_rows(RANGE, WS)


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
        out = [(item * 9.8 * 10000 + 101350) / (850 * 9.8) for item in out]  # переводим в м
        return np.array(out)

    def get_pump(self):
        out = self.get_values()
        out = [split_comma(item) for item in out]
        return out

    def get_flow(self):
        out = self.get_values()
        out = [(item * 1000 * 1000) / (850 * 2 * 60 * 60) for item in out]  # переводим в м3/сек
        return np.array(out)


class Handler:
    @classmethod
    def linear_predict(cls, x, y):  # линейная регрессия
        regr = LinearRegression()
        regr.fit(x, y)
        return regr.predict(x)

    @classmethod
    def get_ransac(cls, x, y):  # RANSAC регрессор
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=5)
        ransac.fit(x, y)
        return ransac

    @classmethod
    def ransac_predict(cls, x, y):
        ransac = cls.get_ransac(x, y)
        return ransac.predict(x)

    @classmethod
    def ransac_mask(cls, x, y):
        ransac = cls.get_ransac(x, y)
        in_mask = ransac.inlier_mask_
        out_mask = np.logical_not(in_mask)
        return in_mask, out_mask

    @classmethod
    def get_inliers(cls, x, y):
        in_mask, out_mask = cls.ransac_mask(x, y)
        return x[in_mask], y[in_mask]

    @classmethod
    def get_outliers(cls, x, y):
        in_mask, out_mask = cls.ransac_mask(x, y)
        return x[out_mask], y[out_mask]
