import numpy as np
from openpyxl import load_workbook

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
