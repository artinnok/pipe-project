import numpy as np

from utils import replace_comma, split_comma, split_slash, get_mean


class Data:
    def __init__(self, col: str, row: range, ws):
        self.col = col
        self.row = row
        self.ws = ws
        self.list = [col + str(item) for item in row]

    def filter_data(self):
        out = [self.ws[item].value for item in self.list if self.ws[item].value is not None]
        return out

    def get_pressure(self):
        out = self.filter_data()
        out = [replace_comma(item) for item in out]
        out = [split_slash(item) for item in out]
        out = [get_mean(item) for item in out]
        out = [(item * 9.8 * 10000 + 101350) / (850 * 9.8) for item in out]  # переводим в м
        return np.array(out)

    def get_pump(self):
        out = self.filter_data()
        out = [split_comma(item) for item in out]
        return out

    def get_flow(self):
        out = self.filter_data()
        out = [(item * 1000 * 1000) / (850 * 2 * 60 * 60) for item in out]  # переводим в м3/сек
        return np.array(out)
