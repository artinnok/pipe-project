import numpy as np

from utils import replace_comma, split_comma, split_slash, get_mean


class Data():
    def __init__(self, col: str, row: range, ws):
        self.col = col
        self.row = row
        self.ws = ws
        self.list = [col+str(item) for item in row]

    def filter_data(self):
        output = [self.ws[item].value for item in self.list if self.ws[item].value is not None]
        return output

    def get_pressure(self):
        output = self.filter_data()
        output = [replace_comma(item) for item in output]
        output = [split_slash(item) for item in output]
        output = [get_mean(item) for item in output]
        output = [(item*9.8*10000 + 101350)/(850*9.8) for item in output] # переводим в м
        return np.array(output)

    def get_pump(self):
        output = self.filter_data()
        output = [split_comma(item) for item in output]
        return output

    def get_flow(self):
        output = self.filter_data()
        output = [(item*1000*1000)/(850*2*60*60) for item in output] # переводим в м3/сек
        return np.array(output)