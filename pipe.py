from openpyxl import Workbook, load_workbook
import numpy as np

wb = load_workbook(filename='data.xlsm', read_only=True)
ws = wb.active


MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
FINISH_ROW = 101
WS = ws


# general functions
def split_comma(input: str) -> list:
    return [int(item) for item in input.split(',')]


def split_slash(input: str) -> list:
    return [float(item) for item in input.split('/')]


def replace_comma(input: str) -> str:
    return input.replace(',', '.')


def get_mean(input: list) -> float:
    return np.mean(input)


def get_cell_value(col: str, row: int) -> str:
    return WS[col+str(row)].value


# helper classes
class Pressure():
    def __init__(self, col: str, row: range):
        self.col = col
        self.row = row
        self.list = [col+str(item) for item in row]

    def get_data(self):
        output = [WS[item].value for item in self.list if WS[item].value is not None]
        output = [replace_comma(item) for item in output]
        output = [split_slash(item) for item in output]
        output = [get_mean(item) for item in output]
        return output


class Pump():
    def __init__(self, col: str, row: range):
        self.col = col
        self.row = row
        self.list = [col+str(item) for item in row]

    def get_data(self):
        output = [WS[item].value for item in self.list if WS[item].value is not None]
        output = [split_comma(item) for item in output]
        return output


# main classes
class Linear():
    def __init__(self, input: list, output: list):
        self.input = input
        self.output = output


a = Pump('T', range(9, 20))
b = Pressure('Y', range(9,20))
print(a.get_data(), b.get_data())