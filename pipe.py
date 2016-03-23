from openpyxl import Workbook, load_workbook
import numpy as np

wb = load_workbook(filename='data.xlsm', read_only=True)
ws = wb.active


MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
FINISH_ROW = 101
RANGE = range(START_ROW, FINISH_ROW)
WS = ws


# general functions
def split_comma(input: str) -> list: # разделяет по запятым в массив
    if type(input) is not str:
        return input
    return [int(item) for item in input.split(',')]


def split_slash(input: str) -> list: # разделяет по / в массив
    return [float(item) for item in input.split('/')]


def replace_comma(input: str) -> str: # заменяет , на .
    return input.replace(',', '.')


def get_mean(input: list) -> float: # считает среднее
    return np.mean(input)


# helper class
class Data():
    def __init__(self, col: str, row: range):
        self.col = col
        self.row = row
        self.list = [col+str(item) for item in row]

    def filter_data(self):
        output = [WS[item].value for item in self.list if WS[item].value is not None]
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
        return output


# main classes
class Linear():
    def __init__(self, input: list, output: list):
        self.input = input
        self.output = output


flow = Data('K', range(9, 101)) # flow

ukhta_pumps = Data('U', RANGE)
ukhta_pressure = Data('Y', RANGE).get_pressure()

sindor_pumps = Data('AE', RANGE)
sindor_pressure = Data('AG', RANGE).get_pressure()

for item in ukhta_pressure - sindor_pressure:
    print(item)
