from openpyxl import Workbook, load_workbook
from numpy.linalg import lstsq
import numpy as np

wb = load_workbook(filename='data.xlsm', read_only=True)
ws = wb.active

MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
ROWS = []
FINISH_ROW = 101
WS = ws


class NPS:
    def __init__(self, name_col, pump_col, Pin_col, Pout_col):
        self.name = WS[name_col + str(NAME_ROW)].value
        self.pump_col = pump_col
        self.Pin_col = Pin_col
        self.Pout_col = Pout_col

    def get_pump(self, col):
        row_col = []
        output = []
        for i in range(START_ROW, FINISH_ROW + 1):
            if self.check_row(i):
                row_col.append(col + str(i))
        for j in row_col:
            u = [0, 0, 0, 0]
            data = WS[j].value
            if type(data) is str:
                data = data.split(',')
                data = [int(x) for x in data]
            elif type(data) is int:
                data = [data]
            for i in data:
                u[i - 1] = 1
            output.append(u)
        return output

    def get_pressure(self, col):
        row_col = []
        output = []
        for i in range(START_ROW, FINISH_ROW + 1):
            if self.check_row(i):
                row_col.append(col + str(i))
        for j in row_col:
            data = WS[j].value
            data = data.replace(',', '.')
            data = data.split('/')
            data = [float(x) for x in data]
            res = (max(data) + min(data))/len(data)
            output.append(res)
        return output

    def get_data(self, col):
        row_col = []
        output = []
        for i in range(START_ROW, FINISH_ROW + 1):
            if self.check_row(i):
                row_col.append(col + str(i))
        for j in row_col:
            output.append(WS[j].value)
        return output

    def get_raw_result(self):
        output = []
        mode = self.get_data(MODE_COL)
        for i in mode:
            pass


    def check_row(self, row):
        data = WS[MODE_COL + str(row)].value
        if data is not None:
            if data.find('.') != -1:
                return True
        return False


ukhta = NPS('T', 'U', 'W', 'Y')

for key, value in ukhta.get_raw_result().items():
    print(key, '=>', value)
