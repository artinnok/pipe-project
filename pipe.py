import numpy as np
from openpyxl import load_workbook

from helper import Data

wb = load_workbook(filename='data.xlsx', read_only=True)
ws = wb.active

MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
FINISH_ROW = 101
WS = ws


# main classes
class Linear:
    def __init__(self, inp: np.array, out: np.array, flow: np.array):
        self.inp = inp
        self.out = out
        self.flow = flow
        self.diff = inp - out


flow = Data('K', RANGE, ws)

# ukhta_pumps = Data('U', RANGE)
# ukhta_pressure = Data('Y', RANGE).get_pressure()
#
# sindor_pumps = Data('AE', RANGE)
# sindor_pressure = Data('AG', RANGE).get_pressure()
#
#
# us = Linear(ukhta_pressure, sindor_pressure, flow)
