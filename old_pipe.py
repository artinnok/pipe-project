from openpyxl import Workbook, load_workbook
from numpy.linalg import lstsq
import numpy as np

wb = load_workbook(filename='data.xlsm', read_only=True)
ws = wb.active

MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
FINISH_ROW = 101
WS = ws

class NPS:
    def __init__(self, name_col, pump_col, Pin_col, Pout_col):
        self.name = WS[name_col + str(NAME_ROW)].value
        self.pump_col = pump_col
        self.Pin_col = Pin_col
        self.Pout_col = Pout_col
        self.data_pump = self.get_pump(self.pump_col)
        self.data_Pin = self.get_data(self.Pin_col)
        self.data_Pout = self.get_data(self.Pout_col)
        self.data_mode = self.get_data(MODE_COL)
        self.data_flow = self.get_data(FLOW_COL)

    def get_data(self, col):
        rc= []
        output = []
        for item in range(START_ROW, FINISH_ROW + 1):
            rc.append(col + str(item))
        for item in rc:
            output.append(WS[item].value)
        return output

    def get_pump(self, col):
        rc= []
        output = []
        for item in range(START_ROW, FINISH_ROW + 1):
            rc.append(col + str(item))
        for item in rc:
            data = WS[item].value
            out = [0, 0, 0, 0]
            if data is not None:
                if type(data) is str:
                    data = data.split(',')
                    for i in data:
                        out[int(i) - 1] = 1
                elif type(data) is int:
                    out[data - 1] = 1
            output.append(out)
        return output

    def get_final_data(self):
        output = []
        length = len(self.data_mode)
        for i in range(length):
            item = self.data_mode[i]
            if (item is not None):
                if (item.find('.') > -1):
                    data = {}
                    data['Pin'] = self.data_Pin[i]
                    data['Pout'] = self.data_Pout[i]
                    data['pump'] = self.data_pump[i]
                    data['mode'] = self.data_mode[i]
                    data['flow'] = self.data_flow[i]
                    data['name'] = self.name
                    output.append(data)
        return output

    def get_active_pump_count(self):
        output = {'1':0, '2':0, '3':0, '4':0}
        for i in [1, 2, 3, 4]:
            for j in self.data_pump:
                if j[i - 1] is 1:
                    output['%s' % i] += 1
        return output

    def get_pump_flow(self):
        output = {'1':[], '2':[], '3':[], '4':[]}
        for i in [1, 2, 3, 4]:
            length = len(self.data_pump)
            for j in range(length):
                if self.data_pump[j][i - 1] is 1:
                    output['%s' % i].append(self.data_flow[j])
        return output

    def get_filtered_pump(self):
        output = set()
        for key, value in self.get_pump_flow().items():
            max_val = max(value)
            min_val = min(value)
            med = (max_val + min_val)/2
            if (max_val - min_val)/med >= 0.05:
                output.add(key)
        return output

    def get_filtered_data(self):
        output = []
        pump = self.get_filtered_pump()
        data = self.get_final_data()
        excluded_pump = set(['1', '2', '3', '4']) - pump
        for i in excluded_pump:
            for j in data:
                if j['pump'][int(i) - 1] is 1:
                    continue
                else:
                    output.append(j)
        return output

    def get_dP(self):
        Pin = []
        Pout = []
        output = []
        for i in self.data_Pin:
            if i is not None:
                data = i.replace(',', '.')
                data = data.split('/')
                sum = 0
                for j in data:
                    j = float(j)
                    sum += j
                Pin.append(sum/2)
        for i in self.data_Pout:
            if i is not None:
                data = i.replace(',', '.')
                data = data.split('/')
                sum = 0
                for j in data:
                    j = float(j)
                    sum += j
                Pout.append(sum/2)
        length = len(Pin)
        for i in range(length):
            output.append(Pout[i] - Pin[i])
        return output

    def get_flow(self):
        output = []
        for i in self.data_flow:
            if i is not None:
                output.append(i)
        return output


ukhta = NPS('T', 'U', 'W', 'Y')
#sindor = NPS('AD', 'AE', 'AG', 'AI')
#mikun = NPS('AN', 'AO', 'AQ', 'AS')
#urdoma = NPS('AX', 'AY', 'BA', 'BC')

#nps_list = [ukhta, sindor, mikun, urdoma]

F = np.array(ukhta.get_flow())
dP = np.array(ukhta.get_dP())
E = np.ones(len(F))
