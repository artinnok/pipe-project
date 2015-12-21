from openpyxl import Workbook, load_workbook
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

wb = load_workbook(filename='data.xlsm', read_only=True)
ws = wb.active


MODE_COL = 'E'
FLOW_COL = 'K'
NAME_ROW = 6
START_ROW = 9
FINISH_ROW = 101
WS = ws


def get_rows():
    output = []
    for i in range(START_ROW, FINISH_ROW + 1):
        data = WS[MODE_COL + str(i)].value
        if data is not None:
            if data.find('.') != -1:
                output.append(i)
    return output

ROWS = get_rows()

class NPS:
    def __init__(self, name_col, pump_col, Pin_col, Pout_col):
        self.name = WS[name_col + str(NAME_ROW)].value
        self.pump = self.get_pump(pump_col)
        self.Pin = self.get_pressure(Pin_col)
        self.Pout = self.get_pressure(Pout_col)
        self.flow = self.get_data(FLOW_COL)
        self.mode = self.get_data(MODE_COL)
        self.dP = self.get_pressure_diff()

    def get_pump(self, col):
        row_col = []
        output = []
        for i in ROWS:
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
        for i in ROWS:
            row_col.append(col + str(i))
        for j in row_col:
            data = WS[j].value
            data = data.replace(',', '.')
            data = data.split('/')
            data = [float(x) for x in data]
            res = (max(data) + min(data))/len(data)
            res = round(res, 3)
            output.append(res)
        return output

    def get_data(self, col):
        row_col = []
        output = []
        for i in ROWS:
            row_col.append(col + str(i))
        for j in row_col:
            output.append(WS[j].value)
        return output

    def convert_flow(self, data):
        output = []
        for i in data:
            output.append((i*1000*1000)/(850*2*60*60)) # переводим в м3/сек
        return output

    def convert_pressure(self, data):
        output = []
        for i in data:
            output.append((i*9.8*10000 + 101350)/(850*9.8)) # переводим в м
        return output

    def get_pressure_diff(self):
        Pin = np.array(self.Pin)
        Pout = np.array(self.Pout)
        output = Pout - Pin
        return output

    def get_pump_count(self):
        output = {'1':0, '2':0, '3':0, '4':0}
        for i in self.pump:
            for j in range(len(i)):
                output[str(j + 1)] += i[j]
        return output

    def get_pump_flow(self):
        output = {'1':[], '2':[], '3':[], '4':[]}
        pump_len = len(self.pump)
        for i in range(pump_len):
            item = self.pump[i]
            item_len = len(item)
            for j in range(item_len):
                if item[j] != 0:
                    output[str(j+1)].append(self.flow[i])
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

    def get_raw_result(self):
        output = []
        mode = self.get_data(MODE_COL)
        mode_len = len(mode)
        for i in range(mode_len):
            data = {}
            data['name'] = self.name
            data['mode'] = self.mode[i]
            data['flow'] = self.flow[i]
            data['pump'] = self.pump[i]
            data['Pin'] = self.Pin[i]
            data['Pout'] = self.Pout[i]
            data['dP'] = self.dP[i]
            output.append(data)
        return output

    def get_filtered_result(self):
        output = []
        pump = self.get_filtered_pump()
        data = self.get_raw_result()
        excluded_pump = set(['1', '2', '3', '4']) - pump
        for i in data:
            pump = i['pump']
            for j in excluded_pump:
                if pump[int(j) - 1] == 1:
                    break
                else:
                    output.append(i)
                    break
        return output

    def get_filtered_data(self, key):
        output = []
        data = self.get_filtered_result()
        for i in data:
            output.append(i[key])
        return output

    def get_cutted_result(self, a, b):
        output1 = []
        output2 = []
        a = np.array(a)
        mean = a.mean()
        top = mean*1.2
        bottom = mean*0.8
        length = len(a)
        for i in range(length):
            if bottom <= a[i] and a[i] <=top:
                output1.append(a[i])
                output2.append(b[i])
        return output1, output2


ukhta = NPS('T', 'U', 'W', 'Y')































# sindor = NPS('AD', 'AE', 'AG', 'AI')
#mikun = NPS('AN', 'AO', 'AQ', 'AS')
# urdoma = NPS('AX', 'AO', 'BA', 'BC')

#F = np.array(ukhta.get_filtered_data('flow'))
#F.shape = (len(F), 1)
#dP = np.array(ukhta.get_filtered_data('dP'))
#E = np.ones(len(F))

# cln = LinearRegression()
# cln.fit(F, dP)
# print(cln.coef_)
# A = [E, F, E, F]

#print(lstsq(F, dP))

# a = np.array(ukhta.convert_pressure(ukhta.get_filtered_data('dP')))
# b = np.array(ukhta.convert_flow(ukhta.get_filtered_data('flow')))
# a, b = ukhta.get_cutted_result(a,b)
#
#
# c = sindor.convert_pressure(sindor.get_filtered_data('dP'))
# d = sindor.convert_flow(sindor.get_filtered_data('flow'))
#
# e = mikun.convert_pressure(mikun.get_filtered_data('dP'))
# f = mikun.convert_flow(mikun.get_filtered_data('flow'))
#
# g = urdoma.convert_pressure(urdoma.get_filtered_data('dP'))
# h = urdoma.convert_flow(urdoma.get_filtered_data('flow'))

# b = np.array(b)
# a = np.array(a)
# b.shape = (len(b), 1)
# cln = LinearRegression()
# cln.fit(b,a)
# plt.scatter(b,a)
#
# plt.xlabel('Q, м3/с')
# plt.ylabel('H, м')
# plt.plot(b, cln.predict(b))

# data = (a - cln.predict(b))*100/cln.predict(b)
# plt.figure(2)
# plt.xlabel('Отклонение в %')
# plt.ylabel('Количество')
# plt.hist(data)
#
# matplotlib.rcParams.update({'font.size': 30})
# plt.show()