import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from helper import Data, Handler


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)

PUMPS_ROWS = ['B', 'D', 'G', 'J', 'M']
PIN_ROWS = ['E', 'H', 'K', 'N', 'P']
POUT_ROWS = ['C', 'F', 'I', 'L', 'O']

# main classes
class Linear:
    def __init__(self, inp: str, out: str, flow: str):
        self.inp = Data(inp).get_pressure()
        self.out = Data(out).get_pressure()
        self.flow = Data(flow).get_flow()[:, np.newaxis]
        self.diff = self.inp - self.out


class NPS:
    def __init__(self, inp: str, out: str, flow: str, pump: str):
        self.inp = Data(inp).get_pressure()
        self.out = Data(out).get_pressure()
        self.flow = Data(flow).get_flow()[:, np.newaxis]
        self.diff = self.out - self.inp
        self.pump = Data(pump).get_pump()

    def get_pump_count(self) -> dict:  # {1: 4, 2: 0, 3: 21, 4: 7}, {насос: сколько раз встречается}
        inp = []
        [inp.extend(item) for item in self.pump]
        out = {
            1: inp.count(1),
            2: inp.count(2),
            3: inp.count(3),
            4: inp.count(4)
        }
        return out

    def get_flow_diff_of_pump(self, pump: int):  # напор и разница давлений одного насоса
        flow, diff = [], []
        for index, item in enumerate(self.pump):
            if pump in item:
                flow.append(self.flow[index])
                diff.append(self.diff[index])
        return [flow, diff]

    def get_flow_diff_all_pumps(self):  # вернет словарь {насос: [массив из расхода, массив из давления]}
        out = {
            1: self.get_flow_diff_of_pump(1),
            2: self.get_flow_diff_of_pump(2),
            3: self.get_flow_diff_of_pump(3),
            4: self.get_flow_diff_of_pump(4)
        }
        return out

    def get_unique_modes(self) -> list:  # [[1,3], [2,4]] вернет уникальные режимы
        out = []
        [out.append(item) for item in self.pump if item not in out]
        return out

    def get_flow_diff_of_mode(self, mode: list) -> list:   # [[f1, f2 ...], [dp1, dp2 ...]]
        flow, diff = [], []              # вернет расход и давление одного режима [массив из расхода, массив из давлений]
        for index, item in enumerate(self.pump):
            if item == mode:
                flow.append(self.flow[index])
                diff.append(self.diff[index])
        return np.array(flow), np.array(diff)

    def show_modes(self):  # рисуем графики для уникальных режимов
        modes = self.get_unique_modes()
        for index, item in enumerate(modes):
            flow, diff = self.get_flow_diff_of_mode(item)
            plt.figure(index)
            plt.scatter(flow, diff)
            plt.title('Режим ' + str(item))
        plt.show()

    def show(self, title):
        x, y = Handler.get_inliers(self.flow, self.diff)
        plt.scatter(x, y, s=50)
        plt.plot(x, Handler.linear_predict(x, y))
        plt.title(title)
        plt.xlabel('Q, м3/с')
        plt.ylabel('H, м')
        plt.show()


def get_map(modes):
    mapping = []
    out = []
    unique_modes = set(modes)
    for item in unique_modes:
        mapping.append(item)
    for item in modes:
        index = mapping.index(item)
        out.append(index)
    return out

flow = Data('A').get_flow()[3:16]
ukhta_pin = Data('E').get_pressure()[3:16]
ukhta_pout = Data('F').get_pressure()[3:16]
sindor_pin = Data('H').get_pressure()[3:16]
sindor_pout = Data('I').get_pressure()[3:16]

pumps = [Data(i).get_values() for i in ['B', 'D', 'G']]
modes = list(zip(*pumps))[3:16]
modes = get_map(modes)
pressure_start = Data('C').get_pressure()[3:16]
pressure_end = Data('P').get_pressure()[3:16]

ones = np.ones(len(modes))
X = np.array(list(zip(ones, pressure_start, pressure_end, modes)))

# flow
model_flow = sm.OLS(flow, X)
res_flow = model_flow.fit()
flow_predict = res_flow.predict(X)
# print(res_flow.summary())

# ukhta_pin
model_ukhta_pin = sm.OLS(ukhta_pin, X)
res_ukhta_pin = model_ukhta_pin.fit()
ukhta_pin_predict = res_flow.predict(X)
# print(res_ukhta_pin.summary())

# ukhta_pout
model_ukhta_pout = sm.OLS(ukhta_pout, X)
res_ukhta_pout = model_ukhta_pout.fit()
ukhta_pout_predict = res_flow.predict(X)
print(res_ukhta_pout.summary)

# plt.scatter(pressure_start, flow, c='r')
# plt.scatter(pressure_start, flow_predict, c='b')
# plt.show()
