import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from helper import Data


# cyrillic support
font = {'family': 'Verdana', 'weight': 'normal'}
plt.rc('font', **font)


# main classes
class Linear:
    def __init__(self, inp: str, out: str, flow: str):
        self.inp = Data(inp).get_pressure()
        self.out = Data(out).get_pressure()
        self.flow = Data(flow).get_flow()[:, np.newaxis]
        self.diff = self.inp - self.out

    def get_predict(self):
        regr = LinearRegression()
        regr.fit(self.flow, self.diff)
        return regr.predict(self.flow)


class NPS:
    def __init__(self, inp: str, out: str, flow: str, pump: str):
        self.inp = Data(inp).get_pressure()
        self.out = Data(out).get_pressure()
        self.flow = Data(flow).get_flow()[:, np.newaxis]
        self.diff = self.out - self.inp
        self.pump = Data(pump).get_pump()

    def get_predict(self):
        regr = LinearRegression()
        regr.fit(self.flow, self.diff)
        return regr.predict(self.flow)

    def get_pump_count(self) -> dict:  # {1: 4, 2: 0, 3: 21, 4: 7}
        inp = []
        [inp.extend(item) for item in self.pump]
        out = {
            1: inp.count(1),
            2: inp.count(2),
            3: inp.count(3),
            4: inp.count(4)
        }
        return out

    def get_unique_modes(self) -> list:  # [[1,3], [2,4]
        out = []
        [out.append(item) for item in self.pump if item not in out]
        return out

    def get_flow_diff_of_mode(self, mode: list):
        flow = []
        diff = []
        for index, item in enumerate(self.pump):
            if item == mode:
                flow.append(self.flow[index])
                diff.append(self.diff[index])
        return [flow, diff]

    def show_modes(self):
        modes = self.get_unique_modes()
        for index, item in enumerate(modes):
            flow, diff = self.get_flow_diff_of_mode(item)
            plt.figure(index)
            plt.scatter(flow, diff)
            plt.title('Режим ' + str(item))
        plt.show()

    def get_pump_flow(self):
        out = {
            '1': [],
            '2': [],
            '3': [],
            '4': []
        }

us = Linear('Y', 'AG', 'K')

ukhta = NPS('W', 'Y', 'K', 'U')
sindor = NPS('AG', 'AI', 'K', 'AE')

ukhta.show_modes()

# plt.scatter(ukhta.flow, ukhta.diff, s=50)  # исходные точки
# plt.plot(ukhta.flow, ukhta.get_predict())  # линейная регрессия
# plt.title('НПС Ухта')
#
# plt.show()
