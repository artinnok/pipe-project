import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

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

    def linear_predict(self):
        regression = LinearRegression()
        regression.fit(self.flow, self.diff)
        return regression.predict(self.flow)

    def ransac_predict(self):
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(self.flow, self.diff)
        return ransac.predict(self.flow)


class NPS:
    def __init__(self, inp: str, out: str, flow: str, pump: str):
        self.inp = Data(inp).get_pressure()
        self.out = Data(out).get_pressure()
        self.flow = Data(flow).get_flow()[:, np.newaxis]
        self.diff = self.out - self.inp
        self.pump = Data(pump).get_pump()

    @staticmethod
    def linear_predict(x, y):  # линейная регрессия
        regr = LinearRegression()
        regr.fit(x, y)
        return regr.predict(x)

    @staticmethod
    def ransac_predict(x, y):
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(x, y)
        return ransac.predict(x)

    @staticmethod
    def ransac_mask(x, y):
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(x, y)
        in_mask = ransac.inlier_mask_
        out_mask = np.logical_not(in_mask)
        return in_mask, out_mask

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
        return flow, diff

    def show_modes(self):  # рисуем графики для уникальных режимов
        modes = self.get_unique_modes()
        for index, item in enumerate(modes):
            flow, diff = self.get_flow_diff_of_mode(item)
            plt.figure(index)
            plt.scatter(flow, diff)
            plt.title('Режим ' + str(item))
        plt.show()

us = Linear('Y', 'AG', 'K')

ukhta = NPS('W', 'Y', 'K', 'U')
sindor = NPS('AG', 'AI', 'K', 'AE')

x, y = ukhta.get_flow_diff_of_mode([1, 3])
plt.scatter(x, y, s=50)  # исходные точки
in_mask, out_mask = ukhta.ransac_mask(x, y)
x = np.array(x)
y = np.array(y)
plt.plot(x[in_mask], y[in_mask], '.g')
plt.plot(x[out_mask], y[out_mask], '.r', label='Выбросы')
plt.plot(x, ukhta.linear_predict(x, y), label='Линейная регрессия')  # линейная регрессия
plt.plot(x, ukhta.ransac_predict(x, y), label='RANSAC')  # ransac регрессия

plt.title('НПС Ухта режим [1,3]')
plt.legend(loc='lower right')
plt.xlabel('Q, м3/с')
plt.ylabel('H, м')


plt.show()
