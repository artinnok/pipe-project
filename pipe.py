import matplotlib.pyplot as plt
import numpy as np
import locale
from sklearn.linear_model import LinearRegression

from helper import Data

locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')


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


us = Linear('Y', 'AG', 'K')

plt.scatter(us.flow, us.diff, s=50)  # исходные точки
plt.plot(us.flow, us.get_predict())  # линейная регрессия
plt.title('Ukhta - Sindor')

plt.show()
