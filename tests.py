import unittest

import numpy as np
from pipe import Solver

P11 = 2 * 10 ** 5
PN1 = 1 * 10 ** 5
P12 = 3 * 10 ** 5
PN2 = 1 * 10 ** 5
P13 = 1.5 * 10 ** 5
PN3 = 1 * 10 ** 5
X = np.array([
    [[P11], [PN1]]
])

B1 = 10 ** (-3)
B0 = 0
THETA = np.array([
    [B0],
    [B1]
])

h = 10 ** (-6)
E = 0.05


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.x = np.array([
            [[P11], [PN1]]
        ])
        self.xx = np.append(self.x, [[[P12], [PN2]]], axis=0)
        self.xxx = np.append(self.xx, [[[P13], [PN3]]], axis=0)
        self.theta = np.array([
            [B0],
            [B1]
        ])

    def test_solve(self):
        for item in [1, 2, 4]:
            self.assertEqual(
                self.s.solve(np.repeat(self.theta, item), self.x)[0, 0],
                100 / item
            )

    def test_precision_solve(self):
        for item in [1, 2, 4]:
            l = len(self.s.solve(np.repeat(self.theta, item), self.xx))
            self.assertEqual(
                self.s.solve(np.repeat(self.theta, item), self.xx)[0, 0],
                100 / item
            )
            self.assertEqual(
                self.s.solve(np.repeat(self.theta, item), self.xx)[l / 2, 0],
                200 / item
            )

if __name__ == '__main__':
    unittest.main()
