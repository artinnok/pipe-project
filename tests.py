import unittest
from utils import split_comma, split_slash, is_letters, replace_comma, get_mean


class UtilsTest(unittest.TestCase):
    def test_split_comma(self):
        self.assertEqual(split_comma('1,2,3'), [1, 2, 3])

    def test_split_slash(self):
        self.assertEqual(split_slash('1/3'), [1, 3])

    def test_is_letters(self):
        self.assertTrue(is_letters('переход '))

    def test_replace_comma(self):
        self.assertEqual(replace_comma('4,3/9,3'), '4.3/9.3')

    def test_get_mean(self):
        self.assertEqual(get_mean([1, 2, 3, 4]), 2.5)