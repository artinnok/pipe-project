import numpy as np


def split_comma(inp: str) -> list:  # разделяет по запятым в массив
    if type(inp) is not str:
        return inp
    return [int(item) for item in inp.split(',')]


def split_slash(inp: str) -> list:  # разделяет по / в массив
    return [float(item) for item in inp.split('/')]


def replace_comma(inp: str) -> str:  # заменяет , на .
    return inp.replace(',', '.')


def get_mean(inp: list) -> float:  # считает среднее
    return np.mean(inp)


def is_letters(inp: str) -> bool:  # проверка на строку из букв
    return inp.strip().isalpha()


def get_range(inp: range, ws) -> list:  # возвращает массив номеров строк для данных
    output = [ws[item].value for item in input if not is_letters()]
    return list
