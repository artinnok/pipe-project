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
    if type(inp) is str:
        return inp.strip().isalpha()


def get_rows(inp: range, ws) -> list:  # возвращает массив номеров строк для данных
    out = []
    for row in inp:
        val = ws['E' + str(row)].value
        if val is not None and not is_letters(val):
            out.append(row)
    return out
