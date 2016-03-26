import numpy as np

def split_comma(input: str) -> list: # разделяет по запятым в массив
    if type(input) is not str:
        return input
    return [int(item) for item in input.split(',')]


def split_slash(input: str) -> list: # разделяет по / в массив
    return [float(item) for item in input.split('/')]


def replace_comma(input: str) -> str: # заменяет , на .
    return input.replace(',', '.')


def get_mean(input: list) -> float: # считает среднее
    return np.mean(input)


def is_letters(input: str) -> bool: # проверка на строку из букв
    return input.strip().isalpha()


def get_range(input: range, ws) -> list: # возвращает массив номеров строк для данных
    output = [ws[item].value for item in input if not is_letters()]
    return list