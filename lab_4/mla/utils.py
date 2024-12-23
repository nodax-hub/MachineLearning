# coding:utf-8
from itertools import batched
from typing import Iterator

import numpy as np


def one_hot(y: np.ndarray) -> np.ndarray:
    """
    Преобразует массив меток в one-hot представление.

    Args:
        y (np.ndarray): Вектор меток (целые числа).

    Returns:
        np.ndarray: One-hot представление меток.
    """
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def batch_iterator(X: np.ndarray, batch_size: int = 64) -> Iterator[np.ndarray]:
    """
    Генерирует батчи данных заданного размера из массива X.

    Args:
        X (np.ndarray): Входной массив данных.
        batch_size (int): Размер одного батча.

    Returns:
        Iterator[np.ndarray]: Итератор батчей данных из X.
    """
    return (np.array(batch) for batch in batched(X, batch_size))
