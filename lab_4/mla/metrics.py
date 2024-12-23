# coding:utf-8
from typing import Callable

import autograd.numpy as np

EPS = 1e-15


def unhot(function: Callable) -> Callable:
    """
    Декоратор для преобразования one-hot представления в одномерный массив.

    Args:
        function (Callable): Функция метрики, принимающая одномерные массивы.

    Returns:
        Callable: Обёрнутая функция, работающая с one-hot представлением.
    """
    
    def wrapper(actual: np.ndarray, predicted: np.ndarray):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)
    
    return wrapper


def absolute_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Вычисляет абсолютную ошибку между предсказаниями и истинными значениями.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Абсолютные ошибки для каждого элемента.
    """
    return np.abs(actual - predicted)


@unhot
def classification_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет долю ошибок классификации.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Доля неправильных предсказаний.
    """
    return (actual != predicted).sum() / float(actual.shape[0])


@unhot
def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет точность классификации.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Доля правильных предсказаний.
    """
    return 1.0 - classification_error(actual, predicted)


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Средняя абсолютная ошибка.
    """
    return np.mean(absolute_error(actual, predicted))


def squared_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Вычисляет квадратичную ошибку для каждого элемента.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Квадратичные ошибки.
    """
    return (actual - predicted) ** 2


def squared_log_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Вычисляет квадратичную логарифмическую ошибку для каждого элемента.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Квадратичные логарифмические ошибки.
    """
    return (np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1)) ** 2


def mean_squared_log_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет среднюю квадратичную логарифмическую ошибку (MSLE).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Средняя квадратичная логарифмическая ошибка.
    """
    return np.mean(squared_log_error(actual, predicted))


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет среднюю квадратичную ошибку (MSE).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Средняя квадратичная ошибка.
    """
    return np.mean(squared_error(actual, predicted))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет корень из средней квадратичной ошибки (RMSE).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Корень из средней квадратичной ошибки.
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def root_mean_squared_log_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет корень из средней квадратичной логарифмической ошибки (RMSLE).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Корень из средней квадратичной логарифмической ошибки.
    """
    return np.sqrt(mean_squared_log_error(actual, predicted))


def logloss(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет логарифмическую функцию потерь (Log Loss).

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные вероятности.

    Returns:
        float: Логарифмическая функция потерь.
    """
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(actual.shape[0])


def hinge(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет функцию потерь Хинжа.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные значения.

    Returns:
        float: Функция потерь Хинжа.
    """
    return np.mean(np.maximum(1.0 - actual * predicted, 0.0))


def binary_crossentropy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Вычисляет бинарную кроссэнтропию.

    Args:
        actual (np.ndarray): Истинные значения.
        predicted (np.ndarray): Предсказанные вероятности.

    Returns:
        float: Бинарная кроссэнтропия.
    """
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return np.mean(-np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))


# aliases
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error


def get_metric(name: str) -> Callable:
    """
    Возвращает функцию метрики по её имени.

    Args:
        name (str): Имя функции метрики.

    Returns:
        Callable: Функция метрики.

    Raises:
        ValueError: Если метрика с таким именем не найдена.
    """
    metrics = {
        "absolute_error": absolute_error,
        "classification_error": classification_error,
        "accuracy": accuracy,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error,
        "root_mean_squared_log_error": root_mean_squared_log_error,
        "logloss": logloss,
        "hinge": hinge,
        "binary_crossentropy": binary_crossentropy,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }
    if name in metrics:
        return metrics[name]
    raise ValueError(f"Invalid metric function: {name}")
