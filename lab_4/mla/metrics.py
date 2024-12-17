import autograd.numpy as np


def unhot(func):
    """
    Декоратор для преобразования one-hot представления в метки перед вычислением метрик.

    Args:
        func (function): Функция метрики.

    Returns:
        function: Обновлённая функция, обрабатывающая one-hot входы.
    """

    def wrapper(y_true, y_pred, *args, **kwargs):
        if y_true.ndim > 1:  # Если y_true в one-hot формате
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:  # Если y_pred в one-hot формате
            y_pred = np.argmax(y_pred, axis=1)
        return func(y_true, y_pred, *args, **kwargs)

    return wrapper


def unhot_conversion(y):
    """
    Преобразует one-hot представление в столбец с метками.

    Args:
        y (np.ndarray): One-hot представление.

    Returns:
        np.ndarray: Вектор меток.
    """
    return np.argmax(y, axis=1)


def absolute_error(y_true, y_pred):
    """
    Вычисляет абсолютную ошибку.

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Абсолютная ошибка.
    """
    return np.abs(y_true - y_pred)


@unhot
def classification_error(y_true, y_pred):
    """
    Вычисляет ошибку классификации.

    Args:
        y_true (np.ndarray): Истинные метки.
        y_pred (np.ndarray): Предсказанные метки.

    Returns:
        float: Доля неправильных предсказаний.
    """
    return np.mean(y_true != y_pred)


@unhot
def accuracy(y_true, y_pred):
    """
    Вычисляет точность классификации.

    Args:
        y_true (np.ndarray): Истинные метки.
        y_pred (np.ndarray): Предсказанные метки.

    Returns:
        float: Доля правильных предсказаний.
    """
    return np.mean(y_true == y_pred)


def mean_absolute_error(y_true, y_pred):
    """
    Вычисляет среднюю абсолютную ошибку (MAE).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: Средняя абсолютная ошибка.
    """
    return np.mean(absolute_error(y_true, y_pred))


def squared_error(y_true, y_pred):
    """
    Вычисляет квадратичную ошибку.

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        np.ndarray: Квадратичная ошибка.
    """
    return (y_true - y_pred) ** 2


def mean_squared_error(y_true, y_pred):
    """
    Вычисляет среднюю квадратичную ошибку (MSE).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: Средняя квадратичная ошибка.
    """
    return np.mean(squared_error(y_true, y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Вычисляет среднекорневую квадратичную ошибку (RMSE).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_squared_log_error(y_true, y_pred):
    """
    Вычисляет среднюю квадратичную логарифмическую ошибку (MSLE).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: MSLE.
    """
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def root_mean_squared_log_error(y_true, y_pred):
    """
    Вычисляет среднекорневую квадратичную логарифмическую ошибку (RMSLE).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: RMSLE.
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def logloss(y_true, y_pred):
    """
    Логистическая функция потерь (Log Loss).

    Args:
        y_true (np.ndarray): Истинные значения.
        y_pred (np.ndarray): Предсказанные вероятности.

    Returns:
        float: Log Loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def hinge(y_true, y_pred):
    """
    Hinge loss для задач бинарной классификации.

    Args:
        y_true (np.ndarray): Истинные метки (-1 или 1).
        y_pred (np.ndarray): Предсказанные значения.

    Returns:
        float: Значение Hinge loss.
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def binary_crossentropy(y_true, y_pred):
    """
    Бинарная кросс-энтропия.

    Args:
        y_true (np.ndarray): Истинные метки.
        y_pred (np.ndarray): Предсказанные вероятности.

    Returns:
        float: Значение бинарной кросс-энтропии.
    """
    return logloss(y_true, y_pred)


# Aliases
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error


def get_metric(name):
    """
    Возвращает функцию метрики по её имени.

    Args:
        name (str): Название метрики.

    Returns:
        function: Функция метрики.

    Raises:
        ValueError: Если метрика с таким именем не найдена.
    """
    metrics = {
        'absolute_error': absolute_error,
        'classification_error': classification_error,
        'accuracy': accuracy,
        'mean_absolute_error': mean_absolute_error,
        'mean_squared_error': mean_squared_error,
        'root_mean_squared_error': root_mean_squared_error,
        'mean_squared_log_error': mean_squared_log_error,
        'root_mean_squared_log_error': root_mean_squared_log_error,
        'logloss': logloss,
        'hinge': hinge,
        'binary_crossentropy': binary_crossentropy,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
    }

    if name not in metrics:
        raise ValueError(f"Неизвестная метрика: {name}")

    return metrics[name]
