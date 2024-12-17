import numpy as np


def normal(shape, mean=0.0, std=0.01):
    """
    Инициализация весов из нормального распределения.

    Args:
        shape (tuple): Форма массива весов.
        mean (float): Среднее значение нормального распределения.
        std (float): Стандартное отклонение нормального распределения.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    return np.random.normal(loc=mean, scale=std, size=shape)


def uniform(shape, low=-0.05, high=0.05):
    """
    Инициализация весов из равномерного распределения.

    Args:
        shape (tuple): Форма массива весов.
        low (float): Нижняя граница диапазона.
        high (float): Верхняя граница диапазона.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    return np.random.uniform(low=low, high=high, size=shape)


def zero(shape):
    """
    Инициализация весов нулями.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Массив, заполненный нулями.
    """
    return np.zeros(shape)


def one(shape):
    """
    Инициализация весов единицами.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Массив, заполненный единицами.
    """
    return np.ones(shape)


def orthogonal(shape, gain=1.0):
    """
    Инициализация весов с использованием ортогональной матрицы.

    Args:
        shape (tuple): Форма массива весов.
        gain (float): Множитель для масштабирования значений.

    Returns:
        np.ndarray: Ортогонально инициализированный массив.
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q


def _glorot_fan(shape):
    """
    Вычисляет фан-ин и фан-аут для Glorot инициализации.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        tuple: fan_in и fan_out.
    """
    fan_in = shape[1] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out


def glorot_normal(shape):
    """
    Glorot (Xavier) инициализация из нормального распределения.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, fan_out = _glorot_fan(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, mean=0.0, std=std)


def glorot_uniform(shape):
    """
    Glorot (Xavier) инициализация из равномерного распределения.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, fan_out = _glorot_fan(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, low=-limit, high=limit)


def he_normal(shape):
    """
    He инициализация из нормального распределения.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, _ = _glorot_fan(shape)
    std = np.sqrt(2.0 / fan_in)
    return normal(shape, mean=0.0, std=std)


def he_uniform(shape):
    """
    He инициализация из равномерного распределения.

    Args:
        shape (tuple): Форма массива весов.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, _ = _glorot_fan(shape)
    limit = np.sqrt(6.0 / fan_in)
    return uniform(shape, low=-limit, high=limit)


def get_initializer(name):
    """
    Возвращает функцию инициализации по её имени.

    Args:
        name (str): Название функции инициализации.

    Returns:
        function: Функция инициализации.

    Raises:
        ValueError: Если передано неизвестное имя.
    """
    initializers = {
        'normal': normal,
        'uniform': uniform,
        'zero': zero,
        'one': one,
        'orthogonal': orthogonal,
        'glorot_normal': glorot_normal,
        'glorot_uniform': glorot_uniform,
        'he_normal': he_normal,
        'he_uniform': he_uniform
    }

    if name not in initializers:
        raise ValueError(f"Неизвестный инициализатор: {name}")

    return initializers[name]
