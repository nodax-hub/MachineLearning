from typing import Tuple, Callable

import numpy as np

EPSILON = 1e-8


def normal(shape: Tuple[int, ...], scale: float = 0.5) -> np.ndarray:
    """
    Генерирует веса из нормального распределения.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.
        scale (float): Стандартное отклонение распределения.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def uniform(shape: Tuple[int, ...], scale: float = 0.5) -> np.ndarray:
    """
    Генерирует веса из равномерного распределения.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.
        scale (float): Диапазон распределения [-scale, scale].

    Returns:
        np.ndarray: Инициализированный массив.
    """
    return np.random.uniform(low=-scale, high=scale, size=shape)


def zero(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Генерирует массив, заполненный нулями.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Массив из нулей.
    """
    return np.zeros(shape)


def one(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Генерирует массив, заполненный единицами.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Массив из единиц.
    """
    return np.ones(shape)


def orthogonal(shape: Tuple[int, ...], scale: float = 0.5) -> np.ndarray:
    """
    Генерирует ортогональный массив весов.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.
        scale (float): Масштабный коэффициент.

    Returns:
        np.ndarray: Ортогонально инициализированный массив.
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    array = np.random.normal(size=flat_shape)
    u, _, v = np.linalg.svd(array, full_matrices=False)
    array = u if u.shape == flat_shape else v
    return np.reshape(array * scale, shape)


def _glorot_fan(shape: Tuple[int, ...]) -> Tuple[float, float]:
    """
    Вычисляет количество входов (fan_in) и выходов (fan_out) для слоя.

    Args:
        shape (Tuple[int, ...]): Форма слоя.

    Returns:
        Tuple[float, float]: fan_in и fan_out.
    """
    assert len(shape) >= 2, "Shape must have at least two dimensions."
    
    if len(shape) == 4:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in, fan_out = shape[:2]
    return float(fan_in), float(fan_out)


def glorot_normal(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Инициализация по методу Glorot с нормальным распределением.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, scale=s)


def glorot_uniform(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Инициализация по методу Glorot с равномерным распределением.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, scale=s)


def he_normal(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Инициализация по методу He с нормальным распределением.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, _ = _glorot_fan(shape)
    s = np.sqrt(2.0 / fan_in)
    return normal(shape, scale=s)


def he_uniform(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """
    Инициализация по методу He с равномерным распределением.

    Args:
        shape (Tuple[int, ...]): Форма выходного массива.

    Returns:
        np.ndarray: Инициализированный массив.
    """
    fan_in, _ = _glorot_fan(shape)
    s = np.sqrt(6.0 / fan_in)
    return uniform(shape, scale=s)


def get_initializer(name: str) -> Callable:
    """
    Возвращает функцию инициализации по её имени.

    Args:
        name (str): Имя функции инициализации.

    Returns:
        Callable: Функция инициализации.

    Raises:
        ValueError: Если функция инициализации не найдена.
    """
    initializers = {
        "normal": normal,
        "uniform": uniform,
        "zero": zero,
        "one": one,
        "orthogonal": orthogonal,
        "glorot_normal": glorot_normal,
        "glorot_uniform": glorot_uniform,
        "he_normal": he_normal,
        "he_uniform": he_uniform,
    }
    if name in initializers:
        return initializers[name]
    raise ValueError(f"Invalid initialization function: {name}")
