import autograd.numpy as np


def sigmoid(x):
    """
    Сигмоидная функция активации.

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Значения после применения сигмоиды.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Функция активации Softmax, безопасная от числового переполнения.

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Вероятности для классов.
    """
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def linear(x):
    """
    Линейная функция активации (прямое пропускание).

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Те же значения, что и на входе.
    """
    return x


def softplus(x):
    """
    Функция активации Softplus (сглаженная ReLU).

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Значения после применения Softplus.
    """
    return np.log(1 + np.exp(x))


def softsign(x):
    """
    Функция активации Softsign.

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Значения после применения Softsign.
    """
    return x / (1 + np.abs(x))


def tanh(x):
    """
    Гиперболический тангенс.

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Значения после применения tanh.
    """
    return np.tanh(x)


def relu(x):
    """
    Функция активации ReLU (Rectified Linear Unit).

    Args:
        x (np.ndarray): Входные данные.

    Returns:
        np.ndarray: Значения после применения ReLU.
    """
    return np.maximum(0, x)


def leakyrelu(x, alpha=0.01):
    """
    Функция активации Leaky ReLU.

    Args:
        x (np.ndarray): Входные данные.
        alpha (float): Небольшой коэффициент для отрицательных значений.

    Returns:
        np.ndarray: Значения после применения Leaky ReLU.
    """
    return np.where(x > 0, x, alpha * x)


def get_activation(name):
    """
    Возвращает функцию активации по её имени.

    Args:
        name (str): Название функции активации.

    Returns:
        function: Функция активации.

    Raises:
        ValueError: Если передано неизвестное имя.
    """
    activations = {
        'sigmoid': sigmoid,
        'softmax': softmax,
        'linear': linear,
        'softplus': softplus,
        'softsign': softsign,
        'tanh': tanh,
        'relu': relu,
        'leakyrelu': leakyrelu,
    }

    if name not in activations:
        raise ValueError(f"Неизвестная функция активации: {name}")

    return activations[name]
