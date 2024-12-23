import autograd.numpy as np


def sigmoid(z):
    """
    Сигмоидальная активационная функция.

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные после применения сигмоида.
    """
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Функция активации softmax для многоклассовой классификации.

    Args:
        z (ndarray): Входные данные (2D массив).

    Returns:
        ndarray: Вероятности классов, нормализованные вдоль осей.
    """
    # Избегаем численного переполнения, вычитая максимум из z
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def linear(z):
    """
    Линейная активационная функция (возвращает входные данные без изменений).

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные без изменений.
    """
    return z


def softplus(z):
    """
    Гладкая версия функции ReLU (softplus).

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные после применения softplus.
    """
    return np.logaddexp(0.0, z)  # Избегаем переполнения


def softsign(z):
    """
    Функция активации softsign.

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные после применения softsign.
    """
    return z / (1 + np.abs(z))


def tanh(z):
    """
    Гиперболический тангенс (tanh).

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные после применения tanh.
    """
    return np.tanh(z)


def relu(z):
    """
    Функция активации ReLU (Rectified Linear Unit).

    Args:
        z (ndarray): Входные данные.

    Returns:
        ndarray: Выходные данные после применения ReLU.
    """
    return np.maximum(0, z)


def leakyrelu(z, a=0.01):
    """
    Функция активации Leaky ReLU.

    Args:
        z (ndarray): Входные данные.
        a (float, optional): Коэффициент утечки для отрицательных значений. По умолчанию 0.01.

    Returns:
        ndarray: Выходные данные после применения Leaky ReLU.
    """
    return np.maximum(z * a, z)


def get_activation(name):
    """
    Возвращает активационную функцию по её имени.

    Args:
        name (str): Имя функции активации.

    Returns:
        Callable: Соответствующая функция активации.

    Raises:
        ValueError: Если имя функции активации некорректно.
    """
    activations = {
        "sigmoid": sigmoid,
        "softmax": softmax,
        "linear": linear,
        "softplus": softplus,
        "softsign": softsign,
        "tanh": tanh,
        "relu": relu,
        "leakyrelu": leakyrelu,
    }
    if name in activations:
        return activations[name]
    raise ValueError(f"Invalid activation function: {name}")
