import numpy as np


def one_hot(y):
    """
    Преобразует целочисленные метки в one-hot представление.

    Args:
        y (array-like): Целочисленные метки.

    Returns:
        np.ndarray: One-hot представление меток.
    """
    y = np.asarray(y)
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]


def batch_iterator(X, y=None, batch_size=32, shuffle=True):
    """
    Итератор для разбиения данных на батчи фиксированного размера.

    Args:
        X (array-like): Матрица признаков.
        y (array-like, optional): Целевые значения.
        batch_size (int): Размер одного батча.
        shuffle (bool): Флаг, перемешивать ли данные перед разбиением.

    Yields:
        tuple: Батч данных (X_batch, y_batch), если y предоставлен,
               иначе возвращает только X_batch.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]

    if y is not None:
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("Длины X и y должны совпадать.")

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        if y is not None:
            yield X[batch_indices], y[batch_indices]
        else:
            yield X[batch_indices]
