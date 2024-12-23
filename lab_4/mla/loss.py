from typing import Callable

from .metrics import mse, logloss, mae, hinge, binary_crossentropy

# Alias for categorical crossentropy
categorical_crossentropy = logloss


def get_loss(name: str) -> Callable:
    """
    Возвращает функцию потерь по её имени.

    Args:
        name (str): Имя функции потерь.

    Returns:
        Callable: Функция потерь.

    Raises:
        ValueError: Если функция потерь с таким именем не найдена.
    """
    losses = {
        "mse": mse,
        "logloss": logloss,
        "mae": mae,
        "hinge": hinge,
        "binary_crossentropy": binary_crossentropy,
        "categorical_crossentropy": categorical_crossentropy,
    }
    if name in losses:
        return losses[name]
    raise ValueError(f"Invalid loss function: {name}")
