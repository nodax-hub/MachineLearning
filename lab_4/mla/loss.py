from .metrics import binary_crossentropy, hinge, logloss, mae, mse

# Дополнительный алиас для логарифмической функции потерь
categorical_crossentropy = logloss

def get_loss(name):
    """
    Возвращает функцию потерь по её имени.

    Args:
        name (str): Название функции потерь.

    Returns:
        function: Функция потерь.

    Raises:
        ValueError: Если функция потерь с указанным именем не найдена.
    """
    losses = {
        'binary_crossentropy': binary_crossentropy,
        'categorical_crossentropy': categorical_crossentropy,
        'hinge': hinge,
        'logloss': logloss,
        'mae': mae,
        'mse': mse,
    }

    if name not in losses:
        raise ValueError(f"Неизвестная функция потерь: {name}")

    return losses[name]
