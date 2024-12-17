import numpy as np


class Constraint:
    """
    Базовый класс для ограничений на параметры модели.
    """

    def clip(self, weights):
        """
        Применяет ограничение к весам.

        Args:
            weights (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Ограниченные веса.
        """
        raise NotImplementedError("Метод 'clip' должен быть переопределён в подклассах.")


class MaxNorm(Constraint):
    """
    Ограничение максимальной нормы весов.

    Args:
        max_value (float): Максимальное значение нормы.
        axis (int): Ось, вдоль которой вычисляется норма.
    """

    def __init__(self, max_value=2.0, axis=0):
        self.max_value = max_value
        self.axis = axis

    def clip(self, weights):
        """
        Ограничивает норму весов до max_value.

        Args:
            weights (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Весы с ограниченной нормой.
        """
        norms = np.linalg.norm(weights, axis=self.axis, keepdims=True)
        desired = np.clip(norms, 0, self.max_value)
        return weights * (desired / (norms + 1e-8))


class NonNeg(Constraint):
    """
    Ограничение, запрещающее отрицательные значения весов.
    """

    def clip(self, weights):
        """
        Заменяет отрицательные значения на ноль.

        Args:
            weights (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Весы, содержащие только неотрицательные значения.
        """
        return np.maximum(0, weights)


class SmallNorm(Constraint):
    """
    Ограничение нормы весов до небольшого значения.

    Args:
        max_value (float): Максимальное значение нормы.
    """

    def __init__(self, max_value=1.0):
        self.max_value = max_value

    def clip(self, weights):
        """
        Ограничивает норму весов до small_value.

        Args:
            weights (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Ограниченные веса.
        """
        norm = np.linalg.norm(weights)
        if norm > self.max_value:
            return weights * (self.max_value / (norm + 1e-8))
        return weights


class UnitNorm(Constraint):
    """
    Ограничение, приводящее веса к единичной норме.

    Args:
        axis (int): Ось, вдоль которой нормируются веса.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def clip(self, weights):
        """
        Приводит веса к единичной норме.

        Args:
            weights (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Нормализованные веса.
        """
        norms = np.linalg.norm(weights, axis=self.axis, keepdims=True)
        return weights / (norms + 1e-8)
