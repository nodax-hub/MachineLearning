# coding:utf-8
from abc import ABC, abstractmethod

import numpy as np

EPSILON = 1e-8


class Constraint(ABC):
    """
    Абстрактный базовый класс для всех ограничений на веса.
    """
    
    @abstractmethod
    def clip(self, p: np.ndarray) -> np.ndarray:
        """
        Применяет ограничение к заданному массиву весов.

        Args:
            p (np.ndarray): Массив весов.

        Returns:
            np.ndarray: Изменённый массив весов после применения ограничения.
        """
        pass


class MaxNorm(Constraint):
    """
    Ограничение нормы по максимальному значению.
    """
    
    def __init__(self, m: float = 2.0, axis: int = 0):
        """
        Args:
            m (float): Максимальная допустимая норма.
            axis (int): Ось, вдоль которой рассчитывается норма.
        """
        self.axis = axis
        self.m = m
    
    def clip(self, p: np.ndarray) -> np.ndarray:
        norms = np.sqrt(np.sum(p ** 2, axis=self.axis, keepdims=True))
        desired = np.clip(norms, 0, self.m)
        return p * (desired / (EPSILON + norms))


class NonNeg(Constraint):
    """
    Ограничение для обеспечения неотрицательности весов.
    """
    
    def clip(self, p: np.ndarray) -> np.ndarray:
        p = np.maximum(p, 0.0)
        return p


class SmallNorm(Constraint):
    """
    Ограничение для обрезки норм весов в пределах [-5, 5].
    """
    
    def clip(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, -5, 5)


class UnitNorm(Constraint):
    """
    Ограничение для нормализации весов до единичной нормы.
    """
    
    def __init__(self, axis: int = 0):
        """
        Args:
            axis (int): Ось, вдоль которой нормализуются веса.
        """
        self.axis = axis
    
    def clip(self, p: np.ndarray) -> np.ndarray:
        norms = np.sqrt(np.sum(p ** 2, axis=self.axis, keepdims=True))
        return p / (EPSILON + norms)
