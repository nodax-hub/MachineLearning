# coding:utf-8
from abc import ABC, abstractmethod

import numpy as np
from autograd import elementwise_grad


class Regularizer(ABC):
    """
    Базовый класс для регуляризаторов.

    Args:
        C (float): Коэффициент регуляризации. По умолчанию 0.01.
    """
    
    def __init__(self, C: float = 0.01):
        self.C = C
        self._grad = elementwise_grad(self._penalty)
    
    @abstractmethod
    def _penalty(self, weights: np.ndarray) -> np.ndarray:
        """
        Вычисляет штраф (penalty) для весов.

        Args:
            weights (np.ndarray): Веса слоя.

        Returns:
            np.ndarray: Значения штрафа.
        """
        pass
    
    def grad(self, weights: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент штрафа по весам.

        Args:
            weights (np.ndarray): Веса слоя.

        Returns:
            np.ndarray: Градиенты штрафа.
        """
        return self._grad(weights)
    
    def __call__(self, weights: np.ndarray) -> np.ndarray:
        """
        Вызывает метод grad для вычисления градиентов.

        Args:
            weights (np.ndarray): Веса слоя.

        Returns:
            np.ndarray: Градиенты штрафа.
        """
        return self.grad(weights)


class L1(Regularizer):
    """
    L1 регуляризатор.
    """
    
    def _penalty(self, weights: np.ndarray) -> np.ndarray:
        return self.C * np.abs(weights)


class L2(Regularizer):
    """
    L2 регуляризатор.
    """
    
    def _penalty(self, weights: np.ndarray) -> np.ndarray:
        return self.C * weights ** 2


class ElasticNet(Regularizer):
    """
    Линейная комбинация штрафов L1 и L2.
    """
    
    def _penalty(self, weights: np.ndarray) -> np.ndarray:
        return 0.5 * self.C * weights ** 2 + (1.0 - self.C) * np.abs(weights)
