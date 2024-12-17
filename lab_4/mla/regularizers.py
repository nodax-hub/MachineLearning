import numpy as np
from autograd import elementwise_grad


class Regularizer:
    """
    Базовый класс для регуляризаторов.
    """

    def __init__(self, alpha=0.01):
        """
        Инициализация регуляризатора.

        Args:
            alpha (float): Коэффициент регуляризации.
        """
        self.alpha = alpha

    def _penalty(self, weights):
        """
        Вычисляет штраф для регуляризации. Переопределяется в подклассах.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            float: Значение штрафа.
        """
        raise NotImplementedError

    def grad(self, weights):
        """
        Вычисляет градиент регуляризатора по весам.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            np.ndarray: Градиент регуляризации.
        """
        return elementwise_grad(self._penalty)(weights)

    def __call__(self, weights):
        """
        Вычисляет штраф регуляризатора при вызове.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            float: Значение штрафа.
        """
        return self._penalty(weights)


class L1(Regularizer):
    """
    L1 регуляризация: штраф за сумму абсолютных значений весов.
    """

    def _penalty(self, weights):
        """
        Вычисляет L1 штраф.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            float: Значение L1 штрафа.
        """
        return self.alpha * np.sum(np.abs(weights))

    def grad(self, weights):
        """
        Вычисляет градиент L1 регуляризации.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            np.ndarray: Градиент регуляризации.
        """
        return self.alpha * np.sign(weights)


class L2(Regularizer):
    """
    L2 регуляризация: штраф за сумму квадратов весов.
    """

    def _penalty(self, weights):
        """
        Вычисляет L2 штраф.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            float: Значение L2 штрафа.
        """
        return self.alpha * 0.5 * np.sum(weights ** 2)

    def grad(self, weights):
        """
        Вычисляет градиент L2 регуляризации.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            np.ndarray: Градиент регуляризации.
        """
        return self.alpha * weights


class ElasticNet(Regularizer):
    """
    ElasticNet регуляризация: комбинация L1 и L2 штрафов.

    Args:
        alpha (float): Общий коэффициент регуляризации.
        l1_ratio (float): Доля L1 регуляризации в ElasticNet.
    """

    def __init__(self, alpha=0.01, l1_ratio=0.5):
        super().__init__(alpha)
        self.l1_ratio = l1_ratio

    def _penalty(self, weights):
        """
        Вычисляет ElasticNet штраф.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            float: Значение ElasticNet штрафа.
        """
        l1 = self.l1_ratio * np.sum(np.abs(weights))
        l2 = (1 - self.l1_ratio) * 0.5 * np.sum(weights ** 2)
        return self.alpha * (l1 + l2)

    def grad(self, weights):
        """
        Вычисляет градиент ElasticNet регуляризации.

        Args:
            weights (np.ndarray): Веса модели.

        Returns:
            np.ndarray: Градиент регуляризации.
        """
        l1_grad = self.l1_ratio * np.sign(weights)
        l2_grad = (1 - self.l1_ratio) * weights
        return self.alpha * (l1_grad + l2_grad)
