from typing import Self

import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Базовый класс для построения оценщиков (estimators).

    Attributes:
        y_required (bool): Требуется ли наличие целевых значений (y).
        fit_required (bool): Требуется ли обучение перед предсказанием.
        is_fitted (bool): Флаг, указывающий, обучена ли модель.
    """
    y_required = True  # Указывает, что y (целевые значения) обязательны
    fit_required = True  # Указывает, что модель должна быть обучена перед предсказанием

    def __init__(self):
        """
        Инициализация базового оценщика.

        Устанавливает флаг `is_fitted` в значение False.
        """
        self.is_fitted = False  # Флаг, что модель не обучена

    def _setup_input(self, X, y=None):
        """
        Преобразует входные данные в формат numpy.ndarray и проверяет их корректность.

        Если y обязателен (y_required=True), метод проверяет его наличие и длину относительно X.

        Args:
            X (array-like): Матрица признаков (фичей).
            y (array-like, optional): Целевые значения. Обязательно, если y_required=True.

        Returns:
            tuple: Кортеж из (X, y), где оба значения приведены к формату numpy.ndarray.

        Raises:
            ValueError: Если y обязателен, но не предоставлен, или если длины X и y не совпадают.
        """
        X = np.asarray(X)

        if self.y_required:
            if y is None:
                raise ValueError("y обязателен, но не предоставлен.")
            y = np.asarray(y)

            if len(X) != len(y):
                raise ValueError("Длины X и y должны совпадать.")

        return X, y

    @abstractmethod
    def fit(self, X, y=None) -> Self:
        """
        Обучает модель на основе предоставленных данных.

        Args:
            X (array-like): Матрица признаков (фичей).
            y (array-like, optional): Целевые значения. Обязательно, если y_required=True.

        Returns:
            Self: Возвращает текущий экземпляр модели.
        """
        X, y = self._setup_input(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Предсказывает значения на основе обученной модели.

        Args:
            X (array-like): Матрица признаков (фичей).

        Returns:
            array-like: Предсказанные значения.

        Raises:
            ValueError: Если модель не была обучена перед предсказанием.
        """
        if self.fit_required and not self.is_fitted:
            raise ValueError("Модель должна быть обучена перед выполнением предсказаний.")

        return self._predict(X)

    @abstractmethod
    def _predict(self, X):
        """
        Внутренний метод для реализации логики предсказаний.

        Args:
            X (array-like): Матрица признаков (фичей).

        Returns:
            array-like: Предсказанные значения.
        """
    pass
