# coding:utf-8
from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    """
    Базовый класс для построения оценщиков.

    Attributes:
        y_required (bool): Указывает, требуется ли массив y.
        fit_required (bool): Указывает, требуется ли вызов fit перед predict.
        X (ndarray): Входной набор данных после обработки.
        y (ndarray): Целевые значения после обработки.
        n_samples (int): Число образцов в X.
        n_features (int): Число признаков в X.
    """
    
    y_required = True
    fit_required = True
    
    def _setup_input(self, X, y=None):
        """
        Проверяет и подготавливает входные данные для оценщика.

        Преобразует X и y в numpy массивы, если это необходимо, и проверяет
        корректность их размеров. Определяет количество образцов и признаков в X.

        Args:
            X (array-like): Набор признаков.
            y (array-like, optional): Целевые значения. Требуется, если y_required=True.

        Raises:
            ValueError: Если X или y пусты или отсутствует обязательный y.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.size == 0:
            raise ValueError("Получена пустая матрица.")
        
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.size
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        
        self.X = X
        
        if self.y_required:
            if y is None:
                raise ValueError("Отсутствует обязательный аргумент y.")
            
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            if y.size == 0:
                raise ValueError("Массив целевых значений должен быть непустым.")
        
        self.y = y
    
    def fit(self, X, y=None):
        """
        Выполняет подготовку входных данных.

        Args:
            X (array-like): Набор признаков.
            y (array-like, optional): Целевые значения.
        """
        self._setup_input(X, y)
    
    def predict(self, X=None):
        """
        Возвращает предсказания для входных данных.

        Args:
            X (array-like, optional): Набор признаков для предсказания. Если None, используется X из fit.

        Returns:
            ndarray: Предсказания для входных данных.

        Raises:
            ValueError: Если fit не был вызван, а fit_required=True.
        """
        if X is not None and not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("Необходимо вызвать `fit` перед `predict`.")
    
    @abstractmethod
    def _predict(self, X=None):
        """
        Абстрактный метод для реализации предсказания в подклассах.

        Args:
            X (array-like, optional): Набор признаков для предсказания.

        Raises:
            NotImplementedError: Если метод не реализован в подклассе.
        """
        pass
