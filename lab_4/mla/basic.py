# coding:utf-8
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import autograd.numpy as np
from autograd import elementwise_grad

from .activations import get_activation
from .parameters import Parameters

np.random.seed(9999)


class Layer(ABC):
    """
    Абстрактный базовый класс для всех слоев нейронной сети.
    """
    
    @abstractmethod
    def setup(self, X_shape: Tuple[int, ...]) -> None:
        """
        Инициализация параметров слоя.

        Args:
            X_shape (Tuple[int, ...]): Форма входного тензора.
        """
        pass
    
    @abstractmethod
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Прямой проход через слой.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Результат прямого прохода.
        """
        pass
    
    @abstractmethod
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        """
        Обратный проход через слой.

        Args:
            delta (np.ndarray): Градиенты от следующего слоя.

        Returns:
            np.ndarray: Градиенты для предыдущего слоя.
        """
        pass
    
    @abstractmethod
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Возвращает форму выхода слоя.

        Args:
            x_shape (Tuple[int, ...]): Форма входных данных.

        Returns:
            Tuple[int, ...]: Форма выходных данных.
        """
        pass


class ParamMixin:
    """
    Миксин для работы с параметрами слоя.
    """
    
    @property
    def parameters(self) -> Parameters:
        return self._params


class PhaseMixin:
    """
    Миксин для переключения между режимами обучения и тестирования.
    """
    
    _train: bool = False
    
    @property
    def is_training(self) -> bool:
        return self._train
    
    @is_training.setter
    def is_training(self, is_train: bool = True) -> None:
        self._train = is_train
    
    @property
    def is_testing(self) -> bool:
        return not self._train
    
    @is_testing.setter
    def is_testing(self, is_test: bool = True) -> None:
        self._train = not is_test


class Dense(Layer, ParamMixin):
    """
    Полносвязный слой.
    """
    
    def __init__(self, output_dim: int, parameters: Optional[Parameters] = None):
        """
        Args:
            output_dim (int): Размер выходного слоя.
            parameters (Optional[Parameters]): Параметры слоя. Если None, создаются новые.
        """
        self.output_dim = output_dim
        self._params = parameters or Parameters()
        self.last_input = None
    
    def setup(self, x_shape: Tuple[int, ...]) -> None:
        self._params.setup_weights((x_shape[1], self.output_dim))
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.last_input = X
        return self.weight(X)
    
    def weight(self, X: np.ndarray) -> np.ndarray:
        W = np.dot(X, self._params["W"])
        return W + self._params["b"]
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        dW = np.dot(self.last_input.T, delta)
        db = np.sum(delta, axis=0)
        
        self._params.update_grad("W", dW)
        self._params.update_grad("b", db)
        return np.dot(delta, self._params["W"].T)
    
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return x_shape[0], self.output_dim


class Activation(Layer):
    """
    Слой активации.
    """
    
    def setup(self, X_shape: Tuple[int, ...]) -> None:
        pass
    
    def __init__(self, name: str):
        """
        Args:
            name (str): Имя функции активации.
        """
        self.last_input = None
        self.activation = get_activation(name)
        self.activation_d = elementwise_grad(self.activation)
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.last_input = X
        return self.activation(X)
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return self.activation_d(self.last_input) * delta
    
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return x_shape


class Dropout(Layer, PhaseMixin):
    """
    Dropout слой. Зануляет часть входов во время обучения.
    """
    
    def setup(self, X_shape: Tuple[int, ...]) -> None:
        pass
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p (float): Вероятность зануления входов.
        """
        self.p = p
        self._mask = None
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        assert self.p > 0, "Вероятность p должна быть больше 0."
        if self.is_training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            return X * self._mask
        return X * (1.0 - self.p)
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return delta * self._mask
    
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return x_shape


class TimeStepSlicer(Layer):
    """
    Извлекает заданный временной шаг из 3D тензора.
    """
    
    def setup(self, X_shape: Tuple[int, ...]) -> None:
        pass
    
    def __init__(self, step: int = -1):
        """
        Args:
            step (int): Индекс временного шага.
        """
        self.step = step
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.step, :]
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        return np.repeat(delta[:, np.newaxis, :], 2, axis=1)
    
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return x_shape[0], x_shape[2]


class TimeDistributedDense(Layer):
    """
    Применяет Dense слой ко всем временным шагам.
    """
    
    def __init__(self, output_dim: int):
        """
        Args:
            output_dim (int): Размер выходного слоя.
        """
        self.output_dim = output_dim
        self.n_timesteps = None
        self.dense = None
        self.input_dim = None
    
    def setup(self, X_shape: Tuple[int, ...]) -> None:
        self.dense = Dense(self.output_dim)
        self.dense.setup((X_shape[0], X_shape[2]))
        self.input_dim = X_shape[2]
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        n_timesteps = X.shape[1]
        X = X.reshape(-1, X.shape[-1])
        y = self.dense.forward_pass(X)
        return y.reshape((-1, n_timesteps, self.output_dim))
    
    def backward_pass(self, delta: np.ndarray) -> np.ndarray:
        n_timesteps = delta.shape[1]
        X = delta.reshape(-1, delta.shape[-1])
        y = self.dense.backward_pass(X)
        return y.reshape((-1, n_timesteps, self.input_dim))
    
    @property
    def parameters(self) -> Parameters:
        return self.dense.parameters
    
    def shape(self, x_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return x_shape[0], x_shape[1], self.output_dim
