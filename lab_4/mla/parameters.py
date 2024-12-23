# coding:utf-8
from typing import Optional, Dict

import numpy as np

from .initializations import get_initializer


class Parameters:
    """
    Контейнер для параметров слоя.

    Args:
        init_name (str): Имя функции инициализации весов. По умолчанию 'glorot_uniform'.
        scale (float): Масштабный коэффициент для инициализации весов.
        bias (float): Начальные значения для смещений. По умолчанию 1.0.
        regularizers (dict): Регуляризаторы для весов, например {'W': L2()}.
        constraints (dict): Ограничения для весов, например {'b': MaxNorm()}.
    """
    
    def __init__(
            self,
            init_name: str = "glorot_uniform",
            scale: float = 0.5,
            bias: float = 1.0,
            regularizers: Optional[Dict[str, object]] = None,
            constraints: Optional[Dict[str, object]] = None,
    ):
        self.constraints = constraints if constraints else {}
        self.regularizers = regularizers if regularizers else {}
        self.initial_bias = bias
        self.scale = scale
        self.init = get_initializer(init_name)
        
        self._params = {}
        self._grads = {}
    
    def setup_weights(self, W_shape: tuple, b_shape: Optional[tuple] = None):
        """
        Инициализирует веса и смещения слоя.

        Args:
            W_shape (tuple): Форма весов W.
            b_shape (Optional[tuple]): Форма смещений b. Если не указано, используется форма выходного слоя.
        """
        if "W" not in self._params:
            self._params["W"] = self.init(shape=W_shape, scale=self.scale)
            self._params["b"] = (
                np.full(b_shape, self.initial_bias)
                if b_shape is not None
                else np.full(W_shape[1], self.initial_bias)
            )
        self.init_grad()
    
    def init_grad(self):
        """
        Инициализирует массивы градиентов, соответствующих каждому массиву весов.
        """
        for key in self._params.keys():
            if key not in self._grads:
                self._grads[key] = np.zeros_like(self._params[key])
    
    def step(self, name: str, step: np.ndarray):
        """
        Обновляет указанный вес на значение шага.

        Args:
            name (str): Имя параметра.
            step (np.ndarray): Значение шага для обновления.
        """
        self._params[name] += step
        
        if name in self.constraints:
            self._params[name] = self.constraints[name].clip(self._params[name])
    
    def update_grad(self, name: str, value: np.ndarray):
        """
        Обновляет значение градиента для указанного параметра.

        Args:
            name (str): Имя параметра.
            value (np.ndarray): Значение градиента.
        """
        self._grads[name] = value
        
        if name in self.regularizers:
            self._grads[name] += self.regularizers[name](self._params[name])
    
    @property
    def n_params(self) -> int:
        """
        Возвращает количество параметров в слое.

        Returns:
            int: Общее количество параметров.
        """
        return sum(np.prod(self._params[x].shape) for x in self._params.keys())
    
    def keys(self):
        """
        Возвращает ключи параметров.

        Returns:
            Iterable[str]: Ключи параметров.
        """
        return self._params.keys()
    
    @property
    def grad(self) -> Dict[str, np.ndarray]:
        """
        Возвращает словарь с градиентами параметров.

        Returns:
            Dict[str, np.ndarray]: Градиенты параметров.
        """
        return self._grads
    
    def __getitem__(self, item: str) -> np.ndarray:
        """
        Доступ к параметру по имени.

        Args:
            item (str): Имя параметра.

        Returns:
            np.ndarray: Значение параметра.

        Raises:
            ValueError: Если параметр не существует.
        """
        if item in self._params:
            return self._params[item]
        raise ValueError(f"Parameter '{item}' not found.")
    
    def __setitem__(self, key: str, value: np.ndarray):
        """
        Устанавливает значение параметра.

        Args:
            key (str): Имя параметра.
            value (np.ndarray): Значение параметра.
        """
        self._params[key] = value
