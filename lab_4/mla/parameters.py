import numpy as np
from .initializations import get_initializer


class Parameters:
    """
    Контейнер для параметров слоя нейронной сети.

    Attributes:
        init (str): Название функции инициализации весов.
        scale (float): Масштабирующий коэффициент для инициализации весов.
        bias (float): Начальное значение для смещений.
        regularizers (dict): Регуляризаторы весов.
        constraints (dict): Ограничения для весов.
        weights (dict): Словарь текущих параметров.
        grads (dict): Словарь градиентов для параметров.
    """

    def __init__(self, init='glorot_uniform', scale=0.5, bias=1.0, regularizers=None, constraints=None):
        """
        Инициализация контейнера для параметров слоя.

        Args:
            init (str): Название функции инициализации весов.
            scale (float): Масштабирующий коэффициент для инициализации.
            bias (float): Начальное значение для смещений.
            regularizers (dict, optional): Словарь регуляризаторов для весов.
            constraints (dict, optional): Словарь ограничений для весов.
        """
        self.init = init
        self.scale = scale
        self.bias = bias
        self.regularizers = regularizers if regularizers else {}
        self.constraints = constraints if constraints else {}

        self.weights = {}
        self.grads = {}
        self._initialized = False

    def setup_weights(self, shapes):
        """
        Создаёт веса и смещения для слоя на основе их формы и выбранного инициализатора.

        Args:
            shapes (dict): Словарь с формами для весов и смещений. Пример:
                {'W': (input_dim, output_dim), 'b': (output_dim,)}
        """
        initializer = get_initializer(self.init)
        for name, shape in shapes.items():
            if name == 'b':
                self.weights[name] = np.full(shape, self.bias)  # Смещения
            else:
                self.weights[name] = initializer(shape) * self.scale
        self.init_grad()
        self._initialized = True

    def init_grad(self):
        """
        Инициализирует массивы градиентов для каждого параметра.
        """
        for name, value in self.weights.items():
            self.grads[name] = np.zeros_like(value)

    def step(self, learning_rate=0.01):
        """
        Выполняет шаг обновления параметров на основе градиентов.

        Args:
            learning_rate (float): Шаг обучения для обновления параметров.
        """
        for name in self.weights:
            self.weights[name] -= learning_rate * self.grads[name]

    def update_grad(self, name, grad):
        """
        Обновляет градиент для конкретного параметра.

        Args:
            name (str): Название параметра.
            grad (np.ndarray): Значение градиента.
        """
        if name in self.grads:
            self.grads[name] += grad

    @property
    def n_params(self):
        """
        Подсчитывает общее количество параметров в контейнере.

        Returns:
            int: Количество параметров.
        """
        return sum(np.prod(w.shape) for w in self.weights.values())

    def keys(self):
        """
        Возвращает ключи параметров.

        Returns:
            list: Список ключей параметров.
        """
        return list(self.weights.keys())

    @property
    def grad(self):
        """
        Возвращает градиенты параметров.

        Returns:
            dict: Словарь градиентов.
        """
        return self.grads

    def __getitem__(self, key):
        """
        Позволяет получать доступ к параметрам через синтаксис словаря.

        Args:
            key (str): Название параметра.

        Returns:
            np.ndarray: Параметр.
        """
        return self.weights[key]

    def __setitem__(self, key, value):
        """
        Позволяет устанавливать параметры через синтаксис словаря.

        Args:
            key (str): Название параметра.
            value (np.ndarray): Новое значение параметра.
        """
        self.weights[key] = value
