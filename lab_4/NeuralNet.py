import numpy as np

from lab_4.mla.base import BaseEstimator
from lab_4.mla.loss import get_loss
from lab_4.mla.metrics import get_metric
from lab_4.mla.utils import batch_iterator


class NeuralNet(BaseEstimator):
    fit_required = False

    def __init__(self, layers, loss, optimizer, metric, batch_size=32, max_epochs=10, shuffle=True):
        """
        Инициализация нейросети.

        Args:
            layers (list): Список слоёв нейросети.
            loss (str): Название функции потерь.
            optimizer (Optimizer): Оптимизатор для обучения.
            metric (str): Название метрики для оценки качества.
            batch_size (int): Размер батча для обучения.
            max_epochs (int): Максимальное количество эпох.
            shuffle (bool): Перемешивать ли данные перед каждой эпохой.
        """
        super().__init__()
        self.layers = layers
        self.loss_fn = get_loss(loss)
        self.optimizer = optimizer
        self.metric_fn = get_metric(metric)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self._is_training = True

    def _setup_layers(self, x_shape):
        """
        Инициализирует веса и параметры слоёв.

        Args:
            x_shape (tuple): Форма входных данных.
        """
        input_shape = x_shape[1]
        for layer in self.layers:
            if hasattr(layer, "setup"):
                layer.setup(input_shape)
            if hasattr(layer, "shape") and layer.shape():
                input_shape = layer.shape()[1]

    def _find_bprop_entry(self):
        """
        Возвращает последний слой для обратного распространения.
        """
        return self.layers[-1]

    def fit(self, X, y=None):
        """
        Обучает нейросеть.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.
        """
        X, y = self._setup_input(X, y)
        self._setup_layers(X.shape)
        self.optimizer.setup(self)

        for epoch in range(self.max_epochs):
            if self.shuffle:
                X, y = self.shuffle_dataset(X, y)

            epoch_loss = 0
            for X_batch, y_batch in batch_iterator(X, y, self.batch_size):
                self.update(X_batch, y_batch)
                epoch_loss += self.error(X_batch, y_batch)

            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {epoch_loss:.4f}")

    def update(self, X, y):
        """
        Обновляет веса сети на одном батче данных.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.
        """
        y_pred = self.fprop(X)
        loss = self.loss_fn(y, y_pred)
        grads = self.backward_pass(y, y_pred)
        self.optimizer.update(self.parameters, grads)

    def fprop(self, X):
        """
        Прямое распространение входных данных через сеть.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Выходные значения нейросети.
        """
        output = X
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output

    def backward_pass(self, y_true, y_pred):
        """
        Обратное распространение ошибки.

        Args:
            y_true (np.ndarray): Истинные значения.
            y_pred (np.ndarray): Предсказанные значения.

        Returns:
            dict: Градиенты параметров.
        """
        grads = {}
        # TODO: loss_fn
        error = y_pred - y_true
        for i, layer in reversed(list(enumerate(self.layers))):
            error = layer.backward_pass(error)
            if hasattr(layer, "parameters"):
                grads[f"layer_{i}"] = {
                    name: grad for name, grad in layer.parameters.grad.items()
                }
        return grads

    def _predict(self, X=None):
        """
        Предсказание на новых данных.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Предсказания.
        """
        return self.fprop(X)

    @property
    def parametric_layers(self):
        """
        Возвращает слои с параметрами (например, Dense).

        Returns:
            list: Список слоёв с параметрами.
        """
        return [layer for layer in self.layers if hasattr(layer, "parameters")]

    @property
    def parameters(self):
        """
        Возвращает параметры всех слоёв.

        Returns:
            dict: Словарь параметров вида {"layer_0": {"W": ..., "b": ...}, ...}.
        """
        params = {}
        for i, layer in enumerate(self.parametric_layers):
            params[f"layer_{i}"] = layer.parameters.weights  # Используем `weights`, чтобы получить вложенные параметры.
        return params

    def error(self, X=None, y=None):
        """
        Вычисляет функцию ошибки на заданных данных.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.

        Returns:
            float: Значение функции ошибки.
        """
        y_pred = self.fprop(X)
        return self.loss_fn(y, y_pred)

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, train):
        self._is_training = train
        for layer in self.layers:
            if hasattr(layer, "is_training"):
                layer.is_training = train

    def shuffle_dataset(self, X, y):
        """
        Перемешивает данные.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.

        Returns:
            tuple: Перемешанные X и y.
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    @property
    def n_layers(self):
        """
        Возвращает количество слоёв.

        Returns:
            int: Количество слоёв в сети.
        """
        return len(self.layers)

    @property
    def n_params(self):
        """
        Возвращает количество обучаемых параметров.

        Returns:
            int: Количество параметров.
        """
        return sum(layer.parameters.n_params for layer in self.parametric_layers)

    def reset(self):
        """
        Сбрасывает параметры сети.
        """
        for layer in self.parametric_layers:
            layer.setup()
