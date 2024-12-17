import logging
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .utils import batch_iterator


class Optimizer:
    """
    Базовый класс для оптимизаторов.
    """

    def __init__(self, learning_rate=0.01):
        """
        Инициализация оптимизатора.

        Args:
            learning_rate (float): Шаг обучения.
        """
        self.learning_rate = learning_rate

    def optimize(self, model, X, y, epochs=1, batch_size=32, shuffle=True):
        """
        Запускает процесс оптимизации модели.

        Args:
            model: Обучаемая модель.
            X (array-like): Входные данные.
            y (array-like): Целевые значения.
            epochs (int): Количество эпох обучения.
            batch_size (int): Размер батча.
            shuffle (bool): Перемешивать ли данные перед каждой эпохой.
        """
        self.setup(model)
        for epoch in range(epochs):
            epoch_loss = self.train_epoch(model, X, y, batch_size, shuffle)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    def update(self, params, grads):
        """
        Выполняет обновление параметров. Должно быть переопределено в подклассах.

        Args:
            params (dict): Словарь параметров модели.
            grads (dict): Словарь градиентов параметров.
        """
        raise NotImplementedError

    def train_epoch(self, model, X, y, batch_size, shuffle):
        """
        Обучает модель в течение одной эпохи.

        Args:
            model: Обучаемая модель.
            X (array-like): Входные данные.
            y (array-like): Целевые значения.
            batch_size (int): Размер батча.
            shuffle (bool): Перемешивать ли данные перед каждой эпохой.

        Returns:
            float: Среднее значение функции потерь за эпоху.
        """
        epoch_loss = 0
        for X_batch, y_batch in batch_iterator(X, y, batch_size, shuffle):
            batch_loss = self.train_batch(model, X_batch, y_batch)
            epoch_loss += batch_loss
        return epoch_loss / (len(X) // batch_size)

    def train_batch(self, model, X_batch, y_batch):
        """
        Обучает модель на одном батче данных.

        Args:
            model: Обучаемая модель.
            X_batch (array-like): Батч входных данных.
            y_batch (array-like): Батч целевых значений.

        Returns:
            float: Значение функции потерь на батче.
        """
        y_pred = model.forward(X_batch)
        loss = model.loss(y_batch, y_pred)
        grads = model.backward(y_batch, y_pred)
        self.update(model.parameters.weights, grads)
        return loss

    def setup(self, model):
        """
        Создаёт дополнительные переменные перед началом оптимизации.
        """
        pass


class SGD(Optimizer):
    """
    Стохастический градиентный спуск (SGD).
    """

    def update(self, params, grads):
        for name in params:
            params[name] -= self.learning_rate * grads[name]


class Adagrad(Optimizer):
    """
    Оптимизатор Adagrad.
    """

    def setup(self, model):
        self.accumulators = {k: np.zeros_like(v) for k, v in model.parameters.weights.items()}

    def update(self, params, grads):
        for name in params:
            self.accumulators[name] += grads[name] ** 2
            params[name] -= self.learning_rate * grads[name] / (np.sqrt(self.accumulators[name]) + 1e-8)


class Adadelta(Optimizer):
    """
    Оптимизатор Adadelta.
    """

    def setup(self, model):
        """
        Инициализирует внутренние переменные оптимизатора.

        Args:
            model: Модель, параметры которой будут оптимизироваться.
        """
        self.accumulators = {}
        self.deltas = {}
        self.rho = 0.95

        for layer_name, params in model.parameters.items():
            for param_name, param_value in params.items():
                key = f"{layer_name}_{param_name}"
                self.accumulators[key] = np.zeros_like(param_value)
                self.deltas[key] = np.zeros_like(param_value)

    def update(self, params, grads):
        """
        Выполняет обновление параметров модели.

        Args:
            params (dict): Параметры модели.
            grads (dict): Градиенты параметров.
        """
        for layer_name, param_group in params.items():
            if layer_name not in grads:  # Пропускаем, если градиенты для слоя отсутствуют
                continue
            for param_name, param_value in param_group.items():
                key = f"{layer_name}_{param_name}"
                self.accumulators[key] = (
                    self.rho * self.accumulators[key]
                    + (1 - self.rho) * grads[layer_name][param_name] ** 2
                )
                update = (
                    grads[layer_name][param_name]
                    * np.sqrt(self.deltas[key] + 1e-8)
                    / np.sqrt(self.accumulators[key] + 1e-8)
                )
                param_group[param_name] -= update
                self.deltas[key] = (
                    self.rho * self.deltas[key] + (1 - self.rho) * update ** 2
                )


class RMSprop(Optimizer):
    """
    Оптимизатор RMSprop.
    """

    def setup(self, model):
        self.accumulators = {k: np.zeros_like(v) for k, v in model.parameters.weights.items()}
        self.rho = 0.9

    def update(self, params, grads):
        for name in params:
            self.accumulators[name] = self.rho * self.accumulators[name] + (1 - self.rho) * grads[name] ** 2
            params[name] -= self.learning_rate * grads[name] / (np.sqrt(self.accumulators[name]) + 1e-8)


class Adam(Optimizer):
    """
    Оптимизатор Adam.
    """

    def setup(self, model):
        self.m = {}
        self.v = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

        for layer_name, params in model.parameters.items():
            for param_name, param_value in params.items():
                self.m[f"{layer_name}_{param_name}"] = np.zeros_like(param_value)
                self.v[f"{layer_name}_{param_name}"] = np.zeros_like(param_value)

    def update(self, params, grads):
        """
        Выполняет обновление параметров модели.

        Args:
            params (dict): Параметры модели.
            grads (dict): Градиенты параметров.
        """
        self.t += 1
        for layer_name, param_group in params.items():
            for param_name, param_value in param_group.items():
                key = f"{layer_name}_{param_name}"
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[layer_name][param_name]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[layer_name][param_name] ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                param_group[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


class Adamax(Optimizer):
    """
    Оптимизатор Adamax.
    """

    def setup(self, model):
        self.m = {k: np.zeros_like(v) for k, v in model.parameters.weights.items()}
        self.u = {k: np.zeros_like(v) for k, v in model.parameters.weights.items()}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for name in params:
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.u[name] = np.maximum(self.beta2 * self.u[name], np.abs(grads[name]))
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            params[name] -= self.learning_rate * m_hat / (self.u[name] + 1e-8)
