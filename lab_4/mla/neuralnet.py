import logging

import numpy as np
from autograd import elementwise_grad

from .base import BaseEstimator
from .loss import get_loss
from .metrics import get_metric
from .optimizers import Optimizer
from .utils import batch_iterator

np.random.seed(9999)


class NeuralNet(BaseEstimator):
    """
    Реализация нейронной сети с обучением методом обратного распространения.

    Args:
        layers (list): Список слоев сети.
        optimizer (Optimizer): Оптимизатор для обновления весов.
        loss_name (str): Название функции потерь.
        max_epochs (int): Максимальное количество эпох обучения.
        batch_size (int): Размер батча для обучения.
        metric_name (str): Метрика для оценки качества модели.
        shuffle (bool): Перемешивать ли данные перед каждой эпохой.
        verbose (bool): Выводить ли лог обучения.
    """
    
    fit_required = False
    
    def __init__(
            self,
            layers,
            optimizer: Optimizer,
            loss_name,
            max_epochs=10,
            batch_size=64,
            metric_name="mse",
            shuffle=False,
            verbose=True,
            very_verbose=False,
    ):
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.shuffle = shuffle
        self.optimizer = optimizer
        
        self.loss = get_loss(loss_name)
        
        if loss_name == "categorical_crossentropy":
            self.loss_grad = lambda actual, predicted: -(actual - predicted)
        else:
            self.loss_grad = elementwise_grad(self.loss, 1)
        
        self.metric = get_metric(metric_name)
        self.layers = layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._n_layers = 0
        self.log_metric = loss_name != metric_name
        self.metric_name = metric_name
        self.bprop_entry = self._find_bprop_entry()
        self._training = False
        self._initialized = False
    
    def _setup_layers(self, x_shape):
        """
        Инициализирует слои модели.

        Args:
            x_shape (tuple): Форма входных данных.
        """
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size
        
        for layer in self.layers:
            layer.setup(x_shape)
            x_shape = layer.shape(x_shape)
        
        self._n_layers = len(self.layers)
        self.optimizer.setup(self)
        self._initialized = True
        logging.info("Total parameters: %s", self.n_params)
    
    def _find_bprop_entry(self):
        """
        Находит входной слой для обратного распространения.

        Returns:
            int: Индекс входного слоя для обратного распространения.
        """
        if len(self.layers) > 0 and not hasattr(self.layers[-1], "parameters"):
            return -1
        return len(self.layers)
    
    def fit(self, X, y=None):
        """
        Обучает модель на заданных данных.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray, optional): Метки данных.
        """
        if not self._initialized:
            self._setup_layers(X.shape)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        self._setup_input(X, y)
        
        self.is_training = True
        self.optimizer.optimize(self)
        self.is_training = False
    
    def update(self, X, y):
        """
        Обновляет параметры модели для одного батча.

        Args:
            X (np.ndarray): Входные данные батча.
            y (np.ndarray): Метки батча.

        Returns:
            float: Значение функции потерь для батча.
        """
        y_pred = self.fprop(X)
        grad = self.loss_grad(y, y_pred)
        for layer in reversed(self.layers[: self.bprop_entry]):
            grad = layer.backward_pass(grad)
        return self.loss(y, y_pred)
    
    def fprop(self, X):
        """
        Прямое распространение.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Выход модели.
        """
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X
    
    def _predict(self, X=None):
        """
        Выполняет предсказание на заданных данных.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Предсказанные значения.
        """
        if not self._initialized:
            self._setup_layers(X.shape)
        
        y = []
        X_batch = batch_iterator(X, self.batch_size)
        for Xb in X_batch:
            y.append(self.fprop(Xb))
        return np.concatenate(y)
    
    @property
    def parametric_layers(self):
        """
        Генерирует параметры для всех слоев.

        Yields:
            object: Параметры слоя.
        """
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                yield layer
    
    @property
    def parameters(self):
        """
        Возвращает список всех параметров.

        Returns:
            list: Список параметров.
        """
        return [layer.parameters for layer in self.parametric_layers]
    
    def error(self, X=None, y=None):
        """
        Вычисляет ошибку модели на заданных данных.

        Args:
            X (np.ndarray, optional): Входные данные.
            y (np.ndarray, optional): Метки данных.

        Returns:
            float: Ошибка модели.
        """
        training_phase = self.is_training
        self.is_training = False
        if X is None and y is None:
            y_pred = self._predict(self.X)
            score = self.metric(self.y, y_pred)
        else:
            y_pred = self._predict(X)
            score = self.metric(y, y_pred)
        self.is_training = training_phase
        return score
    
    @property
    def is_training(self):
        return self._training
    
    @is_training.setter
    def is_training(self, train):
        self._training = train
        for layer in self.layers:
            # TODO: не понятно почему не работает isinstance(layer, ParamMixin)
            if hasattr(layer, 'is_training'):
                layer.is_training = train
    
    def shuffle_dataset(self):
        """
        Перемешивает строки в датасете.
        """
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X = self.X.take(indices, axis=0)
        self.y = self.y.take(indices, axis=0)
    
    @property
    def n_layers(self):
        """
        Возвращает количество слоев в модели.

        Returns:
            int: Количество слоев.
        """
        return self._n_layers
    
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
        Сбрасывает состояние модели.
        """
        self._initialized = False
