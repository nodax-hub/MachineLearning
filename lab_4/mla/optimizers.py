import logging
import time
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from .utils import batch_iterator


class Optimizer:
    """
    Базовый класс для всех оптимизаторов.
    """
    
    def optimize(self, network):
        """
        Запускает процесс оптимизации сети.

        Args:
            network: Экземпляр нейронной сети.

        Returns:
            list: История потерь по эпохам.
        """
        loss_history = []
        
        r = range(network.max_epochs)
        if network.verbose:
            r = tqdm(r)
        
        for i in r:
            if network.shuffle:
                network.shuffle_dataset()
            
            start_time = time.time()
            loss = self.train_epoch(network)
            loss_history.append(loss)
            if network.verbose:
                msg = f"Epoch:{i}, train loss: {loss}"
                if network.log_metric:
                    msg += f", train {network.metric_name}: {network.error()}"
                msg += f", elapsed: {time.time() - start_time:.2f} sec."
                logging.info(msg)
        
        return loss_history
    
    def update(self, network):
        """
        Выполняет обновление параметров сети.

        Args:
            network: Экземпляр нейронной сети.
        """
        raise NotImplementedError("Метод update должен быть реализован в подклассе.")
    
    def train_epoch(self, network):
        """
        Тренирует сеть в течение одной эпохи.

        Args:
            network: Экземпляр нейронной сети.

        Returns:
            float: Средняя ошибка за эпоху.
        """
        losses = []
        
        X_batch = batch_iterator(network.X, network.batch_size)
        y_batch = batch_iterator(network.y, network.batch_size)
        
        batch = zip(X_batch, y_batch)
        
        for X, y in tqdm(batch,
                         total=int(np.ceil(network.n_samples / network.batch_size)),
                         disable=not network.very_verbose):
            loss = np.mean(network.update(X, y))
            self.update(network)
            losses.append(loss)
        
        return np.mean(losses)
    
    def train_batch(self, network, X, y):
        """
        Тренирует сеть на одном батче данных.

        Args:
            network: Экземпляр нейронной сети.
            X: Входные данные.
            y: Истинные значения.

        Returns:
            float: Потеря для данного батча.
        """
        loss = np.mean(network.update(X, y))
        self.update(network)
        return loss
    
    def setup(self, network):
        """
        Инициализирует дополнительные переменные.

        Args:
            network: Экземпляр нейронной сети.
        """
        raise NotImplementedError("Метод setup должен быть реализован в подклассе.")


class SGD(Optimizer):
    """
    Стохастический градиентный спуск (SGD).
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False):
        """
        Args:
            learning_rate (float): Начальная скорость обучения.
            momentum (float): Коэффициент момента.
            decay (float): Коэффициент затухания скорости обучения.
            nesterov (bool): Использовать ли Nesterov momentum.
        """
        self.nesterov = nesterov
        self.decay = decay
        self.momentum = momentum
        self.lr = learning_rate
        self.iteration = 0
        self.velocity = None
    
    def update(self, network):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iteration))
        
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                update = self.momentum * self.velocity[i][n] - lr * grad
                self.velocity[i][n] = update
                if self.nesterov:
                    update = self.momentum * self.velocity[i][n] - lr * grad
                layer.parameters.step(n, update)
        self.iteration += 1
    
    def setup(self, network):
        self.velocity = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.velocity[i][n] = np.zeros_like(layer.parameters[n])


class Adagrad(Optimizer):
    """
    Оптимизатор Adagrad.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Args:
            learning_rate (float): Скорость обучения.
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.eps = epsilon
        self.lr = learning_rate
    
    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] += grad ** 2
                step = self.lr * grad / (np.sqrt(self.accu[i][n]) + self.eps)
                layer.parameters.step(n, -step)
    
    def setup(self, network):
        self.accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])


class Adadelta(Optimizer):
    """
    Оптимизатор Adadelta.
    """
    
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-8):
        """
        Args:
            learning_rate (float): Скорость обучения.
            rho (float): Коэффициент затухания.
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.rho = rho
        self.eps = epsilon
        self.lr = learning_rate
    
    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] = self.rho * self.accu[i][n] + (1.0 - self.rho) * grad ** 2
                step = grad * np.sqrt(self.d_accu[i][n] + self.eps) / np.sqrt(self.accu[i][n] + self.eps)
                
                layer.parameters.step(n, -step * self.lr)
                self.d_accu[i][n] = self.rho * self.d_accu[i][n] + (1.0 - self.rho) * step ** 2
    
    def setup(self, network):
        self.accu = defaultdict(dict)
        self.d_accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])
                self.d_accu[i][n] = np.zeros_like(layer.parameters[n])


class RMSprop(Optimizer):
    """
    Оптимизатор RMSprop.
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        """
        Args:
            learning_rate (float): Скорость обучения.
            rho (float): Коэффициент затухания.
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.eps = epsilon
        self.rho = rho
        self.lr = learning_rate
    
    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.accu[i][n] = (self.rho * self.accu[i][n]) + (1.0 - self.rho) * (grad ** 2)
                step = self.lr * grad / (np.sqrt(self.accu[i][n]) + self.eps)
                layer.parameters.step(n, -step)
    
    def setup(self, network):
        self.accu = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.accu[i][n] = np.zeros_like(layer.parameters[n])


class Adam(Optimizer):
    """
    Оптимизатор Adam.
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate (float): Скорость обучения.
            beta_1 (float): Коэффициент экспоненциального сглаживания для первого момента.
            beta_2 (float): Коэффициент экспоненциального сглаживания для второго момента.
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.epsilon = epsilon
        self.beta_2 = beta_2
        self.beta_1 = beta_1
        self.lr = learning_rate
        self.iterations = 0
        self.t = 1
    
    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.ms[i][n] = (self.beta_1 * self.ms[i][n]) + (1.0 - self.beta_1) * grad
                self.vs[i][n] = (self.beta_2 * self.vs[i][n]) + (1.0 - self.beta_2) * grad ** 2
                lr = self.lr * np.sqrt(1.0 - self.beta_2 ** self.t) / (1.0 - self.beta_1 ** self.t)
                
                step = lr * self.ms[i][n] / (np.sqrt(self.vs[i][n]) + self.epsilon)
                layer.parameters.step(n, -step)
        self.t += 1
    
    def setup(self, network):
        self.ms = defaultdict(dict)
        self.vs = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.ms[i][n] = np.zeros_like(layer.parameters[n])
                self.vs[i][n] = np.zeros_like(layer.parameters[n])


class Adamax(Optimizer):
    """
    Оптимизатор Adamax.
    """
    
    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate (float): Скорость обучения.
            beta_1 (float): Коэффициент экспоненциального сглаживания для первого момента.
            beta_2 (float): Коэффициент экспоненциального сглаживания для второго момента.
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.epsilon = epsilon
        self.beta_2 = beta_2
        self.beta_1 = beta_1
        self.lr = learning_rate
        self.t = 1
    
    def update(self, network):
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                grad = layer.parameters.grad[n]
                self.ms[i][n] = self.beta_1 * self.ms[i][n] + (1.0 - self.beta_1) * grad
                self.us[i][n] = np.maximum(self.beta_2 * self.us[i][n], np.abs(grad))
                
                step = self.lr / (1 - self.beta_1 ** self.t) * self.ms[i][n] / (self.us[i][n] + self.epsilon)
                layer.parameters.step(n, -step)
        self.t += 1
    
    def setup(self, network):
        self.ms = defaultdict(dict)
        self.us = defaultdict(dict)
        for i, layer in enumerate(network.parametric_layers):
            for n in layer.parameters.keys():
                self.ms[i][n] = np.zeros_like(layer.parameters[n])
                self.us[i][n] = np.zeros_like(layer.parameters[n])
