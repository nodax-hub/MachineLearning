import autograd.numpy as np
from autograd import elementwise_grad

from .activations import get_activation
from .parameters import Parameters

np.random.seed(9999)


class Layer:
    """
    Базовый класс для всех слоёв.
    """

    def setup(self, input_dim):
        """
        Инициализирует параметры слоя.

        Args:
            input_dim (int): Размер входных данных.
        """
        pass

    def forward_pass(self, inputs):
        """
        Прямой проход через слой.

        Args:
            inputs (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Выход слоя.
        """
        raise NotImplementedError

    def backward_pass(self, grads):
        """
        Обратный проход через слой.

        Args:
            grads (np.ndarray): Градиенты на входе.

        Returns:
            np.ndarray: Градиенты на выходе.
        """
        raise NotImplementedError

    def shape(self):
        """Возвращает форму текущего слоя."""
        raise NotImplementedError


class ParamMixin:
    """
    Миксин для доступа к параметрам слоя.
    """

    @property
    def parameters(self):
        """Возвращает параметры слоя."""
        return self._parameters


class PhaseMixin:
    """
    Миксин для управления фазами обучения и тестирования.
    """

    def __init__(self):
        self._is_training = True

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, value):
        self._is_training = value

    @property
    def is_testing(self):
        return not self._is_training

    @is_testing.setter
    def is_testing(self, value):
        self._is_training = not value


class Dense(Layer, ParamMixin):
    """
    Полносвязный слой (Dense).

    Args:
        output_dim (int): Количество выходных нейронов.
        parameters (Parameters): Параметры слоя.
    """

    def __init__(self, output_dim, parameters=None):
        self.output_dim = output_dim
        self.activation_name = None
        self.activation = None
        self._parameters = None
        self.input_dim = None
        self._inputs = None

        if self._parameters is None:
            self._parameters = Parameters()

        elif isinstance(parameters, Parameters):
            self._parameters = parameters

        else:
            raise TypeError('Parameters must be a Parameters object.')

    def setup(self, input_dim):
        """
        Инициализирует параметры слоя.

        Args:
            input_dim (int): Количество входных нейронов.
        """
        self.input_dim = input_dim
        self._parameters.setup_weights({'W': (input_dim, self.output_dim), 'b': (self.output_dim,)})

    def forward_pass(self, inputs):
        """
        Прямой проход через Dense слой.

        Args:
            inputs (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Выходные данные.
        """
        self._inputs = inputs
        W, b = self.parameters['W'], self.parameters['b']
        output = np.dot(inputs, W) + b
        return self.activation(output) if self.activation else output

    def backward_pass(self, grads):
        """
        Обратный проход через Dense слой.

        Args:
            grads (np.ndarray): Градиенты на выходе.

        Returns:
            np.ndarray: Градиенты на входе.
        """
        W = self.parameters['W']
        dW = np.dot(self._inputs.T, grads)
        db = np.sum(grads, axis=0)
        dinputs = np.dot(grads, W.T)

        self.parameters.update_grad('W', dW)
        self.parameters.update_grad('b', db)
        return dinputs

    def shape(self):
        return (self.input_dim, self.output_dim)


class Activation(Layer):
    """
    Слой активации.

    Args:
        activation (str): Название функции активации.
    """

    def __init__(self, activation):
        self.activation_name = activation
        self.activation = get_activation(activation)
        self._inputs = None

    def forward_pass(self, inputs):
        self._inputs = inputs
        return self.activation(inputs)

    def backward_pass(self, grads):
        return grads * elementwise_grad(self.activation)(self._inputs)

    def shape(self):
        return None


class Dropout(Layer, PhaseMixin):
    """
    Слой Dropout для регуляризации.

    Args:
        p (float): Вероятность зануления нейрона.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward_pass(self, inputs):
        if self.is_training:
            self._mask = np.random.binomial(1, 1 - self.p, size=inputs.shape) / (1 - self.p)
            return inputs * self._mask
        return inputs

    def backward_pass(self, grads):
        return grads * self._mask if self.is_training else grads

    def shape(self):
        return None


class TimeStepSlicer(Layer):
    """
    Слой для выбора определённого временного шага из 3D тензора.

    Args:
        step (int): Индекс временного шага.
    """

    def __init__(self, step):
        self.step = step

    def forward_pass(self, inputs):
        return inputs[:, self.step, :]

    def backward_pass(self, grads):
        dinputs = np.zeros_like(grads)
        dinputs[:, self.step, :] = grads
        return dinputs

    def shape(self):
        return None


class TimeDistributedDense(Layer):
    """
    Применяет Dense слой к каждому временному шагу тензора.

    Args:
        output_dim (int): Количество выходных нейронов.
    """

    def __init__(self, output_dim):
        self.dense = Dense(output_dim)

    def setup(self, input_dim):
        self.dense.setup(input_dim)

    def forward_pass(self, inputs):
        time_steps = inputs.shape[1]
        outputs = [self.dense.forward_pass(inputs[:, t, :]) for t in range(time_steps)]
        return np.stack(outputs, axis=1)

    def backward_pass(self, grads):
        time_steps = grads.shape[1]
        dinputs = [self.dense.backward_pass(grads[:, t, :]) for t in range(time_steps)]
        return np.stack(dinputs, axis=1)

    @property
    def parameters(self):
        return self.dense.parameters

    def shape(self):
        return self.dense.shape()
