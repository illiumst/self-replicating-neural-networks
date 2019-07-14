from abc import ABC, abstractmethod
import numpy as np

from typing import Tuple


class Task(ABC):

    def __init__(self, input_shape, output_shape, **kwargs):
        assert any([x not in kwargs.keys() for x in ["input_shape", "output_shape"]]), 'Dublicated arguments were given'
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batchsize = kwargs.get('batchsize', 100)

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TaskAdditionOfN(Task):

    def __init__(self, n: int, input_shape=(4,), output_shape=1, **kwargs):
        assert any([x not in kwargs.keys() for x in ["input_shape", "output_shape"]]), 'Dublicated arguments were given'
        assert n <= input_shape[0], f'You cannot Add more values (n={n}) than your input is long (in={input_shape}).'
        kwargs.update(input_shape=input_shape, output_shape=output_shape)
        super(TaskAdditionOfN, self).__init__(**kwargs)
        self.n = n

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((self.batchsize, *self.input_shape))
        x[:, :self.n] = np.random.standard_normal((self.batchsize, self.n)) * 0.5
        y = np.sum(x, axis=1)
        return x, y
