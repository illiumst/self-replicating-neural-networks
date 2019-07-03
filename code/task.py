from abc import ABC, abstractmethod
import numpy as np

from typing import Tuple, List, Union


class Task(ABC):

    def __init__(self, input_shape, output_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batchsize = kwargs.get('batchsize', 100)

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ParticleTaskAdditionOf2(Task):

    def __init__(self, **kwargs):
        super(ParticleTaskAdditionOf2, self).__init__(input_shape=(4,), output_shape=(1, ), **kwargs)

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((self.batchsize, *self.input_shape))
        x[:, :2] = np.random.standard_normal((self.batchsize, 2)) * 0.5
        y = np.sum(x, axis=1)
        return x, y


class SoupTask(Task):

    def __init__(self, input_shape, output_shape):
        super(SoupTask, self).__init__(input_shape, output_shape)
        pass

    def get_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
        # ToDo Hier geht es weiter.