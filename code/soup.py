import random
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Input, Layer, Concatenate, RepeatVector, Reshape
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K

from typing import List, Tuple

# Functions and Operators
from operator import mul
from functools import reduce
from itertools import accumulate

import numpy as np

from task import Task, TaskAdditionOfN

from copy import copy, deepcopy

from network import ParticleDecorator, WeightwiseNeuralNetwork, TrainingNeuralNetworkDecorator, \
    EarlyStoppingByInfNanLoss

from experiment import TaskingSoupExperiment

from math import sqrt


def prng():
    return random.random()


class SlicingLayer(Layer):

    def __init__(self):
        self.kernel: None
        self.inputs: int
        super(SlicingLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = None
        self.inputs = input_shape[-1]
        super(SlicingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        concats = [Concatenate()([x[:, i][..., None]] * self.inputs) for i in range(x.shape[-1].value)]
        return concats

    def compute_output_shape(self, input_shape):
        return [Concatenate()([(None, 1)] * 4) for _ in range(input_shape[-1])]


class Soup(object):

    def __init__(self, size, generator, **kwargs):
        self.size = size
        self.generator = generator
        self.particles = []
        self.historical_particles = {}
        self.soup_params = dict(attacking_rate=0.1, learn_from_rate=0.1, train=0, learn_from_severity=1)
        self.soup_params.update(kwargs)
        self.time = 0
        self.is_seeded = False
        self.is_compiled = False

    def __len__(self):
        return len(self.particles)

    def __copy__(self):
        copy_ = Soup(self.size, self.generator, **self.soup_params)
        copy_.__dict__ = {attr: self.__dict__[attr] for attr in self.__dict__ if
                          attr not in ['particles', 'historical_particles']}
        return copy_

    def without_particles(self):
        self_copy = copy(self)
        # self_copy.particles = [particle.states for particle in self.particles]
        self_copy.historical_particles = {key: val.states for key, val in self.historical_particles.items()}
        return self_copy

    def with_soup_params(self, **kwargs):
        self.soup_params.update(kwargs)
        return self

    def generate_particle(self):
        new_particle = ParticleDecorator(self.generator())
        self.historical_particles[new_particle.get_uid()] = new_particle
        return new_particle

    def get_particle(self, uid, otherwise=None):
        return self.historical_particles.get(uid, otherwise)

    def seed(self):
        if not self.is_seeded:
            self.particles = []
            for _ in range(self.size):
                self.particles += [self.generate_particle()]
        else:
            print('already seeded!')
        self.is_seeded = True
        return self

    def evolve(self, iterations=1):
        for _ in range(iterations):
            self.time += 1
            for particle_id, particle in enumerate(self.particles):
                description = {'time': self.time}
                if prng() < self.soup_params.get('attacking_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.attack(other_particle)
                    description['action'] = 'attacking'
                    description['counterpart'] = other_particle.get_uid()

                if prng() < self.soup_params.get('learn_from_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    for _ in range(self.soup_params.get('learn_from_severity', 1)):
                        particle.learn_from(other_particle)
                    description['action'] = 'learn_from'
                    description['counterpart'] = other_particle.get_uid()

                for _ in range(self.soup_params.get('train', 0)):
                    # callbacks on save_state are broken for TrainingNeuralNetwork
                    loss = particle.train(store_states=False)
                    description['fitted'] = self.soup_params.get('train', 0)
                    description['loss'] = loss
                    description['action'] = 'train_self'
                    description['counterpart'] = None

                if self.soup_params.get('remove_divergent') and particle.is_diverged():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['action'] = 'divergent_dead'
                    description['counterpart'] = new_particle.get_uid()

                if self.soup_params.get('remove_zero') and particle.is_zero():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['action'] = 'zweo_dead'
                    description['counterpart'] = new_particle.get_uid()
                particle.save_state(**description)

    def count(self):
        counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        for particle in self.particles:
            if particle.is_diverged():
                counters['divergent'] += 1
            elif particle.is_fixpoint():
                if particle.is_zero():
                    counters['fix_zero'] += 1
                else:
                    counters['fix_other'] += 1
            elif particle.is_fixpoint(2):
                counters['fix_sec'] += 1
            else:
                counters['other'] += 1
        return counters

    def print_all(self):
        for particle in self.particles:
            particle.print_weights()
            print(particle.is_fixpoint())


class TaskingSoup(Soup):

    @staticmethod
    def weights_to_flat_array(weights: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([d.ravel() for d in weights])

    @staticmethod
    def reshape_flat_array(array, shapes: List[Tuple[int]]) -> List[np.ndarray]:
        # Same thing, but with an additional np call
        # sizes: List[int] = [int(np.prod(shape)) for shape in shapes]

        sizes = [reduce(mul, shape) for shape in shapes]
        # Split the incoming array into slices for layers
        slices = [array[x: y] for x, y in zip(accumulate([0] + sizes), accumulate(sizes))]
        # reshape them in accordance to the given shapes
        weights = [np.reshape(weight_slice, shape) for weight_slice, shape in zip(slices, shapes)]
        return weights

    def __init__(self, population_size: int, task: Task, particle_generator, sparsity_rate=0.1, use_bias=False,
                 safe=True, **kwargs):

        if safe:
            input_shape_error_message = f'The population size must be devideable by {task.input_shape[-1]}'
            assert population_size % task.input_shape[-1] == 0, input_shape_error_message
            assert population_size % 2 == 0, 'The population size needs to be of even value'

        super(TaskingSoup, self).__init__(population_size, particle_generator, **kwargs)
        self.task = task
        self.model: Sequential

        self.network_params = dict(sparsity_rate=sparsity_rate, early_nan_stopping=True, use_bias=use_bias,
                                   depth=population_size // task.input_shape[-1])
        self.network_params.update(kwargs.get('network_params', {}))
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.compile_params.update(kwargs.get('compile_params', {}))

    def with_network_params(self, **params):
        self.network_params.update(params)

    def _generate_model(self):
        particle_idx_list = list(range(len(self)))
        particles_per_layer = len(self) // self.network_params.get('depth')
        task_input = Input(self.task.input_shape, name='Task_Input')
        # First layer, which is conected to the input layer and independently trainable / not trainable at all.
        input_neurons = particles_per_layer * self.task.output_shape
        x = Dense(input_neurons, use_bias=self.network_params.get('use_bias'))(task_input)
        x = SlicingLayer()(x)

        for layer_num in range(self.network_params.get('depth')):
            # This needs to be tensors, because particles come as keras models that applicable
            x = [self.particles[layer_num*particles_per_layer + i].get_model()(x[i]) for
                 i in range(particles_per_layer)]
            x = [RepeatVector(particles_per_layer)(x[i]) for i in range(particles_per_layer)]
            x = [Reshape((particles_per_layer,))(x[i]) for i in range(particles_per_layer)]
        x = Concatenate()(x)
        x = Dense(self.task.output_shape, use_bias=self.network_params.get('use_bias'), activation='linear')(x)

        model = Model(inputs=task_input, outputs=x)
        return model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)

    def set_intermediate_weights(self, weights: List[np.ndarray]):
        all_weights = self.get_weights()
        all_weights[1:-1] = weights
        self.set_weights(all_weights)

    def get_intermediate_weights(self):
        return self.get_weights()[1:-1]

    def seed(self):
        K.clear_session()
        self.is_compiled = False
        super(TaskingSoup, self).seed()
        self.model = self._generate_model()
        pass

    def compile_model(self, **kwargs):
        if not self.is_compiled:
            compile_params = deepcopy(self.compile_params)
            compile_params.update(kwargs)
            return self.model.compile(**compile_params)
        else:
            raise BrokenPipeError('This Model is not compiled yet! Something went wrong in the Pipeline!')

    def get_total_weight_amount(self):
        if self.is_seeded:
            return sum([x.get_amount_of_weights() for x in self.particles])
        else:
            return 0

    def get_shapes(self):
        return [x.shape for x in self.get_weights()]

    def get_intermediate_shapes(self):
        weights = [x.shape for x in self.get_weights()]
        return weights[2:-2] if self.network_params.get('use_bias') else weights[1:-1]

    def predict(self, x):
        return self.model.predict(x)

    def evolve(self, iterations=1):
        for iteration in range(iterations):
            super(TaskingSoup, self).evolve(iterations=1)
            self.train_particles()

    def get_particle_weights(self):
        return np.concatenate([x.get_weights_flat() for x in self.particles])

    def get_particle_input_shape(self):
        if self.is_seeded:
            return tuple([x if x else -1 for x in self.particles[0].get_model().input_shape])
        else:
            return -1

    def set_particle_weights(self, weights):
        particle_weight_shape = self.particles[0].shapes(self.particles[0].get_weights())
        sizes = [x.get_amount_of_weights() for x in self.particles]
        flat_weights = self.weights_to_flat_array(weights)
        slices = [flat_weights[x: y] for x, y in zip(accumulate([0] + sizes), accumulate(sizes))]
        for particle, slice in zip(self.particles, slices):
            new_weights = self.reshape_flat_array(slice, particle_weight_shape)
            particle.set_weights(new_weights)
        return True

    def compiled(self, **kwargs):
        if not self.is_compiled:
            self.compile_model(**kwargs)
            self.is_compiled = True
        return self

    def train(self, batchsize=1, epoch=0):
        self.compiled()
        x, y = self.task.get_samples()
        callbacks = []
        if self.network_params.get('early_nan_stopping'):
            callbacks.append(EarlyStoppingByInfNanLoss())

        # 'or' does not work on empty lists
        callbacks = callbacks if callbacks else None
        """
        Please Note:

        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached."""
        history = self.model.fit(x=x, y=y, initial_epoch=epoch, epochs=epoch + 1, verbose=0,
                                 batch_size=batchsize, callbacks=callbacks)
        return history.history['loss'][-1]

    def train_particles(self, **kwargs):
        self.compiled()
        weights = self.get_particle_weights()
        shaped_weights = self.reshape_flat_array(weights, self.get_intermediate_shapes())
        self.set_intermediate_weights(shaped_weights)
        _ = self.train(**kwargs)  # This returns the loss values
        new_weights = self.get_intermediate_weights()
        self.set_particle_weights(new_weights)
        return


if __name__ == '__main__':
    if True:
        from task import TaskAdditionOfN

        net_generator = lambda: TrainingNeuralNetworkDecorator(
            WeightwiseNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
        )
        soup_generator = lambda: TaskingSoup(20, TaskAdditionOfN(4), net_generator)
        with TaskingSoupExperiment(soup_generator, name='solving_soup') as exp:
            exp.run_exp(reset_model=False)

    if False:
        soup_generator = lambda: Soup(10, net_generator).with_soup_params(remove_divergent=True, remove_zero=True)
        with SoupExperiment(soup_generator, name='soup') as exp:
            net_generator = lambda: TrainingNeuralNetworkDecorator(
                WeightwiseNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
            )

            exp.run_exp(net_generator)

            # net_generator = lambda: FFTNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
            # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
            # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
            # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()

    if False:
        soup_generator = lambda: Soup(10, net_generator).with_soup_params(remove_divergent=True, remove_zero=True)
        with SoupExperiment(soup_generator, name='soup') as exp:
            net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(2, 2)) \
                .with_keras_params(activation='linear').with_params(epsilon=0.0001)

            exp.run_exp(net_generator)

            # net_generator = lambda: TrainingNeuralNetworkDecorator(AggregatingNeuralNetwork(4, 2, 2))
            # .with_keras_params(activation='linear')\
            # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
            # net_generator = lambda: TrainingNeuralNetworkDecorator(FFTNeuralNetwork(4, 2, 2))\
            #     .with_keras_params(activation='linear')\
            # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
            # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()

