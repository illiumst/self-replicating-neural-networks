# Librarys
import numpy as np
from abc import abstractmethod, ABC
from typing import List, Tuple
from types import FunctionType
import warnings

import os

# Functions and Operators
from operator import mul
from functools import reduce
from itertools import accumulate
from copy import deepcopy

# Deep learning Framework
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import SimpleRNN, Dense

# Experiment Class
from task import TaskAdditionOfN
from experiment import TaskExperiment

# Supress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SaveStateCallback(Callback):
    def __init__(self, network, epoch=0):
        super(SaveStateCallback, self).__init__()
        self.net = network
        self.init_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        description = dict(time=epoch+self.init_epoch)
        description['action'] = 'train_self'
        description['counterpart'] = None
        self.net.save_state(**description)
        return


class EarlyStoppingByInfNanLoss(Callback):
    def __init__(self, monitor='loss', verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs: dict = None):
        logs = logs or dict()
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f'Early stopping requires {self.monitor} available!', RuntimeWarning)
            pass

        if np.isnan(current) or np.isinf(current):
            if self.verbose > 0:
                print(f'Epoch {epoch}: early stopping THR')
            self.model.stop_training = True


class NeuralNetwork(ABC):
    """
    This is the Base Network Class, including abstract functions that must be implemented.
    """

    @staticmethod
    def are_weights_diverged(weights: List[np.ndarray]) -> bool:
        return any([any((np.isnan(x).any(), np.isinf(x).any())) for x in weights])

    @staticmethod
    def are_weights_within_bounds(weights: List[np.ndarray], lower_bound: float, upper_bound: float) -> bool:
        return any([((lower_bound < x) & (x < upper_bound)).any() for x in weights])

    @staticmethod
    def get_weight_amount(weights: List[np.ndarray]):
        return sum([x.size for x in weights])

    @staticmethod
    def shapes(weights: List[np.ndarray]):
        return [x.shape for x in weights]

    @staticmethod
    def num_layers(weights: List[np.ndarray]):
        return len(weights)

    def repr(self, weights: List[np.ndarray]):
        return f'Weights({self.weights_to_flat_array(weights).tolist()})'

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

    def __init__(self, **params):
        super().__init__()
        self.params = dict(epsilon=0.00000000000001, early_nan_stopping=True, store_states=False)
        self.params.update(params)
        self.name = params.get('name', self.__class__.__name__)
        self.keras_params = dict(activation='linear', use_bias=False)
        self.states = []
        self.model: Sequential

    def get_params(self) -> dict:
        return self.params

    def get_keras_params(self) -> dict:
        return self.keras_params

    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def with_keras_params(self, **kwargs):
        self.keras_params.update(kwargs)
        return self

    def print_weights(self, weights=None):
        print(self.repr(weights or self.get_weights()))

    def get_amount_of_weights(self):
        return self.get_weight_amount(self.get_weights())

    def get_model(self):
        return self.model

    def get_weights(self) -> List[np.ndarray]:
        return self.get_model().get_weights()

    def get_weights_flat(self) -> np.ndarray:
        return self.weights_to_flat_array(self.get_weights())

    def reshape_flat_array_like(self, array, weights: List[np.ndarray]) -> List[np.ndarray]:
        return self.reshape_flat_array(array, self.shapes(weights))

    def set_weights(self, new_weights: List[np.ndarray]):
        return self.model.set_weights(new_weights)

    def apply_to_network(self, other_network) -> List[np.ndarray]:
        """
        Take a networks weights and apply _this_ networks function.
        :param other_network:
        :return:
        """
        new_weights = self.apply_to_weights(other_network.get_weights())
        return new_weights

    def is_diverged(self):
        return self.are_weights_diverged(self.get_weights())

    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.get_params().get('epsilon')
        return self.are_weights_within_bounds(self.get_weights(), -epsilon, epsilon)

    def is_fixpoint(self, degree: int = 1, epsilon: float = None) -> bool:
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        new_weights = deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if self.are_weights_diverged(new_weights):
                return False

        flat_new = self.weights_to_flat_array(new_weights)
        flat_old = self.weights_to_flat_array(self.get_weights())
        biggerEpsilon = (np.abs(flat_new - flat_old) >= epsilon).any()

        # Boolean Value needs to be flipped to answer "is_fixpoint"
        return not biggerEpsilon

    def aggregate_weights_by(self, weights: List[np.ndarray], func: FunctionType, num_aggregates: int):
        collection_sizes = self.get_weight_amount(weights) // num_aggregates
        flat = self.weights_to_flat_array(weights)
        array_for_aggregation = flat[:collection_sizes * num_aggregates].reshape((num_aggregates, -1))
        left_overs = flat[collection_sizes * num_aggregates:]
        aggregated_weights = func(array_for_aggregation, num_aggregates)
        return aggregated_weights, left_overs

    def shuffle_weights(self, weights: List[np.ndarray]):
        flat = self.weights_to_flat_array(weights)
        np.random.shuffle(flat)
        return self.reshape_flat_array_like(flat, weights)

    @abstractmethod
    def get_samples(self, **kwargs):
        # TODO: add a dogstring, telling the user what this does, e.g. what is a sample?
        raise NotImplementedError

    @abstractmethod
    def apply_to_weights(self, old_weights) -> List[np.ndarray]:
        """
        Take weights as inputs; retunr the evaluation of _this_ network.
        "Apply this function".

        :param old_weights:
        :return:
        """
        raise NotImplementedError


class ParticleDecorator:
    next_uid = 0

    def __init__(self, network):

        # ToDo: Add DocString, What does it do?

        self.uid = self.__class__.next_uid
        self.__class__.next_uid += 1
        self.network = network
        self.states = []
        self.save_state(time=0, action='init', counterpart=None)

    def __getattr__(self, name):
        return getattr(self.network, name)

    def get_uid(self):
        return self.uid

    def make_state(self, **kwargs):
        if self.network.is_diverged():
            return None
        state = {'class': self.network.__class__.__name__, 'weights': self.network.get_weights_flat()}
        state.update(kwargs)
        return state

    def save_state(self, **kwargs):
        state = self.make_state(**kwargs)
        if state is not None:
            self.states += [state]
        else:
            pass
        return True

    def update_state(self, number, **kwargs):
        raise NotImplementedError('Result is vague')
        # if number < len(self.states):
        #     self.states[number] = self.make_state(**kwargs)
        # else:
        #     for i in range(len(self.states), number):
        #         self.states += [None]
        #     self.states += self.make_state(**kwargs)

    def get_states(self):
        return self.states

    def attack(self, other_network, iterations: int = 1):
        """
        Set a networks weights based on the output of the application of my function to its weights.
        "Alter a networks weights based on my evaluation"
        :param other_network:
        :param iterations:
        :return:
        """
        for _ in range(iterations):
            other_network.set_weights(self.apply_to_network(other_network))
        return self

    def self_attack(self, iterations: int = 1):
        """
        Set my weights based on the output of the application of my function to its weights.
        "Alter my network weights based on my evaluation"
        :param iterations:
        :return:
        """
        for _ in range(iterations):
            self.attack(self)
        return self


class TaskDecorator(TaskAdditionOfN):

    def __init__(self, network, **kwargs):
        super(TaskDecorator, self).__init__(**kwargs)
        self.network = network
        self.batchsize = self.network.get_amount_of_weights()

    def __getattr__(self, name):
        return getattr(self.network, name)

    def get_samples(self, task_samples=False, self_samples=False, **kwargs):
        # XOR, cannot be true at the same time
        assert not all([task_samples, self_samples])

        if task_samples:
            return super(TaskDecorator, self).get_samples()

        elif self_samples:
            return self.network.get_samples()

        else:
            self_x, self_y = self.network.get_samples()
            # Super class = Task
            task_x, task_y = super(TaskDecorator, self).get_samples()

            amount_of_weights = self.network.get_amount_of_weights()
            random_idx = np.random.choice(np.arange(amount_of_weights), amount_of_weights//2)

            x = self_x[random_idx] = task_x[random_idx]
            y = self_y[random_idx] = task_y[random_idx]

            return x, y


class WeightwiseNeuralNetwork(NeuralNetwork):

    def __init__(self, width, depth, **kwargs):
        # ToDo: Insert Docstring
        super().__init__(**kwargs)
        self.width: int = width
        self.depth: int = depth
        self.model = Sequential()
        self.model.add(Dense(units=self.width, input_dim=4, **self.keras_params))
        for _ in range(self.depth-1):
            self.model.add(Dense(units=self.width, **self.keras_params))
        self.model.add(Dense(units=1, **self.keras_params))

    def apply(self, inputs):
        # TODO: Write about it... What does it do?
        return self.model.predict(inputs)

    def get_samples(self, **kwargs: List[np.ndarray]):
        weights = kwargs.get('weights', self.get_weights())
        sample = np.asarray([
            [weight, idx, *x] for idx, layer in enumerate(weights) for x, weight in np.ndenumerate(layer)
        ])
        # normalize [layer, cell, position]
        for idx in range(1, sample.shape[1]):
            sample[:, idx] = sample[:, idx] / np.max(sample[:, idx])
        return sample, sample[:, 0]

    def apply_to_weights(self, weights) -> List[np.ndarray]:
        # ToDo: Insert DocString
        # Transform the weight matrix in an horizontal stack as: array([[weight, layer, cell, position], ...])
        transformed_weights, _ = self.get_samples(weights=weights)
        new_flat_weights = self.apply(transformed_weights)
        # use the original weight shape to transform the new tensor
        return self.reshape_flat_array_like(new_flat_weights, weights)


class AggregatingNeuralNetwork(NeuralNetwork):

    @staticmethod
    def aggregate_fft(array: np.ndarray, aggregates: int):
        flat = array.flatten()
        # noinspection PyTypeChecker
        fft_reduction = np.fft.fftn(flat, aggregates)
        return fft_reduction

    @staticmethod
    def aggregate_average(array, _):
        return np.average(array, axis=1)

    @staticmethod
    def aggregate_max(array, _):
        return np.max(array, axis=1)

    @staticmethod
    def deaggregate_identically(aggregate, amount):
        return np.repeat(aggregate, amount, axis=0)

    @staticmethod
    def shuffle_not(weights: List[np.ndarray]):
        """
        Doesn't do a thing. f(x)

        :param weights: A List of Weights
        :type weights: Weights
        :return: The same old weights.
        :rtype: Weights
        """
        return weights

    def shuffle_random(self, weights: List[np.ndarray]):
        weights = self.shuffle_weights(weights)
        return weights

    def __init__(self, aggregates, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.aggregates = aggregates
        self.width = width
        self.depth = depth
        self.model = Sequential()
        self.model.add(Dense(units=width, input_dim=self.aggregates, **self.keras_params))
        for _ in range(depth-1):
            self.model.add(Dense(units=width, **self.keras_params))
        self.model.add(Dense(units=self.aggregates, **self.keras_params))

    def get_aggregator(self):
        return self.params.get('aggregator', self.aggregate_average)

    def get_deaggregator(self):
        return self.params.get('deaggregator', self.deaggregate_identically)

    def get_shuffler(self):
        return self.params.get('shuffler', self.shuffle_not)

    def apply(self, inputs):
        # You need to add an dimension here... "..." copies array values
        return self.model.predict(inputs[None, ...])

    def get_aggregated_weights(self):
        return self.aggregate_weights_by(self.get_weights(), self.get_aggregator(), self.aggregates)

    def apply_to_weights(self, old_weights) -> List[np.ndarray]:

        # build aggregations of old_weights
        old_aggregations, leftovers = self.get_aggregated_weights()

        # call network
        new_aggregations = self.apply(old_aggregations)
        collection_sizes = self.get_amount_of_weights() // self.aggregates
        new_aggregations = self.deaggregate_identically(new_aggregations, collection_sizes)
        # generate new weights
        # only include leftovers if there are some then coonvert them to Weight on base of th old shape
        complete_weights = new_aggregations if not leftovers.shape[0] else np.hstack((new_aggregations, leftovers))
        new_weights = self.reshape_flat_array_like(complete_weights, old_weights)

        # maybe shuffle
        new_weights = self.get_shuffler()(new_weights)
        return new_weights

    def get_samples(self, **kwargs):
        aggregations, _ = self.get_aggregated_weights()
        # What did that do?
        # sample = np.transpose(np.array([[aggregations[i]] for i in range(self.aggregates)]))
        return aggregations, aggregations

    def is_fixpoint_after_aggregation(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        old_aggregations, _ = self.get_aggregated_weights()
        new_weights = deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if self.are_weights_diverged(new_weights):
                return False

        new_aggregations, leftovers = self.get_aggregated_weights()

        # ToDo: Explain This, why are you additionally checking tolerances of aggregated weights?
        biggerEpsilon = (np.abs(np.asarray(old_aggregations) - np.asarray(new_aggregations)) >= epsilon).any()

        # Boolean value has to be flipped to answer the question.
        return True, not biggerEpsilon


class RecurrentNeuralNetwork(NeuralNetwork):

    def __init__(self, width, depth, **kwargs):
        raise NotImplementedError
        super(RecurrentNeuralNetwork, self).__init__()
        self.features = 1
        self.width = width
        self.depth = depth
        self.model = Sequential()
        self.model.add(SimpleRNN(units=width, input_dim=self.features, return_sequences=True, **self.keras_params))
        for _ in range(depth-1):
            self.model.add(SimpleRNN(units=width, return_sequences=True, **self.keras_params))
        self.model.add(SimpleRNN(units=self.features, return_sequences=True, **self.keras_params))

    def apply(self, *inputs):
        stuff = np.transpose(np.array([[[inputs[i]] for i in range(len(inputs))]]))
        return self.model.predict(stuff)[0].flatten()

    def apply_to_weights(self, old_weights):
        # build list from old weights
        new_weights = deepcopy(old_weights)
        old_weights_list = []
        for layer_id, layer in enumerate(old_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    old_weights_list += [weight]

        # call network
        new_weights_list = self.apply(*old_weights_list)

        # write back new weights from list of rnn returns
        current_weight_id = 0
        for layer_id, layer in enumerate(new_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights_list[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
        return new_weights

    def compute_samples(self):
        # build list from old weights
        old_weights_list = []
        for layer_id, layer in enumerate(self.get_weights()):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    old_weights_list += [weight]
        sample = np.asarray(old_weights_list)[None, ..., None]
        return sample, sample


class TrainingNeuralNetworkDecorator:

    def __init__(self, network):
        self.network = network
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.model_compiled = False

    def __getattr__(self, name):
        return getattr(self.network, name)

    def with_params(self, **kwargs):
        self.network.with_params(**kwargs)
        return self

    def with_keras_params(self, **kwargs):
        self.network.with_keras_params(**kwargs)
        return self

    def get_compile_params(self):
        return self.compile_params

    def with_compile_params(self, **kwargs):
        self.compile_params.update(kwargs)
        return self

    def compile_model(self, **kwargs):
        compile_params = deepcopy(self.compile_params)
        compile_params.update(kwargs)
        return self.network.model.compile(**compile_params)

    def compiled(self, **kwargs):
        if not self.model_compiled:
            self.compile_model(**kwargs)
            self.model_compiled = True
        return self

    def train(self, batchsize=1, epoch=0):
        self.compiled()
        x, y = self.network.get_samples()
        callbacks = []
        if self.get_params().get('store_states'):
            callbacks.append(SaveStateCallback(network=self, epoch=epoch))
        if self.get_params().get('early_nan_stopping'):
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
        history = self.network.model.fit(x=x, y=y, initial_epoch=epoch, epochs=epoch+1, verbose=0,
                                         batch_size=batchsize, callbacks=callbacks)
        return history.history['loss'][-1]

    def learn_from(self, other_network, batchsize=1):
        self.compiled()
        other_network.compiled()
        x, y = other_network.network.get_samples()
        history = self.network.model.fit(x=x, y=y, verbose=0, batch_size=batchsize)
        return history.history['loss'][-1]

    def evaluate(self, x=None, y=None, batchsize=1):
        self.compiled()
        x, y = x, y if x is not None and y is not None else self.network.get_samples()
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
        loss = self.network.model.evaluate(x=x, y=y, verbose=0, batch_size=batchsize)
        return loss


if __name__ == '__main__':

    if True:
        # WeightWise Neural Network
        with TaskExperiment().with_params(application_steps=10, trains_per_application=1000, exp_iterations=30) as exp:
            net_generator = lambda: TrainingNeuralNetworkDecorator(TaskDecorator(
                WeightwiseNeuralNetwork(width=2, depth=2))
            ).with_keras_params(activation='linear')
            exp.run_exp(net_generator, reset_model=True)

    if False:
        # Aggregating Neural Network
        net_generator = lambda: AggregatingNeuralNetwork(aggregates=4, width=2, depth=2)
        with MixedFixpointExperiment() as exp:
            exp.run_exp(net_generator, 10)
            exp.reset_all()

    if False:
        # FFT Aggregation
        net_generator = lambda: AggregatingNeuralNetwork(
            aggregates=4, width=2, depth=2, aggregator=AggregatingNeuralNetwork.aggregate_fft)
        with FixpointExperiment() as exp:
            exp.run_exp(net_generator, 10)
            exp.log(exp.counters)
            exp.reset_model()
            exp.reset_all()

    if False:
        # ok so this works quite realiably
        run_count = 1000
        net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(
            width=2, depth=2).with_params(epsilon=0.0001)).with_keras_params(optimizer='sgd')
        with MixedFixpointExperiment() as exp:
            for run_id in tqdm(range(run_count+1)):
                exp.run_exp(net_generator, 1)
                if run_id % 100 == 0:
                    exp.run_exp(net_generator, 1)
            K.clear_session()

    if False:
        with FixpointExperiment() as exp:
            run_count = 100
            net = TrainingNeuralNetworkDecorator(
                AggregatingNeuralNetwork(4, width=2, depth=2).with_params(epsilon=0.1e-6))
            for run_id in tqdm(range(run_count+1)):
                current_loss = net.compiled().train()
                if run_id % 100 == 0:
                    net.print_weights()
                    old_aggs, _ = net.get_aggregated_weights()
                    print("old weights agg: " + str(old_aggs))
                    fp, new_aggs = net.is_fixpoint_after_aggregation(epsilon=0.0001)
                    print("new weights agg: " + str(new_aggs))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Fixpoint after Agg? " + str(fp))
                    print("Loss " + str(current_loss))
                    print()

    if False:
        # this explodes in our faces completely... NAN everywhere
        # TODO: Wtf is happening here?
        with FixpointExperiment() as exp:
            run_count = 10000
            net = TrainingNeuralNetworkDecorator(RecurrentNeuralNetwork(width=2, depth=2)
                                                 ).with_keras_params(optimizer='sgd', activation='linear')
            for run_id in tqdm(range(run_count+1)):
                current_loss = net.compiled().train()
                if run_id % 500 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Loss " + str(current_loss))
                    print()
