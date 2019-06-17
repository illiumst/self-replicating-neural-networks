import numpy as np
from abc import abstractmethod, ABC
from typing import List, Union, Tuple
from types import FunctionType

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import SimpleRNN, Dense

from experiment import *

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


class WeightToolBox:

    def __init__(self):
        """
        Weight class, for easy manipulation of weight vectors from Keras models
        """

        # TODO: implement a way to access the cells directly
        # self.cells = len(self)
        # TODO: implement a way to access the weights directly
        # self.weights = self.to_flat_array() ?

    @staticmethod
    def max(weights: List[np.ndarray]):
        np.max(weights)

    @staticmethod
    def avg(weights: List[np.ndarray]):
        return np.average(weights)

    @staticmethod
    def weight_amount(weights: List[np.ndarray]):
        return np.sum([x.size for x in weights])

    @staticmethod
    def len(weights: List[np.ndarray]):
        return sum([x.size for x in weights])

    @staticmethod
    def shapes(weights: List[np.ndarray]):
        return [x.shape for x in weights]

    @staticmethod
    def num_layers(weights: List[np.ndarray]):
        return len(weights)

    def repr(self, weights: List[np.ndarray]):
        return f'Weights({self.to_flat_array(weights).tolist()})'

    @staticmethod
    def to_flat_array(weights: List[np.ndarray]) -> np.ndarray:
        return np.hstack([weight.flatten() for weight in weights])

    @staticmethod
    def reshape_flat_array(array, shapes) -> List[np.ndarray]:
        sizes: List[int] = [int(np.prod(shape)) for shape in shapes]
        # Split the incoming array into slices for layers
        slices = [array[x: y] for x, y in zip(np.cumsum([0] + sizes), np.cumsum([0] + sizes)[1:])]
        # reshape them in accordance to the given shapes
        weights = [np.reshape(weight_slice, shape) for weight_slice, shape in zip(slices, shapes)]
        return weights

    def reshape_flat_array_like(self, array, weights: List[np.ndarray]) -> List[np.ndarray]:
        return self.reshape_flat_array(array, self.shapes(weights))

    def shuffle_weights(self, weights: List[np.ndarray]):
        flat = self.to_flat_array(weights)
        np.random.shuffle(flat)
        return self.reshape_flat_array_like(flat, weights)

    @staticmethod
    def are_diverged(weights: List[np.ndarray]) -> bool:
        return any([np.isnan(x).any() for x in weights]) or any([np.isinf(x).any() for x in weights])

    @staticmethod
    def are_within_bounds(weights: List[np.ndarray], lower_bound: float, upper_bound: float) -> bool:
        return bool(sum([((lower_bound < x) & (x > upper_bound)).size for x in weights]))

    def aggregate_weights_by(self, weights: List[np.ndarray], func: FunctionType, num_aggregates: int):
        collection_sizes = self.len(weights) // num_aggregates
        weights = self.to_flat_array(weights)[:collection_sizes * num_aggregates].reshape((num_aggregates, -1))
        aggregated_weights = func(weights, num_aggregates)
        left_overs = self.to_flat_array(weights)[collection_sizes * num_aggregates:]
        return aggregated_weights, left_overs


class NeuralNetwork(ABC):
    """
    This is the Base Network Class, including abstract functions that must be implemented.
    """

    def __init__(self, **params):
        super().__init__()
        self.params = dict(epsilon=0.00000000000001)
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

    def get_weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def get_weights_flat(self) -> np.ndarray:
        return weightToolBox.to_flat_array(self.get_weights())

    def set_weights(self, new_weights: List[np.ndarray]):
        return self.model.set_weights(new_weights)

    @abstractmethod
    def get_samples(self):
        # TODO: add a dogstring, telling the user what this does, e.g. what is a sample?
        raise NotImplementedError

    @abstractmethod
    def apply_to_weights(self, old_weights) -> List[np.ndarray]:
        # TODO: add a dogstring, telling the user what this does, e.g. what is applied?
        raise NotImplementedError

    def apply_to_network(self, other_network) -> List[np.ndarray]:
        # TODO: add a dogstring, telling the user what this does, e.g. what is applied?
        new_weights = self.apply_to_weights(other_network.get_weights())
        return new_weights

    def attack(self, other_network):
        # TODO: add a dogstring, telling the user what this does, e.g. what is an attack?
        other_network.set_weights(self.apply_to_network(other_network))
        return self

    def fuck(self, other_network):
        # TODO: add a dogstring, telling the user what this does, e.g. what is fucking?
        self.set_weights(self.apply_to_network(other_network))
        return self

    def self_attack(self, iterations=1):
        # TODO: add a dogstring, telling the user what this does, e.g. what is self attack?
        for _ in range(iterations):
            self.attack(self)
        return self

    def meet(self, other_network):
        # TODO: add a dogstring, telling the user what this does, e.g. what is meeting?
        new_other_network = copy.deepcopy(other_network)
        return self.attack(new_other_network)

    def is_diverged(self):
        return weightToolBox.are_diverged(self.get_weights())

    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.get_params().get('epsilon')
        return weightToolBox.are_within_bounds(self.get_weights(), -epsilon, epsilon)

    def is_fixpoint(self, degree: int = 1, epsilon: float = None) -> bool:
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        new_weights = copy.deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if weightToolBox.are_diverged(new_weights):
                return False

        biggerEpsilon = (np.abs(weightToolBox.to_flat_array(new_weights) - weightToolBox.to_flat_array(self.get_weights()))
                         >= epsilon).any()

        # Boolean Value needs to be flipped to answer "is_fixpoint"
        return not biggerEpsilon

    def print_weights(self, weights=None):
        print(weightToolBox.repr(weights or self.get_weights()))


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

    def get_samples(self, weights: List[np.ndarray] = None):
        weights = weights or self.get_weights()
        sample = np.asarray([
            [weight, idx, *x] for idx, layer in enumerate(weights) for x, weight in np.ndenumerate(layer)
        ])
        # normalize [layer, cell, position]
        for idx in range(1, sample.shape[1]):
            sample[:, idx] = sample[:, idx] / np.max(sample[:, idx])
        return sample, sample

    def apply_to_weights(self, weights) -> List[np.ndarray]:
        # ToDo: Insert DocString
        # Transform the weight matrix in an horizontal stack as: array([[weight, layer, cell, position], ...])
        transformed_weights = self.get_samples(weights)[0]
        new_flat_weights = self.apply(transformed_weights)
        # use the original weight shape to transform the new tensor
        return weightToolBox.reshape_flat_array_like(new_flat_weights, weights)


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
        # ToDo: Find a better way than using the a hardcoded [0]
        return np.hstack([aggregate for _ in range(amount)])[0]

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

    @staticmethod
    def shuffle_random(weights: List[np.ndarray]):
        weights = weightToolBox.shuffle_weights(weights)
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

    def get_amount_of_weights(self):
        return weightToolBox.weight_amount(self.get_weights())

    def apply(self, inputs):
        # You need to add an dimension here... "..." copies array values
        return self.model.predict(inputs[None, ...])

    def get_aggregated_weights(self):
        return weightToolBox.aggregate_weights_by(self.get_weights(), self.get_aggregator(), self.aggregates)

    def apply_to_weights(self, old_weights) -> List[np.ndarray]:

        # build aggregations of old_weights
        old_aggregations, leftovers = self.get_aggregated_weights()

        # call network
        new_aggregations = self.apply(old_aggregations)
        collection_sizes = self.get_amount_of_weights() // self.aggregates
        new_aggregations = self.deaggregate_identically(new_aggregations, collection_sizes)
        # generate new weights
        # only include leftovers if there are some then coonvert them to Weight on base of th old shape
        complete_weights =  new_aggregations if not leftovers.shape[0] else np.hstack((new_aggregations, leftovers))
        new_weights = weightToolBox.reshape_flat_array_like(complete_weights, old_weights)

        # maybe shuffle
        new_weights = self.get_shuffler()(new_weights)
        return new_weights

    def get_samples(self):
        aggregations, _ = self.get_aggregated_weights()
        # What did that do?
        # sample = np.transpose(np.array([[aggregations[i]] for i in range(self.aggregates)]))
        return aggregations, aggregations

    def is_fixpoint_after_aggregation(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        old_aggregations, _ = self.get_aggregated_weights()
        new_weights = copy.deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if weightToolBox.are_diverged(new_weights):
                return False

        new_aggregations, leftovers = self.get_aggregated_weights()

        # ToDo: Explain This, why are you additionally checking tolerances of aggregated weights?
        biggerEpsilon = (np.abs(np.asarray(old_aggregations) - np.asarray(new_aggregations)) >= epsilon).any()

        # Boolean value has to be flipped to answer the question.
        return True, not biggerEpsilon


class RecurrentNeuralNetwork(NeuralNetwork):

    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
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
        new_weights = copy.deepcopy(old_weights)
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
        compile_params = copy.deepcopy(self.compile_params)
        compile_params.update(kwargs)
        return self.network.model.compile(**compile_params)

    def compiled(self, **kwargs):
        if not self.model_compiled:
            self.compile_model(**kwargs)
            self.model_compiled = True
        return self

    def train(self, batchsize=1, store_states=True, epoch=0):
        self.compiled()
        x, y = self.network.get_samples()
        savestatecallback = [SaveStateCallback(network=self, epoch=epoch)] if store_states else None
        history = self.network.model.fit(x=x, y=y, epochs=epoch+1, verbose=0,
                                         batch_size=batchsize, callbacks=savestatecallback,
                                         initial_epoch=epoch)
        return history.history['loss'][-1]

    def learn_from(self, other_network, batchsize=1):
        self.compiled()
        other_network.compiled()
        x, y = other_network.network.get_samples()
        history = self.network.model.fit(x=x, y=y, verbose=0, batch_size=batchsize)

        return history.history['loss'][-1]


weightToolBox = WeightToolBox()

if __name__ == '__main__':

    if False:
        # WeightWise Neural Network
        net_generator = lambda: ParticleDecorator(
            WeightwiseNeuralNetwork(width=2, depth=2
                                    ).with_keras_params(activation='linear'))
        with FixpointExperiment() as exp:
            exp.run_exp(net_generator, 10, logging=True)
            exp.reset_all()

    if False:
        # Aggregating Neural Network
        net_generator = lambda: ParticleDecorator(
            AggregatingNeuralNetwork(aggregates=4, width=2, depth=2
                                     ).with_keras_params())
        with FixpointExperiment() as exp:
            exp.run_exp(net_generator, 10, logging=True)
            exp.reset_all()

    if False:
        # FFT Aggregation
        net_generator = lambda: ParticleDecorator(
            AggregatingNeuralNetwork(
                aggregates=4, width=2, depth=2, aggregator=AggregatingNeuralNetwork.aggregate_fft
            ).with_keras_params(activation='linear'))
        with FixpointExperiment() as exp:
            exp.run_exp(net_generator, 10)
            exp.log(exp.counters)
            exp.reset_model()
            exp.reset_all()

    if True:
        # ok so this works quite realiably
        run_count = 10000
        net_generator = lambda: TrainingNeuralNetworkDecorator(
            ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2)
                              )).with_params(epsilon=0.0001).with_keras_params(optimizer='sgd')
        with MixedFixpointExperiment() as exp:
            for run_id in tqdm(range(run_count+1)):
                exp.run_exp(net_generator, 1)
                if run_id % 100 == 0:
                    exp.run_exp(net_generator, 1)
            K.clear_session()

    if True:
        with FixpointExperiment() as exp:
            run_count = 100
            net = TrainingNeuralNetworkDecorator(AggregatingNeuralNetwork(4, width=2, depth=2)).with_params(epsilon=0.1e-6)
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 100 == 0:
                    net.print_weights()
                    old_aggs, _ = net.net.get_aggregated_weights()
                    print("old weights agg: " + str(old_aggs))
                    fp, new_aggs = net.net.is_fixpoint_after_aggregation(epsilon=0.0001)
                    print("new weights agg: " + str(new_aggs))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Fixpoint after Agg? " + str(fp))
                    print("Loss " + str(loss))
                    print()

    if False:
        # this explodes in our faces completely... NAN everywhere
        # TODO: Wtf is happening here?
        with FixpointExperiment() as exp:
            run_count = 10000
            net = TrainingNeuralNetworkDecorator(RecurrentNeuralNetwork(width=2, depth=2)) \
                .with_params(epsilon=0.1e-2).with_keras_params(optimizer='sgd', activation='linear')
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 500 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Loss " + str(loss))
                    print()
