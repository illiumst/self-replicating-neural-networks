import numpy as np
from abc import abstractmethod, ABC
from typing import List, Union

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import SimpleRNN, Dense
from tensorflow.python.keras import backend as K

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


class Weights:

    @staticmethod
    def __reshape_flat_array__(array, shapes):
        sizes: List[int] = [int(np.prod(shape)) for shape in shapes]
        # Split the incoming array into slices for layers
        slices = [array[x: y] for x, y in zip(np.cumsum([0]+sizes), np.cumsum([0]+sizes)[1:])]
        # reshape them in accordance to the given shapes
        weights = [np.reshape(weight_slice, shape) for weight_slice, shape in zip(slices, shapes)]
        return weights

    def __init__(self, weight_vector: Union[List[np.ndarray], np.ndarray], flat_array_shape=None):
        """
        Weight class, for easy manipulation of weight vectors from Keras models

        :param weight_vector: A numpy array holding weights
        :type weight_vector: List[np.ndarray]
        """
        self.__iter_idx = [0, 0]
        if flat_array_shape:
            weight_vector = self.__reshape_flat_array__(weight_vector, flat_array_shape)

        self.layers = weight_vector

        # TODO: implement a way to access the cells directly
        # self.cells = len(self)
        # TODO: implement a way to access the weights directly
        # self.weights = self.to_flat_array() ?

    def __iter__(self):
        self.__iter_idx = [0, 0]
        return self

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return sum([x.size for x in self.layers])

    def shapes(self):
        return [x.shape for x in self.layers]

    def num_layers(self):
        return len(self.layers)

    def __copy__(self):
        return copy.deepcopy(self)

    def __next__(self):
        # ToDo: Check iteration progress over layers
        # ToDo: There is still a problem interation, currently only cell level is the last loop stage.
        # Do we need this?
        if self.__iter_idx[0] >= len(self.layers):
            if self.__iter_idx[1] >= len(self.layers[self.__iter_idx[0]]):
                raise StopIteration
        result = self.layers[self.__iter_idx[0]][self.__iter_idx[1]]

        if self.__iter_idx[1] >= len(self.layers[self.__iter_idx[0]]):
            self.__iter_idx[0] += 1
            self.__iter_idx[1] = 0
        else:
            self.__iter_idx[1] += 1
        return result

    def __repr__(self):
        return f'Weights({self.to_flat_array().tolist()})'

    def to_flat_array(self) -> np.ndarray:
        return np.hstack([weight.flatten() for weight in self.layers])

    def from_flat_array(self, array):
        new_weights = self.__reshape_flat_array__(array, self.shapes())
        return new_weights

    def are_diverged(self):
        return any([np.isnan(x).any() for x in self.layers]) or any([np.isinf(x).any() for x in self.layers])

    def are_within_bounds(self, lower_bound: float, upper_bound: float):
        return bool(sum([((lower_bound < x) & (x > upper_bound)).size for x in self.layers]))

    def apply_new_weights(self, weights: np.ndarray):
        # TODO: Make this more Pythonic
        new_weights = copy.deepcopy(self.layers)
        current_weight_id = 0
        for layer_id, layer in enumerate(new_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = weights[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
        return new_weights


class NeuralNetwork(ABC):
    """
    This is the Base Network Class, including abstract functions that must be implemented.
    """

    def __init__(self, **params):
        super().__init__()
        self.params = dict(epsilon=0.00000000000001)
        self.params.update(params)
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

    def get_weights(self) -> Weights:
        return Weights(self.model.get_weights())

    def get_weights_flat(self) -> np.ndarray:
        return self.get_weights().to_flat_array()

    def set_weights(self, new_weights: Weights):
        return self.model.set_weights(new_weights)

    @abstractmethod
    def apply_to_weights(self, old_weights) -> Weights:
        # TODO: add a dogstring, telling the user what this does, e.g. what is applied?
        raise NotImplementedError

    def apply_to_network(self, other_network) -> Weights:
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
        return self.get_weights().are_diverged()

    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.get_params().get('epsilon')
        return self.get_weights().are_within_bounds(-epsilon, epsilon)

    def is_fixpoint(self, degree: int = 1, epsilon: float = None) -> bool:
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        new_weights = copy.deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if new_weights.are_diverged():
                return False

        biggerEpsilon = (np.abs(new_weights.to_flat_array() - self.get_weights().to_flat_array()) >= epsilon).any()

        # Boolean Value needs to be flipped to answer "is_fixpoint"
        return not biggerEpsilon

    def print_weights(self, weights=None):
        print(weights or self.get_weights())


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
        if number < len(self.states):
            self.states[number] = self.make_state(**kwargs)
        else:
            for i in range(len(self.states), number):
                self.states += [None]
            self.states += self.make_state(**kwargs)

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

    def apply_to_weights(self, weights) -> Weights:
        # ToDo: Insert DocString
        # Transform the weight matrix in an horizontal stack as: array([[weight, layer, cell, position], ...])
        transformed_weights = np.asarray([
            [weight, idx, *x] for idx, layer in enumerate(weights.layers) for x, weight in np.ndenumerate(layer)
                         ])
        # normalize [layer, cell, position]
        for idx in range(1, transformed_weights.shape[1]):
            transformed_weights[:, idx] = transformed_weights[:, idx] / np.max(transformed_weights[:, idx])
        new_weights = self.apply(transformed_weights)
        # use the original weight shape to transform the new tensor
        return Weights(new_weights, flat_array_shape=weights.shapes())


class AggregatingNeuralNetwork(NeuralNetwork):

    @staticmethod
    def aggregate_average(weights):
        total = 0
        count = 0
        for weight in weights:
            total += float(weight)
            count += 1
        return total / float(count)

    @staticmethod
    def aggregate_max(weights):
        max_found = weights[0]
        for weight in weights:
            max_found = weight > max_found and weight or max_found
        return max_found

    @staticmethod
    def deaggregate_identically(aggregate, amount):
        return [aggregate for _ in range(amount)]

    @staticmethod
    def shuffle_not(weights_list):
        return weights_list

    @staticmethod
    def shuffle_random(weights_list):
        import random
        random.shuffle(weights_list)
        return weights_list

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
        total_weights = 0
        for layer_id, layer in enumerate(self.get_weights()):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    total_weights += 1
        return total_weights

    def apply(self, *inputs):
        stuff = np.transpose(np.array([[inputs[i]] for i in range(self.aggregates)]))
        return self.model.predict(stuff)[0]

    def apply_to_weights(self, old_weights):
        # build aggregations from old_weights
        collection_size = self.get_amount_of_weights() // self.aggregates
        collections, leftovers = self.collect_weights(old_weights, collection_size)

        # call network
        old_aggregations = [self.get_aggregator()(collection) for collection in collections]
        new_aggregations = self.apply(*old_aggregations)

        # generate list of new weights
        new_weights_list = []
        for aggregation_id, aggregation in enumerate(new_aggregations):
            if aggregation_id == self.aggregates - 1:
                new_weights_list += self.get_deaggregator()(aggregation, collection_size + leftovers)
            else:
                new_weights_list += self.get_deaggregator()(aggregation, collection_size)
        new_weights_list = self.get_shuffler()(new_weights_list)

        # write back new weights
        new_weights = self.fill_weights(old_weights, new_weights_list)

        # return results
        # if self.params.get("print_all_weight_updates", False) and not self.is_silent():
        #     print("updated old weight aggregations " + str(old_aggregations))
        #     print("to new weight aggregations      " + str(new_aggregations))
        #     print("resulting in network weights ...")
        #     print(self.weights_to_string(new_weights))
        return new_weights

    @staticmethod
    def collect_weights(all_weights, collection_size):
        collections = []
        next_collection = []
        current_weight_id = 0
        for layer_id, layer in enumerate(all_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    next_collection += [weight]
                    if (current_weight_id + 1) % collection_size == 0:
                        collections += [next_collection]
                        next_collection = []
                    current_weight_id += 1
        collections[-1] += next_collection
        leftovers = len(next_collection)
        return collections, leftovers

    def get_collected_weights(self):
        collection_size = self.get_amount_of_weights() // self.aggregates
        return self.collect_weights(self.get_weights(), collection_size)

    def get_aggregated_weights(self):
        collections, leftovers = self.get_collected_weights()
        aggregations = [self.get_aggregator()(collection) for collection in collections]
        return aggregations, leftovers

    def compute_samples(self):
        aggregations, _ = self.get_aggregated_weights()
        sample = np.transpose(np.array([[aggregations[i]] for i in range(self.aggregates)]))
        return [sample], [sample]

    def is_fixpoint_after_aggregation(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')

        old_aggregations, _ = self.get_aggregated_weights()
        new_weights = copy.deepcopy(self.get_weights())

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
            if new_weights.are_diverged():
                return False

        # ToDo: Explain This, what the heck is happening?
        collection_size = self.get_amount_of_weights() // self.aggregates
        collections, leftovers = self.__class__.collect_weights(new_weights, collection_size)
        new_aggregations = [self.get_aggregator()(collection) for collection in collections]

        # ToDo: Explain This, why are you additionally checking tolerances of aggregated weights?
        biggerEpsilon = (np.abs(np.asarray(old_aggregations) - np.asarray(new_aggregations)) >= epsilon).any()
        # Boolean value hast to be flipped to answer the question.
        return True, not biggerEpsilon


class FFTNeuralNetwork(NeuralNetwork):

    @staticmethod
    def aggregate_fft(weights, dims):
        flat = np.hstack([weight.flatten() for weight in weights])
        fft_reduction = np.fft.fftn(flat, dims)[None, ...]
        return fft_reduction

    @staticmethod
    def deaggregate_identically(aggregate, dims):
        fft_inverse = np.fft.ifftn(aggregate, dims)
        return fft_inverse

    @staticmethod
    def shuffle_not(weights_list):
        return weights_list

    @staticmethod
    def shuffle_random(weights_list):
        import random
        random.shuffle(weights_list)
        return weights_list

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

    def get_shuffler(self):
        return self.params.get('shuffler', self.shuffle_not)

    def get_amount_of_weights(self):
        total_weights = 0
        for layer_id, layer in enumerate(self.get_weights()):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    total_weights += 1
        return total_weights

    def apply(self, inputs):
        sample = np.asarray(inputs)
        return self.model.predict(sample)[0]

    def apply_to_weights(self, old_weights):
        # build aggregations from old_weights
        weights = self.get_weights_flat()

        # call network
        old_aggregation = self.aggregate_fft(weights, self.aggregates)
        new_aggregation = self.apply(old_aggregation)

        # generate list of new weights
        new_weights_list = self.deaggregate_identically(new_aggregation, self.get_amount_of_weights())

        new_weights_list = self.get_shuffler()(new_weights_list)

        # write back new weights
        new_weights = self.fill_weights(old_weights, new_weights_list)

        # return results
        # if self.params.get("print_all_weight_updates", False) and not self.is_silent():
        #     print("updated old weight aggregations " + str(old_aggregation))
        #     print("to new weight aggregations      " + str(new_aggregation))
        #     print("resulting in network weights ...")
        #     print(self.weights_to_string(new_weights))
        return new_weights

    def compute_samples(self):
        weights = self.get_weights()
        sample = np.asarray(weights)[None, ...]
        return [sample], [sample]


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


class TrainingNeuralNetworkDecorator():

    def __init__(self, net, **kwargs):
        self.net = net
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.model_compiled = False

    def __getattr__(self, name):
        return getattr(self.net, name)

    def with_params(self, **kwargs):
        self.net.with_params(**kwargs)
        return self

    def with_keras_params(self, **kwargs):
        self.net.with_keras_params(**kwargs)
        return self

    def get_compile_params(self):
        return self.compile_params

    def with_compile_params(self, **kwargs):
        self.compile_params.update(kwargs)
        return self

    def compile_model(self, **kwargs):
        compile_params = copy.deepcopy(self.compile_params)
        compile_params.update(kwargs)
        return self.net.model.compile(**compile_params)

    def compiled(self, **kwargs):
        if not self.model_compiled:
            self.compile_model(**kwargs)
            self.model_compiled = True
        return self

    def train(self, batchsize=1, store_states=True, epoch=0):
        self.compiled()
        x, y = self.net.compute_samples()
        savestatecallback = [SaveStateCallback(net=self, epoch=epoch)] if store_states else None
        history = self.net.model.fit(x=x, y=y, epochs=epoch+1, verbose=0, batch_size=batchsize, callbacks=savestatecallback, initial_epoch=epoch)
        return history.history['loss'][-1]

    def learn_from(self, other_network, batchsize=1):
        self.compiled()
        other_network.compiled()
        x, y = other_network.net.compute_samples()
        history = self.net.model.fit(x=x, y=y, verbose=0, batch_size=batchsize)

        return history.history['loss'][-1]


if __name__ == '__main__':
    def run_exp(net, prints=False):
        # INFO Run_ID needs to be more than 0, so that exp stores the trajectories!
        exp.run_net(net, 100, run_id=run_id + 1)
        exp.historical_particles[run_id] = net
        if prints:
            print("Fixpoint? " + str(net.is_fixpoint()))
            print("Loss " + str(loss))

    if True:
        # WeightWise Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2) \
                                        .with_keras_params(activation='linear'))
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)

    if False:
        # Aggregating Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(AggregatingNeuralNetwork(aggregates=4, width=2, depth=2) \
                                        .with_keras_params())
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)

    if False:
        #FFT Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(FFTNeuralNetwork(aggregates=4, width=2, depth=2) \
                                        .with_keras_params(activation='linear'))
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)

    if False:
        # ok so this works quite realiably
        with FixpointExperiment() as exp:
            for i in range(1):
                run_count = 1000
                net = TrainingNeuralNetworkDecorator(ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2)))
                net.with_params(epsilon=0.0001).with_keras_params(optimizer='sgd')
                for run_id in tqdm(range(run_count+1)):
                    net.compiled()
                    loss = net.train(epoch=run_id)
                    if run_id % 100 == 0:
                        run_exp(net)
                K.clear_session()

    if False:
        with FixpointExperiment() as exp:
            run_count = 1000
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
            net = TrainingNeuralNetworkDecorator(RecurrentNeuralNetwork(width=2, depth=2))\
                .with_params(epsilon=0.1e-2).with_keras_params(optimizer='sgd', activation='linear')
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 500 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Loss " + str(loss))
                    print()
    if False:
        # and this gets somewhat interesting... we can still achieve non-trivial fixpoints
        # over multiple applications when training enough in-between
        with MixedFixpointExperiment() as exp:
            for run_id in range(10):
                net = TrainingNeuralNetworkDecorator(FFTNeuralNetwork(2, width=2, depth=2))\
                    .with_params(epsilon=0.0001, activation='sigmoid')
                exp.run_net(net, 500, 10)

                net.print_weights()

                print("Fixpoint? " + str(net.is_fixpoint()))
            exp.log(exp.counters)
