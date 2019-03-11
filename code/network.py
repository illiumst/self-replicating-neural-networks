import numpy as np

from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import SimpleRNN, Dense
import keras.backend as K

from util import *
from experiment import *

# Supress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SaveStateCallback(Callback):
    def __init__(self, net, epoch=0):
        super(SaveStateCallback, self).__init__()
        self.net = net
        self.init_epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        description = dict(time=epoch+self.init_epoch)
        description['action'] = 'train_self'
        description['counterpart'] = None
        self.net.save_state(**description)
        return


class NeuralNetwork(PrintingObject):

    @staticmethod
    def weights_to_string(weights):
        s = ""
        for layer_id, layer in enumerate(weights):
            for cell_id, cell in enumerate(layer):
                s += "[ "
                for weight_id, weight in enumerate(cell):
                    s += str(weight) + " "
                s += "]"
            s += "\n"
        return s

    @staticmethod
    def are_weights_diverged(network_weights):
        for layer_id, layer in enumerate(network_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    if np.isnan(weight):
                        return True
                    if np.isinf(weight):
                        return True
        return False

    @staticmethod
    def are_weights_within(network_weights, lower_bound, upper_bound):
        for layer_id, layer in enumerate(network_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    # could be a chain comparission "lower_bound <= weight <= upper_bound"
                    if not (lower_bound <= weight and weight <= upper_bound):
                        return False
        return True

    @staticmethod
    def fill_weights(old_weights, new_weights_list):
        new_weights = copy.deepcopy(old_weights)
        current_weight_id = 0
        for layer_id, layer in enumerate(new_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights_list[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
        return new_weights

    def __init__(self, **params):
        super().__init__()
        self.params = dict(epsilon=0.00000000000001)
        self.params.update(params)
        self.keras_params = dict(activation='linear', use_bias=False)
        self.states = []

    def get_model(self):
        raise NotImplementedError

    def get_params(self):
        return self.params

    def get_keras_params(self):
        return self.keras_params

    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def with_keras_params(self, **kwargs):
        self.keras_params.update(kwargs)
        return self

    def get_weights(self):
        return self.model.get_weights()

    def get_weights_flat(self):
        return np.hstack([weight.flatten() for weight in self.get_weights()])

    def set_weights(self, new_weights):
        return self.model.set_weights(new_weights)

    def apply_to_weights(self, old_weights):
        raise NotImplementedError

    def apply_to_network(self, other_network):
        new_weights = self.apply_to_weights(other_network.get_weights())
        return new_weights

    def attack(self, other_network):
        other_network.set_weights(self.apply_to_network(other_network))
        return self

    def fuck(self, other_network):
        self.set_weights(self.apply_to_network(other_network))
        return self

    def self_attack(self, iterations=1):
        for _ in range(iterations):
            self.attack(self)
        return self

    def meet(self, other_network):
        new_other_network = copy.deepcopy(other_network)
        return self.attack(new_other_network)

    def is_diverged(self):
        return self.are_weights_diverged(self.get_weights())

    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.get_params().get('epsilon')
        return self.are_weights_within(self.get_weights(), -epsilon, epsilon)

    def is_fixpoint(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')
        old_weights = self.get_weights()
        new_weights = copy.deepcopy(old_weights)

        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)

        if NeuralNetwork.are_weights_diverged(new_weights):
            return False
        for layer_id, layer in enumerate(old_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights[layer_id][cell_id][weight_id]
                    if abs(new_weight - weight) >= epsilon:
                        return False
        return True

    def repr_weights(self, weights=None):
        return self.weights_to_string(weights or self.get_weights())

    def print_weights(self, weights=None):
        print(self.repr_weights(weights))


class ParticleDecorator:
    next_uid = 0

    def __init__(self, net):
        self.uid = self.next_uid
        self.next_uid += 1
        self.net = net
        self.states = []

    def __getattr__(self, name):
        return getattr(self.net, name)

    def get_uid(self):
        return self.uid

    def make_state(self, **kwargs):
        weights = self.net.get_weights_flat()
        if any(np.isinf(weights)):
            return None
        state = {'class': self.net.__class__.__name__, 'weights': weights}
        state.update(kwargs)
        return state

    def save_state(self, **kwargs):
        state = self.make_state(**kwargs)
        if state is not None:
            self.states += [state]
        else:
            pass

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

    @staticmethod
    def normalize_id(value, norm):
        if norm > 1:
            return float(value) / float(norm)
        else:
            return float(value)

    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.model = Sequential()
        self.model.add(Dense(units=self.width, input_dim=4, **self.keras_params))
        for _ in range(self.depth-1):
            self.model.add(Dense(units=self.width, **self.keras_params))
        self.model.add(Dense(units=1, **self.keras_params))

    def get_model(self):
        return self.model

    def apply(self, *inputs):
        stuff = np.transpose(np.array([[inputs[0]], [inputs[1]], [inputs[2]], [inputs[3]]]))
        return self.model.predict(stuff)[0][0]

    @classmethod
    def compute_all_duplex_weight_points(cls, old_weights):
        points = []
        normal_points = []
        max_layer_id = len(old_weights) - 1
        for layer_id, layer in enumerate(old_weights):
            max_cell_id = len(layer) - 1
            for cell_id, cell in enumerate(layer):
                max_weight_id = len(cell) - 1
                for weight_id, weight in enumerate(cell):
                    normal_layer_id = cls.normalize_id(layer_id, max_layer_id)
                    normal_cell_id = cls.normalize_id(cell_id, max_cell_id)
                    normal_weight_id = cls.normalize_id(weight_id, max_weight_id)

                    points += [[weight, layer_id, cell_id, weight_id]]
                    normal_points += [[weight, normal_layer_id, normal_cell_id, normal_weight_id]]
        return points, normal_points

    @classmethod
    def compute_all_weight_points(cls, all_weights):
        return cls.compute_all_duplex_weight_points(all_weights)[0]

    @classmethod
    def compute_all_normal_weight_points(cls, all_weights):
        return cls.compute_all_duplex_weight_points(all_weights)[1]

    def apply_to_weights(self, old_weights):
        new_weights = copy.deepcopy(self.get_weights())
        for (weight_point, normal_weight_point) in zip(*self.__class__.compute_all_duplex_weight_points(old_weights)):
            weight, layer_id, cell_id, weight_id = weight_point
            _, normal_layer_id, normal_cell_id, normal_weight_id = normal_weight_point

            new_weight = self.apply(*normal_weight_point)
            new_weights[layer_id][cell_id][weight_id] = new_weight

            if self.params.get("print_all_weight_updates", False) and not self.is_silent():
                print("updated old weight {weight}\t @ ({layer},{cell},{weight_id}) "
                      "to new value {new_weight}\t calling @ ({normal_layer},{normal_cell},{normal_weight_id})").format(
                    weight=weight, layer=layer_id, cell=cell_id, weight_id=weight_id, new_weight=new_weight,
                    normal_layer=normal_layer_id, normal_cell=normal_cell_id, normal_weight_id=normal_weight_id)
        return new_weights

    def compute_samples(self):
        samples = []
        for normal_weight_point in self.compute_all_normal_weight_points(self.get_weights()):
            weight, normal_layer_id, normal_cell_id, normal_weight_id = normal_weight_point

            sample = np.transpose(np.array([[weight], [normal_layer_id], [normal_cell_id], [normal_weight_id]]))
            samples += [sample[0]]
        samples_array = np.asarray(samples)
        return samples_array, samples_array[:, 0]


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

    def get_model(self):
        return self.model

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
        if self.params.get("print_all_weight_updates", False) and not self.is_silent():
            print("updated old weight aggregations " + str(old_aggregations))
            print("to new weight aggregations      " + str(new_aggregations))
            print("resulting in network weights ...")
            print(self.weights_to_string(new_weights))
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

        old_weights = self.get_weights()
        old_aggregations, _ = self.get_aggregated_weights()

        new_weights = copy.deepcopy(old_weights)
        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
        if NeuralNetwork.are_weights_diverged(new_weights):
            return False
        collection_size = self.get_amount_of_weights() // self.aggregates
        collections, leftovers = self.__class__.collect_weights(new_weights, collection_size)
        new_aggregations = [self.get_aggregator()(collection) for collection in collections]

        for aggregation_id, old_aggregation in enumerate(old_aggregations):
            new_aggregation = new_aggregations[aggregation_id]
            if abs(new_aggregation - old_aggregation) >= epsilon:
                return False, new_aggregations
        return True, new_aggregations


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

    def get_model(self):
        return self.model

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
        if self.params.get("print_all_weight_updates", False) and not self.is_silent():
            print("updated old weight aggregations " + str(old_aggregation))
            print("to new weight aggregations      " + str(new_aggregation))
            print("resulting in network weights ...")
            print(self.__class__.weights_to_string(new_weights))
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

    def get_model(self):
        return self.model

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
        savestatecallback = SaveStateCallback(net=self.net, epoch=epoch) if store_states else None
        history = self.net.model.fit(x=x, y=y, verbose=0, batch_size=batchsize, callbacks=[savestatecallback], initial_epoch=epoch)
        return history.history['loss'][-1]

    def train_other(self, other_network, batchsize=1):
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
            # print(net.apply_to_network(net))
            print("Fixpoint? " + str(net.is_fixpoint()))
            print("Loss " + str(loss))
        K.clear_session()

    if False:
        # WeightWise Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2) \
                                        .with_keras_params(activation='linear'))
                run_exp(net)
            exp.log(exp.counters)

    if False:
        # Aggregating Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(AggregatingNeuralNetwork(aggregates=4, width=2, depth=2) \
                                        .with_keras_params())
                run_exp(net)
            exp.log(exp.counters)

    if False:
        #FFT Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                net = ParticleDecorator(FFTNeuralNetwork(aggregates=4, width=2, depth=2) \
                                        .with_keras_params(activation='linear'))
                run_exp(net)
            exp.log(exp.counters)

    if True:
        # ok so this works quite realiably
        with FixpointExperiment() as exp:
            for i in range(1):
                run_count = 1000
                net = ParticleDecorator(TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(width=2, depth=2)))
                net.with_params(epsilon=0.0001).with_keras_params(optimizer='sgd')
                for run_id in tqdm(range(run_count+1)):
                    net.compiled()
                    loss = net.train(epoch=run_id)
                    if run_id % 100 == 0:
                        run_exp(net)

    if False:
        # this does not work as the aggregation function screws over the fixpoint computation....
        # TODO: check for fixpoint in aggregated space...
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
