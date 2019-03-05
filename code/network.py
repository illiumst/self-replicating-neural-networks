import math
import copy
import os
import numpy as np

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

from experiment import *

# Supress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def normalize_id(value, norm):
    if norm > 1:
        return float(value) / float(norm)
    else:
        return float(value)   


def are_weights_diverged(network_weights):
    for layer_id, layer in enumerate(network_weights):
        for cell_id, cell in enumerate(layer):
            for weight_id, weight in enumerate(cell):
                if math.isnan(weight):
                    return True
                if math.isinf(weight):
                    return True
    return False


def are_weights_within(network_weights, lower_bound, upper_bound):
    for layer_id, layer in enumerate(network_weights):
        for cell_id, cell in enumerate(layer):
            for weight_id, weight in enumerate(cell):
                if not (lower_bound <= weight <= upper_bound):
                    return False
    return True

class PrintingObject():

    class SilenceSignal():
        def __init__(self, obj, value):
            self.obj = obj
            self.new_silent = value
        def __enter__(self):
            self.old_silent = self.obj.get_silence()
            self.obj.set_silence(self.new_silent)
        def __exit__(self, exception_type, exception_value, traceback):
            self.obj.set_silence(self.old_silent)
    
    def __init__(self):
        self.silent = True
    
    def is_silent(self):
        return self.silent
    
    def get_silence(self):
        return self.is_silent()
    
    def set_silence(self, value=True):
        self.silent = value
        return self
    
    def unset_silence(self):
        self.silent = False
        return self
        
    def with_silence(self, value=True):
        self.set_silence(value)
        return self
        
    def silence(self, value=True):
        return self.__class__.SilenceSignal(self, value)
        
    def _print(self, *args, **kwargs):
        if not self.silent:
            print(*args, **kwargs)
        

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
    
    def __init__(self, **params):
        super().__init__()
        self.model = Sequential()
        self.params = dict(epsilon=0.00000000000001)
        self.params.update(params)
        self.keras_params = dict(activation='linear', use_bias=False)

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
        
    def get_model(self):
        return self.model
    
    def get_weights(self):
        return self.get_model().get_weights()
        
    def set_weights(self, new_weights):
        return self.get_model().set_weights(new_weights)

    def apply_to_weights(self, old_weights):
        # placeholder, overwrite in subclass
        return old_weights

    def apply_to_network(self, other_network):
        new_weights = self.apply_to_weights(other_network.get_weights())
        return new_weights
        
    def attack(self, other_network):
        other_network.set_weights(self.apply_to_network(other_network))
        return self
        
    def self_attack(self, iterations=1):
        for _ in range(iterations):
            self.attack(self)
        return self
    
    def meet(self, other_network):
        new_other_network = copy.deepcopy(other_network)
        return self.attack(new_other_network)
        
    def self_meet(self, iterations=1):
        new_me = copy.deepcopy(self)
        return new_me.self_attack(iterations)
    
    def is_diverged(self):
        return are_weights_diverged(self.get_weights())
        
    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.params.get('epsilon')
        return are_weights_within(self.get_weights(), -epsilon, epsilon)

    def is_fixpoint(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')
        old_weights = self.get_weights()
        new_weights = copy.deepcopy(old_weights)
        
        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
                
        if are_weights_diverged(new_weights):
            return False
        for layer_id, layer in enumerate(old_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights[layer_id][cell_id][weight_id]
                    if abs(new_weight - weight) >= epsilon:
                        return False
        return True
    
    def repr_weights(self):
        return self.__class__.weights_to_string(self.get_weights())
    
    def print_weights(self):
        print(self.repr_weights())


class WeightwiseNeuralNetwork(NeuralNetwork):
    
    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.model.add(Dense(units=self.width, input_dim=4, **self.keras_params))
        for _ in range(self.depth-1):
            self.model.add(Dense(units=self.width, **self.keras_params))
        self.model.add(Dense(units=1, **self.keras_params))
    
    def apply(self, *inputs):
        stuff = np.transpose(np.array([[inputs[0]], [inputs[1]], [inputs[2]], [inputs[3]]]))
        return self.model.predict(stuff)[0][0]
        
    def apply_to_weights(self, old_weights):
        new_weights = copy.deepcopy(old_weights)
        max_layer_id = len(old_weights) - 1

        for layer_id, layer in enumerate(old_weights):
            max_cell_id = len(layer) - 1

            for cell_id, cell in enumerate(layer):
                max_weight_id = len(cell) - 1

                for weight_id, weight in enumerate(cell):
                    normal_layer_id = normalize_id(layer_id, max_layer_id)
                    normal_cell_id = normalize_id(cell_id, max_cell_id)
                    normal_weight_id = normalize_id(weight_id, max_weight_id)

                    new_weight = self.apply(weight, normal_layer_id, normal_cell_id, normal_weight_id)
                    new_weights[layer_id][cell_id][weight_id] = new_weight

                    if self.params.get("print_all_weight_updates", False) and not self.is_silent():
                        print("updated old weight {weight}\t @ ({layer},{cell},{weight_id}) "
                              "to new value {new_weight}\t calling @ ({normal_layer},{normal_cell},{normal_weight_id})").format(
                            weight=weight, layer=layer_id, cell=cell_id, weight_id=weight_id, new_weight=new_weight,
                            normal_layer=normal_layer_id, normal_cell=normal_cell_id, normal_weight_id=normal_weight_id)
        return new_weights
        
    def compute_samples(self):
        samples = []
        new_weights = copy.deepcopy(self.get_weights())
        max_layer_id = len(self.get_weights()) - 1
        for layer_id, layer in enumerate(self.get_weights()):
            max_cell_id = len(layer) - 1
            for cell_id, cell in enumerate(layer):
                max_weight_id = len(cell) - 1
                for weight_id, weight in enumerate(cell):
                    normal_layer_id = normalize_id(layer_id, max_layer_id)
                    normal_cell_id = normalize_id(cell_id, max_cell_id)
                    normal_weight_id = normalize_id(weight_id, max_weight_id)
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
        self.model.add(Dense(units=width, input_dim=self.aggregates, **self.keras_params))
        for _ in range(depth-1):
            self.model.add(Dense(units=width, **self.keras_params))
        self.model.add(Dense(units=self.aggregates, **self.keras_params))
    
    def get_aggregator(self):
        return self.params.get('aggregator', self.__class__.aggregate_average)
        
    def get_deaggregator(self):
        return self.params.get('deaggregator', self.__class__.deaggregate_identically)
        
    def get_shuffler(self):
        return self.params.get('shuffler', self.__class__.shuffle_not)
    
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
        collections = []
        next_collection = []
        current_weight_id = 0
        for layer_id, layer in enumerate(old_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    next_collection += [weight]
                    if (current_weight_id + 1) % collection_size == 0:
                        collections += [next_collection]
                        next_collection = []
                    current_weight_id += 1
        collections[-1] += next_collection
        leftovers = len(next_collection)
        
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
        new_weights = copy.deepcopy(old_weights)
        current_weight_id = 0
        for layer_id, layer in enumerate(new_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights_list[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
                    
        # return results
        if self.params.get("print_all_weight_updates", False) and not self.is_silent():
            print("updated old weight aggregations " + str(old_aggregations))
            print("to new weight aggregations      " + str(new_aggregations))
            print("resulting in network weights ...")
            print(self.__class__.weights_to_string(new_weights))
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
        return self.__class__.collect_weights(self.get_weights(), collection_size)
        
    def get_aggregated_weights(self):
        collections, leftovers = self.get_collected_weights()
        aggregations = [self.get_aggregator()(collection) for collection in collections]
        return aggregations, leftovers
    
    def compute_samples(self):
        aggregations, _ = self.get_aggregated_weights()
        sample = np.transpose(np.array([[aggregations[i]] for i in range(self.aggregates)]))
        return [sample], [sample]
        
    def is_fixpoint(self, degree=1, epsilon=None):
        assert degree >= 1, "degree must be >= 1"
        epsilon = epsilon or self.get_params().get('epsilon')
        old_weights = self.get_weights()
        new_weights = copy.deepcopy(old_weights)
        
        for _ in range(degree):
            new_weights = self.apply_to_weights(new_weights)
                
        if are_weights_diverged(new_weights):
            return False
        for layer_id, layer in enumerate(old_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    new_weight = new_weights[layer_id][cell_id][weight_id]
                    if abs(new_weight - weight) >= epsilon:
                        return False
        return True


class RecurrentNeuralNetwork(NeuralNetwork):
    
    def __init__(self, width, depth, **kwargs):
        super().__init__(**kwargs)
        self.features = 1
        self.width = width
        self.depth = depth
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
        sample = np.transpose(np.array([[[old_weights_list[i]] for i in range(len(old_weights_list))]]))
        return sample, sample
        


class LearningNeuralNetwork(NeuralNetwork):

    @staticmethod
    def mean_reduction(weights, features):
        single_dim_weights = np.hstack([w.flatten() for w in weights])
        shaped_weights = np.reshape(single_dim_weights, (1, features, -1))
        x = np.mean(shaped_weights, axis=-1)
        return x

    @staticmethod
    def fft_reduction(weights, features):
        single_dim_weights = np.hstack([w.flatten() for w in weights])
        x = np.fft.fft(single_dim_weights, n=features)[None, ...]
        return x

    @staticmethod
    def random_reduction(_, features):
        x = np.random.rand(features)[None, ...]
        return x

    def __init__(self, width, depth, features, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.features = features
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.model.add(Dense(units=self.width, input_dim=self.features, **self.keras_params))
        for _ in range(self.depth-1):
            self.model.add(Dense(units=self.width, **self.keras_params))
        self.model.add(Dense(units=self.features, **self.keras_params))
        self.model.compile(**self.compile_params)

    def apply_to_weights(self, old_weights):
        raise NotImplementedException

    def with_compile_params(self, **kwargs):
        self.compile_params.update(kwargs)
        return self

    def learn(self, epochs, reduction, batchsize=1):
        with tqdm(total=epochs, ascii=True,
                  desc='Type: {t} @ Epoch:'.format(t=self.__class__.__name__),
                  postfix=["Loss", dict(value=0)]) as bar:
            for epoch in range(epochs):
                old_weights = self.get_weights()
                x = reduction(old_weights, self.features)
                history = self.model.fit(x=x, y=x, verbose=0, batch_size=batchsize)
                bar.postfix[1]["value"] = history.history['loss'][-1]
                bar.update()

class TrainingNeuralNetworkDecorator(NeuralNetwork):
    
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.model = None
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.model_compiled = False

    def get_params(self):
        return self.net.get_params()
    
    def get_keras_params(self):
        return self.net.get_keras_params()
        
    def get_compile_params(self):
        return self.net.get_compile_params()
    
    def with_params(self, **kwargs):
        self.net.with_params(**kwargs)
        return self
    
    def with_keras_params(self, **kwargs):
        self.net.with_keras_params(**kwargs)
        return self
    
    def with_compile_params(self, **kwargs):
        self.compile_params.update(kwargs)
        return self
    
    def get_model(self):
        return self.net.get_model()
        
    def apply_to_weights(self, old_weights):
        return self.net.apply_to_weights(old_weights)
        
    def compile_model(self, **kwargs):
        compile_params = copy.deepcopy(self.compile_params)
        compile_params.update(kwargs)
        return self.get_model().compile(**compile_params)
        
    def compiled(self, **kwargs):
        if not self.model_compiled:
            self.compile_model(**kwargs)
            self.model_compiled = True
        return self
        
    def train(self, batchsize=1):
        self.compiled()
        x, y = self.net.compute_samples()
        history = self.net.model.fit(x=x, y=y, verbose=0, batch_size=batchsize)
        return history.history['loss'][-1]
        


if __name__ == '__main__':
    if False:
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                # net = WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation='linear')
                net = AggregatingNeuralNetwork(aggregates=4, width=2, depth=2).with_keras_params(activation='linear')\
                    .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random,
                                 print_all_weight_updates=False, use_bias=True)
                # net = RecurrentNeuralNetwork(width=2, depth=2).with_keras_params(activation='linear')\
                # .with_params(print_all_weight_updates=True)

                # net.print_weights()
                exp.run_net(net, 100)
            exp.log(exp.counters)

    if False: # is_fixpoint was wrong because it trivially returned the old weights
        with IdentLearningExperiment() as exp:
            net = LearningNeuralNetwork(width=2, depth=2, features=2, )\
                .with_keras_params(activation='sigmoid', use_bias=False, ) \
                .with_params(print_all_weight_updates=False)
            net.print_weights()
            time.sleep(1)
            print(net.is_fixpoint(epsilon=0.1e-6))
            print()
            net.learn(1, reduction=LearningNeuralNetwork.fft_reduction)
            import time
            time.sleep(1)
            net.print_weights()
            time.sleep(1)
            print(net.is_fixpoint(epsilon=0.1e-6))
    if False: # ok so this works quite realiably
        with FixpointExperiment() as exp:
            run_count = 1000
            net = TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(width=2, depth=2)).with_params(epsilon=0.1e-6)
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 100 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint(epsilon=0.0001)))
                    print("Loss " + str(loss))
                    print()
    if False: # this does not work as the aggregation function screws over the fixpoint computation.... TODO: check for fixpoint in aggregated space... 
        with FixpointExperiment() as exp:
            run_count = 1000
            net = TrainingNeuralNetworkDecorator(AggregatingNeuralNetwork(4, width=2, depth=2)).with_params(epsilon=0.1e-6)
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 100 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint(epsilon=0.0001)))
                    print("Loss " + str(loss))
                    print()
    if False: # this explodes in our faces completely... NAN everywhere TODO: Wtf is happening here?
        with FixpointExperiment() as exp:
            run_count = 10
            net = TrainingNeuralNetworkDecorator(RecurrentNeuralNetwork(width=2, depth=2)).with_params(epsilon=0.1e-6)
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 1 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint(epsilon=0.0001)))
                    print("Loss " + str(loss))
                    print()
    if True: # and this gets somewhat interesting... we can still achieve non-trivial fixpoints over multiple applications when training enough in-between
        with MixedFixpointExperiment() as exp:
            for run_id in range(1):
                net = TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(width=2, depth=2)).with_params(epsilon=0.0001)
                exp.run_net(net, 500, 10)
                net.print_weights()
                print("Fixpoint? " + str(net.is_fixpoint()))
                print()
            exp.log(exp.counters)
