import math
import copy

import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, Dense
from keras.layers import Input, TimeDistributed
from tqdm import tqdm

from experiment import FixpointExperiment


def normalize_id(value, norm):
    if norm > 1:
        return float(value) / float(norm)
    else:
        return float(value)   

def are_weights_diverged(network_weights):
    for layer_id,layer in enumerate(network_weights):
        for cell_id,cell in enumerate(layer):
            for weight_id,weight in enumerate(cell):
                if math.isnan(weight):
                    return True
                if math.isinf(weight):
                    return True
    return False

def are_weights_within(network_weights, lower_bound, upper_bound):
    for layer_id,layer in enumerate(network_weights):
        for cell_id,cell in enumerate(layer):
            for weight_id,weight in enumerate(cell):
                if not (lower_bound <= weight and weight <= upper_bound):
                    return False
    return True



class NeuralNetwork:

    @staticmethod 
    def weights_to_string(weights):
        s = ""
        for layer_id,layer in enumerate(weights):
            for cell_id,cell in enumerate(layer):
                s += "[ "
                for weight_id,weight in enumerate(cell):
                    s += str(weight) + " "
                s += "]"
            s += "\n"
        return s
    
    def __init__(self, **params):
        self.params = dict(epsilon=0.00000000000001)
        self.params.update(params)
        self.keras_params = dict(activation='linear', use_bias=False)
        self.silent = True
    
    def silence(self):
        self.silent = True
        return self
    
    def unsilence(self):
        self.silent = False
        return self
    
    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self
        
    def with_keras_params(self, **kwargs):
        self.keras_params.update(kwargs)
        return self
    
    def get_weights(self):
        return self.model.get_weights()
        
    def set_weights(self, new_weights):
        return self.model.set_weights(new_weights)

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
    
    def is_diverged(self):
        return are_weights_diverged(self.get_weights())
        
    def is_zero(self, epsilon=None):
        epsilon = epsilon or self.params.get('epsilon')
        return are_weights_within(self.get_weights(), -epsilon, epsilon)

    def is_fixpoint(self, degree=1, epsilon=None):
        epsilon = epsilon or self.params.get('epsilon')
        old_weights = self.get_weights()
        self.silence()
        for _ in range(degree):
            new_weights = self.apply_to_network(self)
        self.unsilence()
        if are_weights_diverged(new_weights):
            return False
        for layer_id,layer in enumerate(old_weights):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
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
        self.model = Sequential()
        self.model.add(Dense(units=width, input_dim=4, **self.keras_params))
        for _ in range(depth-1):
            self.model.add(Dense(units=width, **self.keras_params))
        self.model.add(Dense(units=1, **self.keras_params))
    
    def apply(self, *input):
        stuff = np.transpose(np.array([[input[0]], [input[1]], [input[2]], [input[3]]]))
        return self.model.predict(stuff)[0][0]
        
    def apply_to_weights(self, old_weights):
        new_weights = copy.deepcopy(old_weights)
        max_layer_id = len(old_weights) - 1
        for layer_id,layer in enumerate(old_weights):
            max_cell_id = len(layer) - 1
            for cell_id,cell in enumerate(layer):
                max_weight_id = len(cell) - 1
                for weight_id,weight in enumerate(cell):
                    normal_layer_id = normalize_id(layer_id, max_layer_id)
                    normal_cell_id = normalize_id(cell_id, max_cell_id)
                    normal_weight_id = normalize_id(weight_id, max_weight_id)
                    new_weight = self.apply(weight, normal_layer_id, normal_cell_id, normal_weight_id)
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    if self.params.get("print_all_weight_updates", False) and not self.silent:
                        print("updated old weight " + str(weight) + "\t @ (" + str(layer_id) + "," + str(cell_id) + "," + str(weight_id) + ") to new value " + str(new_weight) + "\t calling @ (" + str(normal_layer_id) + "," + str(normal_cell_id) + "," + str(normal_weight_id) + ")")
        return new_weights


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
        return self.params.get('aggregator', self.__class__.aggregate_average)
        
    def get_deaggregator(self):
        return self.params.get('deaggregator', self.__class__.deaggregate_identically)
        
    def get_shuffler(self):
        return self.params.get('shuffler', self.__class__.shuffle_not)
    
    def get_amount_of_weights(self):
        total_weights = 0
        for layer_id,layer in enumerate(self.get_weights()):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
                    total_weights += 1
        return total_weights
    
    def apply(self, *input):
        stuff = np.transpose(np.array([[input[i]] for i in range(self.aggregates)]))
        return self.model.predict(stuff)[0]
        
    def apply_to_weights(self, old_weights):
        # build aggregations from old_weights
        collection_size = self.get_amount_of_weights() // self.aggregates
        collections = []
        next_collection = []
        current_weight_id = 0
        for layer_id,layer in enumerate(old_weights):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
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
        for aggregation_id,aggregation in enumerate(new_aggregations):
            if aggregation_id == self.aggregates - 1:
                new_weights_list += self.get_deaggregator()(aggregation, collection_size + leftovers)
            else:
                new_weights_list += self.get_deaggregator()(aggregation, collection_size)
        new_weights_list = self.get_shuffler()(new_weights_list)
        # write back new weights
        new_weights = copy.deepcopy(old_weights)
        current_weight_id = 0
        for layer_id,layer in enumerate(new_weights):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
                    new_weight = new_weights_list[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
        # return results
        if self.params.get("print_all_weight_updates", False) and not self.silent:
            print("updated old weight aggregations " + str(old_aggregations))
            print("to new weight aggregations      " + str(new_aggregations))
            print("resulting in network weights ...")
            print(self.__class__.weights_to_string(new_weights))
        return new_weights                



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
    
    def apply(self, *input):
        stuff = np.transpose(np.array([[[input[i]] for i in range(len(input))]]))
        return self.model.predict(stuff)[0].flatten()
        
    def apply_to_weights(self, old_weights):
        # build list from old weights
        new_weights = copy.deepcopy(old_weights)
        old_weights_list = []
        for layer_id,layer in enumerate(old_weights):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
                    old_weights_list += [weight]
        # call network
        new_weights_list = self.apply(*old_weights_list)
        # write back new weights from list of rnn returns
        current_weight_id = 0
        for layer_id,layer in enumerate(new_weights):
            for cell_id,cell in enumerate(layer):
                for weight_id,weight in enumerate(cell):
                    new_weight = new_weights_list[current_weight_id]
                    new_weights[layer_id][cell_id][weight_id] = new_weight
                    current_weight_id += 1
        return new_weights
                    

if __name__ == '__main__':
    if True:
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(100)):
                # net = WeightwiseNeuralNetwork(2, 2).with_keras_params(activation='linear')
                net = AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='linear').with_params(shuffler=AggregatingNeuralNetwork.shuffle_random, print_all_weight_updates=False)
                # net = RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params(print_all_weight_updates=True)
                # net.print_weights()
                exp.run_net(net, 100)
            exp.log(exp.counters)
            
