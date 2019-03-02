import math
import copy

import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, Dense
from keras.layers import Input, TimeDistributed
from tqdm import tqdm

from experiment import Experiment


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
    
    def __init__(self, width, depth, **keras_params):
        self.width = width
        self.depth = depth
        self.params = dict(epsilon=0.00000000000001)
        self.keras_params = dict(activation='linear', use_bias=False)
        self.keras_params.update(keras_params)
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
        
    def set_keras_params(self, **kwargs):
        self.keras_param.update(kwargs)
    
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

    def is_fixpoint(self, epsilon=None):
        epsilon = epsilon or self.params.get('epsilon')
        old_weights = self.get_weights()
        new_weights = self.apply_to_network(self)
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
        s = ""
        for layer_id,layer in enumerate(self.get_weights()):
            for cell_id,cell in enumerate(layer):
                s += "[ "
                for weight_id,weight in enumerate(cell):
                    s += str(weight) + " "
                s += "]"
            s += "\n"
        return s
    
    def print_weights(self):
        print(self.repr_weights())



class WeightwiseNeuralNetwork(NeuralNetwork):
    
    def __init__(self, width, depth, **keras_params):
        super().__init__(width, depth, **keras_params)
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
                    if self.params.get("print_all_weight_updates", False):
                        print("updated old weight " + str(weight) + "\t @ (" + str(layer_id) + "," + str(cell_id) + "," + str(weight_id) + ") to new value " + str(new_weight) + "\t calling @ (" + str(normal_layer_id) + "," + str(normal_cell_id) + "," + str(normal_weight_id) + ")")
        return new_weights

   


if __name__ == '__main__':
    with Experiment() as exp:
        counts = dict(divergent=0, fix_zero=0, fix_other=0, other=0)
        for run_id in tqdm(range(10)):
            activation = 'linear'
            net = WeightwiseNeuralNetwork(2, 2, activation='linear')
            # net.set_params(print_all_weight_updates=True)
            # net.model.summary()
            # net.print_weights()
            # print()
            # print(net.apply(1, 1, 1))
            i = 0
            while i < 100 and not net.is_diverged() and not net.is_fixpoint():
                net.self_attack()
                # net.print_weights()
                # print()
                i += 1
            if net.is_diverged():
                counts['divergent'] += 1
            elif net.is_fixpoint():
                if net.is_zero():
                    counts['fix_zero'] += 1
                else:
                    counts['fix_other'] += 1
                    exp.log(net.repr_weights())
                    net.self_attack()
                    exp.log(net.repr_weights())
            else:
                counts['other'] += 1
        exp.log(counts) 
