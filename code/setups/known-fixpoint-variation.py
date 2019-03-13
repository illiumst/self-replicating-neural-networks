import sys

import os

# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')


from util import *
from experiment import *
from network import *
from soup import prng

import keras.backend


from statistics import mean
avg = mean
    
def generate_fixpoint_weights():
    return [
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32)
    ]
    
def generate_fixpoint_net():
    net = WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation='sigmoid')
    net.set_weights(generate_fixpoint_weights())
    return net

def vary(old_weights, e=1.0):
    new_weights = copy.deepcopy(old_weights)
    for layer_id, layer in enumerate(new_weights):
        for cell_id, cell in enumerate(layer):
            for weight_id, weight in enumerate(cell):
                if prng() < 0.5:
                    new_weights[layer_id][cell_id][weight_id] = weight + prng() * e
                else:
                    new_weights[layer_id][cell_id][weight_id] = weight - prng() * e
    return new_weights

with Experiment('known-fixpoint-variation') as exp:
    exp.depth = 10
    exp.trials = 100
    exp.max_steps = 100
    exp.epsilon = 1e-4
    exp.xs = []
    exp.ys = []
    exp.zs = []
    exp.notable_nets = []
    current_scale = 1.0
    for _ in range(exp.depth):
        print('variation scale ' + str(current_scale))
        for _ in tqdm(range(exp.trials)):
            net = generate_fixpoint_net().with_params(epsilon=exp.epsilon)
            net.set_weights(vary(net.get_weights(), current_scale))
            time_to_something = 0
            time_as_fixpoint = 0
            still_fixpoint = True
            for _ in range(exp.max_steps):
                net.self_attack()
                if net.is_zero() or net.is_diverged():
                    break
                if net.is_fixpoint():
                    if still_fixpoint:
                        time_as_fixpoint += 1
                    else:
                        print('remarkable')
                        exp.notable_nets += [net.get_weights()]
                        still_fixpoint = True
                else:
                    still_fixpoint = False
                time_to_something += 1
            exp.xs += [current_scale]
            exp.ys += [time_to_something] #time steps taken to reach divergence or zero (reaching another fix-point is basically never happening)
            exp.zs += [time_as_fixpoint] #time steps still regarded as sthe initial fix-point
            keras.backend.clear_session()
        current_scale /= 10.0
    for d in range(exp.depth):
        exp.log('variation 10e-' + str(d))
        exp.log('avg time to vergence ' + str(avg(exp.ys[d*exp.trials:(d+1)*exp.trials])))
        exp.log('avg time as fixpoint ' + str(avg(exp.zs[d*exp.trials:(d+1)*exp.trials])))
        
