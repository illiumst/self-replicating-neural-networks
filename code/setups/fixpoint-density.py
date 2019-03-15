import sys
import os
# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')

from util import *
from experiment import *
from network import *

import keras.backend

def generate_counters():
    return {'divergent': 0, 'fix_zero': 0, 'fix_other': 0, 'fix_sec': 0, 'other': 0}

def count(counters, net, notable_nets=[]):
    if net.is_diverged():
        counters['divergent'] += 1
    elif net.is_fixpoint():
        if net.is_zero():
            counters['fix_zero'] += 1
        else:
            counters['fix_other'] += 1
            notable_nets += [net]
    elif net.is_fixpoint(2):
        counters['fix_sec'] += 1
        notable_nets += [net]
    else:
        counters['other'] += 1
    return counters, notable_nets


if __name__ == '__main__':
    with Experiment('fixpoint-density') as exp:
        #NOTE: settings could/should stay this way
        #FFT doesn't work though
        exp.trials = 100000
        exp.epsilon = 1e-4
        net_generators = []
        for activation in ['linear']:
            net_generators += [lambda activation=activation: WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=False)]
            net_generators += [lambda activation=activation: AggregatingNeuralNetwork(aggregates=4, width=2, depth=2).with_keras_params(activation=activation, use_bias=False)]
            # net_generators += [lambda activation=activation: FFTNeuralNetwork(aggregates=4, width=2, depth=2).with_keras_params(activation=activation, use_bias=False)]
            # net_generators += [lambda activation=activation: RecurrentNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=False)]
        all_counters = []
        all_notable_nets = []
        all_names = []
        for net_generator_id, net_generator in enumerate(net_generators):
            counters = generate_counters()
            notable_nets = []
            for _ in tqdm(range(exp.trials)):
                net = net_generator().with_params(epsilon=exp.epsilon)
                net = ParticleDecorator(net)
                name = str(net.__class__.__name__) + " activiation='" + str(net.get_keras_params().get('activation')) + "' use_bias='" + str(net.get_keras_params().get('use_bias')) + "'"
                count(counters, net, notable_nets)
                keras.backend.clear_session()
            all_counters += [counters]
            # all_notable_nets += [notable_nets]
            all_names += [name]
        exp.save(all_counters=all_counters)
        exp.save(all_notable_nets=all_notable_nets)
        exp.save(all_names=all_names)
        for exp_id, counter in enumerate(all_counters):
            exp.log(all_names[exp_id])
            exp.log(all_counters[exp_id])
            exp.log('\n')

    print('Done')
