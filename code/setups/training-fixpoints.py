import sys
import os

# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')

from util import *
from experiment import *
from network import *

import keras.backend as K

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

    with Experiment('training_fixpoint') as exp:
        exp.trials = 20
        exp.run_count = 500
        exp.epsilon = 1e-4
        net_generators = []
        for activation in ['linear']:  # , 'sigmoid', 'relu']:
            for use_bias in [False]:
                net_generators += [lambda activation=activation, use_bias=use_bias: WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]
                # net_generators += [lambda activation=activation, use_bias=use_bias: AggregatingNeuralNetwork(aggregates=4, width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]
                # net_generators += [lambda activation=activation, use_bias=use_bias: RecurrentNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]
        all_counters = []
        all_notable_nets = []
        all_names = []
        for net_generator_id, net_generator in enumerate(net_generators):
            counters = generate_counters()
            notable_nets = []
            for _ in tqdm(range(exp.trials)):
                net = ParticleDecorator(net_generator())
                net = TrainingNeuralNetworkDecorator(net).with_params(epsilon=exp.epsilon)
                name = str(net.net.__class__.__name__) + " activiation='" + str(net.get_keras_params().get('activation')) + "' use_bias=" + str(net.get_keras_params().get('use_bias'))
                for run_id in range(exp.run_count):
                    loss = net.compiled().train(epoch=run_id+1)
                count(counters, net, notable_nets)
            all_counters += [counters]
            all_notable_nets += [notable_nets]
            all_names += [name]
            K.clear_session()
        exp.save(all_counters=all_counters) #net types reached in the end
        # exp.save(all_notable_nets=all_notable_nets)
        exp.save(all_names=all_names) #experiment setups
        for exp_id, counter in enumerate(all_counters):
            exp.log(all_names[exp_id])
            exp.log(all_counters[exp_id])
            exp.log('\n')
