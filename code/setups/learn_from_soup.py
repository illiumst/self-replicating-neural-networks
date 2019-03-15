import sys
import os

# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')

from typing import Tuple

from util import *
from experiment import *
from network import *
from soup import *


import keras.backend

from statistics import mean
avg = mean


def generate_counters():
    """
    Initial build of the counter dict, to store counts.

    :rtype: dict
    :return: dictionary holding counter for: 'divergent', 'fix_zero', 'fix_sec', 'other'
    """
    return {'divergent': 0, 'fix_zero': 0, 'fix_other': 0, 'fix_sec': 0, 'other': 0}


def count(counters, soup, notable_nets=[]):
    """
    Count the occurences ot the types of weight trajectories.

    :param counters:      A counter dictionary.
    :param soup:          A Soup
    :param notable_nets:  A list to store and save intersting candidates

    :rtype      Tuple[dict, list]
    :return:    Both the counter dictionary and the list of interessting nets.
    """

    for net in soup.particles:
        if net.is_diverged():
            counters['divergent'] += 1
        elif net.is_fixpoint():
            if net.is_zero():
                counters['fix_zero'] += 1
            else:
                counters['fix_other'] += 1
                # notable_nets += [net]
        # elif net.is_fixpoint(2):
        #     counters['fix_sec'] += 1
        #     notable_nets += [net]
        else:
            counters['other'] += 1
    return counters, notable_nets


with SoupExperiment('learn-from-soup') as exp:
    exp.soup_size = 10
    exp.soup_life = 100
    exp.trials = 10
    exp.learn_from_severity_values = [10 * i for i in range(11)]
    exp.epsilon = 1e-4
    net_generators = []
    for activation in ['sigmoid']: #['linear', 'sigmoid', 'relu']:
        for use_bias in [False]:
            # net_generators += [lambda activation=activation, use_bias=use_bias: WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]
            net_generators += [lambda activation=activation, use_bias=use_bias: AggregatingNeuralNetwork(aggregates=4, width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]
            # net_generators += [lambda activation=activation, use_bias=use_bias: RecurrentNeuralNetwork(width=2, depth=2).with_keras_params(activation=activation, use_bias=use_bias)]

    all_names = []
    all_data = []
    for net_generator_id, net_generator in enumerate(net_generators):
        xs = []
        ys = []
        zs = []
        notable_nets = []
        for learn_from_severity in exp.learn_from_severity_values:
            counters = generate_counters()
            results = []
            for _ in tqdm(range(exp.trials)):
                soup = Soup(exp.soup_size, lambda net_generator=net_generator,exp=exp: TrainingNeuralNetworkDecorator(net_generator()).with_params(epsilon=exp.epsilon))
                soup.with_params(attacking_rate=-1, learn_from_rate=0.1, train=0, learn_from_severity=learn_from_severity)
                soup.seed()
                name = str(soup.particles[0].net.__class__.__name__) + " activiation='" + str(soup.particles[0].get_keras_params().get('activation')) + "' use_bias=" + str(soup.particles[0].get_keras_params().get('use_bias'))
                for time in range(exp.soup_life):
                    soup.evolve()
                count(counters, soup, notable_nets)
                keras.backend.clear_session()

            xs += [learn_from_severity]
            ys += [float(counters['fix_zero']) / float(exp.trials)]
            zs += [float(counters['fix_other']) / float(exp.trials)]        
        all_names += [name]
        # xs: learn_from_intensity according to exp.learn_from_intensity_values
        # ys: zero-fixpoints after life time
        # zs: non-zero-fixpoints after life time
        all_data += [{'xs':xs, 'ys':ys, 'zs':zs}]

    exp.save(all_names=all_names)
    exp.save(all_data=all_data)
    exp.save(soup=soup.without_particles())
    for exp_id, name in enumerate(all_names):
        exp.log(all_names[exp_id])
        exp.log(all_data[exp_id])
        exp.log('\n')
