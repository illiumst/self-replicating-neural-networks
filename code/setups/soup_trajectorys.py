import sys
import os

# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')

from soup import *
from experiment import *


if __name__ == '__main__':
    if True:
        with SoupExperiment("soup") as exp:
            for run_id in range(1):
                net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(2, 2)) \
                     .with_keras_params(activation='linear').with_params(epsilon=0.0001)
                # net_generator = lambda: TrainingNeuralNetworkDecorator(AggregatingNeuralNetwork(4, 2, 2))\
                #     .with_keras_params(activation='linear')
                # net_generator = lambda: TrainingNeuralNetworkDecorator(FFTNeuralNetwork(4, 2, 2))\
                #    .with_keras_params(activation='linear')
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(20, net_generator).with_params(remove_divergent=True, remove_zero=True,
                                                           train=30,
                                                           learn_from_rate=-1)
                soup.seed()
                for _ in tqdm(range(100)):
                    soup.evolve()
                exp.log(soup.count())
                # you can access soup.historical_particles[particle_uid].states[time_step]['loss']
                #             or soup.historical_particles[particle_uid].states[time_step]['weights']
                # from soup.dill
                exp.save(soup=soup.without_particles())
