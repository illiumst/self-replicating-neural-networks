import random
import copy

from tqdm import tqdm

from experiment import *
from network import *


def prng():
    return random.random()


class Soup:
    
    def __init__(self, size, generator, **kwargs):
        self.size = size
        self.generator = generator
        self.particles = []
        self.params = dict(meeting_rate=0.1, train_other_rate=0.1, train=0)
        self.params.update(kwargs)
    
    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self
    
    def seed(self):
        self.particles = []
        for _ in range(self.size):
            self.particles += [self.generator()]
        return self       
    
    def evolve(self, iterations=1):
        for _ in range(iterations):
            for particle_id, particle in enumerate(self.particles):
                if prng() < self.params.get('meeting_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.attack(other_particle)
                if prng() < self.params.get('train_other_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.train_other(other_particle)
                try:
                    for _ in range(self.params.get('train', 0)):
                        particle.compiled().train()
                except AttributeError:
                    pass
                if self.params.get('remove_divergent') and particle.is_diverged():
                    self.particles[particle_id] = self.generator()
                if self.params.get('remove_zero') and particle.is_zero():
                    self.particles[particle_id] = self.generator()

    def count(self):
        counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        for particle in self.particles:
            if particle.is_diverged():
                counters['divergent'] += 1
            elif particle.is_fixpoint():
                if particle.is_zero():
                    counters['fix_zero'] += 1
                else:
                    counters['fix_other'] += 1
            elif particle.is_fixpoint(2):
                counters['fix_sec'] += 1
            else:
                counters['other'] += 1
        return counters


class LearningSoup(Soup):

    def __init__(self, *args, **kwargs):
        super(LearningSoup, self).__init__(**kwargs)


if __name__ == '__main__':
    if False:
        with SoupExperiment() as exp:
            for run_id in range(1):
                net_generator = lambda: WeightwiseNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
                # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(100, net_generator).with_params(remove_divergent=True, remove_zero=True)
                soup.seed()
                for _ in tqdm(range(100)):
                    soup.evolve()
                exp.log(soup.count())

    if True:
        with SoupExperiment() as exp:
            for run_id in range(1):
                net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(2, 2)).with_keras_params(
                    activation='linear')
                # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
                # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(10, net_generator).with_params(remove_divergent=True, remove_zero=True).with_params(train=500)
                soup.seed()
                for _ in tqdm(range(10)):
                    soup.evolve()
                exp.log(soup.count())

