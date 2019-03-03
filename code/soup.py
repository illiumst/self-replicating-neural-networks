import random
import copy

from experiment import *
from network import *

def prng():
    return random.random()

class Soup:
    
    def __init__(self, size, generator, **kwargs):
        self.size = size
        self.generator = generator
        self.particles = []
        self.params = dict(meeting_rate=0.1)
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
            for particle_id,particle in enumerate(self.particles):
                if prng() < self.params.get('meeting_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.attack(other_particle)
    
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
        

if __name__ == '__main__':
    with SoupExperiment() as exp:
        for run_id in tqdm(range(1)):
            net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='linear').with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
            soup = Soup(100, net_generator)
            soup.seed()
            soup.evolve(100)
            exp.log(soup.count())