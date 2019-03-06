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
        self.historical_particles = {}
        self.params = dict(attacking_rate=0.1, train_other_rate=0.1, train=0)
        self.params.update(kwargs)
        self.time = 0
    
    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self
    
    def generate_particle(self):
        new_particle = ParticleDecorator(self.generator())
        self.historical_particles[new_particle.get_uid()] = new_particle
        return new_particle
        
    def get_particle(self, uid, otherwise=None):
        return self.historical_particles.get(uid, otherwise)
    
    def seed(self):
        self.particles = []
        for _ in range(self.size):
            self.particles += [self.generate_particle()]
        return self       
    
    def evolve(self, iterations=1):
        for _ in range(iterations):
            self.time += 1
            for particle_id, particle in enumerate(self.particles):
                description = {'time': self.time}
                if prng() < self.params.get('attacking_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.attack(other_particle)
                    description['attacking'] = other_particle.get_uid()
                if prng() < self.params.get('train_other_rate'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.train_other(other_particle)
                    description['training'] = other_particle.get_uid()
                for _ in range(self.params.get('train', 0)):
                    loss = particle.compiled().train()
                    description['fitted'] = self.params.get('train', 0)
                    description['loss'] = loss
                if self.params.get('remove_divergent') and particle.is_diverged():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['died'] = True
                    description['cause'] = 'divergent'
                    description['substitute'] = new_particle.get_uid()
                if self.params.get('remove_zero') and particle.is_zero():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['died'] = True
                    description['cause'] = 'zero'
                    description['substitute'] = new_particle.get_uid()
                particle.save_state(**description)

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
        
    def print_all(self):
        for particle in self.particles:
            particle.print_weights()
            print(particle.is_fixpoint())

class ParticleDecorator:
    
    next_uid = 0
    
    def __init__(self, net):
        self.uid = self.__class__.next_uid
        self.__class__.next_uid += 1
        self.net = net
        self.states = []
        
    def __getattr__(self, name):
        return getattr(self.net, name)
    
    def get_uid(self):
        return self.uid
    
    def make_state(self, **kwargs):
        state = {'class': self.net.__class__.__name__, 'weights': self.net.get_weights()}
        state.update(kwargs)
        return state
    
    def save_state(self, **kwargs):
        state = self.make_state(**kwargs)
        self.states += [state]
        
    def update_state(self, number, **kwargs):
        if number < len(self.states):
            self.states[number] = self.make_state(**kwargs)
        else:
            for i in range(len(self.states), number):
                self.states += [None]
            self.states += self.make_state(**kwargs)
    
    def get_states(self):
        return self.states
        


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
        with SoupExperiment("soup") as exp:
            for run_id in range(1):
                net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(2, 2)).with_keras_params(
                    activation='sigmoid').with_params(epsilon=0.0001)

                # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
                # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(10, net_generator).with_params(remove_divergent=True, remove_zero=True, train=200)
                soup.seed()
                for _ in tqdm(range(10)):
                    soup.evolve()
                    soup.print_all()
                exp.log(soup.count())
                exp.save(soup=soup) # you can access soup.historical_particles[particle_uid].states[time_step]['loss']
                                     #             or soup.historical_particles[particle_uid].states[time_step]['weights'] from soup.dill

