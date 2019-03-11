import random

from network import *


def prng():
    return random.random()


class Soup(object):
    
    def __init__(self, size, generator, **kwargs):
        self.size = size
        self.generator = generator
        self.particles = []
        self.historical_particles = {}
        self.params = dict(attacking_rate=0.1, train_other_rate=0.1, train=0)
        self.params.update(kwargs)
        self.time = 0

    def __copy__(self):
        copy_ = Soup(self.size, self.generator, **self.params)
        copy_.__dict__ = {attr: self.__dict__[attr] for attr in self.__dict__ if
                          attr not in ['particles', 'historical_particles']}
        return copy_

    def without_particles(self):
        self_copy = copy.copy(self)
        # self_copy.particles = [particle.states for particle in self.particles]
        self_copy.historical_particles = {key: val.states for key, val in self.historical_particles.items()}
        return self_copy

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
                    description['action'] = 'attacking'
                    description['counterpart'] = other_particle.get_uid()
                if prng() < self.params.get('train_other_rate') and hasattr(self, 'train_other'):
                    other_particle_id = int(prng() * len(self.particles))
                    other_particle = self.particles[other_particle_id]
                    particle.train_other(other_particle)
                    description['action'] = 'train_other'
                    description['counterpart'] = other_particle.get_uid()
                for _ in range(self.params.get('train', 0)):
                    loss = particle.compiled().train()
                    description['fitted'] = self.params.get('train', 0)
                    description['loss'] = loss
                    description['action'] = 'train_self'
                    description['counterpart'] = None
                if self.params.get('remove_divergent') and particle.is_diverged():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['action'] = 'divergent_dead'
                    description['counterpart'] = new_particle.get_uid()
                if self.params.get('remove_zero') and particle.is_zero():
                    new_particle = self.generate_particle()
                    self.particles[particle_id] = new_particle
                    description['action'] = 'zweo_dead'
                    description['counterpart'] = new_particle.get_uid()
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


if __name__ == '__main__':
    if True:
        with SoupExperiment() as exp:
            for run_id in range(1):
                net_generator = lambda: WeightwiseNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
                # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(100, net_generator).with_params(remove_divergent=True, remove_zero=True)
                soup.seed()
                for _ in tqdm(range(1000)):
                    soup.evolve()
                exp.log(soup.count())

    if False:
        with SoupExperiment("soup") as exp:
            for run_id in range(1):
                net_generator = lambda: TrainingNeuralNetworkDecorator(WeightwiseNeuralNetwork(2, 2)).with_keras_params(
                    activation='sigmoid').with_params(epsilon=0.0001)

                # net_generator = lambda: AggregatingNeuralNetwork(4, 2, 2).with_keras_params(activation='sigmoid')\
                # .with_params(shuffler=AggregatingNeuralNetwork.shuffle_random)
                # net_generator = lambda: RecurrentNeuralNetwork(2, 2).with_keras_params(activation='linear').with_params()
                soup = Soup(10, net_generator).with_params(remove_divergent=True, remove_zero=True, train=10)
                soup.seed()
                for _ in tqdm(range(100)):
                    soup.evolve()
                    soup.print_all()
                exp.log(soup.count())
                exp.save(soup=soup.without_particles())  # you can access soup.historical_particles[particle_uid].states[time_step]['loss']
                                     #             or soup.historical_particles[particle_uid].states[time_step]['weights'] from soup.dill
