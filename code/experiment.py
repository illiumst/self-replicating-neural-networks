import os
import time
import dill
from tqdm import tqdm
import copy

from tensorflow.python.keras import backend as K

from abc import ABC, abstractmethod


class IllegalArgumentError(ValueError):
    pass


class Experiment(ABC):

    @staticmethod
    def from_dill(path):
        with open(path, "rb") as dill_file:
            return dill.load(dill_file)

    @staticmethod
    def reset_model():
        K.clear_session()

    def __init__(self, name=None, ident=None):
        self.experiment_id = f'{ident or ""}_{time.time()}'
        self.experiment_name = name or 'unnamed_experiment'
        self.next_iteration = 0
        self.log_messages = list()
        self.historical_particles = dict()

    def __enter__(self):
        self.dir = os.path.join('experiments', f'exp-{self.experiment_name}-{self.experiment_id}-{self.next_iteration}')
        os.makedirs(self.dir)
        print(f'** created {self.dir} **')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save(experiment=self.without_particles())
        self.save_log()
        self.next_iteration += 1

    def log(self, message, **kwargs):
        self.log_messages.append(message)
        print(message, **kwargs)

    def save_log(self, log_name="log"):
        with open(os.path.join(self.dir, f"{log_name}.txt"), "w") as log_file:
            for log_message in self.log_messages:
                print(str(log_message), file=log_file)

    def __copy__(self):
        self_copy = self.__class__(name=self.experiment_name,)
        self_copy.__dict__ = {attr: self.__dict__[attr] for attr in self.__dict__ if
                              attr not in ['particles', 'historical_particles']}
        return self_copy

    def without_particles(self):
        self_copy = copy.copy(self)
        # self_copy.particles = [particle.states for particle in self.particles]
        self_copy.historical_particles = {key: val.states for key, val in self.historical_particles.items()}
        return self_copy

    def save(self, **kwargs):
        for name, value in kwargs.items():
            with open(os.path.join(self.dir, f"{name}.dill"), "wb") as dill_file:
                dill.dump(value, dill_file)

    @abstractmethod
    def run_net(self, net, trains_per_application=100, step_limit=100, run_id=0, **kwargs):
        raise NotImplementedError
        pass

    def run_exp(self, network_generator, exp_iterations, step_limit=100, prints=False, reset_model=False):
        # INFO Run_ID needs to be more than 0, so that exp stores the trajectories!
        for run_id in range(exp_iterations):
            network = network_generator()
            self.run_net(network, step_limit, run_id=run_id + 1)
            self.historical_particles[run_id] = network
            if prints:
                print("Fixpoint? " + str(network.is_fixpoint()))
        if reset_model:
            self.reset_model()

    def reset_all(self):
        self.reset_model()


class FixpointExperiment(Experiment):

    def __init__(self, **kwargs):
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        super().__init__(**kwargs)
        self.counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        self.interesting_fixpoints = []

    def run_exp(self, network_generator, exp_iterations, logging=True, **kwargs):
        kwargs.update(reset_model=False)
        super(FixpointExperiment, self).run_exp(network_generator, exp_iterations, **kwargs)
        if logging:
            self.log(self.counters)
        self.reset_model()

    def run_net(self, net, step_limit=100, run_id=0, **kwargs):
        if len(kwargs):
            raise IllegalArgumentError
        for i in range(step_limit):
            if net.is_diverged() or net.is_fixpoint():
                break
            net.self_attack()
            if run_id:
                net.save_state(time=i)
        self.count(net)

    def count(self, net):
        if net.is_diverged():
            self.counters['divergent'] += 1
        elif net.is_fixpoint():
            if net.is_zero():
                self.counters['fix_zero'] += 1
            else:
                self.counters['fix_other'] += 1
                self.interesting_fixpoints.append(net.get_weights())
        elif net.is_fixpoint(2):
            self.counters['fix_sec'] += 1
        else:
            self.counters['other'] += 1

    def reset_counters(self):
        for key in self.counters.keys():
            self.counters[key] = 0
        return True

    def reset_all(self):
        super(FixpointExperiment, self).reset_all()
        self.reset_counters()


class MixedFixpointExperiment(FixpointExperiment):

    def __init__(self, **kwargs):
        super(MixedFixpointExperiment, self).__init__(name=kwargs.get('name', self.__class__.__name__))

    def run_net(self, net, step_limit=100, run_id=0, **kwargs):
        for i in range(step_limit):
            if net.is_diverged() or net.is_fixpoint():
                break
            net.self_attack()
            with tqdm(postfix=["Loss", dict(value=0)]) as bar:
                for _ in range(kwargs.get('trains_per_application', 100)):
                    loss = net.train()
                    bar.postfix[1]["value"] = loss
                    bar.update()
            if run_id:
                net.save_state()
        self.count(net)


class SoupExperiment(Experiment):

    def __init__(self, **kwargs):
        super(SoupExperiment, self).__init__(name=kwargs.get('name', self.__class__.__name__))

    def run_exp(self, network_generator, exp_iterations, soup_generator=None, soup_iterations=0, prints=False):
        for i in range(soup_iterations):
            soup = soup_generator()
            soup.seed()
            for _ in tqdm(exp_iterations):
                soup.evolve()
            self.log(soup.count())
            self.save(soup=soup.without_particles())

    def run_net(self, net, trains_per_application=100, step_limit=100, run_id=0, **kwargs):
        raise NotImplementedError
        pass


class IdentLearningExperiment(Experiment):

    def __init__(self, **kwargs):
        super(IdentLearningExperiment, self).__init__(name=kwargs.get('name', self.__class__.__name__))

    def run_net(self, net, trains_per_application=100, step_limit=100, run_id=0, **kwargs):
        pass
