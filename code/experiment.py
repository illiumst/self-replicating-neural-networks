import os
import time
import dill
from tqdm import tqdm
from copy import copy

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

    def __init__(self, name=None, ident=None, **kwargs):
        self.experiment_id = f'{ident or ""}_{time.time()}'
        self.experiment_name = name or 'unnamed_experiment'
        self.iteration = 0
        self.log_messages = list()
        self.historical_particles = dict()
        self.params = dict(exp_iterations=100, application_steps=100, prints=True, trains_per_application=100)
        self.with_params(**kwargs)

    def __copy__(self, *args, **kwargs):
        params = self.params
        params.update(name=self.experiment_name)
        params.update(**kwargs)
        self_copy = self.__class__(*args, **params)
        return self_copy

    def __enter__(self):
        self.dir = os.path.join('experiments', f'exp-{self.experiment_name}-{self.experiment_id}-{self.iteration}')
        os.makedirs(self.dir)
        print(f'** created {self.dir} **')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save(experiment=self.without_particles())
        self.save_log()
        # Clean Exit
        self.reset_all()
        # self.iteration += 1  Taken From here!

    def with_params(self, **kwargs):
        # Make them your own
        self.params.update(kwargs)
        return self

    def log(self, message, **kwargs):
        self.log_messages.append(message)
        print(message, **kwargs)

    def save_log(self, log_name="log"):
        with open(os.path.join(self.dir, f"{log_name}.txt"), "w") as log_file:
            for log_message in self.log_messages:
                print(str(log_message), file=log_file)

    def without_particles(self):
        self_copy = copy(self)
        # Check if attribute exists
        if hasattr(self, 'historical_particles'):
            # Check if it is empty.
            if self.historical_particles:
                # Do the Update
                # self_copy.particles = [particle.states for particle in self.particles]
                self_copy.historical_particles = {key: val.states for key, val in self.historical_particles.items()}
        return self_copy

    def save(self, **kwargs):
        for name, value in kwargs.items():
            with open(os.path.join(self.dir, f"{name}.dill"), "wb") as dill_file:
                dill.dump(value, dill_file)

    def reset_log(self):
        self.log_messages = list()

    @abstractmethod
    def run_net(self, net, **kwargs):
        raise NotImplementedError
        pass

    def run_exp(self, network_generator, reset_model=False, **kwargs):
        # INFO Run_ID needs to be more than 0, so that exp stores the trajectories!
        for run_id in range(self.params.get('exp_iterations')):
            network = network_generator()
            self.run_net(network, **kwargs)
            self.historical_particles[run_id] = network
            if self.params.get('prints'):
                print("Fixpoint? " + str(network.is_fixpoint()))
            self.iteration += 1
            if reset_model:
                self.reset_model()

    def reset_all(self):
        self.reset_log()
        self.reset_model()


class FixpointExperiment(Experiment):

    def __init__(self, **kwargs):
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        super().__init__(**kwargs)
        self.counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        self.interesting_fixpoints = []

    def run_exp(self, network_generator, logging=True, reset_model=False, **kwargs):
        kwargs.update(reset_model=False)
        super(FixpointExperiment, self).run_exp(network_generator, **kwargs)
        if logging:
            self.log(self.counters)
        if reset_model:
            self.reset_model()

    def run_net(self, net, **kwargs):
        if len(kwargs):
            raise IllegalArgumentError
        for i in range(self.params.get('application_steps')):
            if net.is_diverged() or net.is_fixpoint():
                break
            net.set_weights(net.apply_to_weights(net.get_weights()))
            if self.iteration and hasattr(self, 'save_state'):
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
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        super(MixedFixpointExperiment, self).__init__(**kwargs)

    def run_net(self, net, **kwargs):
        assert hasattr(net, 'train'), 'This Network must be trainable, i.e. use the "TrainingNeuralNetworkDecorator"!'

        for application in range(self.params.get('application_steps')):
            epoch_num = self.params.get('trains_per_application') * application
            net.set_weights(net.apply_to_weights(net.get_weights()))
            if net.is_diverged() or net.is_fixpoint():
                break
            barformat = "Experiment Iteration: {postfix[iteration]} | "
            barformat += "Evolution Step:{postfix[step]}| "
            barformat += "Training Epoch:{postfix[epoch]}| "
            barformat += "Loss: {postfix[loss]} | {bar}"
            with tqdm(total=self.params.get('trains_per_application'),
                      postfix={'step': 0, 'loss': 0, 'iteration': self.iteration, 'epoch': 0, None: None},
                      bar_format=barformat) as bar:
                # This iterates for self.trains_per_application times, the addition is just for  epoch enumeration
                for epoch in range(epoch_num, epoch_num + self.params.get('trains_per_application')):
                    if net.is_diverged():
                        print('Network diverged to either inf or nan... breaking')
                        break
                    loss = net.train(epoch=epoch)
                    if epoch % 10 == 0:
                        bar.postfix.update(step=application, epoch=epoch, loss=loss, iteration=self.iteration)
                    bar.update()
                    epoch_num += 1
            if self.iteration and hasattr(net, 'save_sate'):
                net.save_state()
        self.count(net)


class TaskExperiment(MixedFixpointExperiment):

    def __init__(self, **kwargs):
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        super(TaskExperiment, self).__init__(**kwargs)

    def run_exp(self, network_generator, reset_model=False, logging=True, **kwargs):
        kwargs.update(reset_model=False, logging=logging)
        super(FixpointExperiment, self).run_exp(network_generator, **kwargs)
        if reset_model:
            self.reset_model()
        pass

    def run_net(self, net, **kwargs):
        assert hasattr(net, 'evaluate')
        super(TaskExperiment, self).run_net(net, **kwargs)

        # Get Performance without Training
        task_performance = net.evaluate(*net.get_samples(task_samples=True),
                                        batchsize=net.get_amount_of_weights())
        self_performance = net.evaluate(*net.get_samples(self_samples=True),
                                        batchsize=net.get_amount_of_weights())

        current_performance = dict(task_performance=task_performance,
                                   self_performance=self_performance,
                                   counters=self.counters, id=self.iteration)

        self.log(current_performance)
        pass


class SoupExperiment(Experiment):

    def __init__(self, soup_generator, **kwargs):
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        self.soup_generator = soup_generator
        super(SoupExperiment, self).__init__(**kwargs)

    def run_exp(self, network_generator, **kwargs):
        for i in range(self.params.get('exp_iterations')):
            soup = self.soup_generator()
            soup.seed()
            for _ in tqdm(range(self.params.get('application_steps'))):
                soup.evolve()
            self.log(soup.count())
            self.save(soup=soup.without_particles())
            K.clear_session()

    def run_net(self, net, **kwargs):
        raise NotImplementedError
        pass


class TaskingSoupExperiment(Experiment):

    def __init__(self, soup_generator, **kwargs):
        kwargs['name'] = self.__class__.__name__ if 'name' not in kwargs else kwargs['name']
        super(TaskingSoupExperiment, self).__init__(**kwargs)
        self.soup_generator = soup_generator

    def __copy__(self):
        super(TaskingSoupExperiment, self).__copy__(self.soup_generator)

    def run_exp(self, **kwargs):
        for i in range(self.params.get('exp_iterations')):
            soup = self.soup_generator()
            soup.seed()
            for _ in tqdm(range(self.params.get('application_steps'))):
                soup.evolve()
            self.log(soup.count())
            self.save(soup=soup.without_particles())
            K.clear_session()

    def run_net(self, net, **kwargs):
        raise NotImplementedError()
        pass


if __name__ == '__main__':
    pass