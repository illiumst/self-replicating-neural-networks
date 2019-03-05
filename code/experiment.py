import os
import time
import dill
from tqdm import tqdm

from collections import defaultdict


class Experiment:
    
    @staticmethod
    def from_dill(path):
        with open(path, "rb") as dill_file:
            return dill.load(dill_file)
    
    def __init__(self, name=None, ident=None):
        self.experiment_id = ident or time.time()
        self.experiment_name = name or 'unnamed_experiment'
        self.base_dir = self.experiment_name
        self.next_iteration = 0
        self.log_messages = []
        self.data_storage = defaultdict(list)
    
    def __enter__(self):
        self.dir = os.path.join(self.base_dir, 'experiments', 'exp-{name}-{id}-{it}'.format(
            name=self.experiment_name, id=self.experiment_id, it=self.next_iteration)
                                )

        os.makedirs(self.dir)
        print("** created {dir} **".format(dir=self.dir))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.save(experiment=self)
        self.save_log()
        self.next_iteration += 1
    
    def log(self, message, **kwargs):
        self.log_messages.append(message)
        print(message, **kwargs)
    
    def save_log(self, log_name="log"):
        with open(os.path.join(self.dir, "{name}.txt".format(name=log_name)), "w") as log_file:
            for log_message in self.log_messages:
                print(str(log_message), file=log_file)
    
    def save(self, **kwargs):
        for name, value in kwargs.items():
            with open(os.path.join(self.dir, "{name}.dill".format(name=name)), "wb") as dill_file:
                dill.dump(value, dill_file)

    def add_trajectory_segment(self, run_id, trajectory):
        self.data_storage[run_id].append(trajectory)
        return


class FixpointExperiment(Experiment):
    
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        self.interesting_fixpoints = []

    def run_net(self, net, step_limit=100, run_id=0):
        i = 0
        while i < step_limit and not net.is_diverged() and not net.is_fixpoint():
            net.self_attack()
            i += 1
            if run_id:
                weights = net.get_weights_flat()
                self.add_trajectory_segment(run_id, weights)
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


class MixedFixpointExperiment(FixpointExperiment):

    def run_net(self, net, trains_per_application=100, step_limit=100, run_id=0):
        # TODO Where to place the trajectory storage ?
        # weights = net.get_weights()
        # self.add_trajectory_segment(run_id, weights)

        i = 0
        while i < step_limit and not net.is_diverged() and not net.is_fixpoint():
            net.self_attack()
            with tqdm(postfix=["Loss", dict(value=0)]) as bar:
                for _ in range(trains_per_application):
                    loss = net.compiled().train()
                    bar.postfix[1]["value"] = loss
                    bar.update()
            i += 1
        self.count(net)
    
            
class SoupExperiment(Experiment):
    pass


class IdentLearningExperiment(Experiment):
    pass
