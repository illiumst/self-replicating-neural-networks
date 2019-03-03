import sys
import os
import time
import copy
import dill

class Experiment:
    
    @staticmethod
    def from_dill(path):
        with open(path, "rb") as dill_file:
            return dill.load(dill_file)
    
    def __init__(self, name=None, id=None):
        self.experiment_id = id or time.time()
        this_file = os.path.realpath(os.getcwd() + "/" + sys.argv[0])
        self.experiment_name = name or os.path.basename(this_file)
        self.base_dir = os.path.realpath((os.path.dirname(this_file) + "/..")) + "/"
        self.next_iteration = 0
        self.log_messages = []
        self.initialize_more()
        
    def initialize_more(self):
        pass
    
    def __enter__(self):
        self.dir = self.base_dir + "experiments/exp-" + str(self.experiment_name) + "-" + str(self.experiment_id) + "-" + str(self.next_iteration) + "/"
        os.mkdir(self.dir)
        print("** created " + str(self.dir))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.save(experiment=self)
        self.save_log()
        self.next_iteration += 1
    
    def log(self, message, **kwargs):
        self.log_messages.append(message)
        print(message, **kwargs)
    
    def save_log(self, log_name="log"):
        with open(self.dir + "/" + str(log_name) + ".txt", "w") as log_file:
            for log_message in self.log_messages:
                print(str(log_message), file=log_file)
    
    def save(self, **kwargs):
        for name,value in kwargs.items():
            with open(self.dir + "/" + str(name) + ".dill", "wb") as dill_file:
                dill.dump(value, dill_file)


class FixpointExperiment(Experiment):
    
    def initialize_more(self):
        self.counters = dict(divergent=0, fix_zero=0, fix_other=0, fix_sec=0, other=0)
        self.interesting_fixpoints = []
    def run_net(self, net, step_limit=100):
        i = 0
        while i < step_limit and not net.is_diverged() and not net.is_fixpoint():
            net.self_attack()
            i += 1
        self.count(net)
    def count(self, net):
        if net.is_diverged():
            self.counters['divergent'] += 1
        elif net.is_fixpoint():
            if net.is_zero():
                self.counters['fix_zero'] += 1
            else:
                self.counters['fix_other'] += 1
                self.interesting_fixpoints.append(net)
                self.log(net.repr_weights())
                net.self_attack()
                self.log(net.repr_weights())
        elif net.is_fixpoint(2):
            self.counters['fix_sec'] += 1
        else:
            self.counters['other'] += 1
            
            
class SoupExperiment(Experiment):
    pass
