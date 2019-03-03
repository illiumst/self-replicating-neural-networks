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
