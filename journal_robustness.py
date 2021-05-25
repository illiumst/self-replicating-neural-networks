import pickle

import pandas as pd
import torch
import random
import copy
import numpy as np

from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

from functionalities_test import is_identity_function, is_zero_fixpoint, test_for_fixpoints, is_divergent
from network import Net
from torch.nn import functional as F
from visualization import plot_loss, bar_chart_fixpoints
import seaborn as sns
from matplotlib import pyplot as plt


def prng():
    return random.random()

def generate_perfekt_synthetic_fixpoint_weights():
    return torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                         [1.0], [0.0], [0.0], [0.0],
                         [1.0], [0.0]
                         ], dtype=torch.float32)


class RobustnessComparisonExperiment:

    @staticmethod
    def apply_noise(network, noise: int):
        """ Changing the weights of a network to values + noise """
        for layer_id, layer_name in enumerate(network.state_dict()):
            for line_id, line_values in enumerate(network.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(network.state_dict()[layer_name][line_id]):
                    #network.state_dict()[layer_name][line_id][weight_id] = weight_value + noise
                    if prng() < 0.5:
                        network.state_dict()[layer_name][line_id][weight_id] = weight_value + noise
                    else:
                        network.state_dict()[layer_name][line_id][weight_id] = weight_value - noise

        return network

    def __init__(self, population_size, log_step_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                 epochs, st_steps, synthetic, directory) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size
        self.net_learning_rate = net_learning_rate
        self.epochs = epochs
        self.ST_steps = st_steps
        self.loss_history = []
        self.synthetic = synthetic
        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.id_functions = []
        self.nets = self.populate_environment()
        self.count_fixpoints()
        self.time_to_vergence, self.time_as_fixpoint = self.test_robustness()

        self.save()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        nets = []
        if self.synthetic:
            ''' Either use perfect / hand-constructed fixpoint ... '''
            net_name = f"net_{str(0)}_synthetic"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
            net.apply_weights(generate_perfekt_synthetic_fixpoint_weights())
            nets.append(net)

        else:
            for i in loop_population_size:
                loop_population_size.set_description("Populating experiment %s" % i)

                ''' .. or use natural approach to train fixpoints from random initialisation. '''
                net_name = f"net_{str(i)}"
                net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
                for _ in range(self.epochs):
                    net.self_train(self.ST_steps, self.log_step_size, self.net_learning_rate)
                nets.append(net)
        return nets

    def test_robustness(self, print_it=True, noise_levels=10, seeds=10):
        assert (len(self.id_functions) == 1 and seeds > 1) or (len(self.id_functions) > 1 and seeds == 1)
        is_synthetic = True if len(self.id_functions) > 1 and seeds == 1 else False
        avg_time_to_vergence = [[0 for _ in range(noise_levels)] for _ in
                                range(seeds if is_synthetic else len(self.id_functions))]
        avg_time_as_fixpoint = [[0 for _ in range(noise_levels)] for _ in
                                range(seeds if is_synthetic else len(self.id_functions))]
        row_headers = []
        data_pos = 0
        # This checks wether to use synthetic setting with multiple seeds
        #   or multi network settings with a singlee seed

        df = pd.DataFrame(columns=['setting', 'noise_level', 'application_step', 'absolute_loss', 'time_to_vergence'])
        for i, fixpoint in enumerate(self.id_functions): #1 / n
            row_headers.append(fixpoint.name)
            for seed in range(seeds): #n / 1
                for noise_level in range(noise_levels):
                    self_application_steps = 0
                    clone = Net(fixpoint.input_size, fixpoint.hidden_size, fixpoint.out_size,
                                f"{fixpoint.name}_clone_noise10e-{noise_level}")
                    clone.load_state_dict(copy.deepcopy(fixpoint.state_dict()))
                    rand_noise = prng() * pow(10, -noise_level) #n / 1
                    clone = self.apply_noise(clone, rand_noise)

                    while not is_zero_fixpoint(clone) and not is_divergent(clone):
                        # -> before
                        clone_weight_pre_application = clone.input_weight_matrix()
                        target_data_pre_application = clone.create_target_weights(clone_weight_pre_application)

                        clone.self_application(1, self.log_step_size)
                        avg_time_to_vergence[i][noise_level] += 1
                        # -> after
                        clone_weight_post_application = clone.input_weight_matrix()
                        target_data_post_application = clone.create_target_weights(clone_weight_post_application)

                        absolute_loss = F.l1_loss(target_data_pre_application, target_data_post_application).item()

                        setting = i if is_synthetic else seed

                        if is_identity_function(clone):
                            avg_time_as_fixpoint[i][noise_level] += 1
                            # When this raises a Type Error, we found a second order fixpoint!
                            self_application_steps += 1
                        else:
                            self_application_steps = pd.NA  # Not a Number!

                        df.loc[df.shape[0]] = [setting, noise_level, self_application_steps,
                                               absolute_loss, avg_time_to_vergence[i][noise_level]]


        # calculate the average:
        # df = df.replace([np.inf, -np.inf], np.nan)
        # df = df.dropna()
        bf = sns.boxplot(data=df, y='self_application_steps', x='noise_level', )
        bf.set_title('Robustness as self application steps per noise level')
        plt.tight_layout()

        # sns.set(rc={'figure.figsize': (10, 50)})
        # bx = sns.catplot(data=df[df['absolute_loss'] < 1], y='absolute_loss', x='application_step', kind='box',
        #                  col='noise_level', col_wrap=3, showfliers=False)
        directory = Path('output') / 'robustness'
        filename = f"absolute_loss_perapplication_boxplot_grid.png"
        filepath = directory / filename

        plt.savefig(str(filepath))

        if print_it:
            col_headers = [str(f"10e-{d}") for d in range(noise_levels)]

            print(f"\nAppplications steps until divergence / zero: ")
            print(tabulate(avg_time_to_vergence, showindex=row_headers, headers=col_headers, tablefmt='orgtbl'))

            print(f"\nTime as fixpoint: ")
            print(tabulate(avg_time_as_fixpoint, showindex=row_headers, headers=col_headers, tablefmt='orgtbl'))

        return avg_time_as_fixpoint, avg_time_to_vergence

    def count_fixpoints(self):
        exp_details = f"ST steps: {self.ST_steps}"
        self.id_functions = test_for_fixpoints(self.fixpoint_counters, self.nets)
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory, self.net_learning_rate,
                            exp_details)

    def visualize_loss(self):
        for i in range(len(self.nets)):
            net_loss_history = self.nets[i].loss_history
            self.loss_history.append(net_loss_history)
        plot_loss(self.loss_history, self.directory)

    def save(self):
        pickle.dump(self, open(f"{self.directory}/experiment_pickle.p", "wb"))
        print(f"\nSaved experiment to {self.directory}.")


if __name__ == "__main__":
    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    ST_steps = 1000
    ST_epochs = 5
    ST_log_step_size = 10
    ST_population_size = 5
    ST_net_hidden_size = 2
    ST_net_learning_rate = 0.04
    ST_name_hash = random.getrandbits(32)
    ST_synthetic = True

    print(f"Running the robustness comparison experiment:")
    RobustnessComparisonExperiment(
        population_size=ST_population_size,
        log_step_size=ST_log_step_size,
        net_input_size=NET_INPUT_SIZE,
        net_hidden_size=ST_net_hidden_size,
        net_out_size=NET_OUT_SIZE,
        net_learning_rate=ST_net_learning_rate,
        epochs=ST_epochs,
        st_steps=ST_steps,
        synthetic=ST_synthetic,
        directory=Path('output') / 'journal_robustness' / f'{ST_name_hash}'
    )
