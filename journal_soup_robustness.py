import copy
import random

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tabulate import tabulate

from functionalities_test import test_for_fixpoints, is_zero_fixpoint, is_divergent, is_identity_function
from network import Net
from visualization import plot_loss, bar_chart_fixpoints, plot_3d_soup, line_chart_fixpoints


def prng():
    return random.random()


class SoupRobustnessExperiment:

    def __init__(self, population_size, net_i_size, net_h_size, net_o_size, learning_rate, attack_chance,
                 train_nets, ST_steps, epochs, log_step_size, directory: Union[str, Path]):
        super().__init__()
        self.population_size = population_size

        self.net_input_size = net_i_size
        self.net_hidden_size = net_h_size
        self.net_out_size = net_o_size
        self.net_learning_rate = learning_rate
        self.attack_chance = attack_chance
        self.train_nets = train_nets
        # self.SA_steps = SA_steps
        self.ST_steps = ST_steps
        self.epochs = epochs
        self.log_step_size = log_step_size

        self.loss_history = []

        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }
        # <self.fixpoint_counters_history> is used for keeping track of the amount of fixpoints in %
        self.fixpoint_counters_history = []
        self.id_functions = []

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.population = []
        self.populate_environment()

        self.evolve()
        self.fixpoint_percentage()
        self.weights_evolution_3d_experiment()
        self.count_fixpoints()
        self.visualize_loss()

        self.time_to_vergence, self.time_as_fixpoint = self.test_robustness()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in tqdm(range(self.population_size)):
            loop_population_size.set_description("Populating soup experiment %s" % i)

            net_name = f"soup_network_{i}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
            self.population.append(net)

    def evolve(self):
        """ Evolving consists of attacking & self-training. """

        loop_epochs = tqdm(range(self.epochs))
        for i in loop_epochs:
            loop_epochs.set_description("Evolving soup %s" % i)

            # A network attacking another network with a given percentage
            if random.randint(1, 100) <= self.attack_chance:
                random_net1, random_net2 = random.sample(range(self.population_size), 2)
                random_net1 = self.population[random_net1]
                random_net2 = self.population[random_net2]
                print(f"\n Attack: {random_net1.name} -> {random_net2.name}")
                random_net1.attack(random_net2)

            #  Self-training each network in the population
            for j in range(self.population_size):
                net = self.population[j]

                for _ in range(self.ST_steps):
                    net.self_train(1, self.log_step_size, self.net_learning_rate)

            # Testing for fixpoints after each batch of ST steps to see relevant data
            if i % self.ST_steps == 0:
                test_for_fixpoints(self.fixpoint_counters, self.population)
                fixpoints_percentage = round(self.fixpoint_counters["identity_func"] / self.population_size, 1)
                self.fixpoint_counters_history.append(fixpoints_percentage)

            # Resetting the fixpoint counter. Last iteration not to be reset -
            #  it is important for the bar_chart_fixpoints().
            if i < self.epochs:
                self.reset_fixpoint_counters()

    def test_robustness(self, print_it=True, noise_levels=10, seeds=10):
        # assert (len(self.id_functions) == 1 and seeds > 1) or (len(self.id_functions) > 1 and seeds == 1)
        is_synthetic = True if len(self.id_functions) > 1 and seeds == 1 else False
        avg_time_to_vergence = [[0 for _ in range(noise_levels)] for _ in
                                range(seeds if is_synthetic else len(self.id_functions))]
        avg_time_as_fixpoint = [[0 for _ in range(noise_levels)] for _ in
                                range(seeds if is_synthetic else len(self.id_functions))]
        row_headers = []
        data_pos = 0
        # This checks wether to use synthetic setting with multiple seeds
        #   or multi network settings with a singlee seed

        df = pd.DataFrame(columns=['seed', 'noise_level', 'application_step', 'absolute_loss'])
        for i, fixpoint in enumerate(self.id_functions):  # 1 / n
            row_headers.append(fixpoint.name)
            for seed in range(seeds):  # n / 1
                for noise_level in range(noise_levels):
                    self_application_steps = 1
                    clone = Net(fixpoint.input_size, fixpoint.hidden_size, fixpoint.out_size,
                                f"{fixpoint.name}_clone_noise10e-{noise_level}")
                    clone.load_state_dict(copy.deepcopy(fixpoint.state_dict()))
                    clone = clone.apply_noise(pow(10, -noise_level))

                    while not is_zero_fixpoint(clone) and not is_divergent(clone):
                        if is_identity_function(clone):
                            avg_time_as_fixpoint[i][noise_level] += 1

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

                        df.loc[data_pos] = [setting, noise_level, self_application_steps, absolute_loss]
                        data_pos += 1
                        self_application_steps += 1

        # calculate the average:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        # sns.set(rc={'figure.figsize': (10, 50)})
        bx = sns.catplot(data=df[df['absolute_loss'] < 1], y='absolute_loss', x='application_step', kind='box',
                         col='noise_level', col_wrap=3, showfliers=False)
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

    def weights_evolution_3d_experiment(self):
        exp_name = f"soup_{self.population_size}_nets_{self.ST_steps}_training_{self.epochs}_epochs"
        return plot_3d_soup(self.population, exp_name, self.directory)

    def count_fixpoints(self):
        self.id_functions = test_for_fixpoints(self.fixpoint_counters, self.population)
        exp_details = f"Evolution steps: {self.epochs} epochs"
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory, self.net_learning_rate,
                            exp_details)

    def fixpoint_percentage(self):
        runs = self.epochs / self.ST_steps
        SA_steps = None
        line_chart_fixpoints(self.fixpoint_counters_history, runs, self.ST_steps, SA_steps, self.directory,
                             self.population_size)

    def visualize_loss(self):
        for i in range(len(self.population)):
            net_loss_history = self.population[i].loss_history
            self.loss_history.append(net_loss_history)

        plot_loss(self.loss_history, self.directory)

    def reset_fixpoint_counters(self):
        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }


if __name__ == "__main__":
    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    soup_epochs = 100
    soup_log_step_size = 5
    soup_ST_steps = 20
    # soup_SA_steps = 10

    # Define number of networks & their architecture
    soup_population_size = 20
    soup_net_hidden_size = 2
    soup_net_learning_rate = 0.04

    # soup_attack_chance in %
    soup_attack_chance = 10

    # not used yet: soup_train_nets has 3 possible values "no", "before_SA", "after_SA".
    soup_train_nets = "no"
    soup_name_hash = random.getrandbits(32)
    soup_synthetic = True

    print(f"Running the robustness comparison experiment:")
    SoupRobustnessExperiment(
        population_size=soup_population_size,
        net_i_size=NET_INPUT_SIZE,
        net_h_size=soup_net_hidden_size,
        net_o_size=NET_OUT_SIZE,
        learning_rate=soup_net_learning_rate,
        attack_chance=soup_attack_chance,
        train_nets=soup_train_nets,
        ST_steps=soup_ST_steps,
        epochs=soup_epochs,
        log_step_size=soup_log_step_size,
        directory=Path('output') / 'robustness' / f'{soup_name_hash}'
    )