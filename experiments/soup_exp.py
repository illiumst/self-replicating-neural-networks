import random
import os.path
import pickle
from pathlib import Path
from typing import Union

from tqdm import tqdm

from experiments.helpers import check_folder, summary_fixpoint_percentage, summary_fixpoint_experiment
from functionalities_test import test_for_fixpoints
from network import Net
from visualization import plot_loss, bar_chart_fixpoints, plot_3d_soup, line_chart_fixpoints


class SoupExperiment:
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

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.population = []
        self.populate_environment()

        self.evolve()
        self.fixpoint_percentage()
        self.weights_evolution_3d_experiment()
        self.count_fixpoints()
        self.visualize_loss()

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

    def weights_evolution_3d_experiment(self):
        exp_name = f"soup_{self.population_size}_nets_{self.ST_steps}_training_{self.epochs}_epochs"
        return plot_3d_soup(self.population, exp_name, self.directory)

    def count_fixpoints(self):
        test_for_fixpoints(self.fixpoint_counters, self.population)
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


def run_soup_experiment(population_size, attack_chance, net_input_size, net_hidden_size, net_out_size,
                        net_learning_rate, epochs, batch_size, runs, run_name, name_hash, ST_steps, train_nets):
    experiments = {}
    fixpoints_percentages = []

    check_folder("soup")

    # Running the experiments
    for i in range(runs):
        # FIXME: Make this a pathlib.Path() Operation
        directory_name = f"experiments/soup/{run_name}_run_{i}_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"

        soup_experiment = SoupExperiment(
            population_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            attack_chance,
            train_nets,
            ST_steps,
            epochs,
            batch_size,
            directory_name
        )
        pickle.dump(soup_experiment, open(f"{directory_name}/full_experiment_pickle.p", "wb"))
        experiments[i] = soup_experiment

        # Building history of fixpoint percentages for summary
        fixpoint_counters_history = soup_experiment.fixpoint_counters_history
        if not fixpoints_percentages:
            fixpoints_percentages = soup_experiment.fixpoint_counters_history
        else:
            # Using list comprehension to make the sum of all the percentages
            fixpoints_percentages = [fixpoints_percentages[i] + fixpoint_counters_history[i] for i in
                                     range(len(fixpoints_percentages))]

    # Creating a folder for the summary of the current runs
    # FIXME: Make this a pathlib.Path() Operation
    directory_name = f"experiments/soup/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"
    os.mkdir(directory_name)

    # Building a summary of all the runs
    summary_pre_title = "soup"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title)
    SA_steps = None
    summary_fixpoint_percentage(runs, epochs, fixpoints_percentages, ST_steps, SA_steps, directory_name,
                                population_size)

