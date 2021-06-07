import os
from pathlib import Path
import pickle
from torch import mean

from tqdm import tqdm
import random
import copy
from functionalities_test import is_identity_function, test_status, test_for_fixpoints, is_zero_fixpoint, is_divergent, is_secondary_fixpoint
from network import Net
from visualization import plot_3d_self_train, plot_loss, plot_3d_soup
import numpy as np
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def prng():
    return random.random()


def l1(tup):
    a, b = tup
    return abs(a - b)


def mean_invariate_manhattan_distance(x, y):
    # One of these one-liners that might be smart or really dumb. Goal is to find pairwise
    # distances of ascending values, ie. sum (abs(min1_X-min1_Y), abs(min2_X-min2Y) ...) / mean.
    # Idea was to find weight sets that have same values but just in different positions, that would
    # make this distance 0.
    return np.mean(list(map(l1, zip(sorted(x.numpy()), sorted(y.numpy())))))


def distance_matrix(nets, distance="MIM", print_it=True):
    matrix = [[0 for _ in range(len(nets))] for _ in range(len(nets))]
    for net in range(len(nets)):
        weights = nets[net].input_weight_matrix()[:, 0]
        for other_net in range(len(nets)):
            other_weights = nets[other_net].input_weight_matrix()[:, 0]
            if distance in ["MSE"]:
                matrix[net][other_net] = MSE(weights, other_weights)
            elif distance in ["MAE"]:
                matrix[net][other_net] = MAE(weights, other_weights)
            elif distance in ["MIM"]:
                matrix[net][other_net] = mean_invariate_manhattan_distance(weights, other_weights)

    if print_it:
        print(f"\nDistance matrix (all to all) [{distance}]:")
        headers = [i.name for i in nets]
        print(tabulate(matrix, showindex=headers, headers=headers, tablefmt='orgtbl'))
    return matrix


def distance_from_parent(nets, distance="MIM", print_it=True):
    list_of_matrices = []
    parents = list(filter(lambda x: "clone" not in x.name and is_identity_function(x), nets))
    distance_range = range(10)
    for parent in parents:
        parent_weights = parent.create_target_weights(parent.input_weight_matrix())
        clones = list(filter(lambda y: parent.name in y.name and parent.name != y.name, nets))
        matrix = [[0 for _ in distance_range] for _ in range(len(clones))]

        for dist in distance_range:
            for idx, clone in enumerate(clones):
                clone_weights = clone.create_target_weights(clone.input_weight_matrix())
                if distance in ["MSE"]:
                    matrix[idx][dist] = MSE(parent_weights, clone_weights) < pow(10, -dist)
                elif distance in ["MAE"]:
                    matrix[idx][dist] = MAE(parent_weights, clone_weights) < pow(10, -dist)
                elif distance in ["MIM"]:
                    matrix[idx][dist] = mean_invariate_manhattan_distance(parent_weights, clone_weights) < pow(10,
                                                                                                               -dist)

        if print_it:
            print(f"\nDistances from parent {parent.name} [{distance}]:")
            col_headers = [str(f"10e-{d}") for d in distance_range]
            row_headers = [str(f"clone_{i}") for i in range(len(clones))]
            print(tabulate(matrix, showindex=row_headers, headers=col_headers, tablefmt='orgtbl'))

        list_of_matrices.append(matrix)

    return list_of_matrices


class SoupSpawnExperiment:

    @staticmethod
    def apply_noise(network, noise: int):
        """ Changing the weights of a network to values + noise """

        for layer_id, layer_name in enumerate(network.state_dict()):
            for line_id, line_values in enumerate(network.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(network.state_dict()[layer_name][line_id]):
                    # network.state_dict()[layer_name][line_id][weight_id] = weight_value + noise
                    if prng() < 0.5:
                        network.state_dict()[layer_name][line_id][weight_id] = weight_value + noise
                    else:
                        network.state_dict()[layer_name][line_id][weight_id] = weight_value - noise

        return network

    def __init__(self, population_size, log_step_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                 epochs, st_steps, attack_chance, nr_clones, noise, directory) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size
        self.net_learning_rate = net_learning_rate
        self.epochs = epochs
        self.ST_steps = st_steps
        self.attack_chance = attack_chance
        self.loss_history = []
        self.nr_clones = nr_clones
        self.noise = noise or 10e-5
        print("\nNOISE:", self.noise)

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Populating environment & evolving entities
        self.parents = []
        self.clones = []
        self.parents_with_clones = []
        self.parents_clones_id_functions = []

        self.populate_environment()

        self.spawn_and_continue()
        self.weights_evolution_3d_experiment(self.parents, "only_parents")
        self.weights_evolution_3d_experiment(self.clones, "only_clones")
        self.weights_evolution_3d_experiment(self.parents_with_clones, "parents_with_clones")
        self.weights_evolution_3d_experiment(self.parents_clones_id_functions, "id_f_with_parents")

        # self.visualize_loss()
        self.distance_matrix = distance_matrix(self.parents_clones_id_functions, print_it=False)
        self.parent_clone_distances = distance_from_parent(self.parents_clones_id_functions, print_it=False)

        self.save()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating experiment %s" % i)

            net_name = f"parent_net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)

            for _ in range(self.ST_steps):
                net.self_train(1, self.log_step_size, self.net_learning_rate)

            self.parents.append(net)
            self.parents_with_clones.append(net)

            if is_identity_function(net):
                self.parents_clones_id_functions.append(net)
                print(f"\nNet {net.name} is identity function")

            if is_divergent(net):
                print(f"\nNet {net.name} is divergent")

            if is_zero_fixpoint(net):
                print(f"\nNet {net.name} is zero fixpoint")

            if is_secondary_fixpoint(net):
                print(f"\nNet {net.name} is secondary fixpoint")

    def evolve(self, population):
        print(f"Clone soup has a population of {len(population)} networks")

        loop_epochs = tqdm(range(self.epochs-1))
        for i in loop_epochs:
            loop_epochs.set_description("\nEvolving clone soup %s" % i)

            # A network attacking another network with a given percentage
            if random.randint(1, 100) <= self.attack_chance:
                random_net1, random_net2 = random.sample(range(len(population)), 2)
                random_net1 = population[random_net1]
                random_net2 = population[random_net2]
                print(f"\n Attack: {random_net1.name} -> {random_net2.name}")
                random_net1.attack(random_net2)

            #  Self-training each network in the population
            for j in range(len(population)):
                net = population[j]

                for _ in range(self.ST_steps):
                    net.self_train(1, self.log_step_size, self.net_learning_rate)

    def spawn_and_continue(self, number_clones: int = None):
        number_clones = number_clones or self.nr_clones

        df = pd.DataFrame(
            columns=['parent', 'MAE_pre', 'MAE_post', 'MSE_pre', 'MSE_post', 'MIM_pre', 'MIM_post', 'noise',
                     'status_post'])

        # MAE_pre, MSE_pre, MIM_pre = 0, 0, 0

        # For every initial net {i} after populating (that is fixpoint after first epoch);
        for i in range(len(self.parents)):
            net = self.parents[i]
            # We set parent start_time to just before this epoch ended, so plotting is zoomed in. Comment out to
            # to see full trajectory (but the clones will be very hard to see).
            # Make one target to compare distances to clones later when they have trained.
            net.start_time = self.ST_steps - 150
            net_input_data = net.input_weight_matrix()
            net_target_data = net.create_target_weights(net_input_data)

            # print(f"\nNet {i} is fixpoint")

            # Clone the fixpoint x times and add (+-)self.noise to weight-sets randomly;
            # To plot clones starting after first epoch (z=ST_steps), set that as start_time!
            # To make sure PCA will plot the same trajectory up until this point, we clone the
            # parent-net's weight history as well.
            for j in range(number_clones):
                clone = Net(net.input_size, net.hidden_size, net.out_size,
                            f"net_{str(i)}_clone_{str(j)}", start_time=self.ST_steps)
                clone.load_state_dict(copy.deepcopy(net.state_dict()))
                rand_noise = prng() * self.noise
                clone = self.apply_noise(clone, rand_noise)
                clone.s_train_weights_history = copy.deepcopy(net.s_train_weights_history)
                clone.number_trained = copy.deepcopy(net.number_trained)

                # Pre Training distances (after noise application of course)
                clone_pre_weights = clone.create_target_weights(clone.input_weight_matrix())
                MAE_pre = MAE(net_target_data, clone_pre_weights)
                MSE_pre = MSE(net_target_data, clone_pre_weights)
                MIM_pre = mean_invariate_manhattan_distance(net_target_data, clone_pre_weights)

                net.children.append(clone)
                self.clones.append(clone)
                self.parents_with_clones.append(clone)

        self.evolve(self.clones)
        # evolve also with the parents together
        # self.evolve(self.parents_with_clones)

        for i in range(len(self.parents)):
            net = self.parents[i]
            net_input_data = net.input_weight_matrix()
            net_target_data = net.create_target_weights(net_input_data)

            for j in range(len(net.children)):
                clone = net.children[j]

                # Post Training distances for comparison
                clone_post_weights = clone.create_target_weights(clone.input_weight_matrix())
                MAE_post = MAE(net_target_data, clone_post_weights)
                MSE_post = MSE(net_target_data, clone_post_weights)
                MIM_post = mean_invariate_manhattan_distance(net_target_data, clone_post_weights)

                # .. log to data-frame and add to nets for 3d plotting if they are fixpoints themselves.
                test_status(clone)
                if is_identity_function(clone):
                    print(f"Clone {j} (of net_{i}) is fixpoint."
                          f"\nMSE({i},{j}): {MSE_post}"
                          f"\nMAE({i},{j}): {MAE_post}"
                          f"\nMIM({i},{j}): {MIM_post}\n")
                    self.parents_clones_id_functions.append(clone)

                df.loc[clone.name] = [net.name, MAE_pre, MAE_post, MSE_pre, MSE_post, MIM_pre, MIM_post, self.noise,
                                      clone.is_fixpoint]

            # Finally take parent net {i} and finish it's training for comparison to clone development.
            for _ in range(self.epochs - 1):
                for _ in range(self.ST_steps):
                    net.self_train(1, self.log_step_size, self.net_learning_rate)
            net_weights_after = net.create_target_weights(net.input_weight_matrix())
            print(f"Parent net's distance to original position."
                  f"\nMSE(OG,new): {MAE(net_target_data, net_weights_after)}"
                  f"\nMAE(OG,new): {MSE(net_target_data, net_weights_after)}"
                  f"\nMIM(OG,new): {mean_invariate_manhattan_distance(net_target_data, net_weights_after)}\n")

        self.df = df

    def weights_evolution_3d_experiment(self, nets_population, suffix):
        exp_name = f"soup_basins_{str(len(nets_population))}_nets_3d_weights_PCA_{suffix}"
        return plot_3d_soup(nets_population, exp_name, self.directory)

    def visualize_loss(self):
        for i in range(len(self.parents)):
            net_loss_history = self.parents[i].loss_history
            self.loss_history.append(net_loss_history)
        plot_loss(self.loss_history, self.directory)

    def save(self):
        pickle.dump(self, open(f"{self.directory}/experiment_pickle.p", "wb"))
        print(f"\nSaved experiment to {self.directory}.")


if __name__ == "__main__":

    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    # Define number of runs & name:
    ST_runs = 1
    ST_runs_name = "test-27"
    soup_ST_steps = 1500
    soup_epochs = 2
    soup_log_step_size = 10

    # Define number of networks & their architecture
    nr_clones = 3
    soup_population_size = 2
    soup_net_hidden_size = 2
    soup_net_learning_rate = 0.04
    soup_attack_chance = 10
    soup_name_hash = random.getrandbits(32)

    print(f"Running the Soup-Spawn experiment:")
    exp_list = []
    for noise_factor in range(2, 5):
        exp = SoupSpawnExperiment(
            population_size=soup_population_size,
            log_step_size=soup_log_step_size,
            net_input_size=NET_INPUT_SIZE,
            net_hidden_size=soup_net_hidden_size,
            net_out_size=NET_OUT_SIZE,
            net_learning_rate=soup_net_learning_rate,
            epochs=soup_epochs,
            st_steps=soup_ST_steps,
            attack_chance=soup_attack_chance,
            nr_clones=nr_clones,
            noise=pow(10, -noise_factor),
            directory=Path('output') / 'soup_spawn_basin' / f'{soup_name_hash}' / f'10e-{noise_factor}'
        )
        exp_list.append(exp)

    # Boxplot with counts of nr_fixpoints, nr_other, nr_etc. on y-axis
    df = pd.concat([exp.df for exp in exp_list])
    sns.countplot(data=df, x="noise", hue="status_post")
    plt.savefig(f"output/soup_spawn_basin/{soup_name_hash}/fixpoint_status_countplot.png")

    # Catplot (either kind="point" or "box") that shows before-after training distances to parent
    mlt = df[["MIM_pre", "MIM_post", "noise"]].melt("noise", var_name="time", value_name='Average Distance')
    sns.catplot(data=mlt, x="time", y="Average Distance", col="noise", kind="point", col_wrap=5, sharey=False)
    plt.savefig(f"output/soup_spawn_basin/{soup_name_hash}/clone_distance_catplot.png")
