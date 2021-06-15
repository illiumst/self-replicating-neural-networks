import os
from pathlib import Path
import pickle
from tqdm import tqdm
import random
import copy
from functionalities_test import is_identity_function, test_status
from network import Net
from visualization import plot_3d_self_train, plot_loss
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


class SpawnExperiment:

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
                 epochs, st_steps, nr_clones, noise, directory) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size
        self.net_learning_rate = net_learning_rate
        self.epochs = epochs
        self.ST_steps = st_steps
        self.loss_history = []
        self.nets = []
        self.nr_clones = nr_clones
        self.noise = noise or 10e-5
        print("\nNOISE:", self.noise)

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.populate_environment()
        self.spawn_and_continue()
        self.weights_evolution_3d_experiment()
        # self.visualize_loss()
        self.distance_matrix = distance_matrix(self.nets, print_it=False)
        self.parent_clone_distances = distance_from_parent(self.nets, print_it=False)
        self.save()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating experiment %s" % i)

            net_name = f"ST_net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)

            for _ in range(self.ST_steps):
                net.self_train(1, self.log_step_size, self.net_learning_rate)

            self.nets.append(net)

    def spawn_and_continue(self, number_clones: int = None):
        number_clones = number_clones or self.nr_clones

        df = pd.DataFrame(
            columns=['parent', 'MAE_pre', 'MAE_post', 'MSE_pre', 'MSE_post', 'MIM_pre', 'MIM_post', 'noise',
                     'status_post'])

        # For every initial net {i} after populating (that is fixpoint after first epoch);
        for i in range(self.population_size):
            net = self.nets[i]
            # We set parent start_time to just before this epoch ended, so plotting is zoomed in. Comment out to
            # to see full trajectory (but the clones will be very hard to see). 
            # Make one target to compare distances to clones later when they have trained.
            net.start_time = self.ST_steps - 150
            net_input_data = net.input_weight_matrix()
            net_target_data = net.create_target_weights(net_input_data)

            if is_identity_function(net):
                print(f"\nNet {i} is fixpoint")

                # Clone the fixpoint x times and add (+-)self.noise to weight-sets randomly;
                # To plot clones starting after first epoch (z=ST_steps), set that as start_time!
                # To make sure PCA will plot the same trajectory up until this point, we clone the
                # parent-net's weight history as well.
                for j in range(number_clones):
                    clone = Net(net.input_size, net.hidden_size, net.out_size,
                                f"ST_net_{str(i)}_clone_{str(j)}", start_time=self.ST_steps)
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

                    # Then finish training each clone {j} (for remaining epoch-1 * ST_steps) ..
                    for _ in range(self.epochs - 1):
                        for _ in range(self.ST_steps):
                            clone.self_train(1, self.log_step_size, self.net_learning_rate)

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
                        self.nets.append(clone)

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

    def weights_evolution_3d_experiment(self):
        exp_name = f"ST_{str(len(self.nets))}_nets_3d_weights_PCA"
        return plot_3d_self_train(self.nets, exp_name, self.directory, self.log_step_size, plot_pca_together=True)

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

    # Define number of runs & name:
    ST_runs = 1
    ST_runs_name = "test-27"
    ST_steps = 2500
    ST_epochs = 2
    ST_log_step_size = 10

    # Define number of networks & their architecture
    nr_clones = 5
    ST_population_size = 2
    ST_net_hidden_size = 2
    ST_net_learning_rate = 0.04
    ST_name_hash = random.getrandbits(32)

    print(f"Running the Spawn experiment:")
    exp_list = []
    for noise_factor in range(2, 5):
        exp = SpawnExperiment(
            population_size=ST_population_size,
            log_step_size=ST_log_step_size,
            net_input_size=NET_INPUT_SIZE,
            net_hidden_size=ST_net_hidden_size,
            net_out_size=NET_OUT_SIZE,
            net_learning_rate=ST_net_learning_rate,
            epochs=ST_epochs,
            st_steps=ST_steps,
            nr_clones=nr_clones,
            noise=pow(10, -noise_factor),
            directory=Path('output') / 'spawn_basin' / f'{ST_name_hash}' / f'10e-{noise_factor}'
        )
        exp_list.append(exp)

    # Boxplot with counts of nr_fixpoints, nr_other, nr_etc. on y-axis
    df = pd.concat([exp.df for exp in exp_list])
    sns.countplot(data=df, x="noise", hue="status_post")
    plt.savefig(f"output/spawn_basin/{ST_name_hash}/fixpoint_status_countplot.png")

    # Catplot (either kind="point" or "box") that shows before-after training distances to parent
    mlt = df[["MIM_pre", "MIM_post", "noise"]].melt("noise", var_name="time", value_name='Average Distance')
    sns.catplot(data=mlt, x="time", y="Average Distance", col="noise", kind="point", col_wrap=5, sharey=False)
    plt.savefig(f"output/spawn_basin/{ST_name_hash}/clone_distance_catplot.png")
