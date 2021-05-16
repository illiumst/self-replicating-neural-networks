import os
from pathlib import Path

from tqdm import tqdm
import random
import copy
from functionalities_test import is_identity_function
from network import Net
from visualization import plot_3d_self_train, plot_loss
import numpy as np

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


def prng():
    return random.random()


def l1(tup):
    a, b = tup
    return abs(a-b)


def mean_invariate_manhattan_distance(x, y):
    # One of these one-liners that might be smart or really dumb. Goal is to find pairwise 
    # distances of ascending values, ie. sum (abs(min1_X-min1_Y), abs(min2_X-min2Y) ...) / mean.  
    # Idea was to find weight sets that have same values but just in different positions, that would
    # make this distance 0.
    return np.mean(list(map(l1, zip(sorted(x), sorted(y)))))


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
        print(f"\nDistance matrix [{distance}]:")
        [print(row) for row in matrix]
    return matrix


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
                 epochs, st_steps, noise, directory) -> None:
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
        self.noise = noise or 10e-5
        print("\nNOISE:", self.noise)

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.populate_environment()
        self.spawn_and_continue()
        self.weights_evolution_3d_experiment()
        # self.visualize_loss()
        distance_matrix(self.nets)

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating experiment %s" % i)

            net_name = f"ST_net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)

            for _ in range(self.ST_steps):
                net.self_train(1, self.log_step_size, self.net_learning_rate)

            # print(f"\nLast weight matrix (epoch: {self.epochs}):\n
            # {net.input_weight_matrix()}\nLossHistory: {net.loss_history[-10:]}")
            self.nets.append(net)

    def spawn_and_continue(self, number_spawns: int = 5):
        # For every initial net {i} after populating (that is fixpoint after first epoch);
        for i in range(self.population_size):
            net = self.nets[i]

            net_input_data = net.input_weight_matrix()
            net_target_data = net.create_target_weights(net_input_data)
            if is_identity_function(net):
                print(f"\nNet {i} is fixpoint")
                # print("\nNet weights before training\n", target_data)

                # Clone the fixpoint x times and add (+-)self.noise to weight-sets randomly;
                # To plot clones starting after first epoch (z=ST_steps), set that as start_time!
                for j in range(number_spawns):
                    clone = Net(net.input_size, net.hidden_size, net.out_size,
                                f"ST_net_{str(i)}_clone_{str(j)}",
                                start_time=self.ST_steps)
                    clone.load_state_dict(copy.deepcopy(net.state_dict()))
                    rand_noise = prng() * self.noise
                    clone = self.apply_noise(clone, rand_noise)

                    # Then finish training each clone {j} (for remaining epoch-1 * ST_steps)
                    # and add to nets for plotting;
                    for _ in range(self.epochs - 1):
                        for _ in range(self.ST_steps):
                            clone.self_train(1, self.log_step_size, self.net_learning_rate)
                    # print(f"clone {j} last weights: {target_data}, noise {noise}")
                    if is_identity_function(clone):
                        input_data = clone.input_weight_matrix()
                        target_data = clone.create_target_weights(input_data)
                        print(f"Clone {j} (of net_{i}) is fixpoint. \nMSE(j,i): "
                              f"{MSE(net_target_data, target_data)}, \nMAE(j,i): {MAE(net_target_data, target_data)}\n")
                    self.nets.append(clone)

                # Finally take parent net {i} and finish it's training for comparison to clone development.
                for _ in range(self.epochs - 1):
                    for _ in range(self.ST_steps):
                        net.self_train(1, self.log_step_size, self.net_learning_rate)
                # print("\nNet weights after training \n", target_data)

        else:
            print("No fixpoints found.")

    def weights_evolution_3d_experiment(self):
        exp_name = f"ST_{str(len(self.nets))}_nets_3d_weights_PCA"
        return plot_3d_self_train(self.nets, exp_name, self.directory, self.log_step_size)

    def visualize_loss(self):
        for i in range(len(self.nets)):
            net_loss_history = self.nets[i].loss_history
            self.loss_history.append(net_loss_history)
        plot_loss(self.loss_history, self.directory)


if __name__ == "__main__":

    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    # Define number of runs & name:
    ST_runs = 1
    ST_runs_name = "test-27"
    ST_steps = 1500
    ST_epochs = 2
    ST_log_step_size = 10

    # Define number of networks & their architecture
    ST_population_size = 1
    ST_net_hidden_size = 2
    ST_net_learning_rate = 0.04
    ST_name_hash = random.getrandbits(32)

    print(f"Running the Spawn experiment:")
    for noise_factor in range(3, 6):
        SpawnExperiment(
            population_size=ST_population_size,
            log_step_size=ST_log_step_size,
            net_input_size=NET_INPUT_SIZE,
            net_hidden_size=ST_net_hidden_size,
            net_out_size=NET_OUT_SIZE,
            net_learning_rate=ST_net_learning_rate,
            epochs=ST_epochs,
            st_steps=ST_steps,
            noise=pow(10, -noise_factor),
            directory=Path('output') / 'spawn_basin' / f'{ST_name_hash}_10e-{noise_factor}'
        )
