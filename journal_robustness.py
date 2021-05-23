import pickle
import torch
import random
import copy

from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from journal_basins import mean_invariate_manhattan_distance as MIM
from functionalities_test import is_identity_function, is_zero_fixpoint, test_for_fixpoints, is_divergent
from network import Net
from torch.nn import functional as F
from visualization import plot_loss, bar_chart_fixpoints


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

        for i in loop_population_size:
            loop_population_size.set_description("Populating experiment %s" % i)

            if self.synthetic:
                ''' Either use perfect / hand-constructed fixpoint ... '''
                net_name = f"net_{str(i)}_synthetic"
                net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
                net.apply_weights(generate_perfekt_synthetic_fixpoint_weights())

            else:
                ''' .. or use natural approach to train fixpoints from random initialisation. '''
                net_name = f"net_{str(i)}"
                net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
                for _ in range(self.epochs):
                    net.self_train(self.ST_steps, self.log_step_size, self.net_learning_rate)
            nets.append(net)
        return nets

    def test_robustness(self, print_it=True):
        avg_time_to_vergence = [[0 for _ in range(10)] for _ in range(len(self.id_functions))]
        avg_time_as_fixpoint = [[0 for _ in range(10)] for _ in range(len(self.id_functions))]
        avg_loss_per_application = [[0 for _ in range(10)] for _ in range(len(self.id_functions))]
        noise_range = range(10)
        row_headers = []

        for i, fixpoint in enumerate(self.id_functions):
            row_headers.append(fixpoint.name)
            loss_per_application = [[0 for _ in range(10)] for _ in range(len(self.id_functions))]
            for seed in range(10):
                for noise_level in noise_range:
                    clone = Net(fixpoint.input_size, fixpoint.hidden_size, fixpoint.out_size,
                                f"{fixpoint.name}_clone_noise10e-{noise_level}")
                    clone.load_state_dict(copy.deepcopy(fixpoint.state_dict()))
                    rand_noise = prng() * pow(10, -noise_level)
                    clone = self.apply_noise(clone, rand_noise)

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

                        loss_per_application[seed][noise_level] = (F.l1_loss(target_data_pre_application,
                                                                             target_data_post_application))


        if print_it:
            col_headers = [str(f"10e-{d}") for d in noise_range]

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
        directory=Path('output') / 'robustness' / f'{ST_name_hash}'
    )
