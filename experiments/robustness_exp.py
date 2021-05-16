import copy
import os.path
import pickle
import random

from tqdm import tqdm

from experiments.helpers import check_folder, summary_fixpoint_experiment
from functionalities_test import test_for_fixpoints, is_identity_function
from network import Net
from visualization import bar_chart_fixpoints, box_plot, write_file


def add_noise(input_data, epsilon = pow(10, -5)):

    output = copy.deepcopy(input_data)
    for k in range(len(input_data)):
        output[k][0] += random.random() * epsilon

    return output


class RobustnessExperiment:
    def __init__(self, population_size, log_step_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                 ST_steps, directory_name) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size

        self.net_learning_rate = net_learning_rate

        self.ST_steps = ST_steps
        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }
        self.id_functions = []

        self.directory_name = directory_name
        os.mkdir(self.directory_name)

        self.nets = []
        # Create population:
        self.populate_environment()
        print("Nets:\n", self.nets)

        self.count_fixpoints()
        [print(net.is_fixpoint) for net in self.nets]
        self.test_robustness()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating robustness experiment %s" % i)

            net_name = f"net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)

            for _ in range(self.ST_steps):
                input_data = net.input_weight_matrix()
                target_data = net.create_target_weights(input_data)
                net.self_train(1, self.log_step_size, self.net_learning_rate)

            self.nets.append(net)

    def test_robustness(self):
        # test_for_fixpoints(self.fixpoint_counters, self.nets, self.id_functions)

        zero_epsilon = pow(10, -5)
        data = [[0 for _ in range(10)] for _ in range(len(self.id_functions))]

        for i in range(len(self.id_functions)):
            for j in range(10):
                original_net = self.id_functions[i]

                # Creating a clone of the network. Not by copying it, but by creating a completely new network
                # and changing its weights to the original ones.
                original_net_clone = Net(original_net.input_size, original_net.hidden_size, original_net.out_size,
                                         original_net.name)
                # Extra safety for the value of the weights
                original_net_clone.load_state_dict(copy.deepcopy(original_net.state_dict()))

                noisy_weights = add_noise(original_net_clone.input_weight_matrix())
                original_net_clone.apply_weights(noisy_weights)

                # Testing if the new net is still an identity function after applying noise
                still_id_func = is_identity_function(original_net_clone, zero_epsilon)

                # If the net is still an id. func. after applying the first run of noise, continue to apply it until otherwise
                while still_id_func and data[i][j] <= 1000:
                    data[i][j] += 1

                    original_net_clone = original_net_clone.self_application(1, self.log_step_size)

                    still_id_func = is_identity_function(original_net_clone, zero_epsilon)

        print(f"Data {data}")

        if data.count(0) == 10:
            print(f"There is no network resisting the robustness test.")
            text = f"For this population of \n {self.population_size} networks \n there is no" \
                   f" network resisting the robustness test."
            write_file(text, self.directory_name)
        else:
            box_plot(data, self.directory_name, self.population_size)

    def count_fixpoints(self):
        exp_details = f"ST steps: {self.ST_steps}"

        self.id_functions = test_for_fixpoints(self.fixpoint_counters, self.nets)
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)


def run_robustness_experiment(population_size, batch_size, net_input_size, net_hidden_size, net_out_size,
                              net_learning_rate, epochs, runs, run_name, name_hash):
    experiments = {}

    check_folder("robustness")

    # Running the experiments
    for i in range(runs):
        ST_directory_name = f"experiments/robustness/{run_name}_run_{i}_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"

        robustness_experiment = RobustnessExperiment(
            population_size,
            batch_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            epochs,
            ST_directory_name
        )
        pickle.dump(robustness_experiment, open(f"{ST_directory_name}/full_experiment_pickle.p", "wb"))
        experiments[i] = robustness_experiment

    # Building a summary of all the runs
    directory_name = f"experiments/robustness/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{str(name_hash)}"
    os.mkdir(directory_name)

    summary_pre_title = "robustness"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title)

if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
