import os.path
import pickle
from pathlib import Path

from tqdm import tqdm

from experiments.helpers import check_folder, summary_fixpoint_experiment
from functionalities_test import test_for_fixpoints
from network import Net
from visualization import plot_loss, bar_chart_fixpoints
from visualization import plot_3d_self_train



class SelfTrainExperiment:
    def __init__(self, population_size, log_step_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                 epochs, directory_name) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size

        self.net_learning_rate = net_learning_rate
        self.epochs = epochs

        self.loss_history = []

        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }

        self.directory_name = directory_name
        os.mkdir(self.directory_name)

        self.nets = []
        # Create population:
        self.populate_environment()

        self.weights_evolution_3d_experiment()
        self.count_fixpoints()
        self.visualize_loss()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating ST experiment %s" % i)

            net_name = f"ST_net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)

            for _ in range(self.epochs):
              input_data = net.input_weight_matrix()
              target_data = net.create_target_weights(input_data)
              net.self_train(1, self.log_step_size, self.net_learning_rate)

            print(f"\nLast weight matrix (epoch: {self.epochs}):\n{net.input_weight_matrix()}\nLossHistory: {net.loss_history[-10:]}")
            self.nets.append(net)

    def weights_evolution_3d_experiment(self):
        exp_name = f"ST_{str(len(self.nets))}_nets_3d_weights_PCA"
        return plot_3d_self_train(self.nets, exp_name, self.directory_name, self.log_step_size)

    def count_fixpoints(self):
        test_for_fixpoints(self.fixpoint_counters, self.nets)
        exp_details = f"Self-train for {self.epochs} epochs"
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)

    def visualize_loss(self):
        for i in range(len(self.nets)):
            net_loss_history = self.nets[i].loss_history
            self.loss_history.append(net_loss_history)

        plot_loss(self.loss_history, self.directory_name)


def run_ST_experiment(population_size, batch_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                      epochs, runs, run_name, name_hash):
    experiments = {}
    logging_directory = Path('output') / 'self_training'
    logging_directory.mkdir(parents=True, exist_ok=True)

    # Running the experiments
    for i in range(runs):
        experiment_name = f"{run_name}_run_{i}_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"
        this_exp_directory = logging_directory / experiment_name
        ST_experiment = SelfTrainExperiment(
            population_size,
            batch_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            epochs,
            this_exp_directory
        )
        with (this_exp_directory / 'full_experiment_pickle.p').open('wb') as f:
            pickle.dump(ST_experiment, f)
        experiments[i] = ST_experiment

    # Building a summary of all the runs
    summary_name = f"/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"
    summary_directory_name = logging_directory / summary_name
    summary_directory_name.mkdir(parents=True, exist_ok=True)

    summary_pre_title = "ST"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, summary_directory_name,
                                summary_pre_title)

if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
