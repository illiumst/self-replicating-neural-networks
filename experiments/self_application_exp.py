import os.path
import pickle

from tqdm import tqdm

from experiments.helpers import check_folder, summary_fixpoint_experiment
from functionalities_test import test_for_fixpoints
from network import Net
from visualization import bar_chart_fixpoints
from visualization import plot_3d_self_application


class SelfApplicationExperiment:
    def __init__(self, population_size, log_step_size, net_input_size, net_hidden_size, net_out_size,
                 net_learning_rate, application_steps, train_nets, directory_name, training_steps
                 ) -> None:
        self.population_size = population_size
        self.log_step_size = log_step_size
        self.net_input_size = net_input_size
        self.net_hidden_size = net_hidden_size
        self.net_out_size = net_out_size

        self.net_learning_rate = net_learning_rate
        self.SA_steps = application_steps  #

        self.train_nets = train_nets
        self.ST_steps = training_steps

        self.directory_name = directory_name
        os.mkdir(self.directory_name)

        """ Creating the nets & making the SA steps & (maybe) also training the networks. """
        self.nets = []
        # Create population:
        self.populate_environment()

        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }

        self.weights_evolution_3d_experiment()
        self.count_fixpoints()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating SA experiment %s" % i)

            net_name = f"SA_net_{str(i)}"

            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name
                      )
            for _ in range(self.SA_steps):
                input_data = net.input_weight_matrix()
                target_data = net.create_target_weights(input_data)

                if self.train_nets == "before_SA":
                    net.self_train(1, self.log_step_size, self.net_learning_rate)
                    net.self_application(self.SA_steps, self.log_step_size)
                elif self.train_nets == "after_SA":
                    net.self_application(self.SA_steps, self.log_step_size)
                    net.self_train(1, self.log_step_size, self.net_learning_rate)
                else:
                    net.self_application(self.SA_steps, self.log_step_size)

            self.nets.append(net)

    def weights_evolution_3d_experiment(self):
        exp_name = f"SA_{str(len(self.nets))}_nets_3d_weights_PCA"
        plot_3d_self_application(self.nets, exp_name, self.directory_name, self.log_step_size)

    def count_fixpoints(self):
        test_for_fixpoints(self.fixpoint_counters, self.nets)
        exp_details = f"{self.SA_steps} SA steps"
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)


def run_SA_experiment(population_size, batch_size, net_input_size, net_hidden_size, net_out_size,
                      net_learning_rate, runs, run_name, name_hash, application_steps, train_nets, training_steps):
    experiments = {}

    check_folder("self_application")

    # Running the experiments
    for i in range(runs):
        directory_name = f"experiments/self_application/{run_name}_run_{i}_{str(population_size)}_nets_{application_steps}_SA_{str(name_hash)}"

        SA_experiment = SelfApplicationExperiment(
            population_size,
            batch_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            application_steps,
            train_nets,
            directory_name,
            training_steps
        )
        pickle.dump(SA_experiment, open(f"{directory_name}/full_experiment_pickle.p", "wb"))
        experiments[i] = SA_experiment

    # Building a summary of all the runs
    directory_name = f"experiments/self_application/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{application_steps}_SA_{str(name_hash)}"
    os.mkdir(directory_name)

    summary_pre_title = "SA"
    summary_fixpoint_experiment(runs, population_size, application_steps, experiments, net_learning_rate,
                                directory_name,
                                summary_pre_title)


if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
