import os.path
import pickle

from tqdm import tqdm

from experiments.helpers import check_folder, summary_fixpoint_experiment, summary_fixpoint_percentage
from functionalities_test import test_for_fixpoints
from network import Net
from visualization import plot_loss, bar_chart_fixpoints, line_chart_fixpoints
from visualization import plot_3d_self_train


class MixedSettingExperiment:
    def __init__(self, population_size, net_i_size, net_h_size, net_o_size, learning_rate, train_nets,
                 epochs, SA_steps, ST_steps_between_SA, log_step_size, directory_name):
        super().__init__()
        self.population_size = population_size

        self.net_input_size = net_i_size
        self.net_hidden_size = net_h_size
        self.net_out_size = net_o_size
        self.net_learning_rate = learning_rate
        self.train_nets = train_nets
        self.epochs = epochs
        self.SA_steps = SA_steps
        self.ST_steps_between_SA = ST_steps_between_SA
        self.log_step_size = log_step_size

        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }

        self.loss_history = []

        self.fixpoint_counters_history = []

        self.directory_name = directory_name
        os.mkdir(self.directory_name)

        self.nets = []
        self.populate_environment()

        self.fixpoint_percentage()
        self.weights_evolution_3d_experiment()
        self.count_fixpoints()
        self.visualize_loss()

    def populate_environment(self):
        loop_population_size = tqdm(range(self.population_size))
        for i in loop_population_size:
            loop_population_size.set_description("Populating mixed experiment %s" % i)

            net_name = f"mixed_net_{str(i)}"
            net = Net(self.net_input_size, self.net_hidden_size, self.net_out_size, net_name)
            self.nets.append(net)

        loop_epochs = tqdm(range(self.epochs))
        for j in loop_epochs:
            loop_epochs.set_description("Running mixed experiment %s" % j)

            for i in loop_population_size:
                net = self.nets[i]

                if self.train_nets == "before_SA":
                    for _ in range(self.ST_steps_between_SA):
                        input_data = net.input_weight_matrix()
                        target_data = net.create_target_weights(input_data)
                        net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)
                    net.self_application(self.SA_steps, self.log_step_size)

                elif self.train_nets == "after_SA":
                    net.self_application(self.SA_steps, self.log_step_size)
                    for _ in range(self.ST_steps_between_SA):
                        input_data = net.input_weight_matrix()
                        target_data = net.create_target_weights(input_data)
                        net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)

                print(
                    f"\nLast weight matrix (epoch: {j}):\n{net.input_weight_matrix()}\nLossHistory: {net.loss_history[-10:]}")
            test_for_fixpoints(self.fixpoint_counters, self.nets)
            # Rounding the result not to run into other problems later regarding the exact representation of floating number
            fixpoints_percentage = round((self.fixpoint_counters["fix_zero"] + self.fixpoint_counters[
                "fix_sec"]) / self.population_size, 1)
            self.fixpoint_counters_history.append(fixpoints_percentage)

            # Resetting the fixpoint counter. Last iteration not to be reset - it is important for the bar_chart_fixpoints().
            if j < self.epochs:
                self.reset_fixpoint_counters()

    def weights_evolution_3d_experiment(self):
        exp_name = f"Mixed {str(len(self.nets))}"

        # This batch size is not relevant for mixed settings because during an epoch there are more steps of SA & ST happening
        # and only they need the batch size. To not affect the number of epochs shown in the 3D plot, will send
        # forward the number "1" for batch size with the variable <irrelevant_batch_size>
        irrelevant_batch_size = 1
        plot_3d_self_train(self.nets, exp_name, self.directory_name, irrelevant_batch_size)

    def count_fixpoints(self):
        exp_details = f"SA steps: {self.SA_steps}; ST steps: {self.ST_steps_between_SA}"

        test_for_fixpoints(self.fixpoint_counters, self.nets)
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)

    def fixpoint_percentage(self):
        line_chart_fixpoints(self.fixpoint_counters_history, self.epochs, self.ST_steps_between_SA,
                             self.SA_steps, self.directory_name, self.population_size)

    def visualize_loss(self):
        for i in range(len(self.nets)):
            net_loss_history = self.nets[i].loss_history
            self.loss_history.append(net_loss_history)

        plot_loss(self.loss_history, self.directory_name)

    def reset_fixpoint_counters(self):
        self.fixpoint_counters = {
            "identity_func": 0,
            "divergent": 0,
            "fix_zero": 0,
            "fix_weak": 0,
            "fix_sec": 0,
            "other_func": 0
        }


def run_mixed_experiment(population_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate, train_nets,
                         epochs, SA_steps, ST_steps_between_SA, batch_size, name_hash, runs, run_name):
    experiments = {}
    fixpoints_percentages = []

    check_folder("mixed")

    # Running the experiments
    for i in range(runs):
        directory_name = f"experiments/mixed/{run_name}_run_{i}_{str(population_size)}_nets_{SA_steps}_SA_{ST_steps_between_SA}_ST_{str(name_hash)}"

        mixed_experiment = MixedSettingExperiment(
            population_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            train_nets,
            epochs,
            SA_steps,
            ST_steps_between_SA,
            batch_size,
            directory_name
        )
        pickle.dump(mixed_experiment, open(f"{directory_name}/full_experiment_pickle.p", "wb"))
        experiments[i] = mixed_experiment

        # Building history of fixpoint percentages for summary
        fixpoint_counters_history = mixed_experiment.fixpoint_counters_history
        if not fixpoints_percentages:
            fixpoints_percentages = mixed_experiment.fixpoint_counters_history
        else:
            # Using list comprehension to make the sum of all the percentages
            fixpoints_percentages = [fixpoints_percentages[i] + fixpoint_counters_history[i] for i in
                                     range(len(fixpoints_percentages))]

    # Building a summary of all the runs
    directory_name = f"experiments/mixed/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{str(name_hash)}"
    os.mkdir(directory_name)

    summary_pre_title = "mixed"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title)
    summary_fixpoint_percentage(runs, epochs, fixpoints_percentages, ST_steps_between_SA, SA_steps, directory_name,
                                population_size)


if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
