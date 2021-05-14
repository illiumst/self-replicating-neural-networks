import copy
import random
import os.path
import pickle
from tokenize import String

from tqdm import tqdm
from functionalities_test import test_for_fixpoints, is_zero_fixpoint, is_identity_function
from network import Net
from visualization import plot_loss, bar_chart_fixpoints, plot_3d_soup, line_chart_fixpoints, box_plot, write_file
from visualization import plot_3d_self_application, plot_3d_self_train


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
              net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)

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
                    net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)
                    net.self_application(input_data, self.SA_steps, self.log_step_size)
                elif self.train_nets == "after_SA":
                    net.self_application(input_data, self.SA_steps, self.log_step_size)
                    net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)
                else:
                    net.self_application(input_data, self.SA_steps, self.log_step_size)

            self.nets.append(net)

    def weights_evolution_3d_experiment(self):
        exp_name = f"SA_{str(len(self.nets))}_nets_3d_weights_PCA"
        plot_3d_self_application(self.nets, exp_name, self.directory_name, self.log_step_size)

    def count_fixpoints(self):
        test_for_fixpoints(self.fixpoint_counters, self.nets)
        exp_details = f"{self.SA_steps} SA steps"
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)


class SoupExperiment:
    def __init__(self, population_size, net_i_size, net_h_size, net_o_size, learning_rate, attack_chance,
                 train_nets, ST_steps, epochs, log_step_size, directory_name):
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

        self.directory_name = directory_name
        os.mkdir(self.directory_name)

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
            chance = random.randint(1, 100)
            if chance <= self.attack_chance:
                random_net1, random_net2 = random.sample(range(self.population_size), 2)
                random_net1 = self.population[random_net1]
                random_net2 = self.population[random_net2]
                print(f"\n Attack: {random_net1.name} -> {random_net2.name}")
                random_net1.attack(random_net2)

            #  Self-training each network in the population
            for j in range(self.population_size):
                net = self.population[j]
                
                for _ in range(self.ST_steps):
                    input_data = net.input_weight_matrix()
                    target_data = net.create_target_weights(input_data)
                    net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)


            # Testing for fixpoints after each batch of ST steps to see relevant data
            if i % self.ST_steps == 0:
                test_for_fixpoints(self.fixpoint_counters, self.population)
                fixpoints_percentage = round((self.fixpoint_counters["fix_zero"] + self.fixpoint_counters["fix_weak"] +
                                              self.fixpoint_counters["fix_sec"]) / self.population_size, 1)
                self.fixpoint_counters_history.append(fixpoints_percentage)

            # Resetting the fixpoint counter. Last iteration not to be reset - it is important for the bar_chart_fixpoints().
            if i < self.epochs:
                self.reset_fixpoint_counters()

    def weights_evolution_3d_experiment(self):
        exp_name = f"soup_{self.population_size}_nets_{self.ST_steps}_training_{self.epochs}_epochs"
        return plot_3d_soup(self.population, exp_name, self.directory_name)

    def count_fixpoints(self):
        test_for_fixpoints(self.fixpoint_counters, self.population)
        exp_details = f"Evolution steps: {self.epochs} epochs"
        bar_chart_fixpoints(self.fixpoint_counters, self.population_size, self.directory_name, self.net_learning_rate,
                            exp_details)

    def fixpoint_percentage(self):
        runs = self.epochs / self.ST_steps
        SA_steps = None
        line_chart_fixpoints(self.fixpoint_counters_history, runs, self.ST_steps, SA_steps, self.directory_name,
                             self.population_size)

    def visualize_loss(self):
        for i in range(len(self.population)):
            net_loss_history = self.population[i].loss_history
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
                    input_data = net.input_weight_matrix()
                    net.self_application(input_data, self.SA_steps, self.log_step_size)
                      
                elif self.train_nets == "after_SA":
                    input_data = net.input_weight_matrix()
                    net.self_application(input_data, self.SA_steps, self.log_step_size)
                    for _ in range(self.ST_steps_between_SA):
                        input_data = net.input_weight_matrix()
                        target_data = net.create_target_weights(input_data)
                        net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)
                
                print(f"\nLast weight matrix (epoch: {j}):\n{net.input_weight_matrix()}\nLossHistory: {net.loss_history[-10:]}") 
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
                net.self_train(1, self.log_step_size, self.net_learning_rate, input_data, target_data)

            self.nets.append(net)

    def test_robustness(self):
        #test_for_fixpoints(self.fixpoint_counters, self.nets, self.id_functions)

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

                input_data = original_net_clone.input_weight_matrix()
                target_data = original_net_clone.create_target_weights(input_data)

                changed_weights = copy.deepcopy(input_data)
                for k in range(len(input_data)):
                    changed_weights[k][0] = changed_weights[k][0] + pow(10, -j)

                # Testing if the new net is still an identity function after applying noise
                still_id_func = is_identity_function(original_net_clone, changed_weights, target_data, zero_epsilon)

                # If the net is still an id. func. after applying the first run of noise, continue to apply it until otherwise
                while still_id_func and data[i][j] <= 1000:
                    data[i][j] += 1

                    input_data = original_net_clone.input_weight_matrix()
                    original_net_clone = original_net_clone.self_application(input_data, 1, self.log_step_size)
                    
                    #new_weights = original_net_clone.create_target_weights(changed_weights)
                    #original_net_clone = original_net_clone.apply_weights(original_net_clone, new_weights)

                    still_id_func = is_identity_function(original_net_clone, input_data, target_data, zero_epsilon)

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


""" ----------------------------------------------- Running the experiments ----------------------------------------------- """


def run_ST_experiment(population_size, batch_size, net_input_size, net_hidden_size, net_out_size, net_learning_rate,
                      epochs, runs, run_name, name_hash):
    experiments = {}

    check_folder("self_training")

    # Running the experiments
    for i in range(runs):
        ST_directory_name = f"experiments/self_training/{run_name}_run_{i}_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"

        ST_experiment = SelfTrainExperiment(
            population_size,
            batch_size,
            net_input_size,
            net_hidden_size,
            net_out_size,
            net_learning_rate,
            epochs,
            ST_directory_name
        )
        pickle.dump(ST_experiment, open(f"{ST_directory_name}/full_experiment_pickle.p", "wb"))
        experiments[i] = ST_experiment

    # Building a summary of all the runs
    directory_name = f"experiments/self_training/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"
    os.mkdir(directory_name)

    summary_pre_title = "ST"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title)


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


def run_soup_experiment(population_size, attack_chance, net_input_size, net_hidden_size, net_out_size,
                        net_learning_rate, epochs, batch_size, runs, run_name, name_hash, ST_steps, train_nets):
    experiments = {}
    fixpoints_percentages = []

    check_folder("soup")

    # Running the experiments
    for i in range(runs):
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
    directory_name = f"experiments/soup/summary_{run_name}_{runs}_runs_{str(population_size)}_nets_{epochs}_epochs_{str(name_hash)}"
    os.mkdir(directory_name)

    # Building a summary of all the runs
    summary_pre_title = "soup"
    summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title)
    SA_steps = None
    summary_fixpoint_percentage(runs, epochs, fixpoints_percentages, ST_steps, SA_steps, directory_name,
                                population_size)


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


""" ----------------------------------------- Methods for summarizing the experiments ------------------------------------------ """


def summary_fixpoint_experiment(runs, population_size, epochs, experiments, net_learning_rate, directory_name,
                                summary_pre_title):
    avg_fixpoint_counters = {
        "avg_identity_func": 0,
        "avg_divergent": 0,
        "avg_fix_zero": 0,
        "avg_fix_weak": 0,
        "avg_fix_sec": 0,
        "avg_other_func": 0
    }

    for i in range(len(experiments)):
        fixpoint_counters = experiments[i].fixpoint_counters

        avg_fixpoint_counters["avg_identity_func"] += fixpoint_counters["identity_func"]
        avg_fixpoint_counters["avg_divergent"] += fixpoint_counters["divergent"]
        avg_fixpoint_counters["avg_fix_zero"] += fixpoint_counters["fix_zero"]
        avg_fixpoint_counters["avg_fix_weak"] += fixpoint_counters["fix_weak"]
        avg_fixpoint_counters["avg_fix_sec"] += fixpoint_counters["fix_sec"]
        avg_fixpoint_counters["avg_other_func"] += fixpoint_counters["other_func"]

    # Calculating the average for each fixpoint
    avg_fixpoint_counters.update((x, y / len(experiments)) for x, y in avg_fixpoint_counters.items())

    # Checking where the data is coming from to have a relevant title in the plot.
    if summary_pre_title not in ["ST", "SA", "soup", "mixed", "robustness"]:
        summary_pre_title = ""

    # Plotting the summary
    source_checker = "summary"
    exp_details = f"{summary_pre_title}: {runs} runs & {epochs} epochs each."
    bar_chart_fixpoints(avg_fixpoint_counters, population_size, directory_name, net_learning_rate, exp_details,
                        source_checker)


def summary_fixpoint_percentage(runs, epochs, fixpoints_percentages, ST_steps, SA_steps, directory_name,
                                population_size):
    fixpoints_percentages = [round(fixpoints_percentages[i] / runs, 1) for i in range(len(fixpoints_percentages))]

    # Plotting summary
    if "soup" in directory_name:
        line_chart_fixpoints(fixpoints_percentages, epochs / ST_steps, ST_steps, SA_steps, directory_name,
                             population_size)
    else:
        line_chart_fixpoints(fixpoints_percentages, epochs, ST_steps, SA_steps, directory_name, population_size)


""" --------------------------------------------------- Miscellaneous ---------------------------------------------------------- """


def check_folder(experiment_folder: String):
    if not os.path.isdir("experiments"): os.mkdir(f"experiments/")
    if not os.path.isdir(f"experiments/{experiment_folder}/"): os.mkdir(f"experiments/{experiment_folder}/")
