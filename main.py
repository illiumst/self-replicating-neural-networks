from experiments import run_ST_experiment, run_SA_experiment, run_soup_experiment, run_mixed_experiment, \
    run_robustness_experiment
import random


# TODO maybe add also SA to the soup

def run_experiments(run_ST, run_SA, run_soup, run_mixed, run_robustness):
    if run_ST:
        print(f"Running the ST experiment:")
        run_ST_experiment(ST_population_size, ST_log_step_size, NET_INPUT_SIZE, ST_net_hidden_size, NET_OUT_SIZE,
                          ST_net_learning_rate,
                          ST_epochs, ST_runs, ST_runs_name, ST_name_hash)
    if run_SA:
        print(f"\n Running the SA experiment:")
        run_SA_experiment(SA_population_size, SA_log_step_size, NET_INPUT_SIZE, SA_net_hidden_size, NET_OUT_SIZE,
                          SA_net_learning_rate, SA_runs, SA_runs_name, SA_name_hash,
                          SA_steps, SA_train_nets, SA_ST_steps)
    if run_soup:
        print(f"\n Running the soup experiment:")
        run_soup_experiment(soup_population_size, soup_attack_chance, NET_INPUT_SIZE, soup_net_hidden_size,
                            NET_OUT_SIZE, soup_net_learning_rate, soup_epochs, soup_log_step_size, soup_runs, soup_runs_name,
                            soup_name_hash, soup_ST_steps, soup_train_nets)
    if run_mixed:
        print(f"\n Running the mixed experiment:")
        run_mixed_experiment(mixed_population_size, NET_INPUT_SIZE, mixed_net_hidden_size, NET_OUT_SIZE,
                             mixed_net_learning_rate, mixed_train_nets, mixed_epochs, mixed_SA_steps,
                             mixed_ST_steps_between_SA, mixed_log_step_size, mixed_name_hash, mixed_total_runs, mixed_runs_name)
    if run_robustness:
        print(f"Running the robustness experiment:")
        run_robustness_experiment(rob_population_size, rob_log_step_size, NET_INPUT_SIZE, rob_net_hidden_size,
                                  NET_OUT_SIZE, rob_net_learning_rate, rob_ST_steps, rob_runs, rob_runs_name, rob_name_hash)
    if not run_ST and not run_SA and not run_soup and not run_mixed and not run_robustness:
        print(f"No experiments to be run.")


if __name__ == '__main__':
    # Constants:
    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    """ ------------------------------------- Self-training (ST) experiment ------------------------------------- """
    run_ST_experiment_bool = True

    # Define number of runs & name:
    ST_runs = 3
    ST_runs_name = "test-27"
    ST_epochs = 500
    ST_log_step_size = 5

    # Define number of networks & their architecture
    ST_population_size = 10
    ST_net_hidden_size = 2

    ST_net_learning_rate = 0.04

    ST_name_hash = random.getrandbits(32)

    """ ----------------------------------- Self-application (SA) experiment ----------------------------------- """

    run_SA_experiment_bool = True

    # Define number of runs, name, etc.:
    SA_runs_name = "test-17"
    SA_runs = 2
    SA_steps = 100
    SA_app_batch_size = 5
    SA_train_batch_size = 5
    SA_log_step_size = 5

    # Define number of networks & their architecture
    SA_population_size = 10
    SA_net_hidden_size = 2

    SA_net_learning_rate = 0.04

    # SA_train_nets has 3 possible values "no", "before_SA", "after_SA".
    SA_train_nets = "no"
    SA_ST_steps = 300

    SA_name_hash = random.getrandbits(32)

    """ -------------------------------------------- Soup experiment -------------------------------------------- """

    run_soup_experiment_bool = True

    # Define number of runs, name, etc.:
    soup_runs = 1
    soup_runs_name = "test-16"
    soup_epochs = 100
    soup_log_step_size = 5
    soup_ST_steps = 20
    # soup_SA_steps = 10

    # Define number of networks & their architecture
    soup_population_size = 5
    soup_net_hidden_size = 2
    soup_net_learning_rate = 0.04

    # soup_attack_chance in %
    soup_attack_chance = 10

    # not used yet: soup_train_nets has 3 possible values "no", "before_SA", "after_SA".
    soup_train_nets = "no"

    soup_name_hash = random.getrandbits(32)

    """ ------------------------------------------- Mixed experiment -------------------------------------------- """

    run_mixed_experiment_bool = True

    # Define number of runs, name, etc.:
    mixed_runs_name = "test-17"
    mixed_total_runs = 2

    # Define number of networks & their architecture
    mixed_population_size = 5
    mixed_net_hidden_size = 2

    mixed_epochs = 10
    # Set the <batch_size> to the same value as <ST_steps_between_SA> to see the weights plotted
    # ONLY after each epoch, and not after a certain amount of steps.
    mixed_log_step_size = 5
    mixed_ST_steps_between_SA = 50
    mixed_SA_steps = 4

    mixed_net_learning_rate = 0.04

    # mixed_train_nets has 2 possible values "before_SA", "after_SA".
    mixed_train_nets = "after_SA"

    mixed_name_hash = random.getrandbits(32)

    """ ----------------------------------------- Robustness experiment ----------------------------------------- """
    run_robustness_bool = True

    # Define number of runs & name:
    rob_runs = 3
    rob_runs_name = "test-07"
    rob_ST_steps = 500
    rob_log_step_size = 10

    # Define number of networks & their architecture
    rob_population_size = 6
    rob_net_hidden_size = 2

    rob_net_learning_rate = 0.04

    rob_name_hash = random.getrandbits(32)

    """ ---------------------------------------- Running the experiment ----------------------------------------- """

    run_experiments(run_ST_experiment_bool, run_SA_experiment_bool, run_soup_experiment_bool, run_mixed_experiment_bool,
                    run_robustness_bool)
