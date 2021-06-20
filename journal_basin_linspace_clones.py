import copy
import itertools
from pathlib import Path
import random
import pickle
import pandas as pd
import numpy as np
import torch

from functionalities_test import is_identity_function, test_status
from journal_basins import SpawnExperiment, prng, mean_invariate_manhattan_distance
from network import Net

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

import seaborn as sns
from matplotlib import pyplot as plt


class SpawnLinspaceExperiment(SpawnExperiment):

    def spawn_and_continue(self, number_clones: int = None):
        number_clones = number_clones or self.nr_clones

        df = pd.DataFrame(
            columns=['parent', 'MAE_pre', 'MAE_post', 'MSE_pre', 'MSE_post', 'MIM_pre', 'MIM_post', 'noise',
                     'status_post'])

        # For every initial net {i} after populating (that is fixpoint after first epoch);
        pairwise_net_list = itertools.combinations(self.parents, 2)
        for net1, net2 in pairwise_net_list:
            # We set parent start_time to just before this epoch ended, so plotting is zoomed in. Comment out to
            # to see full trajectory (but the clones will be very hard to see).
            # Make one target to compare distances to clones later when they have trained.
            net1.start_time = self.ST_steps - 150
            net1_input_data = net1.input_weight_matrix()
            net1_target_data = net1.create_target_weights(net1_input_data)

            net2.start_time = self.ST_steps - 150
            net2_input_data = net2.input_weight_matrix()
            net2_target_data = net2.create_target_weights(net2_input_data)

            if is_identity_function(net1) and is_identity_function(net2):
                # Clone the fixpoint x times and add (+-)self.noise to weight-sets randomly;
                # To plot clones starting after first epoch (z=ST_steps), set that as start_time!
                # To make sure PCA will plot the same trajectory up until this point, we clone the
                # parent-net's weight history as well.
                in_between_weights = np.linspace(net1_target_data, net2_target_data, number_clones,endpoint=False)

                for j, in_between_weight in enumerate(in_between_weights):
                    clone = Net(net1.input_size, net1.hidden_size, net1.out_size,
                                name=f"{net1.name}_clone_{str(j)}", start_time=self.ST_steps)
                    clone.apply_weights(torch.as_tensor(in_between_weight))

                    clone.s_train_weights_history = copy.deepcopy(net1.s_train_weights_history)
                    clone.number_trained = copy.deepcopy(net1.number_trained)

                    # Pre Training distances (after noise application of course)
                    clone_pre_weights = clone.create_target_weights(clone.input_weight_matrix())
                    MAE_pre = MAE(net1_target_data, clone_pre_weights)
                    MSE_pre = MSE(net1_target_data, clone_pre_weights)
                    MIM_pre = mean_invariate_manhattan_distance(net1_target_data, clone_pre_weights)

                    # Then finish training each clone {j} (for remaining epoch-1 * ST_steps) ..
                    for _ in range(self.epochs - 1):
                        for _ in range(self.ST_steps):
                            clone.self_train(1, self.log_step_size, self.net_learning_rate)

                    # Post Training distances for comparison
                    clone_post_weights = clone.create_target_weights(clone.input_weight_matrix())
                    MAE_post = MAE(net1_target_data, clone_post_weights)
                    MSE_post = MSE(net1_target_data, clone_post_weights)
                    MIM_post = mean_invariate_manhattan_distance(net1_target_data, clone_post_weights)

                    # .. log to data-frame and add to nets for 3d plotting if they are fixpoints themselves.
                    test_status(clone)
                    if is_identity_function(clone):
                        print(f"Clone {j} (of net_{net1.name}) is fixpoint."
                              f"\nMSE({net1.name},{j}): {MSE_post}"
                              f"\nMAE({net1.name},{j}): {MAE_post}"
                              f"\nMIM({net1.name},{j}): {MIM_post}\n")
                        self.nets.append(clone)

                    df.loc[clone.name] = [net1.name, MAE_pre, MAE_post, MSE_pre, MSE_post, MIM_pre, MIM_post,
                                          self.noise, clone.is_fixpoint]

        for parent in self.parents:
            for _ in range(self.epochs - 1):
                for _ in range(self.ST_steps):
                    parent.self_train(1, self.log_step_size, self.net_learning_rate)
            

        self.df = df


if __name__ == '__main__':
    NET_INPUT_SIZE = 4
    NET_OUT_SIZE = 1

    # Define number of runs & name:
    ST_runs = 1
    ST_runs_name = "test-27"
    ST_steps = 2000
    ST_epochs = 2
    ST_log_step_size = 10

    # Define number of networks & their architecture
    nr_clones = 3
    ST_population_size = 3
    ST_net_hidden_size = 2
    ST_net_learning_rate = 0.04
    ST_name_hash = random.getrandbits(32)

    print(f"Running the Spawn experiment:")
    exp = SpawnLinspaceExperiment(
        population_size=ST_population_size,
        log_step_size=ST_log_step_size,
        net_input_size=NET_INPUT_SIZE,
        net_hidden_size=ST_net_hidden_size,
        net_out_size=NET_OUT_SIZE,
        net_learning_rate=ST_net_learning_rate,
        epochs=ST_epochs,
        st_steps=ST_steps,
        nr_clones=nr_clones,
        noise=None,
        directory=Path('output') / 'spawn_basin' / f'{ST_name_hash}' / f'linage'
    )
    df = exp.df

    directory = Path('output') / 'spawn_basin' / f'{ST_name_hash}' / 'linage'
    pickle.dump(exp, open(f"{directory}/experiment_pickle_{ST_name_hash}.p", "wb"))
    print(f"\nSaved experiment to {directory}.")

    # Boxplot with counts of nr_fixpoints, nr_other, nr_etc. on y-axis
    sns.countplot(data=df, x="noise", hue="status_post")
    plt.savefig(f"output/spawn_basin/{ST_name_hash}/fixpoint_status_countplot.png")

    # Catplot (either kind="point" or "box") that shows before-after training distances to parent
    mlt = df[["MIM_pre", "MIM_post", "noise"]].melt("noise", var_name="time", value_name='Average Distance')
    sns.catplot(data=mlt, x="time", y="Average Distance", col="noise", kind="point", col_wrap=5, sharey=False)
    plt.savefig(f"output/spawn_basin/{ST_name_hash}/clone_distance_catplot.png")
