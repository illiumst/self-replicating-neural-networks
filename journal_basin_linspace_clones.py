import copy
import itertools
from pathlib import Path
import random
import pickle
import pandas as pd
import numpy as np
import torch

from functionalities_test import is_identity_function, test_status
from journal_basins import SpawnExperiment, mean_invariate_manhattan_distance
from network import Net

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


class SpawnLinspaceExperiment(SpawnExperiment):

    def spawn_and_continue(self, number_clones: int = None):
        number_clones = number_clones or self.nr_clones

        df = pd.DataFrame(
            columns=['clone', 'parent', 'parent2',
                     'MAE_pre', 'MAE_post',
                     'MSE_pre', 'MSE_post',
                     'MIM_pre', 'MIM_post',
                     'noise', 'status_pst'])

        # For every initial net {i} after populating (that is fixpoint after first epoch);
        # parent = self.parents[0]
        # parent_clone = clone = Net(parent.input_size, parent.hidden_size, parent.out_size,
        #                         name=f"{parent.name}_clone_{0}", start_time=self.ST_steps)
        # parent_clone.apply_weights(torch.as_tensor(parent.create_target_weights(parent.input_weight_matrix())))
        # parent_clone = parent_clone.apply_noise(self.noise)
        # self.parents.append(parent_clone)
        pairwise_net_list = list(itertools.combinations(self.parents, 2))
        for net1, net2 in pairwise_net_list:
            # We set parent start_time to just before this epoch ended, so plotting is zoomed in. Comment out to
            # to see full trajectory (but the clones will be very hard to see).
            # Make one target to compare distances to clones later when they have trained.
            net1.start_time = self.ST_steps - 150
            net1_input_data = net1.input_weight_matrix().detach()
            net1_target_data = net1.create_target_weights(net1_input_data).detach()

            net2.start_time = self.ST_steps - 150
            net2_input_data = net2.input_weight_matrix().detach()
            net2_target_data = net2.create_target_weights(net2_input_data).detach()

            if is_identity_function(net1) and is_identity_function(net2):
                # if True:
                # Clone the fixpoint x times and add (+-)self.noise to weight-sets randomly;
                # To plot clones starting after first epoch (z=ST_steps), set that as start_time!
                # To make sure PCA will plot the same trajectory up until this point, we clone the
                # parent-net's weight history as well.

                in_between_weights = np.linspace(net1_target_data, net2_target_data, number_clones, endpoint=False)
                # in_between_weights = np.logspace(net1_target_data, net2_target_data, number_clones, endpoint=False)

                for j, in_between_weight in enumerate(in_between_weights):
                    clone = Net(net1.input_size, net1.hidden_size, net1.out_size,
                                name=f"{net1.name}_{net2.name}_clone_{str(j)}", start_time=self.ST_steps + 100)
                    clone.apply_weights(torch.as_tensor(in_between_weight))

                    clone.s_train_weights_history = copy.deepcopy(net1.s_train_weights_history)
                    clone.number_trained = copy.deepcopy(net1.number_trained)

                    # Pre Training distances (after noise application of course)
                    clone_pre_weights = clone.create_target_weights(clone.input_weight_matrix()).detach()
                    MAE_pre = MAE(net1_target_data, clone_pre_weights)
                    MSE_pre = MSE(net1_target_data, clone_pre_weights)
                    MIM_pre = mean_invariate_manhattan_distance(net1_target_data, clone_pre_weights)

                    try:
                        # Then finish training each clone {j} (for remaining epoch-1 * ST_steps) ..
                        for _ in range(self.epochs - 1):
                            for _ in range(self.ST_steps):
                                clone.self_train(1, self.log_step_size, self.net_learning_rate)
                                if any([torch.isnan(x).any() for x in clone.parameters()]):
                                    raise ValueError
                    except ValueError:
                        print("Ran into nan in 'in beetween weights' array.")
                        df.loc[len(df)] = [j, net1.name, net2.name,
                                           MAE_pre, 0,
                                           MSE_pre, 0,
                                           MIM_pre, 0,
                                           self.noise, clone.is_fixpoint]
                        continue

                    # Post Training distances for comparison
                    clone_post_weights = clone.create_target_weights(clone.input_weight_matrix()).detach()
                    MAE_post = MAE(net1_target_data, clone_post_weights)
                    MSE_post = MSE(net1_target_data, clone_post_weights)
                    MIM_post = mean_invariate_manhattan_distance(net1_target_data, clone_post_weights)

                    # .. log to data-frame and add to nets for 3d plotting if they are fixpoints themselves.
                    test_status(clone)
                    if is_identity_function(clone):
                        print(f"Clone {j} (between {net1.name} and {net2.name}) is fixpoint."
                              f"\nMSE({net1.name},{j}): {MSE_post}"
                              f"\nMAE({net1.name},{j}): {MAE_post}"
                              f"\nMIM({net1.name},{j}): {MIM_post}\n")
                        self.nets.append(clone)

                    df.loc[len(df)] = [j, net1.name, net2.name,
                                       MAE_pre, MAE_post,
                                       MSE_pre, MSE_post,
                                       MIM_pre, MIM_post,
                                       self.noise, clone.is_fixpoint]

        for net1, net2 in pairwise_net_list:
            try:
                value = 'MAE'
                c_selector = [f'{value}_pre', f'{value}_post']
                values = df.loc[(df['parent'] == net1.name) & (df['parent2'] == net2.name)][c_selector]
                this_min, this_max = values.values.min(), values.values.max()
                df.loc[(df['parent'] == net1.name) &
                       (df['parent2'] == net2.name), c_selector] = (values - this_min) / (this_max - this_min)
            except ValueError:
                pass

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
    nr_clones = 25
    ST_population_size = 10
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
        noise=1e-8,
        directory=Path('output') / 'spawn_basin' / f'{ST_name_hash}' / f'linage'
    )
    df = exp.df

    directory = Path('output') / 'spawn_basin' / f'{ST_name_hash}' / 'linage'
    with (directory / f"experiment_pickle_{ST_name_hash}.p").open('wb') as f:
        pickle.dump(exp, f)
    print(f"\nSaved experiment to {directory}.")

    # Boxplot with counts of nr_fixpoints, nr_other, nr_etc. on y-axis
    # sns.countplot(data=df, x="noise", hue="status_post")
    # plt.savefig(f"output/spawn_basin/{ST_name_hash}/fixpoint_status_countplot.png")

    # Catplot (either kind="point" or "box") that shows before-after training distances to parent
    # mlt = df[["MIM_pre", "MIM_post", "noise"]].melt("noise", var_name="time", value_name='Average Distance')
    # sns.catplot(data=mlt, x="time", y="Average Distance", col="noise", kind="point", col_wrap=5, sharey=False)
    # plt.savefig(f"output/spawn_basin/{ST_name_hash}/clone_distance_catplot.png")

    # Pointplot with pre and after parent Distances
    import seaborn as sns
    from matplotlib import pyplot as plt, ticker

    # ptplt = sns.pointplot(data=exp.df, x='MAE_pre', y='MAE_post', join=False)
    ptplt = sns.scatterplot(x=exp.df['MAE_pre'], y=exp.df['MAE_post'])
    # ptplt.set(xscale='log', yscale='log')
    x0, x1 = ptplt.axes.get_xlim()
    y0, y1 = ptplt.axes.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    # This is the x=y line using transforms
    ptplt.plot(lims, lims, 'w', linestyle='dashdot', transform=ptplt.axes.transData)
    ptplt.plot([0, 1], [0, 1], ':k', transform=ptplt.axes.transAxes)
    ptplt.set(xlabel='Mean Absolute Distance before Self-Training',
              ylabel='Mean Absolute Distance after Self-Training')
    # ptplt.axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: round(float(x), 2)))
    # ptplt.xticks(rotation=45)
    #for ind, label in enumerate(ptplt.get_xticklabels()):
    #    if ind % 10 == 0:  # every 10th label is kept
    #        label.set_visible(True)
    #    else:
    #        label.set_visible(False)

    filepath = exp.directory / 'mim_dist_plot.pdf'
    plt.tight_layout()
    plt.savefig(filepath, dpi=600, format='pdf', bbox_inches='tight')
