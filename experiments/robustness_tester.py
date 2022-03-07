import pandas as pd
import torch
import random
import copy

from tqdm import tqdm
from functionalities_test import (is_identity_function, is_zero_fixpoint, test_for_fixpoints, is_divergent,
                                  FixTypes as FT)
from network import Net
from torch.nn import functional as F
import seaborn as sns
from matplotlib import pyplot as plt


def prng():
    return random.random()


def generate_perfekt_synthetic_fixpoint_weights():
    return torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                         [1.0], [0.0], [0.0], [0.0],
                         [1.0], [0.0]
                         ], dtype=torch.float32)

PALETTE = 10 * (
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#e41a1c",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#888888",
    "#a6cee3",
    "#b2df8a",
    "#cab2d6",
    "#fb9a99",
    "#fdbf6f",
)


def test_robustness(model_path, noise_levels=10, seeds=10, log_step_size=10):
    model = torch.load(model_path, map_location='cpu')
    networks = [x for x in model.particles if x.is_fixpoint == FT.identity_func]
    time_to_vergence = [[0 for _ in range(noise_levels)] for _ in range(len(networks))]
    time_as_fixpoint = [[0 for _ in range(noise_levels)] for _ in range(len(networks))]
    row_headers = []

    df = pd.DataFrame(columns=['setting', 'Noise Level', 'Self Train Steps', 'absolute_loss',
                               'Time to convergence', 'Time as fixpoint'])
    with tqdm(total=(seeds * noise_levels * len(networks))) as pbar:
        for setting, fixpoint in enumerate(networks):  # 1 / n
            row_headers.append(fixpoint.name)
            for seed in range(seeds):  # n / 1
                for noise_level in range(noise_levels):
                    steps = 0
                    clone = Net(fixpoint.input_size, fixpoint.hidden_size, fixpoint.out_size,
                                f"{fixpoint.name}_clone_noise_1e-{noise_level}")
                    clone.load_state_dict(copy.deepcopy(fixpoint.state_dict()))
                    clone = clone.apply_noise(pow(10, -noise_level))

                    while not is_zero_fixpoint(clone) and not is_divergent(clone):
                        # -> before
                        clone_weight_pre_application = clone.input_weight_matrix()
                        target_data_pre_application = clone.create_target_weights(clone_weight_pre_application)

                        clone.self_application(1, log_step_size)
                        time_to_vergence[setting][noise_level] += 1
                        # -> after
                        clone_weight_post_application = clone.input_weight_matrix()
                        target_data_post_application = clone.create_target_weights(clone_weight_post_application)

                        absolute_loss = F.l1_loss(target_data_pre_application, target_data_post_application).item()

                        if is_identity_function(clone):
                            time_as_fixpoint[setting][noise_level] += 1
                            # When this raises a Type Error, we found a second order fixpoint!
                        steps += 1

                        df.loc[df.shape[0]] = [f'{setting}_{seed}', fr'$\mathregular{{10^{{-{noise_level}}}}}$',
                                               steps, absolute_loss,
                                               time_to_vergence[setting][noise_level],
                                               time_as_fixpoint[setting][noise_level]]
                    pbar.update(1)

    # Get the measuremts at the highest time_time_to_vergence
    df_sorted = df.sort_values('Self Train Steps', ascending=False).drop_duplicates(['setting', 'Noise Level'])
    df_melted = df_sorted.reset_index().melt(id_vars=['setting', 'Noise Level', 'Self Train Steps'],
                                             value_vars=['Time to convergence', 'Time as fixpoint'],
                                             var_name="Measurement",
                                             value_name="Steps").sort_values('Noise Level')

    df_melted.to_csv(model_path.parent / 'robustness_boxplot.csv', index=False)

    # Plotting
    # plt.rcParams.update({
    #    "text.usetex": True,
    #    "font.family": "sans-serif",
    #    "font.size": 12,
    #    "font.weight": 'bold',
    #    "font.sans-serif": ["Helvetica"]})
    plt.clf()
    sns.set(style='whitegrid', font_scale=1)
    _ = sns.boxplot(data=df_melted, y='Steps', x='Noise Level', hue='Measurement', palette=PALETTE)
    plt.tight_layout()

    # sns.set(rc={'figure.figsize': (10, 50)})
    # bx = sns.catplot(data=df[df['absolute_loss'] < 1], y='absolute_loss', x='application_step', kind='box',
    #                  col='noise_level', col_wrap=3, showfliers=False)

    filename = f"robustness_boxplot.png"
    filepath = model_path.parent / filename
    plt.savefig(str(filepath))
    plt.close('all')
    return time_as_fixpoint, time_to_vergence


if __name__ == "__main__":
    raise NotImplementedError('Get out of here!')
