import numpy as np
import torch
import pandas as pd
import re
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from network import FixTypes


if __name__ == '__main__':
    p = Path(r'experiments\output\mn_st_200_4_alpha_100\trained_model_ckpt_e200.tp')
    m = torch.load(p, map_location=torch.device('cpu'))
    particles = [y for x in m._meta_layer_list for y in x.particles]
    df = pd.DataFrame(columns=['type', 'layer', 'neuron', 'name', 'color'])
    colors = []

    for particle in particles:
        l, c, w = [float(x) for x in re.sub("[^0-9|_]", "", particle.name).split('_')]

        color = sns.color_palette()[0 if particle.is_fixpoint == FixTypes.identity_func else 1]
        # color = 'orange' if particle.is_fixpoint == FixTypes.identity_func else 'blue'
        colors.append(color)
        df.loc[df.shape[0]] = (particle.is_fixpoint, l-1, w, particle.name, color)
        df.loc[df.shape[0]] = (particle.is_fixpoint, l, c, particle.name, color)
    for layer in list(df['layer'].unique()):
        divisor = df.loc[(df['layer'] == layer), 'neuron'].max()
        df.loc[(df['layer'] == layer), 'neuron'] /= divisor

    print('gathered')
    for n, (fixtype, color) in enumerate(zip([FixTypes.other_func, FixTypes.identity_func], ['blue', 'orange'])):
        plt.clf()
        ax = sns.lineplot(y='neuron', x='layer', hue='name', data=df[df['type'] == fixtype],
                          legend=False, estimator=None,
                          palette=[sns.color_palette()[n]] * (df[df['type'] == fixtype].shape[0]//2), lw=1)
        # ax.set(yscale='log', ylabel='Neuron')
        ax.set_title(fixtype)
        plt.show()
        print('plottet')

