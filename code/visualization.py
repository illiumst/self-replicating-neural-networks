import os
import re
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser
from distutils.util import strtobool

import numpy as np
import tensorflow as tf

import plotly as pl
from plotly import tools
import plotly.graph_objs as go

import dill


def build_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--in_file', nargs=1, type=str)
    arg_parser.add_argument('-o', '--out_file', nargs='?', default='out', type=str)
    return arg_parser.parse_args()


def numberFromStrings(string) -> list:
    numberfromstring = [int(x) for x in re.findall('\d+', string)]
    return numberfromstring


def visulize_as_tiled_subplot(plotting_tuple, filename='plot'):
    def norm(val, a=0, b=0.25):
        return (val - a) / (b - a)

    data = np.asarray(plotting_tuple)

    fig = tools.make_subplots(rows=1, cols=3,
                              subplot_titles=('Layers: 1', 'Layers: 2', 'Layers: 3'),
                              horizontal_spacing=0.05)

    for x in range(1, 4):
        # Only select Plots with x Layers
        scatter_slice = data[np.where(data[:, 2] == x)]
        # Only Select Plots with x Cells
        scatter_slice = scatter_slice[np.where(scatter_slice[:, 1] <= 10)]
        # Normalize colors
        colors = scatter_slice[:, 4]
        # colors = np.apply_along_axis(norm, 0, scatter_slice[:, 4])
        scatter = go.Scatter(x=scatter_slice[:, 3],
                             y=scatter_slice[:, 1],
                             hoverinfo='text',
                             text=['Absolute Loss:<br>{}'.format(val) for val in colors],
                             mode='markers',
                             showlegend=False,
                             marker=dict(size=10, color=colors, colorscale='Jet',
                                         # Only plot the colorscale once, use one for all
                                         showscale=True if x == 1 else False,
                                         cmax=0.25, cmin=0,
                                         colorbar=dict(y=0.5, x=1, tickmode='array', ticks='outside',
                                                       tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25],
                                                       ticktext=["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"]
                                                       )
                                         )
                             )
        fig.append_trace(scatter, 1, x,)
        # TODO: Layout Loop
        if x == 1:
            fig['layout']['yaxis{}'.format(x)].update(tickwidth=1, title='Number of Cells')
        if x == 2:
            fig['layout']['xaxis{}'.format(x)].update(tickwidth=1, title='Position -X')

    fig['layout'].update(title='{} - Mean Absolute Loss'.format(os.path.split('DESTINATION_OR_EXPERIMENT_NAME')[-1].upper()),
                         height=300, width=800, margin=dict(l=50, r=0, t=60, b=50))
    # import plotly.io as pio
    # pio.write_image(fig, filename)
    pl.offline.plot(fig, filename=filename)
    pass


def visulize_as_splatter3d(plotting_tuple, filename='plot'):
    # timesteps, cells, layers, positions, val
    _ , cells, layers, position, val = zip(*plotting_tuple)
    text = ['Cells: {}<br>Layers: {}<br>Position: {}<br>Mean(Min()): {}'.format(cells, layers, position, val)
            for _, cells, layers, position, val in plotting_tuple]

    data = [go.Scatter3d(x=cells, y=layers, z=position, text=text, hoverinfo='text', mode='markers',
                         marker=dict(color=val, colorscale='Jet', opacity=0.8,
                                     colorbar=dict(y=0.5, x=0.9, title="Mean(Min(Seeds))"))
                         )]
    layout = go.Layout(scene=dict(aspectratio=dict(x=2, y=2, z=1),
                                  xaxis=dict(tickwidth=1, title='Number of Cells'),
                                  yaxis=dict(tickwidth=1, title='Number of Layers'),
                                  zaxis=dict(tickwidth=1, title='Position -pX')),
                       margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)  # filename='3d-scatter_plot'


def compile_run_name(path: str) -> dict:
    """
    Retrieve all names, extract index positions and group by seeds.

    :param path: Path to the current TB folder of a sinle NN configuration
    :return: List of foldernames to filter for.
    """
    config_keys = ['run_seed', 'timesteps', 'index_position', 'cell_count', 'layers', 'cell_type']
    found_configurations = defaultdict(list)
    for dname in os.listdir(path):
        if os.path.isdir(os.path.join(path, dname)):
            this_config = {key: value for key, value in zip(config_keys, dname.split("_"))}
            found_configurations[this_config['index_position']].append(dname)

    return found_configurations



if __name__ == '__main__':
    args = build_args()
    in_file = args.in_file[0]
    out_file = args.out_file

    with open(in_file, 'rb') as dill_file:
        experiment = dill.load(dill_file)

    print('hi')
