import os

from argparse import ArgumentParser
import numpy as np

import plotly as pl
import plotly.graph_objs as go

import colorlover as cl

import dill

from sklearn.manifold.t_sne import TSNE


def build_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--in_file', nargs=1, type=str)
    arg_parser.add_argument('-o', '--out_file', nargs='?', default='out', type=str)
    return arg_parser.parse_args()


def plot_latent_trajectories(data_dict, filename='latent_trajectory_plot'):

    bupu = cl.scales['9']['seq']['BuPu']
    scale = cl.interp(bupu, len(data_dict)+1)  # Map color scale to N bins

    # Fit the mebedding space
    transformer = TSNE()
    for trajectory_id in data_dict:
        transformer.fit(np.asarray(data_dict[trajectory_id]))

    # Transform data accordingly and plot it
    data = []
    for trajectory_id in data_dict:
        transformed = transformer._fit(np.asarray(data_dict[trajectory_id]))
        line_trace = go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            text='Hovertext goes here'.format(),
            line=dict(color=scale[trajectory_id]),
            # legendgroup='Position -{}'.format(pos),
            # name='Position -{}'.format(pos),
            showlegend=False,
            # hoverinfo='text',
            mode='lines')
        line_start = go.Scatter(mode='markers', x=[transformed[0, 0]], y=[transformed[0, 1]],
                                marker=dict(
                                    color='rgb(255, 0, 0)',
                                    size=4
                                ),
                                showlegend=False
                                )
        line_end = go.Scatter(mode='markers', x=[transformed[-1, 0]], y=[transformed[-1, 1]],
                              marker=dict(
                                  color='rgb(0, 0, 0)',
                                  size=4
                              ),
                              showlegend=False
                              )
        data.extend([line_trace, line_start, line_end])

    layout = dict(title='{} - Latent Trajectory Movement'.format('Penis'),
                  height=800, width=800, margin=dict(l=0, r=0, t=0, b=0))
    # import plotly.io as pio
    # pio.write_image(fig, filename)
    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)
    pass


def plot_latent_trajectories_3D(data_dict, filename='plot'):
    def norm(val, a=0, b=0.25):
        return (val - a) / (b - a)

    bupu = cl.scales['9']['seq']['BuPu']
    scale = cl.interp(bupu, len(data_dict)+1)  # Map color scale to N bins

    max_len = max([len(trajectory) for trajectory in data_dict.values()])

    # Fit the mebedding space
    transformer = TSNE()
    for trajectory_id in data_dict:
        transformer.fit(data_dict[trajectory_id])

    # Transform data accordingly and plot it
    data = []
    for trajectory_id in data_dict:
        transformed = transformer._fit(np.asarray(data_dict[trajectory_id]))
        trace = go.Scatter3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=np.arange(transformed.shape[0]),
            text='Hovertext goes here'.format(),
            line=dict(color=scale[trajectory_id]),
            # legendgroup='Position -{}'.format(pos),
            # name='Position -{}'.format(pos),
            showlegend=False,
            # hoverinfo='text',
            mode='lines')
        data.append(trace)

    layout = go.Layout(scene=dict(aspectratio=dict(x=2, y=2, z=1),
                                  xaxis=dict(tickwidth=1, title='Transformed X'),
                                  yaxis=dict(tickwidth=1, title='transformed Y'),
                                  zaxis=dict(tickwidth=1, title='Epoch')),
                       title='{} - Latent Trajectory Movement'.format('Penis'),
                       width=800, height=800,
                       margin=dict(l=0, r=0, b=0, t=0))

    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)
    pass


def plot_histogram(bars_dict_list, filename='histogram_plot'):
    # catagorical
    ryb = cl.scales['10']['div']['RdYlBu']

    data = []
    for bar_id, bars_dict in bars_dict_list:
        hist = go.Histogram(
            histfunc="count",
            y=bars_dict.get('value', 14),
            x=bars_dict.get('name', 'gimme a name'),
            showlegend=False,
            marker=dict(
                color=ryb[bar_id]
            ),
        )
        data.append(hist)

    layout=dict(title='{} Histogram Plot'.format('Experiment Name Penis'),
                height=400, width=400, margin=dict(l=0, r=0, t=0, b=0))

    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)

    pass


def line_plot(line_dict_list, filename='lineplot'):
    # lines with standard deviation
    # Transform data accordingly and plot it
    data = []
    rdylgn = cl.scales['10']['div']['RdYlGn']
    rdylgn_background = [scale + (0.4,) for scale in cl.to_numeric(rdylgn)]
    for line_id, line_dict in enumerate(line_dict_list):
        name = line_dict.get('name', 'gimme a name')

        upper_bound = go.Scatter(
            name='Upper Bound',
            x=line_dict['x'],
            y=line_dict['upper_y'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor=rdylgn_background[line_id],
        )

        trace = go.Scatter(
            x=line_dict['x'],
            y=line_dict['main_y'],
            mode='lines',
            name=name,
            line=dict(color=line_id),
            fillcolor=rdylgn_background[line_id],
            fill='tonexty')

        lower_bound = go.Scatter(
            name='Lower Bound',
            x=line_dict['x'],
            y=line_dict['lower_y'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines')

        data.extend([upper_bound, trace, lower_bound])

    layout=dict(title='{} Line Plot'.format('Experiment Name Penis'),
                height=800, width=800, margin=dict(l=0, r=0, t=0, b=0))

    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)
    pass


if __name__ == '__main__':
    args = build_args()
    in_file = args.in_file[0]
    out_file = args.out_file

    with open(in_file, 'rb') as in_f:
        experiment = dill.load(in_f)
        plot_latent_trajectories_3D(experiment.data_storage)

    print('aha')
