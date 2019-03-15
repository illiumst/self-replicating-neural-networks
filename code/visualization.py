import os

from experiment import Experiment
# noinspection PyUnresolvedReferences
from soup import Soup

from argparse import ArgumentParser
import numpy as np

import plotly as pl
import plotly.graph_objs as go

import colorlover as cl

import dill

from sklearn.manifold.t_sne import TSNE, PCA


def build_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--in_file', nargs=1, type=str)
    arg_parser.add_argument('-o', '--out_file', nargs='?', default='out', type=str)
    return arg_parser.parse_args()


def build_from_soup_or_exp(soup):
    particles = soup.historical_particles
    particle_list = []
    for particle in particles.values():
        particle_dict = dict(
            trajectory=[event['weights'] for event in particle],
            time=[event['time'] for event in particle],
            action=[event.get('action', None) for event in particle],
            counterpart=[event.get('counterpart', None) for event in particle]
        )
        if any([x is not None for x in particle_dict['counterpart']]):
            print('counterpart')
        particle_list.append(particle_dict)
    return particle_list


def plot_latent_trajectories(soup_or_experiment, filename='latent_trajectory_plot'):
    assert isinstance(soup_or_experiment, (Experiment, Soup))
    bupu = cl.scales['11']['div']['RdYlGn']
    data_dict = build_from_soup_or_exp(soup_or_experiment)
    scale = cl.interp(bupu, len(data_dict)+1)  # Map color scale to N bins

    # Fit the mebedding space
    transformer = TSNE()
    for particle_dict in data_dict:
        array = np.asarray([np.hstack([x.flatten() for x in timestamp]).flatten()
                            for timestamp in particle_dict['trajectory']])
        particle_dict['trajectory'] = array
        transformer.fit(array)

    # Transform data accordingly and plot it
    data = []
    for p_id, particle_dict in enumerate(data_dict):
        transformed = transformer._fit(np.asarray(particle_dict['trajectory']))
        line_trace = go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            text='Hovertext goes here'.format(),
            line=dict(color=scale[p_id]),
            # legendgroup='Position -{}'.format(pos),
            name='Particle - {}'.format(p_id),
            showlegend=True,
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


def plot_latent_trajectories_3D(soup_or_experiment, filename='plot'):
    def norm(val, a=0, b=0.25):
        return (val - a) / (b - a)

    data_list = build_from_soup_or_exp(soup_or_experiment)
    if not data_list:
        return

    base_scale = cl.scales['9']['div']['RdYlGn']
    # base_scale = cl.scales['9']['qual']['Set1']
    scale = cl.interp(base_scale, len(data_list)+1)  # Map color scale to N bins

    # Fit the embedding space
    transformer = PCA(n_components=2)

    array = []
    for particle_dict in data_list:
        array.append(particle_dict['trajectory'])

    transformer.fit(np.vstack(array))

    # Transform data accordingly and plot it
    data = []
    for p_id, particle_dict in enumerate(data_list):
        transformed = transformer.transform(particle_dict['trajectory'])
        line_trace = go.Scatter3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=np.asarray(particle_dict['time']),
            text='Particle: {}<br> It had {} lifes.'.format(p_id, len(particle_dict['trajectory'])),
            line=dict(
                color=scale[p_id],
                width=4
            ),
            # legendgroup='Particle - {}'.format(p_id),
            name='Particle -{}'.format(p_id),
            showlegend=False,
            hoverinfo='text',
            mode='lines')

        line_start = go.Scatter3d(mode='markers', x=[transformed[0, 0]], y=[transformed[0, 1]],
                                  z=np.asarray(particle_dict['time'][0]),
                                  marker=dict(
                                      color='rgb(255, 0, 0)',
                                      size=4
                                  ),
                                  showlegend=False
                                  )

        line_end = go.Scatter3d(mode='markers', x=[transformed[-1, 0]], y=[transformed[-1, 1]],
                                z=np.asarray(particle_dict['time'][-1]),
                                marker=dict(
                                    color='rgb(0, 0, 0)',
                                    size=4
                                ),
                                showlegend=False
                                )

        data.extend([line_trace, line_start, line_end])

    axis_layout = dict(gridcolor='rgb(255, 255, 255)',
                       gridwidth=3,
                       zerolinecolor='rgb(255, 255, 255)',
                       showbackground=True,
                       backgroundcolor='rgb(230, 230,230)',
                       titlefont=dict(
                           color='black',
                           size=30
                           )
                       )

    layout = go.Layout(scene=dict(
        # aspectratio=dict(x=2, y=2, z=2),
        xaxis=dict(title='Transformed X', **axis_layout),
        yaxis=dict(title='Transformed Y', **axis_layout),
        zaxis=dict(title='Epoch', **axis_layout)),
        # title='{} - Latent Trajectory Movement'.format('Soup'),

        width=1024, height=1024,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=data, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename, validate=True)
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


def search_and_apply(absolut_file_or_folder, plotting_function, files_to_look_for=[]):
    if os.path.isdir(absolut_file_or_folder):
        for sub_file_or_folder in os.scandir(absolut_file_or_folder):
            search_and_apply(sub_file_or_folder.path, plotting_function, files_to_look_for=files_to_look_for)
    elif absolut_file_or_folder.endswith('.dill'):
        file_or_folder = os.path.split(absolut_file_or_folder)[-1]
        if file_or_folder in files_to_look_for and not os.path.exists('{}.html'.format(absolut_file_or_folder[:-5])):
            print('Apply Plotting function "{func}" on file "{file}"'.format(func=plotting_function.__name__,
                                                                             file=absolut_file_or_folder)
                  )
            with open(absolut_file_or_folder, 'rb') as in_f:
                exp = dill.load(in_f)
            try:
                plotting_function(exp, filename='{}.html'.format(absolut_file_or_folder[:-5]))
            except ValueError:
                pass
            except AttributeError:
                pass
        else:
            # This was either another FilyType or Plot.html alerady exists.
            pass


if __name__ == '__main__':
    args = build_args()
    in_file = args.in_file[0]
    out_file = args.out_file

    search_and_apply(in_file, plot_latent_trajectories_3D, ["trajectorys.dill", "soup.dill"])
