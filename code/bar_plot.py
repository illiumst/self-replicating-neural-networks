import os

from experiment import Experiment
# noinspection PyUnresolvedReferences
from soup import Soup
from typing import List

from collections import defaultdict

from argparse import ArgumentParser
import numpy as np

import plotly as pl
import plotly.graph_objs as go

import colorlover as cl

import dill


def build_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--in_file', nargs=1, type=str)
    arg_parser.add_argument('-o', '--out_file', nargs='?', default='out', type=str)
    return arg_parser.parse_args()


def plot_histogram(bars_dict_list: List[dict], filename='histogram_plot'):
    # catagorical
    ryb = cl.scales['10']['div']['RdYlBu']

    data = []

    if bars_dict_list:
        keys = bars_dict_list[0].keys()
        keyDict = defaultdict(list)
    else:
        raise IOError('This List is empty, is this intended?')

    for key in keys:
        keyDict[key] = np.mean([bars_dict[key] for bars_dict in bars_dict_list])

    hist = go.Bar(
        y=[keyDict.get(key, 0) for key in keys],
        x=[key for key in keys],
        showlegend=False,
        marker=dict(
            color=[ryb[bar_id] for bar_id in range(len(keys))]
        ),
    )
    data.append(hist)

    layout = dict(title='{} Histogram Plot'.format('Experiment Name Penis'),
                  # height=400, width=400,
                  # margin=dict(l=20, r=20, t=20, b=20)
                  )

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
        if file_or_folder in files_to_look_for and not os.path.exists('{}.html'.format(file_or_folder[:-5])):
            print('Apply Plotting function "{func}" on file "{file}"'.format(func=plotting_function.__name__,
                                                                             file=absolut_file_or_folder)
                  )

            with open(absolut_file_or_folder, 'rb') as in_f:
                exp = dill.load(in_f)

            plotting_function(exp, filename='{}.html'.format(absolut_file_or_folder[:-5]))

        else:
            pass
            # This was not a file i should look for.
    else:
        # This was either another FilyType or Plot.html alerady exists.
        pass


if __name__ == '__main__':
    args = build_args()
    in_file = args.in_file[0]
    out_file = args.out_file

    search_and_apply(in_file, plot_histogram, files_to_look_for=['all_counters.dill'])
    # , 'all_names.dill', 'all_notable_nets.dill'])
