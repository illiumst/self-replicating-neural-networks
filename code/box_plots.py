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


def plot_box(exp: Experiment, filename='histogram_plot'):
    # catagorical
    ryb = cl.scales['10']['div']['RdYlBu']

    data = []

    for d in range(exp.depth):
        names = ['D 10e-{}'.format(d)] * exp.trials
        data.extend(names)

    trace_list = []

    vergence_box = go.Box(
        y=exp.ys,
        x=data,
        name='Time to Vergence',
        boxpoints=False,
        showlegend=True,
        marker=dict(
            color=ryb[3]
        ),
    )
    fixpoint_box = go.Box(
        y=exp.zs,
        x=data,
        name='Time as Fixpoint',
        boxpoints=False,
        showlegend=True,
        marker=dict(
            color=ryb[-1]
        ),
    )

    trace_list.extend([vergence_box, fixpoint_box])

    layout = dict(title='{} Histogram Plot'.format('Experiment Name Penis'),
                  boxmode='group',
                  boxgap=0,
                  # barmode='group',
                  bargap=0,
                  xaxis=dict(showgrid=False,
                             zeroline=True,
                             tickangle=0,
                             showticklabels=True),
                  yaxis=dict(
                      title='Occurences',
                      zeroline=False)
                  # height=400, width=400,
                  # margin=dict(l=20, r=20, t=20, b=20)
                  )

    fig = go.Figure(data=trace_list, layout=layout)
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

    search_and_apply(in_file, plot_box, files_to_look_for=['experiment.dill'])
    # , 'all_names.dill', 'all_notable_nets.dill'])
