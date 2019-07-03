import os
from collections import defaultdict

# noinspection PyUnresolvedReferences
from soup import Soup
from experiment import TaskExperiment

from argparse import ArgumentParser

import plotly as pl
import plotly.graph_objs as go

import colorlover as cl

import dill
import numpy as np


def build_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-i', '--in_file', nargs=1, type=str)
    arg_parser.add_argument('-o', '--out_file', nargs='?', default='out', type=str)
    return arg_parser.parse_args()


def line_plot(exp: TaskExperiment, filename='lineplot'):
    assert isinstance(exp, TaskExperiment), ' This has to be a TaskExperiment!'
    traces, data = [], defaultdict(list)

    color_scale = cl.scales['3']['div']['RdYlBu']

    # Sort data per Key
    for message in exp.log_messages:
        for key in message.keys():
            try:
                data[key].append(-0.1 if np.isnan(message[key]) or np.isinf(message[key]) else message[key])
            except:
                data[key].append(message[key])

    for line_id, key in enumerate(data.keys()):
        if key not in ['counters', 'id']:
            trace = go.Scatter(
                x=[x for x in range(len(data[key]))],
                y=data[key],
                name=key,
                line=dict(
                    color=color_scale[line_id],
                    width=5
                ),
            )

            traces.append(trace)
        else:
            continue

    layout = dict(xaxis=dict(title='Trains per self-application', titlefont=dict(size=20)),
                  yaxis=dict(title='Average amount of fixpoints found',
                             titlefont=dict(size=20),
                             # type='log',
                             # range=[0, 2]
                             ),
                  legend=dict(orientation='h', x=0.3, y=-0.3),
                  # height=800, width=800,
                  margin=dict(b=0)
                  )

    fig = go.Figure(data=traces, layout=layout)
    pl.offline.plot(fig, auto_open=True, filename=filename)
    pass


def search_and_apply(absolut_file_or_folder, plotting_function, files_to_look_for=None, override=False):
    # ToDo: Clean this Mess
    assert os.path.exists(absolut_file_or_folder), f'The given path does not exist! Given: {absolut_file_or_folder}'
    files_to_look_for = files_to_look_for or list()
    if os.path.isdir(absolut_file_or_folder):
        for sub_file_or_folder in os.scandir(absolut_file_or_folder):
            search_and_apply(sub_file_or_folder.path, plotting_function,
                             files_to_look_for=files_to_look_for, override=override)
    elif absolut_file_or_folder.endswith('.dill'):
        file_or_folder = os.path.split(absolut_file_or_folder)[-1]
        if file_or_folder in files_to_look_for or not files_to_look_for:
            if not os.path.exists('{}.html'.format(absolut_file_or_folder[:-5])) or override:
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
                # Plot.html already exists.
                pass
        else:
            # This was a wrong FilyType.
            pass


if __name__ == '__main__':
    args = build_args()
    in_file = args.in_file[0]
    out_file = args.out_file

    search_and_apply(in_file, line_plot, override=True)
