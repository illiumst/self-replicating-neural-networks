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


def line_plot(names_exp_tuple, filename='lineplot'):

    names, line_dict_list = names_exp_tuple

    names = ['Weightwise', 'Aggregating', 'Recurrent']

    if False:
        data = []
        base_scale = cl.scales['10']['div']['RdYlGn']
        scale = cl.interp(base_scale, len(line_dict_list) + 1)  # Map color scale to N bins
        for ld_id, line_dict in enumerate(line_dict_list):
            for data_point in ['ys', 'zs']:
                trace = go.Scatter(
                    x=line_dict['xs'],
                    y=line_dict[data_point],
                    name='{} {}zero-fixpoints'.format(names[ld_id], 'non-' if data_point == 'zs' else ''),
                    line=dict(
                        # color=scale[ld_id],
                        width=5,
                        # dash='dash' if data_point == 'ys' else ''
                    ),
                )

                data.append(trace)
    if True:

        data = []
        base_scale = cl.scales['10']['div']['RdYlGn']
        scale = cl.interp(base_scale, len(line_dict_list) + 1)  # Map color scale to N bins
        for ld_id, line_dict in enumerate(line_dict_list):
            trace = go.Scatter(
                x=line_dict['xs'],
                y=line_dict['ys'],
                name=names[ld_id],
                line=dict(  # color=scale[ld_id],
                          width=5
                ),
            )

            data.append(trace)

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

            names_dill_location = os.path.join(*os.path.split(absolut_file_or_folder)[:-1], 'all_names.dill')
            with open(names_dill_location, 'rb') as in_f:
                names = dill.load(in_f)

            try:
                plotting_function((names, exp), filename='{}.html'.format(absolut_file_or_folder[:-5]))
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

    search_and_apply(in_file, line_plot, ["all_data.dill"])

