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


class DataPlotter:

    def __init__(self, path=None):
        self.path = path or os.getcwd()
        pass

    def search_and_apply(self, plotting_function, files_to_look_for=None, absolut_file_or_folder=None):
        absolut_file_or_folder, files_to_look_for = self.path or absolut_file_or_folder, list() or files_to_look_for
        if os.path.isdir(absolut_file_or_folder):
            for sub_file_or_folder in os.scandir(absolut_file_or_folder):
                self.search_and_apply(plotting_function, files_to_look_for=files_to_look_for,
                                      absolut_file_or_folder=sub_file_or_folder.path)
        elif absolut_file_or_folder.endswith('.dill'):
            file_or_folder = os.path.split(absolut_file_or_folder)[-1]
            if file_or_folder in files_to_look_for and not os.path.exists(
                    '{}.html'.format(absolut_file_or_folder[:-5])):
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
                # This was either another FilyType or Plot.html already exists.
                pass


if __name__ == '__main__':
    plotter = DataPlotter
    pass
