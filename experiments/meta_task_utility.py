import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
import sys
import platform

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize
from tqdm import tqdm

# noinspection DuplicatedCode
if platform.node() == 'CarbonX':
    debug = True
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@ Warning, Debugging Config@!!!!!! @")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
else:
    debug = False
    try:
        # noinspection PyUnboundLocalVariable
        if __package__ is None:
            DIR = Path(__file__).resolve().parent
            sys.path.insert(0, str(DIR.parent))
            __package__ = DIR.name
        else:
            DIR = None
    except NameError:
        DIR = None
        pass

from network import FixTypes as ft
from functionalities_test import test_for_fixpoints

WORKER = 10 if not debug else 0
debug = False
BATCHSIZE = 500 if not debug else 50
EPOCH = 50
VALIDATION_FRQ = 3 if not debug else 1
SELF_TRAIN_FRQ = 1 if not debug else 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True, parents=True)

if debug:
    torch.autograd.set_detect_anomaly(True)


class ToFloat:

    def __init__(self):
        pass

    def __call__(self, x):
        return x.to(torch.float32)


class AddTaskDataset(Dataset):
    def __init__(self, length=int(5e5)):
        super().__init__()
        self.length = length
        self.prng = np.random.default_rng()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ab = self.prng.normal(size=(2,)).astype(np.float32)
        return ab, ab.sum(axis=-1, keepdims=True)


def set_checkpoint(model, out_path, epoch_n, final_model=False):
    epoch_n = str(epoch_n)
    if not final_model:
        ckpt_path = Path(out_path) / 'ckpt' / f'{epoch_n.zfill(4)}_model_ckpt.tp'
    else:
        ckpt_path = Path(out_path) / f'trained_model_ckpt_e{epoch_n}.tp'
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(model, ckpt_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    py_store_path = Path(out_path) / 'exp_py.txt'
    if not py_store_path.exists():
        shutil.copy(__file__, py_store_path)
    return ckpt_path


def validate(checkpoint_path, ratio=0.1):
    checkpoint_path = Path(checkpoint_path)
    import torchmetrics

    # initialize metric
    validmetric = torchmetrics.Accuracy()
    ut = Compose([ToTensor(), ToFloat(), Resize((15, 15)), Flatten(start_dim=0)])

    try:
        datas = MNIST(str(DATA_PATH), transform=ut, train=False)
    except RuntimeError:
        datas = MNIST(str(DATA_PATH), transform=ut, train=False, download=True)
    valid_d = DataLoader(datas, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

    model = torch.load(checkpoint_path, map_location=DEVICE).eval()
    n_samples = int(len(valid_d) * ratio)

    with tqdm(total=n_samples, desc='Validation Run: ') as pbar:
        for idx, (valid_batch_x, valid_batch_y) in enumerate(valid_d):
            valid_batch_x, valid_batch_y = valid_batch_x.to(DEVICE), valid_batch_y.to(DEVICE)
            y_valid = model(valid_batch_x)

            # metric on current batch
            acc = validmetric(y_valid.cpu(), valid_batch_y.cpu())
            pbar.set_postfix_str(f'Acc: {acc}')
            pbar.update()
            if idx == n_samples:
                break

    # metric on all batches using custom accumulation
    acc = validmetric.compute()
    tqdm.write(f"Avg. accuracy on all data: {acc}")
    return acc


def new_storage_df(identifier, weight_count):
    if identifier == 'train':
        return pd.DataFrame(columns=['Epoch', 'Batch', 'Metric', 'Score'])
    elif identifier == 'weights':
        return pd.DataFrame(columns=['Epoch', 'Weight', *(f'weight_{x}' for x in range(weight_count))])


def checkpoint_and_validate(model, out_path, epoch_n, final_model=False):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    result = validate(ckpt_path)
    return result


def plot_training_particle_types(path_to_dataframe):
    plt.clf()
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=False)
    # Set up figure
    fig, ax = plt.subplots()  # initializes figure and plots
    data = df.loc[df['Metric'].isin(ft.all_types())]
    fix_types = data['Metric'].unique()
    data = data.pivot(index='Epoch', columns='Metric', values='Score').reset_index().fillna(0)
    _ = plt.stackplot(data['Epoch'], *[data[fixtype] for fixtype in fix_types], labels=fix_types.tolist())

    ax.set(ylabel='Particle Count', xlabel='Epoch')
    ax.set_title('Particle Type Count')

    fig.legend(loc="center right", title='Particle Type', bbox_to_anchor=(0.85, 0.5))
    plt.tight_layout()
    if debug:
        plt.show()
    else:
        plt.savefig(Path(path_to_dataframe.parent / 'training_particle_type_lp.png'), dpi=300)


def plot_training_result(path_to_dataframe):
    plt.clf()
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=False)

    # Set up figure
    fig, ax1 = plt.subplots()  # initializes figure and plots
    ax2 = ax1.twinx()  # applies twinx to ax2, which is the second y-axis.

    # plots the first set of data
    data = df[(df['Metric'] == 'Task Loss') | (df['Metric'] == 'Self Train Loss')].groupby(['Epoch', 'Metric']).mean()
    palette = sns.color_palette()[1:data.reset_index()['Metric'].unique().shape[0]+1]
    sns.lineplot(data=data.groupby(['Epoch', 'Metric']).mean(), x='Epoch', y='Score', hue='Metric',
                 palette=palette, ax=ax1)

    # plots the second set of data
    data = df[(df['Metric'] == 'Test Accuracy') | (df['Metric'] == 'Train Accuracy')]
    palette = sns.color_palette()[len(palette)+1:data.reset_index()['Metric'].unique().shape[0] + len(palette)+1]
    sns.lineplot(data=data, x='Epoch', y='Score', marker='o', hue='Metric', palette=palette)

    ax1.set(yscale='log', ylabel='Losses')
    ax1.set_title('Training Lineplot')
    ax2.set(ylabel='Accuracy')

    fig.legend(loc="center right", title='Metric', bbox_to_anchor=(0.85, 0.5))
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.tight_layout()
    if debug:
        plt.show()
    else:
        plt.savefig(Path(path_to_dataframe.parent / 'training_lineplot.png'), dpi=300)


def plot_network_connectivity_by_fixtype(path_to_trained_model):
    m = torch.load(path_to_trained_model, map_location=torch.device('cpu')).eval()
    # noinspection PyProtectedMember
    particles = list(m.particles)
    df = pd.DataFrame(columns=['type', 'layer', 'neuron', 'name'])

    for prtcl in particles:
        l, c, w = [float(x) for x in re.sub("[^0-9|_]", "", prtcl.name).split('_')]
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l-1, w, prtcl.name)
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l, c, prtcl.name)
    for layer in list(df['layer'].unique()):
        # Rescale
        divisor = df.loc[(df['layer'] == layer), 'neuron'].max()
        df.loc[(df['layer'] == layer), 'neuron'] /= divisor

    tqdm.write(f'Connectivity Data gathered')
    for n, fixtype in enumerate(ft.all_types()):
        if df[df['type'] == fixtype].shape[0] > 0:
            plt.clf()
            ax = sns.lineplot(y='neuron', x='layer', hue='name', data=df[df['type'] == fixtype],
                              legend=False, estimator=None, lw=1)
            _ = sns.lineplot(y=[0, 1], x=[-1, df['layer'].max()], legend=False, estimator=None, lw=0)
            ax.set_title(fixtype)
            lines = ax.get_lines()
            for line in lines:
                line.set_color(sns.color_palette()[n])
            if debug:
                plt.show()
            else:
                plt.savefig(Path(path_to_trained_model.parent / f'net_connectivity_{fixtype}.png'), dpi=300)
            tqdm.write(f'Connectivity plottet: {fixtype} - n = {df[df["type"] == fixtype].shape[0] // 2}')
        else:
            tqdm.write(f'No Connectivity {fixtype}')


def run_particle_dropout_test(model_path):
    diff_store_path = model_path.parent / 'diff_store.csv'
    latest_model = torch.load(model_path, map_location=DEVICE).eval()
    prtcl_dict = defaultdict(lambda: 0)
    _ = test_for_fixpoints(prtcl_dict, list(latest_model.particles))
    tqdm.write(str(dict(prtcl_dict)))
    diff_df = pd.DataFrame(columns=['Particle Type', 'Accuracy', 'Diff'])

    acc_pre = validate(model_path, ratio=1).item()
    diff_df.loc[diff_df.shape[0]] = ('All Organism', acc_pre, 0)

    for fixpoint_type in ft.all_types():
        new_model = torch.load(model_path, map_location=DEVICE).eval().replace_with_zero(fixpoint_type)
        if [x for x in new_model.particles if x.is_fixpoint == fixpoint_type]:
            new_ckpt = set_checkpoint(new_model, model_path.parent, fixpoint_type, final_model=True)
            acc_post = validate(new_ckpt, ratio=1).item()
            acc_diff = abs(acc_post - acc_pre)
            tqdm.write(f'Zero_ident diff = {acc_diff}')
            diff_df.loc[diff_df.shape[0]] = (fixpoint_type, acc_post, acc_diff)

    diff_df.to_csv(diff_store_path, mode='a', header=not diff_store_path.exists(), index=False)
    return diff_store_path


def plot_dropout_stacked_barplot(mdl_path, diff_store_path):

    diff_df = pd.read_csv(diff_store_path)
    particle_dict = defaultdict(lambda: 0)
    latest_model = torch.load(mdl_path, map_location=DEVICE).eval()
    _ = test_for_fixpoints(particle_dict, list(latest_model.particles))
    tqdm.write(str(dict(particle_dict)))
    plt.clf()
    fig, ax = plt.subplots(ncols=2)
    colors = sns.color_palette()[1:diff_df.shape[0]+1]
    _ = sns.barplot(data=diff_df, y='Accuracy', x='Particle Type', ax=ax[0], palette=colors)

    ax[0].set_title('Accuracy after particle dropout')
    ax[0].set_xlabel('Particle Type')

    ax[1].pie(particle_dict.values(), labels=particle_dict.keys(), colors=list(reversed(colors)), )
    ax[1].set_title('Particle Count')

    plt.tight_layout()
    if debug:
        plt.show()
    else:
        plt.savefig(Path(diff_store_path.parent / 'dropout_stacked_barplot.png'), dpi=300)


def run_particle_dropout_and_plot(model_path):
    diff_store_path = run_particle_dropout_test(model_path)
    plot_dropout_stacked_barplot(model_path, diff_store_path)


def flat_for_store(parameters):
    return (x.item() for y in parameters for x in y.detach().flatten())


def train_self_replication(model, st_stps, **kwargs) -> dict:
    self_train_loss = model.combined_self_train(st_stps, **kwargs)
    # noinspection PyUnboundLocalVariable
    stp_log = dict(Metric='Self Train Loss', Score=self_train_loss.item())
    return stp_log


def train_task(model, optimizer, loss_func, btch_x, btch_y) -> (dict, torch.Tensor):
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    btch_x, btch_y = btch_x.to(DEVICE), btch_y.to(DEVICE)
    y_prd = model(btch_x)

    loss = loss_func(y_prd, btch_y.to(torch.long))
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    stp_log = dict(Metric='Task Loss', Score=loss.item())

    return stp_log, y_prd


if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
