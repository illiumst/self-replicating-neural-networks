import pickle
import re
from collections import defaultdict
from pathlib import Path
import sys
import platform

import pandas as pd
import torchmetrics
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
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

from network import MetaNet, FixTypes as ft
from sparse_net import SparseNetwork
from functionalities_test import test_for_fixpoints

WORKER = 10 if not debug else 2
debug = False
BATCHSIZE = 500 if not debug else 50
EPOCH = 200
VALIDATION_FRQ = 3 if not debug else 1
SELF_TRAIN_FRQ = 1 if not debug else 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if debug:
    torch.autograd.set_detect_anomaly(True)


class ToFloat:

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
    return ckpt_path


def validate(checkpoint_path, ratio=0.1):
    checkpoint_path = Path(checkpoint_path)
    import torchmetrics

    # initialize metric
    validmetric = torchmetrics.Accuracy()
    ut = Compose([ToTensor(), ToFloat(), Resize((15, 15)), Flatten(start_dim=0)])

    try:
        datas = MNIST(str(data_path), transform=ut, train=False)
    except RuntimeError:
        datas = MNIST(str(data_path), transform=ut, train=False, download=True)
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
    palette = sns.color_palette()[0:data.reset_index()['Metric'].unique().shape[0]]
    sns.lineplot(data=data.groupby(['Epoch', 'Metric']).mean(), x='Epoch', y='Score', hue='Metric',
                 palette=palette, ax=ax1)

    # plots the second set of data
    data = df[(df['Metric'] == 'Test Accuracy') | (df['Metric'] == 'Train Accuracy')]
    palette = sns.color_palette()[len(palette):data.reset_index()['Metric'].unique().shape[0] + len(palette)]
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
    m = torch.load(path_to_trained_model, map_location=torch.device('cpu'))
    # noinspection PyProtectedMember
    particles = [y for x in m._meta_layer_list for y in x.particles]
    df = pd.DataFrame(columns=['type', 'layer', 'neuron', 'name'])

    for prtcl in particles:
        l, c, w = [float(x) for x in re.sub("[^0-9|_]", "", prtcl.name).split('_')]
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l-1, w, prtcl.name)
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l, c, prtcl.name)
    for layer in list(df['layer'].unique()):
        # Rescale
        divisor = df.loc[(df['layer'] == layer), 'neuron'].max()
        df.loc[(df['layer'] == layer), 'neuron'] /= divisor

    print('gathered')
    for n, fixtype in enumerate([ft.other_func, ft.identity_func]):
        plt.clf()
        ax = sns.lineplot(y='neuron', x='layer', hue='name', data=df[df['type'] == fixtype],
                          legend=False, estimator=None,
                          palette=[sns.color_palette()[n]] * (df[df['type'] == fixtype].shape[0]//2), lw=1)
        ax.set_title(fixtype)
        plt.show()
        print('plottet')


def run_particle_dropout_test(run_path):
    diff_store_path = run_path / 'diff_store.csv'
    prtcl_dict = defaultdict(lambda: 0)
    _ = test_for_fixpoints(prtcl_dict, list(latest_model.particles))
    tqdm.write(str(dict(prtcl_dict)))

    acc_pre = validate(model_path, ratio=1).item()
    diff_df = pd.DataFrame(columns=['Particle Type', 'Accuracy', 'Diff'])
    for fixpoint_type in ft.all_types():
        new_model = torch.load(model_path, map_location=DEVICE).eval().replace_with_zero(fixpoint_type)
        if [x for x in new_model.particles if x.is_fixpoint == fixpoint_type]:
            new_ckpt = set_checkpoint(new_model, model_path.parent, fixpoint_type, final_model=True)
            acc_post = validate(new_ckpt, ratio=1).item()
            acc_diff = abs(acc_post - acc_pre)
            tqdm.write(f'Zero_ident diff = {acc_diff}')
            diff_df.loc[diff_df.shape[0]] = (fixpoint_type, acc_post, acc_diff)

    diff_df.to_csv(diff_store_path, mode='a', header=not df_store_path.exists(), index=False)
    return diff_store_path


def plot_dropout_stacked_barplot(path_to_diff_df):
    diff_df = pd.read_csv(path_to_diff_df)
    particle_dict = defaultdict(lambda: 0)
    _ = test_for_fixpoints(particle_dict, list(latest_model.particles))
    tqdm.write(str(dict(particle_dict)))
    plt.clf()
    fig, ax = plt.subplots(ncols=2)
    colors = sns.color_palette()[:diff_df.shape[0]]
    barplot = sns.barplot(data=diff_df, y='Accuracy', x='Particle Type', palette=colors, ax=ax[0])
    # noinspection PyUnboundLocalVariable
    for idx, patch in enumerate(barplot.patches):
        if idx != 0:
            # we recenter the bar
            patch.set_x(patch.get_x() + idx * 0.035)

    ax[0].set_title('Accuracy after particle dropout')
    ax[0].set_xlabel('Accuracy')

    ax[1].pie(particle_dict.values(), labels=particle_dict.keys(), colors=colors, )
    ax[1].set_title('Particle Count for ')

    plt.tight_layout()
    if debug:
        plt.show()
    else:
        plt.savefig(Path(path_to_diff_df.parent / 'dropout_stacked_barplot.png'), dpi=300)


def run_particle_dropout_and_plot(run_path):
    diff_store_path = run_particle_dropout_test(run_path)
    plot_dropout_stacked_barplot(diff_store_path)


def flat_for_store(parameters):
    return (x.item() for y in parameters for x in y.detach().flatten())


if __name__ == '__main__':

    self_train = True
    training = True
    train_to_id_first = False
    train_to_task_first = False
    train_to_task_first_sequential = True

    tsk_threshold = 0.855
    self_train_alpha = 1
    batch_train_beta = 1
    weight_hidden_size = 3
    residual_skip = True
    n_seeds = 2

    data_path = Path('data')
    data_path.mkdir(exist_ok=True, parents=True)
    assert not (train_to_task_first and train_to_id_first)

    st_str = f'{"" if self_train else "no_"}st'
    a_str = f'_alpha_{self_train_alpha}' if self_train_alpha != 1 else ''
    res_str = f'{"" if residual_skip else "_no_res"}'
    # dr_str = f'{f"_dr_{dropout}" if dropout != 0 else ""}'
    id_str = f'{f"_StToId" if train_to_id_first else ""}'
    tsk_str = f'{f"_Tsk_{tsk_threshold}" if train_to_task_first else ""}'
    exp_path = Path('output') / f'mn_{st_str}_{EPOCH}_{weight_hidden_size}{a_str}{res_str}{id_str}{tsk_str}'

    for seed in range(n_seeds):
        seed_path = exp_path / str(seed)

        model_path = seed_path / '0000_trained_model.zip'
        df_store_path = seed_path / 'train_store.csv'
        weight_store_path = seed_path / 'weight_store.csv'
        srnn_parameters = dict()

        if training:
            utility_transforms = Compose([ToTensor(), ToFloat(), Resize((15, 15)), Flatten(start_dim=0)])
            try:
                dataset = MNIST(str(data_path), transform=utility_transforms)
            except RuntimeError:
                dataset = MNIST(str(data_path), transform=utility_transforms, download=True)
            d = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

            interface = np.prod(dataset[0][0].shape)
            sparse_metanet = SparseNetwork(interface, depth=5, width=6, out=10, residual_skip=residual_skip,
                                    weight_hidden_size=weight_hidden_size,).to(DEVICE)
            dense_metanet = MetaNet(interface, depth=5, width=6, out=10, residual_skip=residual_skip,
                                    weight_hidden_size=weight_hidden_size,).to(DEVICE)
            meta_weight_count = sum(p.numel() for p in next(dense_metanet.particles).parameters())

            loss_fn = nn.CrossEntropyLoss()
            dense_optimizer = torch.optim.SGD(dense_metanet.parameters(), lr=0.008, momentum=0.9)
            sparse_optimizer = torch.optim.SGD(sparse_metanet.parameters(), lr=0.008, momentum=0.9)

            train_store = new_storage_df('train', None)
            weight_store = new_storage_df('weights', meta_weight_count)
            init_tsk = train_to_task_first
            for epoch in tqdm(range(EPOCH), desc='MetaNet Train - Epochs'):
                is_validation_epoch = epoch % VALIDATION_FRQ == 0 if not debug else True
                is_self_train_epoch = epoch % SELF_TRAIN_FRQ == 0 if not debug else True
                sparse_metanet = sparse_metanet.train()
                dense_metanet = dense_metanet.train()
                if is_validation_epoch:
                    metric = torchmetrics.Accuracy()
                else:
                    metric = None
                init_st = train_to_id_first and not all(x.is_fixpoint == ft.identity_func for x in dense_metanet.particles)
                for batch, (batch_x, batch_y) in tqdm(enumerate(d), total=len(d), desc='MetaNet Train - Batch'):
                    # Self Train
                    if self_train and not init_tsk and (is_self_train_epoch or init_st):
                        # Transfer weights
                        sparse_metanet = sparse_metanet.replace_weights_by_particles(dense_metanet.particles)
                        # Zero your gradients for every batch!
                        sparse_optimizer.zero_grad()
                        self_train_loss = sparse_metanet.combined_self_train() * self_train_alpha
                        self_train_loss.backward()
                        # Adjust learning weights
                        sparse_optimizer.step()
                        step_log = dict(Epoch=epoch, Batch=batch,
                                        Metric='Self Train Loss', Score=self_train_loss.item())
                        train_store.loc[train_store.shape[0]] = step_log
                        # Transfer weights
                        dense_metanet = dense_metanet.replace_particles(sparse_metanet.particle_weights)
                    if not init_st:
                        # Zero your gradients for every batch!
                        dense_optimizer.zero_grad()
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        y_pred = dense_metanet(batch_x)
                        # loss = loss_fn(y, batch_y.unsqueeze(-1).to(torch.float32))
                        loss = loss_fn(y_pred, batch_y.to(torch.long)) * batch_train_beta
                        loss.backward()

                        # Adjust learning weights
                        dense_optimizer.step()

                        step_log = dict(Epoch=epoch, Batch=batch,
                                        Metric='Task Loss', Score=loss.item())
                        train_store.loc[train_store.shape[0]] = step_log
                        if is_validation_epoch:
                            metric(y_pred.cpu(), batch_y.cpu())

                    if batch >= 3 and debug:
                        break

                if is_validation_epoch:
                    dense_metanet = dense_metanet.eval()
                    if train_to_id_first <= epoch:
                        validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                              Metric='Train Accuracy', Score=metric.compute().item())
                        train_store.loc[train_store.shape[0]] = validation_log

                    accuracy = checkpoint_and_validate(dense_metanet, seed_path, epoch).item()
                    validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                          Metric='Test Accuracy', Score=accuracy)
                    train_store.loc[train_store.shape[0]] = validation_log
                    if init_tsk or (train_to_task_first and train_to_task_first_sequential):
                        init_tsk = accuracy <= tsk_threshold
                if init_st or is_validation_epoch:
                    counter_dict = defaultdict(lambda: 0)
                    # This returns ID-functions
                    _ = test_for_fixpoints(counter_dict, list(dense_metanet.particles))
                    for key, value in dict(counter_dict).items():
                        step_log = dict(Epoch=int(epoch), Batch=BATCHSIZE, Metric=key, Score=value)
                        train_store.loc[train_store.shape[0]] = step_log
                if init_st or is_validation_epoch:
                    for particle in dense_metanet.particles:
                        weight_log = (epoch, particle.name, *flat_for_store(particle.parameters()))
                        weight_store.loc[weight_store.shape[0]] = weight_log
                    train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists(), index=False)
                    weight_store.to_csv(weight_store_path, mode='a', header=not weight_store_path.exists(), index=False)
                    train_store = new_storage_df('train', None)
                    weight_store = new_storage_df('weights', meta_weight_count)

            dense_metanet.eval()

            counter_dict = defaultdict(lambda: 0)
            # This returns ID-functions
            _ = test_for_fixpoints(counter_dict, list(dense_metanet.particles))
            for key, value in dict(counter_dict).items():
                step_log = dict(Epoch=int(EPOCH), Batch=BATCHSIZE, Metric=key, Score=value)
                train_store.loc[train_store.shape[0]] = step_log
            accuracy = checkpoint_and_validate(dense_metanet, seed_path, EPOCH, final_model=True)
            validation_log = dict(Epoch=EPOCH, Batch=BATCHSIZE,
                                  Metric='Test Accuracy', Score=accuracy.item())
            for particle in dense_metanet.particles:
                weight_log = (EPOCH, particle.name, *(flat_for_store(particle.parameters())))
                weight_store.loc[weight_store.shape[0]] = weight_log

            train_store.loc[train_store.shape[0]] = validation_log
            train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists(), index=False)
            weight_store.to_csv(weight_store_path, mode='a', header=not weight_store_path.exists(), index=False)

        plot_training_result(df_store_path)
        plot_training_particle_types(df_store_path)

        try:
            model_path = next(seed_path.glob(f'*e{EPOCH}.tp'))
        except StopIteration:
            print('Model pattern did not trigger.')
            print(f'Search path was: {seed_path}:')
            print(f'Found Models are: {list(seed_path.rglob(".tp"))}')
            exit(1)
        latest_model = torch.load(model_path, map_location=DEVICE).eval()

        run_particle_dropout_and_plot(seed_path)
        plot_network_connectivity_by_fixtype(model_path)

    if n_seeds >= 2:
        pass
