import pickle
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

from network import MetaNet
from functionalities_test import test_for_fixpoints

WORKER = 10 if not debug else 2
BATCHSIZE = 500 if not debug else 50
EPOCH = 100 if not debug else 3
VALIDATION_FRQ = 5 if not debug else 1
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


def new_train_storage_df():
    return pd.DataFrame(columns=['Epoch', 'Batch', 'Metric', 'Score'])


def checkpoint_and_validate(model, out_path, epoch_n, final_model=False):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    result = validate(ckpt_path)
    return result


def plot_training_result(path_to_dataframe):
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=0)

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


if __name__ == '__main__':

    self_train = False
    training = False
    plotting = True
    particle_analysis = True
    as_sparse_network_test = True

    data_path = Path('data')
    data_path.mkdir(exist_ok=True, parents=True)

    run_path = Path('output') / 'mnist_self_train_100_NEW_STYLE'
    model_path = run_path / '0000_trained_model.zip'
    df_store_path = run_path / 'train_store.csv'

    if training:
        utility_transforms = Compose([ToTensor(), ToFloat(), Resize((15, 15)), Flatten(start_dim=0)])
        try:
            dataset = MNIST(str(data_path), transform=utility_transforms)
        except RuntimeError:
            dataset = MNIST(str(data_path), transform=utility_transforms, download=True)
        d = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

        interface = np.prod(dataset[0][0].shape)
        metanet = MetaNet(interface, depth=5, width=6, out=10).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(metanet.parameters(), lr=0.008, momentum=0.9)

        train_store = new_train_storage_df()
        for epoch in tqdm(range(EPOCH), desc='MetaNet Train - Epochs'):
            is_validation_epoch = epoch % VALIDATION_FRQ == 0 if not debug else True
            is_self_train_epoch = epoch % SELF_TRAIN_FRQ == 0 if not debug else True
            metanet = metanet.train()
            if is_validation_epoch:
                metric = torchmetrics.Accuracy()
            else:
                metric = None
            for batch, (batch_x, batch_y) in tqdm(enumerate(d), total=len(d), desc='MetaNet Train - Batch'):
                if self_train and is_self_train_epoch:
                    # Zero your gradients for every batch!
                    optimizer.zero_grad()
                    self_train_loss = metanet.combined_self_train()
                    self_train_loss.backward()
                    # Adjust learning weights
                    optimizer.step()
                    step_log = dict(Epoch=epoch, Batch=batch, Metric='Self Train Loss', Score=self_train_loss.item())
                    train_store.loc[train_store.shape[0]] = step_log

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                y = metanet(batch_x)
                # loss = loss_fn(y, batch_y.unsqueeze(-1).to(torch.float32))
                loss = loss_fn(y, batch_y.to(torch.long))
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                step_log = dict(Epoch=epoch, Batch=batch,
                                Metric='Task Loss', Score=loss.item())
                train_store.loc[train_store.shape[0]] = step_log
                if is_validation_epoch:
                    metric(y.cpu(), batch_y.cpu())

                if batch >= 3 and debug:
                    break

            if is_validation_epoch:
                metanet = metanet.eval()
                validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                      Metric='Train Accuracy', Score=metric.compute().item())
                train_store.loc[train_store.shape[0]] = validation_log

                accuracy = checkpoint_and_validate(metanet, run_path, epoch)
                validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                      Metric='Test Accuracy', Score=accuracy.item())
                train_store.loc[train_store.shape[0]] = validation_log
                if particle_analysis:
                    counter_dict = defaultdict(lambda: 0)
                    # This returns ID-functions
                    _ = test_for_fixpoints(counter_dict, list(metanet.particles))
                    for key, value in dict(counter_dict).items():
                        step_log = dict(Epoch=int(epoch), Batch=BATCHSIZE, Metric=key, Score=value)
                        train_store.loc[train_store.shape[0]] = step_log
                train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists())
                # train_store = new_train_storage_df()

        metanet.eval()
        accuracy = checkpoint_and_validate(metanet, run_path, EPOCH, final_model=True)
        validation_log = dict(Epoch=EPOCH, Batch=BATCHSIZE,
                              Metric='Test Accuracy', Score=accuracy.item())

        train_store.loc[train_store.shape[0]] = validation_log
        train_store.to_csv(df_store_path)

    if plotting:
        plot_training_result(df_store_path)

    if particle_analysis:
        model_path = next(run_path.glob(f'*e{EPOCH}.tp'))
        latest_model = torch.load(model_path, map_location=DEVICE).eval()
        counter_dict = defaultdict(lambda: 0)
        _ = test_for_fixpoints(counter_dict, list(latest_model.particles))
        tqdm.write(str(dict(counter_dict)))
        zero_ident = torch.load(model_path, map_location=DEVICE).eval().replace_with_zero('identity_func')
        zero_other = torch.load(model_path, map_location=DEVICE).eval().replace_with_zero('other_func')
        if as_sparse_network_test:
            acc_pre = validate(model_path, ratio=1)
            ident_ckpt = set_checkpoint(zero_ident, model_path.parent, -1, final_model=True)
            ident_acc_post = validate(ident_ckpt, ratio=1)
            tqdm.write(f'Zero_ident diff = {abs(ident_acc_post-acc_pre)}')
            other_ckpt = set_checkpoint(zero_other, model_path.parent, -2, final_model=True)
            other_acc_post = validate(other_ckpt, ratio=1)
            tqdm.write(f'Zero_other diff = {abs(other_acc_post - acc_pre)}')
