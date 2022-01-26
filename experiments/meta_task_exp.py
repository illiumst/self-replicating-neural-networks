import pickle
import time
from pathlib import Path
import sys
import platform

import pandas as pd
import torchmetrics

if platform.node() != 'CarbonX':
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
else:
    debug = True

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from tqdm import tqdm

from network import MetaNet

WORKER = 10 if not debug else 2
BATCHSIZE = 500 if not debug else 50
EPOCH = 50 if not debug else 3
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
    if final_model:
        ckpt_path = Path(out_path) / 'ckpt' / f'{epoch_n.zfill(4)}_model_ckpt.tp'
    else:
        ckpt_path = Path(out_path) / f'trained_model_ckpt.tp'
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(model, ckpt_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return ckpt_path


def validate(checkpoint_path, ratio=0.1):
    checkpoint_path = Path(checkpoint_path)
    import torchmetrics

    # initialize metric
    metric = torchmetrics.Accuracy()

    try:
        dataset = MNIST(str(data_path), transform=utility_transforms, train=False)
    except RuntimeError:
        dataset = MNIST(str(data_path), transform=utility_transforms, train=False, download=True)
    d = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

    model = torch.load(checkpoint_path, map_location=DEVICE).eval()
    n_samples = int(len(d) * ratio)

    with tqdm(total=n_samples, desc='Validation Run: ') as pbar:
        for idx, (batch_x, batch_y) in enumerate(d):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            y = model(batch_x)

            # metric on current batch
            acc = metric(y.cpu(), batch_y.cpu())
            pbar.set_postfix_str(f'Acc: {acc}')
            pbar.update()
            if idx == n_samples:
                break

    # metric on all batches using custom accumulation
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")
    return acc


def checkpoint_and_validate(model, out_path, epoch_n, final_model=False):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    result = validate(ckpt_path)
    return result


def plot_training_result(path_to_dataframe):
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=0)

    fig, ax1 = plt.subplots()  # initializes figure and plots
    ax2 = ax1.twinx()          # applies twinx to ax2, which is the second y axis.

    # plots the first set of data, and sets it to ax1.
    data = df[df['Metric'] == 'BatchLoss']
    # plots the second set, and sets to ax2.
    sns.lineplot(data=data.groupby('Epoch').mean(), x='Epoch', y='Score', legend=True, ax=ax2)
    data = df[df['Metric'] == 'Test Accuracy']
    sns.lineplot(data=data, x='Epoch', y='Score', marker='o', color='red')
    data = df[df['Metric'] == 'Train Accuracy']
    sns.lineplot(data=data, x='Epoch', y='Score', marker='o', color='green')

    ax2.set(yscale='log')
    ax1.set_title('Training Lineplot')
    plt.tight_layout()
    if debug:
        plt.show()
    else:
        plt.savefig(Path(path_to_dataframe.parent / 'training_lineplot.png'))


if __name__ == '__main__':

    self_train = True
    soup_interaction = True
    training = True
    plotting = True

    data_path = Path('data')
    data_path.mkdir(exist_ok=True, parents=True)

    run_path = Path('output') / 'intergrated_self_train'
    model_path = run_path / '0000_trained_model.zip'

    if training:
        utility_transforms = Compose([ToTensor(), ToFloat(), Flatten(start_dim=0)])

        try:
            dataset = MNIST(str(data_path), transform=utility_transforms)
        except RuntimeError:
            dataset = MNIST(str(data_path), transform=utility_transforms, download=True)
        d = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

        interface = np.prod(dataset[0][0].shape)
        metanet = MetaNet(interface, depth=4, width=6, out=10).to(DEVICE).train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(metanet.parameters(), lr=0.004, momentum=0.9)

        train_store = pd.DataFrame(columns=['Epoch', 'Batch', 'Metric', 'Score'])
        for epoch in tqdm(range(EPOCH), desc='MetaNet Train - Epochs'):
            is_validation_epoch = epoch % VALIDATION_FRQ == 0 if not debug else True
            is_self_train_epoch = epoch % SELF_TRAIN_FRQ == 0 if not debug else True
            if is_validation_epoch:
                metric = torchmetrics.Accuracy()
            for batch, (batch_x, batch_y) in tqdm(enumerate(d), total=len(d), desc='MetaNet Train - Batch'):
                if self_train and is_self_train_epoch:
                    # Zero your gradients for every batch!
                    optimizer.zero_grad()
                    combined_self_train_loss = metanet.combined_self_train()
                    combined_self_train_loss.backward()
                    # Adjust learning weights
                    optimizer.step()

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
                                Metric='BatchLoss', Score=loss.item())
                train_store.loc[train_store.shape[0]] = step_log
                if is_validation_epoch:
                    metric(y.cpu(), batch_y.cpu())

                if batch >= 3 and debug:
                    break

            if is_validation_epoch:
                validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                      Metric='Train Accuracy', Score=metric.compute().item())
                train_store.loc[train_store.shape[0]] = validation_log

                accuracy = checkpoint_and_validate(metanet, run_path, epoch)
                validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                      Metric='Test Accuracy', Score=accuracy.item())
                train_store.loc[train_store.shape[0]] = validation_log

        accuracy = checkpoint_and_validate(metanet, run_path, EPOCH, final_model=True)
        validation_log = dict(Epoch=EPOCH, Batch=BATCHSIZE,
                              Metric='Test Accuracy', Score=accuracy.item())
        train_store.loc[train_store.shape[0]] = validation_log

        torch.save(metanet, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        train_store.to_csv(run_path / 'train_store.csv')

    if plotting:
        plot_training_result(run_path / 'train_store.csv')
