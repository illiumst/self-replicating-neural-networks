import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

from tqdm import tqdm

from functionalities_test import test_for_fixpoints, FixTypes as ft
from sanity_check_weights import test_weights_as_model, extract_weights_from_model

WORKER = 10
BATCHSIZE = 500

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True, parents=True)

PALETTE = sns.color_palette()
PALETTE.insert(0, PALETTE.pop(1))  # Orange First

FINAL_CHECKPOINT_NAME = f'trained_model_ckpt_FINAL.tp'


class AddGaussianNoise(object):
    def __init__(self, ratio=1e-4):
        self.ratio = ratio

    def __call__(self, tensor: torch.Tensor):
        return tensor + (torch.randn_like(tensor, device=tensor.device) * self.ratio)

    def __repr__(self):
        return self.__class__.__name__ + f'(ratio={self.ratio}'


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
        if isinstance(epoch_n, str):
            ckpt_path = Path(out_path) / f'{Path(FINAL_CHECKPOINT_NAME).stem}_{epoch_n}.tp'
        else:
            ckpt_path = Path(out_path) / FINAL_CHECKPOINT_NAME
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(model, ckpt_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    py_store_path = Path(out_path) / 'exp_py.txt'
    if not py_store_path.exists():
        shutil.copy(__file__, py_store_path)
    return ckpt_path


# noinspection PyProtectedMember
def validate(checkpoint_path, valid_loader, metric_class=torchmetrics.Accuracy):
    checkpoint_path = Path(checkpoint_path)

    # initialize metric
    validmetric = metric_class()
    model = torch.load(checkpoint_path, map_location=DEVICE).eval()

    with tqdm(total=len(valid_loader), desc='Validation Run: ') as pbar:
        for idx, (valid_batch_x, valid_batch_y) in enumerate(valid_loader):
            valid_batch_x, valid_batch_y = valid_batch_x.to(DEVICE), valid_batch_y.to(DEVICE)
            y_valid = model(valid_batch_x)

            # metric on current batch
            measure = validmetric(y_valid.cpu(), valid_batch_y.cpu())
            pbar.set_postfix_str(f'Measure: {measure}')
            pbar.update()

    # metric on all batches using custom accumulation
    measure = validmetric.compute()
    tqdm.write(f"Avg. {validmetric._get_name()} on all data: {measure}")
    return measure


def new_storage_df(identifier, weight_count):
    if identifier == 'train':
        return pd.DataFrame(columns=['Epoch', 'Batch', 'Metric', 'Score'])
    elif identifier == 'weights':
        return pd.DataFrame(columns=['Epoch', 'Weight', *(f'weight_{x}' for x in range(weight_count))])


def checkpoint_and_validate(model, valid_loader, out_path, epoch_n, keep_n=5, final_model=False,
                            validation_metric=torchmetrics.Accuracy):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    # Clean up Checkpoints
    if keep_n > 0:
        all_ckpts = sorted(list(ckpt_path.parent.iterdir()))
        while len(all_ckpts) > keep_n:
            all_ckpts.pop(0).unlink()
    elif keep_n == 0:
        pass
    else:
        raise ValueError(f'"keep_n" cannot be negative, but was: {keep_n}')

    result = validate(ckpt_path, valid_loader, metric_class=validation_metric)
    return result


def plot_training_particle_types(path_to_dataframe):
    plt.close('all')
    plt.clf()
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=False).sort_values('Metric')
    # Set up figure
    fig, ax = plt.subplots()  # initializes figure and plots
    data = df.loc[df['Metric'].isin(ft.all_types())]
    fix_types = data['Metric'].unique()
    data = data.pivot(index='Epoch', columns='Metric', values='Score').reset_index().fillna(0)
    _ = plt.stackplot(data['Epoch'], *[data[fixtype] for fixtype in fix_types],
                      labels=fix_types.tolist(), colors=PALETTE)

    ax.set(ylabel='Particle Count', xlabel='Epoch')
    # ax.set_title('Particle Type Count')

    fig.legend(loc="center right", title='Particle Type', bbox_to_anchor=(0.85, 0.5))
    plt.tight_layout()
    plt.savefig(Path(path_to_dataframe.parent / 'training_particle_type_lp.png'), dpi=300)


def plot_training_result(path_to_dataframe, metric_name='Accuracy', plot_name=None):
    plt.clf()
    # load from Drive
    df = pd.read_csv(path_to_dataframe, index_col=False).sort_values('Metric')

    # Check if this is a single lineplot or if aggregated
    group = ['Epoch', 'Metric']
    if 'Seed' in df.columns:
        group.append('Seed')

    # Set up figure
    fig, ax1 = plt.subplots()  # initializes figure and plots
    ax2 = ax1.twinx()  # applies twinx to ax2, which is the second y-axis.

    # plots the first set of data
    data = df[(df['Metric'] == 'Task Loss') | (df['Metric'] == 'Self Train Loss')].groupby(['Epoch', 'Metric']).mean()
    grouped_for_lineplot = data.groupby(group).mean()
    palette_len_1 = len(grouped_for_lineplot.droplevel(0).reset_index().Metric.unique())

    sns.lineplot(data=grouped_for_lineplot, x='Epoch', y='Score', hue='Metric',
                 palette=PALETTE[:palette_len_1], ax=ax1, ci='sd')

    # plots the second set of data
    data = df[(df['Metric'] == f'Test {metric_name}') | (df['Metric'] == f'Train {metric_name}')]
    palette_len_2 = len(data.Metric.unique())
    sns.lineplot(data=data, x='Epoch', y='Score', hue='Metric',
                 palette=PALETTE[palette_len_1:palette_len_2+palette_len_1], ci='sd')

    ax1.set(yscale='log', ylabel='Losses')
    # ax1.set_title('Training Lineplot')
    ax2.set(ylabel=metric_name)
    if metric_name != 'Accuracy':
        ax2.set(yscale='log')

    fig.legend(loc="center right", title='Metric', bbox_to_anchor=(0.85, 0.5))
    for ax in [ax1, ax2]:
        if legend := ax.get_legend():
            legend.remove()
    plt.tight_layout()
    plt.savefig(Path(path_to_dataframe.parent / ('training_lineplot.png' if plot_name is None else plot_name)), dpi=300)


def plot_network_connectivity_by_fixtype(path_to_trained_model):
    m = torch.load(path_to_trained_model, map_location=DEVICE).eval()
    # noinspection PyProtectedMember
    particles = list(m.particles)
    df = pd.DataFrame(columns=['Type', 'Layer', 'Neuron', 'Name'])

    for prtcl in particles:
        l, c, w = [float(x) for x in re.sub("[^0-9|_]", "", prtcl.name).split('_')]
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l-1, w, prtcl.name)
        df.loc[df.shape[0]] = (prtcl.is_fixpoint, l, c, prtcl.name)
    for layer in list(df['Layer'].unique()):
        # Rescale
        divisor = df.loc[(df['Layer'] == layer), 'Neuron'].max()
        df.loc[(df['Layer'] == layer), 'Neuron'] /= divisor

    tqdm.write(f'Connectivity Data gathered')
    df = df.sort_values('Type')
    n = 0
    for fixtype in ft.all_types():
        if df[df['Type'] == fixtype].shape[0] > 0:
            plt.clf()
            ax = sns.lineplot(y='Neuron', x='Layer', hue='Name', data=df[df['Type'] == fixtype],
                              legend=False, estimator=None, lw=1)
            _ = sns.lineplot(y=[0, 1], x=[-1, df['Layer'].max()], legend=False, estimator=None, lw=0)
            ax.set_title(fixtype)
            lines = ax.get_lines()
            for line in lines:
                line.set_color(PALETTE[n])
            plt.savefig(Path(path_to_trained_model.parent / f'net_connectivity_{fixtype}.png'), dpi=300)
            tqdm.write(f'Connectivity plottet: {fixtype} - n = {df[df["Type"] == fixtype].shape[0] // 2}')
            n += 1
        else:
            # tqdm.write(f'No Connectivity {fixtype}')
            pass


# noinspection PyProtectedMember
def run_particle_dropout_test(model_path, valid_loader, metric_class=torchmetrics.Accuracy):
    diff_store_path = model_path.parent / 'diff_store.csv'
    latest_model = torch.load(model_path, map_location=DEVICE).eval()
    prtcl_dict = defaultdict(lambda: 0)
    _ = test_for_fixpoints(prtcl_dict, list(latest_model.particles))
    tqdm.write(str(dict(prtcl_dict)))
    diff_df = pd.DataFrame(columns=['Particle Type', metric_class()._get_name(), 'Diff'])

    acc_pre = validate(model_path, valid_loader, metric_class=metric_class).item()
    diff_df.loc[diff_df.shape[0]] = ('All Organism', acc_pre, 0)

    for fixpoint_type in ft.all_types():
        new_model = torch.load(model_path, map_location=DEVICE).eval().replace_with_zero(fixpoint_type)
        if [x for x in new_model.particles if x.is_fixpoint == fixpoint_type]:
            new_ckpt = set_checkpoint(new_model, model_path.parent, fixpoint_type, final_model=True)
            acc_post = validate(new_ckpt, valid_loader, metric_class=metric_class).item()
            acc_diff = abs(acc_post - acc_pre)
            tqdm.write(f'Zero_ident diff = {acc_diff}')
            diff_df.loc[diff_df.shape[0]] = (fixpoint_type, acc_post, acc_diff)

    diff_df.to_csv(diff_store_path, mode='w', header=True, index=False)
    return diff_store_path


# noinspection PyProtectedMember
def plot_dropout_stacked_barplot(mdl_path, diff_store_path, metric_class=torchmetrics.Accuracy):
    metric_name = metric_class()._get_name()
    diff_df = pd.read_csv(diff_store_path).sort_values('Particle Type')
    particle_dict = defaultdict(lambda: 0)
    latest_model = torch.load(mdl_path, map_location=DEVICE).eval()
    _ = test_for_fixpoints(particle_dict, list(latest_model.particles))
    particle_dict = dict(particle_dict)
    sorted_particle_dict = dict(sorted(particle_dict.items()))
    tqdm.write(str(sorted_particle_dict))
    plt.clf()
    fig, ax = plt.subplots(ncols=2)
    colors = PALETTE.copy()
    colors.insert(0, colors.pop(-1))
    palette_len = len(diff_df['Particle Type'].unique())
    _ = sns.barplot(data=diff_df, y=metric_name, x='Particle Type', ax=ax[0], palette=colors[:palette_len], ci=None)

    ax[0].set_title(f'{metric_name} after particle dropout')
    ax[0].set_xlabel('Particle Type')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30)

    ax[1].pie(sorted_particle_dict.values(), labels=sorted_particle_dict.keys(),
              colors=PALETTE[:len(sorted_particle_dict)])
    ax[1].set_title('Particle Count')

    plt.tight_layout()
    plt.savefig(Path(diff_store_path.parent / 'dropout_stacked_barplot.png'), dpi=300)


def run_particle_dropout_and_plot(model_path, valid_loader, metric_class=torchmetrics.Accuracy):
    diff_store_path = run_particle_dropout_test(model_path, valid_loader=valid_loader, metric_class=metric_class)
    plot_dropout_stacked_barplot(model_path, diff_store_path, metric_class=metric_class)


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


def highlight_fixpoints_vs_mnist_mean(mdl_path, dataloader):
    latest_model = torch.load(mdl_path, map_location=DEVICE).eval()
    activation_vector = torch.as_tensor([[0, 0, 0, 0, 1]], dtype=torch.float32, device=DEVICE)
    binary_images = []
    real_images = []
    with torch.no_grad():
        # noinspection PyProtectedMember
        for cell in latest_model._meta_layer_first.meta_cell_list:
            cell_image_binary = torch.zeros((len(cell.meta_weight_list)), device=DEVICE)
            cell_image_real = torch.zeros((len(cell.meta_weight_list)), device=DEVICE)
            for idx, particle in enumerate(cell.particles):
                if particle.is_fixpoint == ft.identity_func:
                    cell_image_binary[idx] += 1
                    cell_image_real[idx] = particle(activation_vector).abs().squeeze().item()
            binary_images.append(cell_image_binary.reshape((15, 15)))
            real_images.append(cell_image_real.reshape((15, 15)))

        binary_images = torch.stack(binary_images)
        real_images = torch.stack(real_images)

        binary_image = torch.sum(binary_images, keepdim=True, dim=0)
        real_image = torch.sum(real_images, keepdim=True, dim=0)

    mnist_images = [x for x, _ in dataloader]
    mnist_mean = torch.cat(mnist_images).reshape(10000, 15, 15).abs().sum(dim=0)

    fig, axs = plt.subplots(1, 3)

    for idx, image in enumerate([binary_image, real_image, mnist_mean]):
        img = axs[idx].imshow(image.squeeze().detach().cpu())
        img.axes.axis('off')

    plt.tight_layout()
    plt.savefig(mdl_path.parent / 'heatmap.png', dpi=300)
    plt.clf()
    plt.close('all')


def plot_training_results_over_n_seeds(exp_path, df_train_store_name='train_store.csv', metric_name='Accuracy'):
    combined_df_store_path = exp_path / f'comb_train_{exp_path.stem[:-1]}n.csv'
    # noinspection PyUnboundLocalVariable
    found_train_stores = exp_path.rglob(df_train_store_name)
    train_dfs = []
    for found_train_store in found_train_stores:
        train_store_df = pd.read_csv(found_train_store, index_col=False)
        train_store_df['Seed'] = int(found_train_store.parent.name)
        train_dfs.append(train_store_df)
    combined_train_df = pd.concat(train_dfs)
    combined_train_df.to_csv(combined_df_store_path, index=False)
    plot_training_result(combined_df_store_path, metric_name=metric_name,
                         plot_name=f"{combined_df_store_path.stem}.png"
                         )
    plt.clf()
    plt.close('all')


def sanity_weight_swap(exp_path, dataloader, metric_class=torchmetrics.Accuracy):
    # noinspection PyProtectedMember
    metric_name = metric_class()._get_name()
    found_models = exp_path.rglob(f'*{FINAL_CHECKPOINT_NAME}')
    df = pd.DataFrame(columns=['Seed', 'Model', metric_name])
    for model_idx, found_model in enumerate(found_models):
        model = torch.load(found_model, map_location=DEVICE).eval()
        weights = extract_weights_from_model(model)

        results = test_weights_as_model(model, weights, dataloader, metric_class=metric_class)
        for model_name, measurement in results.items():
            df.loc[df.shape[0]] = (model_idx, model_name, measurement)
        df.loc[df.shape[0]] = (model_idx, 'Difference', np.abs(np.subtract(*results.values())))

    df.to_csv(exp_path / 'sanity_weight_swap.csv', index=False)
    _ = sns.boxplot(data=df, x='Model', y=metric_name)
    plt.tight_layout()
    plt.savefig(exp_path / 'sanity_weight_swap.png', dpi=300)
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    raise NotImplementedError('Test this here!!!')
