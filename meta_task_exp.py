# # # Imports
from collections import defaultdict
from pathlib import Path

import platform

import torchmetrics
import numpy as np
import torch

from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize
from tqdm import tqdm

# noinspection DuplicatedCode
from experiments.meta_task_utility import (ToFloat, new_storage_df, train_task, checkpoint_and_validate, flat_for_store,
                                           plot_training_result, plot_training_particle_types,
                                           plot_network_connectivity_by_fixtype, run_particle_dropout_and_plot,
                                           highlight_fixpoints_vs_mnist_mean, AddGaussianNoise,
                                           plot_training_results_over_n_seeds, sanity_weight_swap,
                                           FINAL_CHECKPOINT_NAME)
from experiments.robustness_tester import test_robustness
from plot_3d_trajectories import plot_single_3d_trajectories_by_layer, plot_grouped_3d_trajectories_by_layer

if platform.node() == 'CarbonX':
    debug = True
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@ Warning, Debugging Config@!!!!!! @")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
else:
    debug = False

from network import MetaNet, FixTypes
from functionalities_test import test_for_fixpoints

utility_transforms = Compose([ToTensor(), ToFloat(), Resize((15, 15)), Flatten(start_dim=0), AddGaussianNoise()])
WORKER = 10 if not debug else 2
debug = False
BATCHSIZE = 2000 if not debug else 50
EPOCH = 200
VALIDATION_FRQ = 3 if not debug else 1
VAL_METRIC_CLASS = torchmetrics.Accuracy
# noinspection PyProtectedMember
VAL_METRIC_NAME = VAL_METRIC_CLASS()._get_name()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True, parents=True)

try:
    plot_dataset = MNIST(str(DATA_PATH), transform=utility_transforms, train=False)
except RuntimeError:
    plot_dataset = MNIST(str(DATA_PATH), transform=utility_transforms, train=False, download=True)
plot_loader = DataLoader(plot_dataset, batch_size=BATCHSIZE, shuffle=False,
                         drop_last=True, num_workers=WORKER)


if __name__ == '__main__':

    training = False
    plotting = False
    robustnes = True    # EXPENSIV!!!!!!!
    n_st = 300          # per batch !!
    activation = None   # nn.ReLU()

    for weight_hidden_size in [3]:

        weight_hidden_size = weight_hidden_size
        residual_skip = True
        n_seeds = 3
        depth = 5
        width = 3
        out = 10

        data_path = Path('data')
        data_path.mkdir(exist_ok=True, parents=True)

        # noinspection PyUnresolvedReferences
        ac_str = f'_{activation.__class__.__name__}' if activation is not None else ''
        res_str = f'{"" if residual_skip else "_no_res"}'
        st_str = f'_nst_{n_st}'

        config_str = f'{res_str}{ac_str}{st_str}'
        exp_path = Path('output') / f'mn_st_{EPOCH}_{weight_hidden_size}{config_str}_gauss'

        for seed in range(n_seeds):
            seed_path = exp_path / str(seed)

            df_store_path = seed_path / 'train_store.csv'
            weight_store_path = seed_path / 'weight_store.csv'
            srnn_parameters = dict()

            if training:
                # Check if files do exist on project location, warn and break.
                for path in [df_store_path, weight_store_path]:
                    assert not path.exists(), f'Path "{path}" already exists. Check your configuration!'
                try:
                    train_dataset = MNIST(str(DATA_PATH), transform=utility_transforms)
                except RuntimeError:
                    train_dataset = MNIST(str(DATA_PATH), transform=utility_transforms, download=True)
                train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,
                                          drop_last=True, num_workers=WORKER)
                try:
                    valid_dataset = MNIST(str(DATA_PATH), transform=utility_transforms, train=False)
                except RuntimeError:
                    valid_dataset = MNIST(str(DATA_PATH), transform=utility_transforms, train=False, download=True)
                valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True,
                                          drop_last=True, num_workers=WORKER)

                interface = np.prod(train_dataset[0][0].shape)
                metanet = MetaNet(interface, depth=depth, width=width, out=out,
                                  residual_skip=residual_skip, weight_hidden_size=weight_hidden_size,
                                  activation=activation
                                  ).to(DEVICE)

                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(metanet.parameters(), lr=0.004, momentum=0.9)

                train_store = new_storage_df('train', None)
                weight_store = new_storage_df('weights', metanet.particle_parameter_count)

                for epoch in tqdm(range(EPOCH), desc=f'Train - Epochs'):
                    is_validation_epoch = epoch % VALIDATION_FRQ == 0 if not debug else True
                    metanet = metanet.train()

                    # Init metrics, even we do not need:
                    metric = VAL_METRIC_CLASS()
                    n_st_per_batch = max(n_st // len(train_loader), 1)

                    if is_validation_epoch:
                        for particle in metanet.particles:
                            weight_log = (epoch, particle.name, *flat_for_store(particle.parameters()))
                            weight_store.loc[weight_store.shape[0]] = weight_log

                    for batch, (batch_x, batch_y) in tqdm(enumerate(train_loader),
                                                          total=len(train_loader), desc='MetaNet Train - Batch'
                                                          ):
                        # Self Train
                        self_train_loss = metanet.combined_self_train(n_st_per_batch,
                                                                      reduction='mean', per_particle=False)
                        # noinspection PyUnboundLocalVariable
                        st_step_log = dict(Metric='Self Train Loss', Score=self_train_loss.item())
                        st_step_log.update(dict(Epoch=epoch, Batch=batch))
                        train_store.loc[train_store.shape[0]] = st_step_log

                        # Task Train
                        tsk_step_log, y_pred = train_task(metanet, optimizer, loss_fn, batch_x, batch_y)
                        tsk_step_log.update(dict(Epoch=epoch, Batch=batch))
                        train_store.loc[train_store.shape[0]] = tsk_step_log
                        metric(y_pred.cpu(), batch_y.cpu())

                    if is_validation_epoch:
                        metanet = metanet.eval()
                        try:
                            validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                                  Metric=f'Train {VAL_METRIC_NAME}', Score=metric.compute().item())
                            train_store.loc[train_store.shape[0]] = validation_log
                        except RuntimeError:
                            pass

                        accuracy = checkpoint_and_validate(metanet, valid_loader, seed_path, epoch).item()
                        validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                              Metric=f'Test {VAL_METRIC_NAME}', Score=accuracy)
                        train_store.loc[train_store.shape[0]] = validation_log

                    if is_validation_epoch:
                        counter_dict = defaultdict(lambda: 0)
                        # This returns ID-functions
                        _ = test_for_fixpoints(counter_dict, list(metanet.particles))
                        counter_dict = dict(counter_dict)
                        for key, value in counter_dict.items():
                            val_step_log = dict(Epoch=int(epoch), Batch=BATCHSIZE, Metric=key, Score=value)
                            train_store.loc[train_store.shape[0]] = val_step_log
                        tqdm.write(f'Fixpoint Tester Results: {counter_dict}')

                    # FLUSH to disk
                    if is_validation_epoch:
                        train_store.to_csv(df_store_path, mode='a',
                                           header=not df_store_path.exists(), index=False)
                        weight_store.to_csv(weight_store_path, mode='a',
                                            header=not weight_store_path.exists(), index=False)
                        train_store = new_storage_df('train', None)
                        weight_store = new_storage_df('weights', metanet.particle_parameter_count)

                ###########################################################
                # EPOCHS endet
                metanet = metanet.eval()

                counter_dict = defaultdict(lambda: 0)
                # This returns ID-functions
                _ = test_for_fixpoints(counter_dict, list(metanet.particles))
                for key, value in dict(counter_dict).items():
                    step_log = dict(Epoch=int(EPOCH)+1, Batch=BATCHSIZE, Metric=key, Score=value)
                    train_store.loc[train_store.shape[0]] = step_log
                accuracy = checkpoint_and_validate(metanet, valid_loader, seed_path, EPOCH, final_model=True)
                validation_log = dict(Epoch=EPOCH, Batch=BATCHSIZE,
                                      Metric=f'Test {VAL_METRIC_NAME}', Score=accuracy.item())
                train_store.loc[train_store.shape[0]] = validation_log
                for particle in metanet.particles:
                    weight_log = (EPOCH, particle.name, *(flat_for_store(particle.parameters())))
                    weight_store.loc[weight_store.shape[0]] = weight_log

                # FLUSH to disk
                train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists(), index=False)
                weight_store.to_csv(weight_store_path, mode='a', header=not weight_store_path.exists(), index=False)

            try:
                model_path = next(seed_path.glob(f'*{FINAL_CHECKPOINT_NAME}'))
            except StopIteration:
                print('Model pattern did not trigger.')
                print(f'Search path was: {seed_path}:')
                print(f'Found Models are: {list(seed_path.rglob(".tp"))}')
                exit(1)

            if plotting:
                highlight_fixpoints_vs_mnist_mean(model_path, plot_loader)

                plot_training_result(df_store_path)
                plot_training_particle_types(df_store_path)

                try:
                    # noinspection PyUnboundLocalVariable
                    run_particle_dropout_and_plot(model_path, valid_loader=plot_loader, metric_class=VAL_METRIC_CLASS)
                except (ValueError, NameError) as e:
                    print(e)
                try:
                    plot_network_connectivity_by_fixtype(model_path)
                except (ValueError, NameError) as e:
                    print(e)

                highlight_fixpoints_vs_mnist_mean(model_path, plot_loader)

                try:
                    for fixtype in FixTypes.all_types():
                        plot_single_3d_trajectories_by_layer(model_path, weight_store_path, status_type=fixtype)
                        plot_grouped_3d_trajectories_by_layer(model_path, weight_store_path, status_type=fixtype)

                except ValueError as e:
                    print('ERROR:', e)
            if robustnes:
                try:
                    test_robustness(model_path, seeds=1)
                except ValueError as e:
                    print('ERROR:', e)

        if 2 <= n_seeds <= sum(list(x.is_dir() for x in exp_path.iterdir())):
            if plotting:

                plot_training_results_over_n_seeds(exp_path, metric_name=VAL_METRIC_NAME)

                sanity_weight_swap(exp_path, plot_loader, VAL_METRIC_CLASS)
