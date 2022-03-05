from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.meta_task_small_utility import AddTaskDataset, train_task
from network import MetaNet
from functionalities_test import test_for_fixpoints, FixTypes as ft
from experiments.meta_task_utility import new_storage_df, flat_for_store, plot_training_result, \
    plot_training_particle_types, run_particle_dropout_and_plot, plot_network_connectivity_by_fixtype, \
    checkpoint_and_validate
from plot_3d_trajectories import plot_single_3d_trajectories_by_layer, plot_grouped_3d_trajectories_by_layer

WORKER = 0
BATCHSIZE = 50
EPOCH = 30
VALIDATION_FRQ = 3
VALIDATION_METRIC = torchmetrics.MeanAbsoluteError
# noinspection PyProtectedMember
VAL_METRIC_NAME = VALIDATION_METRIC()._get_name()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    training = True
    plotting = True
    n_st = 700
    activation = None  # nn.ReLU()

    for weight_hidden_size in [3, 4, 5, 6]:

        tsk_threshold = 0.85
        weight_hidden_size = weight_hidden_size
        residual_skip = True
        n_seeds = 10
        depth = 3
        width = 3
        out = 1

        data_path = Path('data')
        data_path.mkdir(exist_ok=True, parents=True)

        # noinspection PyUnresolvedReferences
        ac_str = f'_{activation.__class__.__name__}' if activation is not None else ''
        res_str = f'{"" if residual_skip else "_no_res"}'
        # dr_str = f'{f"_dr_{dropout}" if dropout != 0 else ""}'

        config_str = f'{res_str}'
        exp_path = Path('output') / f'add_st_{EPOCH}_{weight_hidden_size}{config_str}{ac_str}'

        # if not training:
        #     # noinspection PyRedeclaration
        #     exp_path = Path('output') / f'add_st_{n_st}_{weight_hidden_size}'

        for seed in range(n_seeds):
            seed_path = exp_path / str(seed)

            model_path = seed_path / '0000_trained_model.zip'
            df_store_path = seed_path / 'train_store.csv'
            weight_store_path = seed_path / 'weight_store.csv'
            srnn_parameters = dict()

            valid_data = AddTaskDataset()
            vali_load = DataLoader(valid_data, batch_size=BATCHSIZE, shuffle=True,
                                   drop_last=True, num_workers=WORKER)

            if training:
                # Check if files do exist on project location, warn and break.
                for path in [model_path, df_store_path, weight_store_path]:
                    assert not path.exists(), f'Path "{path}" already exists. Check your configuration!'

                train_data = AddTaskDataset()
                train_load = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True,
                                        drop_last=True, num_workers=WORKER)

                interface = np.prod(train_data[0][0].shape)
                metanet = MetaNet(interface, depth=depth, width=width, out=out,
                                  residual_skip=residual_skip, weight_hidden_size=weight_hidden_size,
                                  activation=activation
                                  ).to(DEVICE)

                loss_fn = nn.MSELoss()
                optimizer = torch.optim.SGD(metanet.parameters(), lr=0.004, momentum=0.9)

                train_store = new_storage_df('train', None)
                weight_store = new_storage_df('weights', metanet.particle_parameter_count)

                for epoch in tqdm(range(EPOCH), desc=f'Train - Epochs'):
                    is_validation_epoch = epoch % VALIDATION_FRQ == 0
                    metanet = metanet.train()

                    # Init metrics, even we do not need:
                    metric = VALIDATION_METRIC()
                    n_st_per_batch = n_st // len(train_load)

                    for batch, (batch_x, batch_y) in tqdm(enumerate(train_load),
                                                          total=len(train_load), desc='MetaNet Train - Batch'
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
                        if metric.total.item():
                            validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                                  Metric=f'Train {VAL_METRIC_NAME}', Score=metric.compute().item())
                            train_store.loc[train_store.shape[0]] = validation_log

                        mae = checkpoint_and_validate(metanet, vali_load, seed_path, epoch,
                                                      validation_metric=VALIDATION_METRIC).item()
                        validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                              Metric=f'Test {VAL_METRIC_NAME}', Score=mae)
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
                            for particle in metanet.particles:
                                weight_log = (epoch, particle.name, *flat_for_store(particle.parameters()))
                                weight_store.loc[weight_store.shape[0]] = weight_log
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
                    step_log = dict(Epoch=int(EPOCH), Batch=BATCHSIZE, Metric=key, Score=value)
                    train_store.loc[train_store.shape[0]] = step_log
                accuracy = checkpoint_and_validate(metanet, vali_load, seed_path, EPOCH, final_model=True,
                                                   validation_metric=VALIDATION_METRIC)
                validation_log = dict(Epoch=EPOCH, Batch=BATCHSIZE,
                                      Metric=f'Test {VAL_METRIC_NAME}', Score=accuracy.item())
                for particle in metanet.particles:
                    weight_log = (EPOCH, particle.name, *(flat_for_store(particle.parameters())))
                    weight_store.loc[weight_store.shape[0]] = weight_log

                train_store.loc[train_store.shape[0]] = validation_log
                train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists(), index=False)
                weight_store.to_csv(weight_store_path, mode='a', header=not weight_store_path.exists(), index=False)
            if plotting:

                plot_training_result(df_store_path, metric=VAL_METRIC_NAME)
                plot_training_particle_types(df_store_path)

                try:
                    model_path = next(seed_path.glob(f'*e{EPOCH}.tp'))
                except StopIteration:
                    print('####################################################')
                    print('ERROR: Model pattern did not trigger.')
                    print(f'INFO: Search path was: {seed_path}:')
                    print(f'INFO: Found Models are: {list(seed_path.rglob(".tp"))}')
                    print('####################################################')
                    exit(1)

                try:
                    run_particle_dropout_and_plot(model_path, valid_loader=vali_load, metric_class=VALIDATION_METRIC)
                except ValueError as e:
                    print('ERROR:', e)
                try:
                    plot_network_connectivity_by_fixtype(model_path)
                except ValueError as e:
                    print('ERROR:', e)
                try:
                    plot_single_3d_trajectories_by_layer(model_path, weight_store_path, status_type=ft.identity_func)
                    plot_single_3d_trajectories_by_layer(model_path, weight_store_path, status_type=ft.other_func)
                    plot_grouped_3d_trajectories_by_layer(model_path, weight_store_path, status_type=ft.identity_func)
                    plot_grouped_3d_trajectories_by_layer(model_path, weight_store_path, status_type=ft.other_func)
                except ValueError as e:
                    print('ERROR:', e)

    if n_seeds >= 2:
        combined_df_store_path = exp_path.parent / f'comb_train_{exp_path.stem[:-1]}n.csv'
        # noinspection PyUnboundLocalVariable
        found_train_stores = exp_path.rglob(df_store_path.name)
        train_dfs = []
        for found_train_store in found_train_stores:
            train_store_df = pd.read_csv(found_train_store, index_col=False)
            train_store_df['Seed'] = int(found_train_store.parent.name)
            train_dfs.append(train_store_df)
        combined_train_df = pd.concat(train_dfs)
        combined_train_df.to_csv(combined_df_store_path, index=False)
        plot_training_result(combined_df_store_path, metric=VAL_METRIC_NAME,
                             plot_name=f"{combined_df_store_path.stem}.png"
                             )
