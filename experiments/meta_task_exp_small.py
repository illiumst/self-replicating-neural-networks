import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.utils.data import Dataset, DataLoader
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
from experiments.meta_task_exp import new_storage_df, train_self_replication, train_task, set_checkpoint, \
    flat_for_store, plot_training_result, plot_training_particle_types, run_particle_dropout_and_plot, \
    plot_network_connectivity_by_fixtype

WORKER = 10 if not debug else 2
debug = False
BATCHSIZE = 50 if not debug else 50
EPOCH = 10
VALIDATION_FRQ = 1 if not debug else 1
SELF_TRAIN_FRQ = 1 if not debug else 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AddTaskDataset(Dataset):
    def __init__(self, length=int(1e5)):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ab = torch.randn(size=(2,)).to(torch.float32)
        return ab, ab.sum(axis=-1, keepdims=True)


def validate(checkpoint_path, valid_d, ratio=1, validmetric=torchmetrics.MeanAbsoluteError()):
    checkpoint_path = Path(checkpoint_path)
    import torchmetrics

    # initialize metric
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
    tqdm.write(f"Avg. Accuracy on all data: {acc}")
    return acc


def checkpoint_and_validate(model, out_path, epoch_n, valid_d, final_model=False):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    result = validate(ckpt_path, valid_d)
    return result


if __name__ == '__main__':

    training = True
    train_to_id_first = False
    train_to_task_first = False
    seq_task_train = True
    force_st_for_epochs_n = 5
    n_st_per_batch = 10
    activation = None  # nn.ReLU()

    use_sparse_network = False

    for weight_hidden_size in [3, 4]:

        tsk_threshold = 0.85
        weight_hidden_size = weight_hidden_size
        residual_skip = False
        n_seeds = 3
        depth = 3
        width = 3
        out = 1

        data_path = Path('data')
        data_path.mkdir(exist_ok=True, parents=True)
        assert not (train_to_task_first and train_to_id_first)

        ac_str = f'_{activation.__class__.__name__}' if activation is not None else ''
        s_str = f'_n_{n_st_per_batch}' if n_st_per_batch > 1 else ""
        res_str = f'{"" if residual_skip else "_no_res"}'
        # dr_str = f'{f"_dr_{dropout}" if dropout != 0 else ""}'
        id_str = f'{f"_StToId" if train_to_id_first else ""}'
        tsk_str = f'{f"_Tsk_{tsk_threshold}" if train_to_task_first and tsk_threshold != 1 else ""}'
        sprs_str = '_sprs' if use_sparse_network else ''
        f_str = f'_f_{force_st_for_epochs_n}' if \
            force_st_for_epochs_n and seq_task_train and train_to_task_first else ""
        config_str = f'{s_str}{res_str}{id_str}{tsk_str}{f_str}{sprs_str}'
        exp_path = Path('output') / f'add_st_{EPOCH}_{weight_hidden_size}{config_str}{ac_str}'

        if not training:
            # noinspection PyRedeclaration
            exp_path = Path('output') / 'mn_st_n_2_100_4'

        for seed in range(n_seeds):
            seed_path = exp_path / str(seed)

            model_path = seed_path / '0000_trained_model.zip'
            df_store_path = seed_path / 'train_store.csv'
            weight_store_path = seed_path / 'weight_store.csv'
            srnn_parameters = dict()

            if training:
                # Check if files do exist on project location, warn and break.
                for path in [model_path, df_store_path, weight_store_path]:
                    assert not path.exists(), f'Path "{path}" already exists. Check your configuration!'

                train_data = AddTaskDataset()
                valid_data = AddTaskDataset()
                train_load = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True,
                                        drop_last=True, num_workers=WORKER)
                vali_load = DataLoader(valid_data, batch_size=BATCHSIZE, shuffle=False,
                                       drop_last=True, num_workers=WORKER)

                interface = np.prod(train_data[0][0].shape)
                dense_metanet = MetaNet(interface, depth=depth, width=width, out=out,
                                        residual_skip=residual_skip, weight_hidden_size=weight_hidden_size,
                                        activation=activation
                                        ).to(DEVICE)
                sparse_metanet = SparseNetwork(interface, depth=depth, width=width, out=out,
                                               residual_skip=residual_skip, weight_hidden_size=weight_hidden_size,
                                               activation=activation
                                               ).to(DEVICE) if use_sparse_network else dense_metanet
                if use_sparse_network:
                    sparse_metanet = sparse_metanet.replace_weights_by_particles(dense_metanet.particles)

                loss_fn = nn.MSELoss()
                dense_optimizer = torch.optim.SGD(dense_metanet.parameters(), lr=0.004, momentum=0.9)
                sparse_optimizer = torch.optim.SGD(
                    sparse_metanet.parameters(), lr=0.001, momentum=0.9
                                                   ) if use_sparse_network else dense_optimizer

                dense_weights_updated = False
                sparse_weights_updated = False

                train_store = new_storage_df('train', None)
                weight_store = new_storage_df('weights', dense_metanet.particle_parameter_count)

                init_tsk = train_to_task_first
                for epoch in tqdm(range(EPOCH), desc=f'Train - Epochs'):
                    is_validation_epoch = epoch % VALIDATION_FRQ == 0 if not debug else True
                    is_self_train_epoch = epoch % SELF_TRAIN_FRQ == 0 if not debug else True
                    sparse_metanet = sparse_metanet.train()
                    dense_metanet = dense_metanet.train()

                    # Init metrics, even we do not need:
                    metric = torchmetrics.MeanAbsoluteError()

                    # Define what to train in this epoch:
                    do_tsk_train = train_to_task_first
                    force_st    = (force_st_for_epochs_n >= (EPOCH - epoch)) and force_st_for_epochs_n
                    init_st     = (train_to_id_first and not dense_metanet.count_fixpoints() > 200)
                    do_st_train = init_st or is_self_train_epoch or force_st

                    for batch, (batch_x, batch_y) in tqdm(enumerate(train_load),
                                                          total=len(train_load), desc='MetaNet Train - Batch'
                                                          ):

                        # Self Train
                        if do_st_train:
                            # Transfer weights
                            if dense_weights_updated:
                                sparse_metanet = sparse_metanet.replace_weights_by_particles(dense_metanet.particles)
                                dense_weights_updated = False
                            st_steps = n_st_per_batch if not init_st else n_st_per_batch * 10
                            step_log = train_self_replication(sparse_metanet, sparse_optimizer, st_steps)
                            step_log.update(dict(Epoch=epoch, Batch=batch))
                            train_store.loc[train_store.shape[0]] = step_log
                            if use_sparse_network:
                                sparse_weights_updated = True

                        # Task Train
                        if not init_st:
                            # Transfer weights
                            if sparse_weights_updated:
                                dense_metanet = dense_metanet.replace_particles(sparse_metanet.particle_weights)
                                sparse_weights_updated = False
                            step_log, y_pred = train_task(dense_metanet, dense_optimizer, loss_fn, batch_x, batch_y)

                            step_log.update(dict(Epoch=epoch, Batch=batch))
                            train_store.loc[train_store.shape[0]] = step_log
                            if use_sparse_network:
                                dense_weights_updated = True
                            metric(y_pred.cpu(), batch_y.cpu())

                    if is_validation_epoch:
                        if sparse_weights_updated:
                            dense_metanet = dense_metanet.replace_particles(sparse_metanet.particle_weights)
                            sparse_weights_updated = False

                        dense_metanet = dense_metanet.eval()
                        if do_tsk_train:
                            validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                                  Metric='Train Accuracy', Score=metric.compute().item())
                            train_store.loc[train_store.shape[0]] = validation_log

                        accuracy = checkpoint_and_validate(dense_metanet, seed_path, epoch, vali_load).item()
                        validation_log = dict(Epoch=int(epoch), Batch=BATCHSIZE,
                                              Metric='Test Accuracy', Score=accuracy)
                        train_store.loc[train_store.shape[0]] = validation_log
                        if init_tsk or (train_to_task_first and seq_task_train):
                            init_tsk = accuracy <= tsk_threshold
                    if init_st or is_validation_epoch:
                        if dense_weights_updated:
                            sparse_metanet = sparse_metanet.replace_weights_by_particles(dense_metanet.particles)
                            dense_weights_updated = False
                        counter_dict = defaultdict(lambda: 0)
                        # This returns ID-functions
                        _ = test_for_fixpoints(counter_dict, list(dense_metanet.particles))
                        counter_dict = dict(counter_dict)
                        for key, value in counter_dict.items():
                            step_log = dict(Epoch=int(epoch), Batch=BATCHSIZE, Metric=key, Score=value)
                            train_store.loc[train_store.shape[0]] = step_log
                        tqdm.write(f'Fixpoint Tester Results: {counter_dict}')
                        if sum(x.is_fixpoint == ft.identity_func for x in dense_metanet.particles) > 200:
                            train_to_id_first = False
                        # Reset Diverged particles
                        sparse_metanet.reset_diverged_particles()
                        if use_sparse_network:
                            sparse_weights_updated = True

                        # FLUSH to disk
                        if is_validation_epoch:
                            for particle in dense_metanet.particles:
                                weight_log = (epoch, particle.name, *flat_for_store(particle.parameters()))
                                weight_store.loc[weight_store.shape[0]] = weight_log
                            train_store.to_csv(df_store_path, mode='a', header=not df_store_path.exists(), index=False)
                            weight_store.to_csv(weight_store_path, mode='a', header=not weight_store_path.exists(), index=False)
                            train_store = new_storage_df('train', None)
                            weight_store = new_storage_df('weights', dense_metanet.particle_parameter_count)

                ###########################################################
                # EPOCHS endet
                dense_metanet = dense_metanet.eval()

                counter_dict = defaultdict(lambda: 0)
                # This returns ID-functions
                _ = test_for_fixpoints(counter_dict, list(dense_metanet.particles))
                for key, value in dict(counter_dict).items():
                    step_log = dict(Epoch=int(EPOCH), Batch=BATCHSIZE, Metric=key, Score=value)
                    train_store.loc[train_store.shape[0]] = step_log
                accuracy = checkpoint_and_validate(dense_metanet, seed_path, EPOCH, vali_load, final_model=True)
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

            try:
                run_particle_dropout_and_plot(model_path)
            except ValueError as e:
                print(e)
            try:
                plot_network_connectivity_by_fixtype(model_path)
            except ValueError as e:
                print(e)

    if n_seeds >= 2:
        pass
