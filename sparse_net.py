from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn

import functionalities_test
from network import Net
from functionalities_test import is_identity_function, test_for_fixpoints, epsilon_error_margin
from tqdm import tqdm, trange
import numpy as np
from pathlib import Path
import torch
from torch.nn import Flatten
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize


def xavier_init(m):
    if isinstance(m, nn.Linear):
        return nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, torch.Tensor):
        return nn.init.xavier_uniform_(m)


class SparseLayer(nn.Module):
    def __init__(self, nr_nets, interface=5, depth=3, width=2, out=1):
        super(SparseLayer, self).__init__()

        self.nr_nets = nr_nets
        self.interface_dim = interface
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
        dummy_net = Net(self.interface_dim, self.hidden_dim, self.out_dim)
        self.dummy_net_shapes = [list(x.shape) for x in dummy_net.parameters()]
        self.dummy_net_weight_pos_enc = dummy_net._weight_pos_enc

        self.sparse_sub_layer = list()
        self.indices = list()
        self.diag_shapes = list()
        self.weights = nn.ParameterList()
        self._particles = None

        for layer_id in range(self.depth_dim):
            indices, weights, diag_shape = self.coo_sparse_layer(layer_id)
            self.indices.append(indices)
            self.diag_shapes.append(diag_shape)
            self.weights.append(weights)
        self.apply(xavier_init)

    def coo_sparse_layer(self, layer_id):
        with torch.no_grad():
            layer_shape = self.dummy_net_shapes[layer_id]
            sparse_diagonal = np.eye(self.nr_nets).repeat(layer_shape[0], axis=-2).repeat(layer_shape[1], axis=-1)
            indices = torch.Tensor(np.argwhere(sparse_diagonal == 1).T, )
        values = torch.nn.Parameter(torch.randn((np.prod((*layer_shape, self.nr_nets)).item())), requires_grad=True)

        return indices, values, sparse_diagonal.shape

    def get_self_train_inputs_and_targets(self):
        # view weights of each sublayer in equal chunks, each column representing weights of one selfrepNN
        # i.e., first interface*hidden weights of layer1, first hidden*hidden weights of layer2
        #  and first hidden*out weights of layer3 = first net
        # [nr_layers*[nr_net*nr_weights_layer_i]]
        with torch.no_grad():
            weights = [layer.view(-1, int(len(layer)/self.nr_nets)).detach() for layer in self.weights]
            # [nr_net*[nr_weights]]
            weights_per_net = [torch.cat([layer[i] for layer in weights]).view(-1, 1) for i in range(self.nr_nets)]
            # (16, 25)

            encoding_matrix, mask = self.dummy_net_weight_pos_enc
            weight_device = weights_per_net[0].device
            if weight_device != encoding_matrix.device or weight_device != mask.device:
                encoding_matrix, mask = encoding_matrix.to(weight_device), mask.to(weight_device)
                self.dummy_net_weight_pos_enc = encoding_matrix, mask

            inputs = torch.hstack(
                [encoding_matrix * mask + weights_per_net[i].expand(-1, encoding_matrix.shape[-1]) * (1 - mask)
                 for i in range(self.nr_nets)]
            )
            targets = torch.hstack(weights_per_net)
            return inputs.T, targets.T

    @property
    def particles(self):
        if self._particles is None:
            self._particles = [Net(self.interface_dim, self.hidden_dim, self.out_dim) for _ in range(self.nr_nets)]
            pass
        else:
            pass

        # Particle Weight Update
        all_weights = [layer.view(-1, int(len(layer) / self.nr_nets)) for layer in self.weights]
        weights_per_net = [torch.cat([layer[i] for layer in all_weights]).view(-1, 1) for i in
                           range(self.nr_nets)]  # [nr_net*[nr_weights]]
        for particles, weights in zip(self._particles, weights_per_net):
            particles.apply_weights(weights)
        return self._particles

    def reset_diverged_particles(self):
        for weights in self.weights:
            if torch.isinf(weights).any() or torch.isnan(weights).any():
                with torch.no_grad():
                    where_nan = torch.nan_to_num(weights, -99, -99, -99)
                    mask = torch.where(where_nan == -99, 0, 1)
                    weights[:] = (where_nan * mask + torch.randn_like(weights) * (1 - mask))[:]

    @property
    def particle_weights(self):
        all_weights = [layer.view(-1, int(len(layer) / self.nr_nets)) for layer in self.weights]
        weights_per_net = [torch.cat([layer[i] for layer in all_weights]).view(-1, 1) for i in
                           range(self.nr_nets)]  # [nr_net*[nr_weights]]
        return weights_per_net

    def replace_weights_by_particles(self, particles):
        assert len(particles) == self.nr_nets
        with torch.no_grad():
            # Particle Weight Update
            all_weights = [list(particle.parameters()) for particle in particles]
            all_weights = [torch.cat(x).view(-1) for x in zip(*all_weights)]
            # [layer.view(-1, int(len(layer) / self.nr_nets)) for layer in self.weights]
            for weights, parameters in zip(all_weights, self.parameters()):
                parameters[:] = weights[:]
        return self

    def __call__(self, x):
        for indices, diag_shapes, weights in zip(self.indices, self.diag_shapes, self.weights):
            s = torch.sparse_coo_tensor(indices, weights, diag_shapes, requires_grad=True, device=x.device)
            x = torch.sparse.mm(s, x)
        return x

    def to(self, *args, **kwargs):
        super(SparseLayer, self).to(*args, **kwargs)
        self.sparse_sub_layer = [sparse_sub_layer.to(*args, **kwargs) for sparse_sub_layer in self.sparse_sub_layer]
        return self


def test_sparse_layer():
    net = SparseLayer(1000)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.008, momentum=0.9)
    # optimizer = torch.optim.SGD([layer.coalesce().values() for layer in net.sparse_sub_layer], lr=0.004, momentum=0.9)
    df = pd.DataFrame(columns=['Epoch', 'Func Type', 'Count'])
    train_iterations = 20000

    for train_iteration in trange(train_iterations):
        optimizer.zero_grad()
        X, Y = net.get_self_train_inputs_and_targets()
        output = net(X)

        loss = loss_fn(output, Y) * 100

        # loss = sum([loss_fn(out, target) for out, target in zip(output, Y)]) / len(output) * 10

        loss.backward()
        optimizer.step()

        if train_iteration % 500 == 0:
            counter = defaultdict(lambda: 0)
            id_functions = functionalities_test.test_for_fixpoints(counter, list(net.particles))
            counter = dict(counter)
            tqdm.write(f"identity_fn after {train_iteration + 1} self-train epochs: {counter}")
            for key, value in counter.items():
                df.loc[df.shape[0]] = (train_iteration, key, value)

    counter = defaultdict(lambda: 0)
    id_functions = functionalities_test.test_for_fixpoints(counter, list(net.particles))
    counter = dict(counter)
    tqdm.write(f"identity_fn after {train_iterations} self-train epochs: {counter}")
    for key, value in counter.items():
        df.loc[df.shape[0]] = (train_iterations, key, value)
    df.to_csv('counter.csv', mode='w')

    c = pd.read_csv('counter.csv', index_col=0)
    sns.lineplot(data=c, x='Epoch', y='Count', hue='Func Type')
    plt.savefig('counter.png', dpi=300)


def embed_batch(x, repeat_dim):
    # x of shape (batchsize, flat_img_dim)

    # (batchsize, flat_img_dim, 1)
    x = x.unsqueeze(-1)
    # (batchsize, flat_img_dim, encoding_dim*repeat_dim)
    # torch.sparse_coo_tensor(indices, weights, diag_shapes, requires_grad=True, device=x.device)
    return torch.cat((torch.zeros(x.shape[0], x.shape[1], 4, device=x.device), x), dim=2).repeat(1, 1, repeat_dim)

def embed_vector(x, repeat_dim):
    # x of shape [flat_img_dim]
    x = x.unsqueeze(-1)  # (flat_img_dim, 1)
    # (flat_img_dim,  encoding_dim*repeat_dim)
    return torch.cat((torch.zeros(x.shape[0], 4), x), dim=1).repeat(1,repeat_dim)


class SparseNetwork(nn.Module):

    @property
    def nr_nets(self):
        return sum(x.nr_nets for x in self.sparselayers)

    def __init__(self, input_dim, depth, width, out, residual_skip=True, activation=None,
                 weight_interface=5, weight_hidden_size=2, weight_output_size=1
                 ):
        super(SparseNetwork, self).__init__()
        self.residual_skip = residual_skip
        self.input_dim = input_dim
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
        self.activation = activation
        self.first_layer = SparseLayer(self.input_dim  * self.hidden_dim,
                                       interface=weight_interface, width=weight_hidden_size, out=weight_output_size)
        self.last_layer = SparseLayer(self.hidden_dim * self.out_dim,
                                      interface=weight_interface, width=weight_hidden_size, out=weight_output_size)
        self.hidden_layers = nn.ModuleList([
            SparseLayer(self.hidden_dim * self.hidden_dim,
                        interface=weight_interface, width=weight_hidden_size, out=weight_output_size
                        ) for _ in range(self.depth_dim - 2)])

    def __call__(self, x):

        tensor = self.sparse_layer_forward(x, self.first_layer)
        if self.activation:
            tensor = self.activation(tensor)
        for nl_idx, network_layer in enumerate(self.hidden_layers):
            # if idx % 2 == 1 and self.residual_skip:
            if self.residual_skip:
                residual = tensor
            tensor = self.sparse_layer_forward(tensor, network_layer)
            # if idx % 2 == 0 and self.residual_skip:
            if self.residual_skip:
                tensor = tensor + residual
        tensor = self.sparse_layer_forward(tensor, self.last_layer, view_dim=self.out_dim)
        return tensor

    def sparse_layer_forward(self, x, sparse_layer, view_dim=None):
        view_dim = view_dim if view_dim else self.hidden_dim
        # batch pass (one by one, sparse bmm doesn't support grad)
        if len(x.shape) > 1:
            embedded_inpt = embed_batch(x, sparse_layer.nr_nets)
            # [batchsize, hidden*inpt_dim, feature_dim]
            x = torch.stack([sparse_layer(inpt.T).sum(dim=1).view(view_dim, x.shape[1]).sum(dim=1) for inpt in
                             embedded_inpt])
        # vector
        else:
            embedded_inpt = embed_vector(x, sparse_layer.nr_nets)
            x = sparse_layer(embedded_inpt.T).sum(dim=1).view(view_dim, x.shape[1]).sum(dim=1)
        return x

    @property
    def particles(self):
        #particles = []
        #particles.extend(self.first_layer.particles)
        #for layer in self.hidden_layers:
        #    particles.extend(layer.particles)
        #particles.extend(self.last_layer.particles)
        return (x for y in (self.first_layer.particles,
                            *(l.particles for l in self.hidden_layers),
                            self.last_layer.particles) for x in y)

    @property
    def particle_weights(self):
        return (x for y in self.sparselayers for x in y.particle_weights)

    def reset_diverged_particles(self):
        for layer in self.sparselayers:
            layer.reset_diverged_particles()

    def to(self, *args, **kwargs):
        super(SparseNetwork, self).to(*args, **kwargs)
        self.first_layer = self.first_layer.to(*args, **kwargs)
        self.last_layer = self.last_layer.to(*args, **kwargs)
        self.hidden_layers = nn.ModuleList([hidden_layer.to(*args, **kwargs) for hidden_layer in self.hidden_layers])
        return self

    @property
    def sparselayers(self):
        return (x for x in (self.first_layer, *self.hidden_layers, self.last_layer))

    def combined_self_train(self, optimizer, reduction='mean'):
        losses = []
        loss_fn = nn.MSELoss(reduction=reduction)
        for layer in self.sparselayers:
            optimizer.zero_grad()
            x, target_data = layer.get_self_train_inputs_and_targets()
            output = layer(x)
            # loss = sum([loss_fn(out, target) for out, target in zip(output, target_data)]) / len(output)

            loss = loss_fn(output, target_data) * layer.nr_nets

            losses.append(loss.detach())
            loss.backward()
            optimizer.step()

        return sum(losses)

    def replace_weights_by_particles(self, particles):
        particles = list(particles)
        for layer in self.sparselayers:
            layer.replace_weights_by_particles(particles[:layer.nr_nets])
            del particles[:layer.nr_nets]
        return self


def test_sparse_net():
    utility_transforms = Compose([ Resize((10, 10)), ToTensor(), Flatten(start_dim=0)])
    data_path = Path('data')
    WORKER = 8
    BATCHSIZE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MNIST(str(data_path), transform=utility_transforms)
    d = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=WORKER)

    data_dim = np.prod(dataset[0][0].shape)
    metanet = SparseNetwork(data_dim, depth=3, width=5, out=10)
    batchx, batchy = next(iter(d))
    out = metanet(batchx)

    result = sum([torch.allclose(out[i], batchy[i], rtol=0, atol=epsilon_error_margin) for i in range(metanet.nr_nets)])
    # print(f"identity_fn after {train_iteration+1} self-train iterations: {result} /{net.nr_nets}")


def test_sparse_net_sef_train():
    sparse_metanet = SparseNetwork(15*15, 5, 6, 10).to('cuda')
    init_st_store_path = Path('counter.csv')
    optimizer = torch.optim.SGD(sparse_metanet.parameters(), lr=0.004, momentum=0.9)
    init_st_epochs = 10000
    init_st_df = pd.DataFrame(columns=['Epoch', 'Func Type', 'Count'])

    for st_epoch in trange(init_st_epochs):
        _ = sparse_metanet.combined_self_train(optimizer)

        if st_epoch % 500 == 0:
            counter = defaultdict(lambda: 0)
            id_functions = test_for_fixpoints(counter, list(sparse_metanet.particles))
            counter = dict(counter)
            tqdm.write(f"identity_fn after {st_epoch} self-train epochs: {counter}")
            for key, value in counter.items():
                init_st_df.loc[init_st_df.shape[0]] = (st_epoch, key, value)
        sparse_metanet.reset_diverged_particles()

    counter = defaultdict(lambda: 0)
    id_functions = test_for_fixpoints(counter, list(sparse_metanet.particles))
    counter = dict(counter)
    tqdm.write(f"identity_fn after {init_st_epochs} self-train epochs: {counter}")
    for key, value in counter.items():
        init_st_df.loc[init_st_df.shape[0]] = (init_st_epochs, key, value)
    init_st_df.to_csv(init_st_store_path, mode='w', index=False)

    c = pd.read_csv(init_st_store_path)
    sns.lineplot(data=c, x='Epoch', y='Count', hue='Func Type')
    plt.savefig(init_st_store_path, dpi=300)


def test_manual_for_loop():
    nr_nets = 500
    nets = [Net(5,2,1) for _ in range(nr_nets)]
    loss_fn = torch.nn.MSELoss(reduction="sum")
    rounds = 1000

    for net in tqdm(nets):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        for i in range(rounds):
            optimizer.zero_grad()
            input_data = net.input_weight_matrix()
            target_data = net.create_target_weights(input_data)
            output = net(input_data)
            loss = loss_fn(output, target_data)
            loss.backward()
            optimizer.step()

    sum([is_identity_function(net) for net in nets])


if __name__ == '__main__':
    # test_sparse_layer()
    test_sparse_net_sef_train()
    # test_sparse_net()
    # for comparison
    # test_manual_for_loop()