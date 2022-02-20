from torch import nn

from network import Net
from functionalities_test import is_identity_function
from tqdm import tqdm,trange
import numpy as np
from pathlib import Path
import torch
from torch.nn import Flatten
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize


class SparseLayer(nn.Module):
    def __init__(self, nr_nets, interface=5, depth=3, width=2, out=1):
        super(SparseLayer, self).__init__()

        self.nr_nets = nr_nets
        self.interface_dim = interface
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
        self.dummy_net = Net(self.interface_dim, self.hidden_dim, self.out_dim)

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

    def coo_sparse_layer(self, layer_id):
        layer_shape = list(self.dummy_net.parameters())[layer_id].shape
        sparse_diagonal = np.eye(self.nr_nets).repeat(layer_shape[0], axis=-2).repeat(layer_shape[1], axis=-1)
        indices = torch.Tensor(np.argwhere(sparse_diagonal == 1).T)
        values = torch.nn.Parameter(
            torch.randn((self.nr_nets * (layer_shape[0]*layer_shape[1]))), requires_grad=True
        )

        return indices, values, sparse_diagonal.shape

    def get_self_train_inputs_and_targets(self):
        encoding_matrix, mask = self.dummy_net._weight_pos_enc

        # view weights of each sublayer in equal chunks, each column representing weights of one selfrepNN
        # i.e., first interface*hidden weights of layer1, first hidden*hidden weights of layer2
        #  and first hidden*out weights of layer3 = first net
        # [nr_layers*[nr_net*nr_weights_layer_i]]
        weights = [layer.view(-1, int(len(layer)/self.nr_nets)) for layer in self.weights]
        # [nr_net*[nr_weights]]
        weights_per_net = [torch.cat([layer[i] for layer in weights]).view(-1, 1) for i in range(self.nr_nets)]
        # (16, 25)
        inputs = torch.hstack(
            [encoding_matrix * mask + weights_per_net[i].expand(-1, encoding_matrix.shape[-1]) * (1 - mask)
             for i in range(self.nr_nets)]
        )
        targets = torch.hstack(weights_per_net)
        return inputs.T.detach(), targets.T.detach()

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
    net = SparseLayer(500) #50 parallel nets
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(net.weights, lr=0.004, momentum=0.9)
    # optimizer = torch.optim.SGD([layer.coalesce().values() for layer in net.sparse_sub_layer], lr=0.004, momentum=0.9)

    for train_iteration in trange(1000):
        optimizer.zero_grad()
        X,Y = net.get_self_train_inputs_and_targets()
        out = net(X)

        loss = loss_fn(out, Y)

        # print("X:", X.shape, "Y:", Y.shape)
        # print("OUT", out.shape)
        # print("LOSS", loss.item())

        loss.backward(retain_graph=True)
        optimizer.step()

    epsilon=pow(10, -5)
    # is each of the networks self-replicating?
    print(f"identity_fn after {train_iteration+1} self-train iterations: {sum([torch.allclose(out[i], Y[i], rtol=0, atol=epsilon) for i in range(net.nr_nets)])}/{net.nr_nets}")


def embed_batch(x, repeat_dim):
    # x of shape (batchsize, flat_img_dim)
    x = x.unsqueeze(-1) #(batchsize, flat_img_dim, 1)
    return torch.cat((torch.zeros(x.shape[0], x.shape[1], 4, device=x.device), x), dim=2).repeat(1, 1, repeat_dim) #(batchsize, flat_img_dim, encoding_dim*repeat_dim)


def embed_vector(x, repeat_dim):
    # x of shape [flat_img_dim]
    x = x.unsqueeze(-1)  # (flat_img_dim, 1)
    # (flat_img_dim,  encoding_dim*repeat_dim)
    return torch.cat((torch.zeros(x.shape[0], 4), x), dim=1).repeat(1,repeat_dim)


class SparseNetwork(nn.Module):
    def __init__(self, input_dim, depth, width, out, residual_skip=True,
                 weight_interface=5, weight_hidden_size=2, weight_output_size=1
                 ):
        super(SparseNetwork, self).__init__()
        self.residual_skip = residual_skip
        self.input_dim = input_dim
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
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
        for nl_idx, network_layer in enumerate(self.hidden_layers):
            if nl_idx % 2 == 0 and self.residual_skip:
                residual = tensor.clone()
            # Sparse Layer pass
            tensor = self.sparse_layer_forward(tensor, network_layer)

            if nl_idx % 2 != 0 and self.residual_skip:
                # noinspection PyUnboundLocalVariable
                tensor += residual
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
        particles = []
        particles.extend(self.first_layer.particles)
        for layer in self.hidden_layers:
            particles.extend(layer.particles)
        particles.extend(self.last_layer.particles)
        return iter(particles)

    def to(self, *args, **kwargs):
        super(SparseNetwork, self).to(*args, **kwargs)
        self.first_layer = self.first_layer.to(*args, **kwargs)
        self.last_layer = self.last_layer.to(*args, **kwargs)
        self.hidden_layers = nn.ModuleList([hidden_layer.to(*args, **kwargs) for hidden_layer in self.hidden_layers])
        return self

    def combined_self_train(self):
        import time
        t = time.time()
        losses = []
        for layer in [self.first_layer, *self.hidden_layers, self.last_layer]:
            x, target_data = layer.get_self_train_inputs_and_targets()
            output = layer(x)

            losses.append(F.mse_loss(output, target_data))
        print('Time Taken:', time.time() - t)
        return torch.hstack(losses).sum(dim=-1, keepdim=True)


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
    metanet(batchx)


def test_manual_for_loop():
    nr_nets = 500
    nets = [Net(5,2,1) for _ in range(nr_nets)]
    loss_fn = torch.nn.MSELoss(reduction="sum")
    rounds = 1000

    for net in tqdm(nets):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.9)
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
    test_sparse_layer()
    # test_sparse_net()
    # for comparison
    test_manual_for_loop()