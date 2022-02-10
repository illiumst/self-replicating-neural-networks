from network import Net
from typing import List
from functionalities_test import is_identity_function
from tqdm import tqdm,trange
import numpy as np
from pathlib import Path
import torch
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize


class SparseLayer():
    def __init__(self, nr_nets, interface=5, depth=3, width=2, out=1):
        self.nr_nets = nr_nets
        self.interface_dim = interface
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
        self.dummy_net = Net(self.interface_dim, self.hidden_dim, self.out_dim)
        
        self.sparse_sub_layer = []
        self.weights = []
        for layer_id in range(depth):
            layer, weights = self.coo_sparse_layer(layer_id)
            self.sparse_sub_layer.append(layer)
            self.weights.append(weights)
 
    def coo_sparse_layer(self, layer_id):
        layer_shape = list(self.dummy_net.parameters())[layer_id].shape
        #print(layer_shape) #(out_cells, in_cells) -> (2,5), (2,2), (1,2)

        sparse_diagonal = np.eye(self.nr_nets).repeat(layer_shape[0], axis=-2).repeat(layer_shape[1], axis=-1)
        indices = np.argwhere(sparse_diagonal == 1).T
        values = torch.nn.Parameter(torch.randn((self.nr_nets * (layer_shape[0]*layer_shape[1]) )))
        #values = torch.randn((self.nr_nets * layer_shape[0]*layer_shape[1] ))
        s = torch.sparse_coo_tensor(indices, values, sparse_diagonal.shape, requires_grad=True)
        print(f"L{layer_id}:", s.shape)
        return s, values

    def get_self_train_inputs_and_targets(self):
        encoding_matrix, mask = self.dummy_net._weight_pos_enc

        # view weights of each sublayer in equal chunks, each column representing weights of one selfrepNN
        # i.e., first interface*hidden weights of layer1, first hidden*hidden weights of layer2 and first hidden*out weights of layer3 = first net
        weights = [layer.view(-1, int(len(layer)/self.nr_nets)) for layer in self.weights]    #[nr_layers*[nr_net*nr_weights_layer_i]]
        weights_per_net = [torch.cat([layer[i] for layer in weights]).view(-1,1) for i in range(self.nr_nets)]   #[nr_net*[nr_weights]]
        inputs = torch.hstack([encoding_matrix * mask + weights_per_net[i].expand(-1, encoding_matrix.shape[-1]) * (1 - mask) for i in range(self.nr_nets)]) #(16, 25)
        targets = torch.hstack(weights_per_net)
        return inputs.T, targets.T

    def __call__(self, x):
        X1 = torch.sparse.mm(self.sparse_sub_layer[0], x)
        #print("X1", X1.shape)

        X2 = torch.sparse.mm(self.sparse_sub_layer[1], X1)
        #print("X2", X2.shape)

        X3 = torch.sparse.mm(self.sparse_sub_layer[2], X2)
        #print("X3", X3.shape)
        
        return X3


def test_sparse_layer():
    net = SparseLayer(500) #50 parallel nets
    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD([weight for weight in net.weights], lr=0.004, momentum=0.9)
    #optimizer = torch.optim.SGD([layer for layer in net.sparse_sub_layer], lr=0.004, momentum=0.9)
    
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
    return torch.cat( (torch.zeros( x.shape[0], x.shape[1], 4), x), dim=2).repeat(1,1,repeat_dim) #(batchsize, flat_img_dim, encoding_dim*repeat_dim)

def embed_vector(x, repeat_dim):
    # x of shape [flat_img_dim]
    x = x.unsqueeze(-1) #(flat_img_dim, 1)
    return torch.cat( (torch.zeros( x.shape[0], 4), x), dim=1).repeat(1,repeat_dim) #(flat_img_dim,  encoding_dim*repeat_dim)

class SparseNetwork():
    def __init__(self, input_dim, depth, width, out):
        self.input_dim = input_dim
        self.depth_dim = depth
        self.hidden_dim = width
        self.out_dim = out
        self.sparse_layers = []
        self.sparse_layers.append(  SparseLayer( self.input_dim  * self.hidden_dim  ))
        self.sparse_layers.extend([ SparseLayer( self.hidden_dim * self.hidden_dim  ) for layer_idx in range(self.depth_dim - 2)])
        self.sparse_layers.append(  SparseLayer( self.hidden_dim * self.out_dim     ))

    def __call__(self, x):
        
        for sparse_layer in self.sparse_layers[:-1]:
            # batch pass (one by one, sparse bmm doesn't support grad)
            if len(x.shape) > 1:
                embedded_inpt = embed_batch(x, sparse_layer.nr_nets)
                x = torch.stack([sparse_layer(inpt.T).sum(dim=1).view(self.hidden_dim, x.shape[1]).sum(dim=1) for inpt in embedded_inpt]) #[batchsize, hidden*inpt_dim, feature_dim]
            # vector
            else:
                embedded_inpt = embed_vector(x, sparse_layer.nr_nets)
                x = sparse_layer(embedded_inpt.T).sum(dim=1).view(self.hidden_dim, x.shape[1]).sum(dim=1)
            print("out", x.shape)
        
        # output layer
        sparse_layer = self.sparse_layers[-1]
        if len(x.shape) > 1:
            embedded_inpt = embed_batch(x, sparse_layer.nr_nets)
            x = torch.stack([sparse_layer(inpt.T).sum(dim=1).view(self.out_dim, x.shape[1]).sum(dim=1) for inpt in embedded_inpt]) #[batchsize, hidden*inpt_dim, feature_dim]
        else:
            embedded_inpt = embed_vector(x, sparse_layer.nr_nets)
            x = sparse_layer(embedded_inpt.T).sum(dim=1).view(self.out_dim, x.shape[1]).sum(dim=1)
        print("out", x.shape)
        return x


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
    batchx.shape, batchy.shape
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
    test_sparse_net()
    #for comparison
    test_manual_for_loop()