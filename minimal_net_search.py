from tqdm import tqdm
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Grayscale
import torchmetrics
import pickle

from network import MetaNetCompareBaseline

WORKER = 0
BATCHSIZE = 500
EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MNIST_TRANSFORM = Compose([ Resize((10, 10)), ToTensor(), Normalize((0.1307,), (0.3081,)), Flatten(start_dim=0)])
CIFAR10_TRANSFORM = Compose([ Grayscale(num_output_channels=1), Resize((10, 10)), ToTensor(), Normalize((0.48,), (0.25,)), Flatten(start_dim=0)])


def train_and_test(testnet, optimizer, loss, trainset, testset):
    d_train = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=False, drop_last=True, num_workers=WORKER)
    d_test = DataLoader(testset, batch_size=BATCHSIZE, shuffle=False, drop_last=True, num_workers=WORKER)
    
    # train
    for epoch in tqdm(range(EPOCH), desc='MetaNet Train - Epoch'):
        for batch, (batch_x, batch_y) in enumerate(d_train):
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            y = testnet(batch_x)
            loss = loss_fn(y, batch_y)
            loss.backward()
            optimizer.step()

    # test
    testnet.eval()
    metric = torchmetrics.Accuracy()
    with tqdm(desc='Test Batch: ') as pbar:
        for batch, (batch_x, batch_y) in tqdm(enumerate(d_test), total=len(d_test), desc='MetaNet Test - Batch'):
            y = testnet(batch_x)
            loss = loss_fn(y, batch_y)
            acc = metric(y.cpu(), batch_y.cpu())
            pbar.set_postfix_str(f'Acc: {acc}')
            pbar.update()

        acc = metric.compute()
        tqdm.write(f"Avg. accuracy on all data: {acc}")
    return acc

if __name__ == '__main__':
    torch.manual_seed(42)
    data_path = Path('data')
    data_path.mkdir(exist_ok=True, parents=True)
    mnist_train = MNIST(str(data_path), transform=MNIST_TRANSFORM, download=True, train=True)
    mnist_test = MNIST(str(data_path), transform=MNIST_TRANSFORM, download=True, train=False)
    cifar10_train = CIFAR10(str(data_path), transform=CIFAR10_TRANSFORM, download=True, train=True)
    cifar10_test = CIFAR10(str(data_path), transform=CIFAR10_TRANSFORM, download=True, train=False)
    loss_fn = nn.CrossEntropyLoss()
    frame = pd.DataFrame(columns=['Dataset', 'Neurons', 'Layers', 'Parameters', 'Accuracy'])

    for name, trainset, testset in [("MNIST",mnist_train,mnist_test), ("CIFAR10",cifar10_train,cifar10_test)]:
        best_acc = 0
        neuron_count = 0
        layer_count = 0
        
        # find upper bound (in steps of 10, neurons/layer > 200 will start back from 10 with layers+1)
        while best_acc <= 0.95:
            neuron_count += 10
            if neuron_count >= 210:
                neuron_count = 10
                layer_count += 1
            net = MetaNetCompareBaseline(100, layer_count, neuron_count, out=10)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.9)
            acc = train_and_test(net, optimizer, loss_fn, trainset, testset)
            if acc > best_acc:
                best_acc = acc
            
            num_params = sum(p.numel() for p in net._meta_layer_list.parameters())
            frame.loc[frame.shape[0]] = dict(Dataset=name, Neurons=neuron_count, Layers=layer_count, Parameters=num_params, Accuracy=acc)
            print(f"> {name}\t| {neuron_count} neurons\t| {layer_count} h.-layer(s)\t| {num_params} params\n")

        print(frame)
        pickle.dump(frame, "min_net_search_df.pkl")