from collections import defaultdict

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
from network import MetaNet, MetaNetCompareBaseline

def extract_weights_from_model(model:MetaNet)->dict:
    inpt = torch.zeros(5)
    inpt[-1] = 1
    inpt.long()

    weights = defaultdict(list)
    layers = [layer.particles for layer in [model._meta_layer_first, *model._meta_layer_list, model._meta_layer_last]]
    for i, layer in enumerate(layers):
        for net in layer:
            weights[i].append(net(inpt).detach())
    return dict(weights)


def test_weights_as_model(model, new_weights:dict, data):
    TransferNet = MetaNetCompareBaseline(model.interface, depth=model.depth, width=model.width, out=model.out,
                                         residual_skip=True)

    with torch.no_grad():
        for weights, parameters in zip(new_weights.values(), TransferNet.parameters()):
            parameters[:] = torch.Tensor(weights).view(parameters.shape)[:]

    TransferNet.eval()
    metric = torchmetrics.Accuracy()
    with tqdm(desc='Test Batch: ') as pbar:
        for batch, (batch_x, batch_y) in tqdm(enumerate(data), total=len(data), desc='MetaNet Sanity Check'):
            y = TransferNet(batch_x)
            acc = metric(y.cpu(), batch_y.cpu())
            pbar.set_postfix_str(f'Acc: {acc}')
            pbar.update()
                
        # metric on all batches using custom accumulation
        acc = metric.compute()
        tqdm.write(f"Avg. accuracy on all data: {acc}")
        return acc


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WORKER = 0
    BATCHSIZE = 500
    MNIST_TRANSFORM = Compose([Resize((15, 15)), ToTensor(), Flatten(start_dim=0)])
    torch.manual_seed(42)
    data_path = Path('data')
    data_path.mkdir(exist_ok=True, parents=True)
    mnist_test = MNIST(str(data_path), transform=MNIST_TRANSFORM, download=True, train=False)
    d_test = DataLoader(mnist_test, batch_size=BATCHSIZE, shuffle=False, drop_last=True, num_workers=WORKER)
    
    model = torch.load(Path('experiments/output/trained_model_ckpt_e50.tp'), map_location=DEVICE).eval()
    weights = extract_weights_from_model(model)
    test_weights_as_model(model, weights, d_test)
    
