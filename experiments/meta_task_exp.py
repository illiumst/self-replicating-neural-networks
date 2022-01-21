import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from network import MetaNet


class TaskDataset(Dataset):
    def __init__(self, length=int(5e5)):
        super().__init__()
        self.length = length
        self.prng = np.random.default_rng()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ab = self.prng.normal(size=(2,)).astype(np.float32)
        return ab, ab.sum(axis=-1, keepdims=True)


if __name__ == '__main__':
    metanet = MetaNet(2, 3, 4, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(metanet.parameters(), lr=0.004)

    d = DataLoader(TaskDataset(), batch_size=50, shuffle=True, drop_last=True)
    # metanet.train(True)
    losses = []
    for batch_x, batch_y in tqdm(d, total=len(d)):
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        y = metanet(batch_x)
        loss = loss_fn(y, batch_y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        losses.append(loss.item())

    sns.lineplot(y=np.asarray(losses), x=np.arange(len(losses)))
    plt.show()


