import sys
from collections import defaultdict
from pathlib import Path
import platform

import pandas as pd
import torch.optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate


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

import functionalities_test
from network import Net


class MultiplyByXTaskDataset(Dataset):
    def __init__(self, x=0.23, length=int(5e5)):
        super().__init__()
        self.length = length
        self.x = x
        self.prng = np.random.default_rng()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ab = self.prng.normal(size=(1,)).astype(np.float32)
        return ab, ab * self.x


if __name__ == '__main__':
    net = Net(5, 4, 1, lr=0.004)
    multiplication_target = 0.03
    st_steps = 0

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

    train_frame = pd.DataFrame(columns=['Epoch', 'Batch', 'Metric', 'Score'])

    dataset = MultiplyByXTaskDataset(x=multiplication_target, length=1000000)
    dataloader = DataLoader(dataset=dataset, batch_size=8000, num_workers=0)
    for epoch in trange(30):
        mean_batch_loss = []
        mean_self_tain_loss = []

        for batch, (batch_x, batch_y) in tenumerate(dataloader):
            self_train_loss, _ = net.self_train(1000 // 20, save_history=False)
            is_fixpoint = functionalities_test.is_identity_function(net)
            if not is_fixpoint:
                st_steps += 2

            if is_fixpoint:
                tqdm.write(f'is fixpoint after st : {is_fixpoint}, first reached after st_steps: {st_steps}')
                tqdm.write(f'is fixpoint after tsk: {functionalities_test.is_identity_function(net)}')

            #mean_batch_loss.append(loss.detach())
            mean_self_tain_loss.append(self_train_loss.detach())

        train_frame.loc[train_frame.shape[0]] = dict(Epoch=epoch, Batch=batch,
                                                     Metric='Self Train Loss', Score=np.average(mean_self_tain_loss))
        train_frame.loc[train_frame.shape[0]] = dict(Epoch=epoch, Batch=batch,
                                                         Metric='Batch Loss', Score=np.average(mean_batch_loss))

    counter = defaultdict(lambda: 0)
    functionalities_test.test_for_fixpoints(counter, nets=[net])
    print(dict(counter), self_train_loss)
    sanity = net(torch.Tensor([0,0,0,0,1])).detach()
    print(sanity)
    print(abs(sanity - multiplication_target))
    sns.lineplot(data=train_frame, x='Epoch', y='Score', hue='Metric')
    outpath = Path('output') / 'sanity' / 'test.png'
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath)







