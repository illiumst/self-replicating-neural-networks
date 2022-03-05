import torch
from torch.utils.data import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AddTaskDataset(Dataset):
    def __init__(self, length=int(1e3)):
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ab = torch.randn(size=(2,)).to(torch.float32)
        return ab, ab.sum(axis=-1, keepdims=True)


def train_task(model, optimizer, loss_func, btch_x, btch_y) -> (dict, torch.Tensor):
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    btch_x, btch_y = btch_x.to(DEVICE), btch_y.to(DEVICE)
    y_prd = model(btch_x)

    loss = loss_func(y_prd, btch_y.to(torch.float))
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    stp_log = dict(Metric='Task Loss', Score=loss.item())

    return stp_log, y_prd


if __name__ == '__main__':
    raise(NotImplementedError('Get out of here'))
