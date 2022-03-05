from pathlib import Path

import torch
import torchmetrics

from torch.utils.data import Dataset
from tqdm import tqdm

from experiments.meta_task_utility import set_checkpoint

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


def validate(checkpoint_path, valid_d, ratio=1, validmetric=torchmetrics.MeanAbsoluteError()):
    checkpoint_path = Path(checkpoint_path)

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


def checkpoint_and_validate(model, out_path, epoch_n, valid_d, final_model=False):
    out_path = Path(out_path)
    ckpt_path = set_checkpoint(model, out_path, epoch_n, final_model=final_model)
    result = validate(ckpt_path, valid_d)
    return result


if __name__ == '__main__':
    raise(NotImplementedError('Get out of here'))
