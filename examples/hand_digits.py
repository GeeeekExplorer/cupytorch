import gzip
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import cupytorch as ct
from cupytorch import nn
from cupytorch.optim import Optimizer, SGD
from cupytorch.optim.lr_scheduler import StepLR
from cupytorch.utils.data import TensorDataset, DataLoader


def load(path: Path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], byteorder='big')
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = {8: 'uint8', 9: 'int8', 11: 'int16',
         12: 'int32', 13: 'float32', 14: 'float64'}
    s = [int.from_bytes(data[4 * (i + 1): 4 * (i + 2)], byteorder='big')
         for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[ty], offset=(4 * (nd + 1)))
    if 'images' in str(path):
        parsed = parsed.astype('float32') / 255
        parsed = (parsed - 0.1307) / 0.3081
    elif 'labels':
        parsed = parsed.astype('int64')
    return ct.tensor(parsed.reshape(*s))


class Net(nn.Module):

    def __init__(self, num_pixel: int, num_class: int):
        super().__init__()
        self.num_pixel = num_pixel
        self.fc1 = nn.Linear(num_pixel, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_class)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, input: ct.Tensor) -> ct.Tensor:
        output = input.view(-1, self.num_pixel)
        output = self.drop(self.act(self.fc1(output)))
        output = self.drop(self.act(self.fc2(output)))
        return self.fc3(output)


def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer):
    model.train()
    losses = 0
    for step, (x, y) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if step % 500 == 0:
            yield step, losses / 500
            losses = 0


@ct.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    losses, acc = 0, 0
    for x, y in dataloader:
        z = model(x)
        loss = criterion(z, y)
        losses += loss.item()
        acc += z.argmax(-1).eq(y).mean().item()
    return losses / len(dataloader), acc / len(dataloader)


if __name__ == '__main__':
    epochs = 10
    path = Path('../datasets/MNIST')
    train_loader = DataLoader(TensorDataset(load(path / 'train-images-idx3-ubyte.gz'),
                                            load(path / 'train-labels-idx1-ubyte.gz')),
                              batch_size=20, shuffle=True)
    test_loader = DataLoader(TensorDataset(load(path / 't10k-images-idx3-ubyte.gz'),
                                           load(path / 't10k-labels-idx1-ubyte.gz')),
                             batch_size=20, shuffle=False)

    model = Net(28 * 28, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # scheduler = StepLR(optimizer, 3, 0.5)

    print(model)
    print(optimizer)
    print(criterion)

    train_acc, eval_acc = [], []

    for epoch in range(epochs):

        for step, loss in train(model, train_loader, criterion, optimizer):
            print(f'Epoch: {epoch}, Train Step: {step}, Train Loss: {loss:.6f}')

        loss, acc = evaluate(model, train_loader, criterion)
        print(f'Epoch: {epoch}, Train Loss: {loss:.6f}, Train Accuracy: {acc * 100:.4f}%')
        train_acc.append(acc)

        loss, acc = evaluate(model, test_loader, criterion)
        print(f'Epoch: {epoch}, Test Loss: {loss:.6f}, Test Accuracy: {acc * 100:.4f}%')
        eval_acc.append(acc)

        # scheduler.step()

    plt.figure()
    plt.plot(range(1, epochs + 1), train_acc, label='train acc')
    plt.plot(range(1, epochs + 1), eval_acc, label='eval acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy VS Test Accuracy')
    plt.legend()
    plt.savefig('acc.png')
