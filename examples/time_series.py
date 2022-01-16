import sys
from pathlib import Path

import numpy as np

sys.path.append('..')
import cupytorch as ct
from cupytorch import nn
from cupytorch.optim import Optimizer, SGD, Adam
from cupytorch.utils.data import TensorDataset, DataLoader


def load(path: Path, input_len: int, output_len: int):
    raw_train_data = np.loadtxt(path / 'train.csv', delimiter=',')
    raw_test_data = np.loadtxt(path / 'test.csv', delimiter=',')

    mu = raw_train_data.mean(1, keepdims=True)
    sigma = raw_train_data.std(1, keepdims=True)
    raw_train_data = (raw_train_data - mu) / sigma
    raw_test_data = (raw_test_data - mu) / sigma
    globals()['mu'] = ct.tensor(mu)
    globals()['sigma'] = ct.tensor(sigma)

    test_data = raw_train_data[:, -input_len:]
    test_labels = raw_test_data
    train_data = np.concatenate([raw_train_data[:, i:i + input_len]
                                 for i in range(raw_train_data.shape[1] - input_len - output_len)])
    train_labels = np.concatenate([raw_train_data[:, i:i + output_len]
                                   for i in range(input_len, raw_train_data.shape[1] - output_len)])

    return [ct.tensor(t) for t in [train_data, train_labels, test_data, test_labels]]


class Net(nn.Module):

    def __init__(self, input_len: int, output_len: int, hidden_size: int):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hidden_size
        self.encoder = nn.LSTMCell(1, hidden_size)
        self.decoder = nn.LSTMCell(1, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input: ct.Tensor) -> ct.Tensor:
        batch_size = input.shape[0]
        input = input.t().view(self.input_len, batch_size, 1)
        h = c = ct.zeros(batch_size, self.hidden_size)
        for i in range(self.input_len):
            h, c = self.encoder(input[i], (h, c))
        o = ct.zeros(batch_size, 1)
        output = []
        for i in range(self.output_len):
            h, c = self.decoder(o, (h, c))
            o = self.fc(h)
            output.append(o)
        output = ct.stack(output).view(self.output_len, batch_size).t()
        return output


def metric(pred: ct.Tensor, target: ct.Tensor, indices: ct.Tensor):
    global mu, sigma
    mu, sigma = mu[indices], sigma[indices]
    pred = pred * sigma + mu
    target = target * sigma + mu
    return 2 * ((pred - target).abs() / (pred.abs() + target.abs())).mean()


def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer):
    model.train()
    losses = 0
    for step, (x, y, _) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if step % 10 == 0:
            yield step, losses / 10
            losses = 0


@ct.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    losses, smape = 0, 0
    for x, y, i in dataloader:
        z = model(x)
        loss = criterion(z, y)
        losses += loss.item()
        smape += metric(z, y, i).item()
    return losses / len(dataloader), smape / len(dataloader)


if __name__ == '__main__':
    input_len = 28
    output_len = 56

    path = Path('../datasets/NN5/')
    train_data, train_labels, test_data, test_labels = load(path, input_len, output_len)

    indices = np.random.permutation(len(train_data))
    train_indices = indices[:-len(indices) // 5]
    eval_indices = indices[-len(indices) // 5:]

    eval_data, eval_labels = train_data[eval_indices], train_labels[eval_indices]
    train_data, train_labels = train_data[train_indices], train_labels[train_indices]

    train_indices = ct.tensor(train_indices % len(test_data))
    eval_indices = ct.tensor(eval_indices % len(test_data))
    test_indices = ct.arange(len(test_data))
    train_loader = DataLoader(TensorDataset(train_data, train_labels, train_indices), batch_size=200, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_data, eval_labels, eval_indices), batch_size=200, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data, test_labels, test_indices), batch_size=200, shuffle=False)

    model = Net(input_len, output_len, 20)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    print(model)
    print(optimizer)
    print(criterion)

    for epoch in range(2):

        for step, loss in train(model, train_loader, criterion, optimizer):
            print(f'Epoch: {epoch}, Train Step: {step}, Train Loss: {loss:.6f}')

        loss, smape = evaluate(model, train_loader, criterion)
        print(f'Epoch: {epoch}, Train Loss: {loss:.6f}, Train sMAPE: {smape * 100:.4f}%')
        loss, smape = evaluate(model, eval_loader, criterion)
        print(f'Epoch: {epoch}, Eval Loss: {loss:.6f}, Eval sMAPE: {smape * 100:.4f}%')
        loss, smape = evaluate(model, test_loader, criterion)
        print(f'Epoch: {epoch}, Test Loss: {loss:.6f}, Test sMAPE: {smape * 100:.4f}%')
