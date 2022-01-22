# CuPyTorch

CuPyTorch是一个小型[PyTorch](https://pytorch.org/)，名字来源于：

1. 不同于已有的几个使用[NumPy](https://numpy.org/)实现PyTorch的开源项目，本项目通过[CuPy](https://cupy.dev/)支持cuda计算
2. 发音与Cool PyTorch接近，因为使用不超过1000行纯Python代码实现PyTorch确实很cool

CuPyTorch支持numpy和cupy两种计算后端，实现大量PyTorch常用功能，力求99%兼容PyTorch语法语义，并能轻松扩展，以下列出已经完成的功能：

* `tensor`: 
  * `tensor`: 创建张量
  * `arange`: 区间等差张量
  * `stack`: 堆叠张量
  * `ones/zeros`, `ones/zeros_like`: 全1/0张量
  * `rand/randn`, `rand/randn_like`: 0~1均匀分布/高斯分布张量
  * `+`, `-`, `*`, `/`, `@`, `**`: 双目数值运算及其右值和原地操作
  * `>`, `<`, `==`, `>=`, `<=`, `!=`: 比较运算
  * `&`, `|`, `^`: 双目逻辑运算
  * `~`, `-`: 取反/取负运算
  * `[]`: 基本和花式索引和切片操作
  * `abs`, `exp`, `log`, `sqrt`: 数值运算
  * `sum`, `mean`: 数据归约操作
  * `max/min`, `amax/amin`, `argmax/argmin`: 最大/小值及其索引计算
  
* `autograd`: 支持以上所有非整数限定运算的自动微分

* `nn`:
  * `Module`: 模型基类，管理参数，格式化打印
  * `activation`: `ReLU`, `GeLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`
  * `loss`: `L1Loss`, `MSELoss`, `NLLLoss`, `CrossEntropyLoss`
  * `layer`: `Linear`, `Dropout` ,`LSTM`

* `optim`:
  * `Optimizer`: 优化器基类，管理参数，格式化打印
  * `SGD`, `Adam`: 两个最常见的优化器
  * `lr_scheduler`: `LambdaLR`和`StepLR`学习率调度器

* `utils.data`:
  * `DataLoader`: 批量迭代`Tensor`数据，支持随机打乱
  * `Dataset`:  数据集基类，用于继承
  * `TensorDataset`: 纯用`Tensor`构成的数据集

[cloc](https://github.com/AlDanial/cloc)的代码统计结果：

| Language | files | blank | comment | code |
| :------: | :---: | :---: | :-----: | :--: |
|  Python  |  22   |  353  |   27    | 1000  |

自动微分示例：

```python
import cupytorch as ct

a = ct.tensor([[-1., 2], [-3., 4.]], requires_grad=True)
b = ct.tensor([[4., 3.], [2., 1.]], requires_grad=True)
c = ct.tensor([[1., 2.], [0., 2.]], requires_grad=True)
d = ct.tensor([1., -2.], requires_grad=True)
e = a @ b.T
f = (c.max(1)[0].exp() + e[:, 0] + b.pow(2) + 2 * d.reshape(2, 1).abs()).mean()
print(f)
f.backward()
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)

# tensor(18.889057, grad_fn=<MeanBackward>)
# tensor([[2.  1.5]
#         [2.  1.5]])
# tensor([[0.  4.5]
#         [1.  0.5]])
# tensor([[0.       3.694528]
#         [0.       3.694528]])
# tensor([ 1. -1.])
```

手写数字识别示例：

```python
from pathlib import Path
import cupytorch as ct
from cupytorch import nn
from cupytorch.optim import SGD
from cupytorch.optim.lr_scheduler import StepLR
from cupytorch.utils.data import TensorDataset, DataLoader


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


def load(path: Path):
    # define how to load data as tensor
    pass


path = Path('../datasets/MNIST')
train_dl = DataLoader(TensorDataset(load(path / 'train-images-idx3-ubyte.gz'),
                                    load(path / 'train-labels-idx1-ubyte.gz')),
                      batch_size=20, shuffle=True)
test_dl = DataLoader(TensorDataset(load(path / 't10k-images-idx3-ubyte.gz'),
                                   load(path / 't10k-labels-idx1-ubyte.gz')),
                     batch_size=20, shuffle=False)
model = Net(28 * 28, 10)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = StepLR(optimizer, 5, 0.5)

print(model)
print(optimizer)
print(criterion)

for epoch in range(10):
    losses = 0
    for step, (x, y) in enumerate(train_dl, 1):
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if step % 500 == 0:
            losses /= 500
            print(f'Epoch: {epoch}, Train Step: {step}, Train Loss: {losses:.6f}')
            losses = 0
    scheduler.step()
```

`examples`文件夹中提供了两个完整示例：

* 在[MNIST](http://yann.lecun.com/exdb/mnist/)数据集上使用MLP做手写数字分类
* 在[NN5](http://www.neural-forecasting-competition.com/downloads/NN5/datasets/download.htm)数据集上使用LSTM做ATM机取款预测

参考：

* [pytorch](https://github.com/pytorch/pytorch)
* [minitorch](https://github.com/zhouzaida/minitorch)
* [tinygrad](https://github.com/geohot/tinygrad)

