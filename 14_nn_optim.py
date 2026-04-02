import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
# 优化器
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Run(nn.Module):
    def __init__(self):
        super(Run, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
run = Run()
optim = torch.optim.SGD(run.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    # 对所有数据进行了一轮的学习
    # 想要让模型对每个数据的损失降低，要进行多轮epoch
    for data in dataloader:
        imgs, targets = data
        outputs = run(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()# 将优化器的每个参数清零
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)





















