import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Run(nn.Module):
    def __init__(self):
        super(Run, self).__init__()
        self.linear1 = Linear(196608, 10)# 输入维度和输出维度

    def forward(self,input):
        output = self.linear1(input)
        return output

run = Run()

for data in dataloader:
    imgs,targets = data
    # print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,-1))
    print(output.shape)
    output = run(output)
    print(output.shape)
    break