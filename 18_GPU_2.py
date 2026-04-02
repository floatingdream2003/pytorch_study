# 完整的训练流程
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from model_17_2 import *
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果 train_data_size = 10，训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

run = Run()
run = run.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(run.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

best_accuracy = 0

# tensorboard
# writer = SummaryWriter("./logs_train")


for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i + 1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = run(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad() #梯度清零
        loss.backward() #反向传播
        optimizer.step() #优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss))
            # writer.add_scalar("train_loss",loss.item(),total_train_step)
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():# 测试中，不需要对梯度调整，设置梯度都为0
        for data in test_dataloader:
            imgs,targets = data
            outputs = run(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试数据集上的loss:{}".format(total_test_loss))
    print("整体的正确率：{}".format(total_accuracy/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    if total_accuracy>best_accuracy:
        best_accuracy = total_accuracy
        torch.save(run,"run_{}.pth".format(i))
        print("模型已保存")
# writer.close()




