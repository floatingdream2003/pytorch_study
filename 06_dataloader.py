import torchvision

# 准备测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)
# dataloader(batch_size=4)它会从dataset中取4个数据

# writer = SummaryWriter("./logs/logs_dataloader")
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs, targets = data
        # writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step+1
# writer.close()

# 终端输入: tensorboard --logdir=dataloader
