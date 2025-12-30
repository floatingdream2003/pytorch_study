from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
"""
tensorboard的使用
用于在训练中对训练过程进行可视化
"""

writer = SummaryWriter("./logs/logs")
image_path = "dataset/train/cat/img_1.png"
img_PIL = Image.open(image_path)
image_array = np.array(img_PIL)
# print(type(image_array))
# print(image_array.shape)

writer.add_image("train",image_array,1,dataformats='HWC')

for i in range(20):
    writer.add_scalar("y=2x",3*i,i)
"""
这里在目录下创建了一个logs文件夹
打开logs下的文件：
            tensorboard --logdir=logs
设置指定端口（防止多人使用服务器导致端口冲突）：
            tensorboard --logdir=logs --port = xxxx
"""
writer.close()