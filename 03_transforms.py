import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# transforms是一个工具箱，用来处理图片
# 通过transforms.ToTensor去看两个问题
# 1、transforms如何使用
# 2、为什么需要Tensor数据集

# 绝对路径 C:\Users\91274\Desktop\pytorch_study\dataset\train\cat\img.png
# 相对路径 dataset/train/cat/img.png
img_path = "dataset/train/cat/img.png"
# img = cv2.imread(img_path)
img = Image.open(img_path)

writer = SummaryWriter("./logs/logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
