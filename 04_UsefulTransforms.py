import cv2 as cv
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# 04 常见的transforms工具类讲解
# ----------目录----------
# 1.Totensor
# 2.Normalize
# 3.Resize
# -----------------------

writer = SummaryWriter("./logs/logs")
# opencv的图片通道是BGR，而TensorBoard期望的是RGB，会导致图片颜色改变（opencv也有函数可以解决）
# img = cv.imread("dataset/train/cat/img.png")

img = Image.open("dataset/train/cat/img.png")
print(img)

# Totensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor",img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])# 两个参数分别是每个通道的均值喝标准差
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img --> resize --> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL --> totensor --> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
# 组合两个函数，compose只能输入数组，注意前后函数的输出输入格式是否一致，否则报错
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随即裁剪
trans_random = transforms.RandomCrop(500, 1000)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHi", img_crop, i)

writer.close()