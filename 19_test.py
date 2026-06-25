import torch
import torchvision
from PIL import Image
# 完整的模型验证套路

image_path = "imgs/airplane.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)
# print(image.shape)

model = torch.load("run_9.pth", weights_only=False)
# print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval() # 模型切换到评估模式
with torch.no_grad(): # 关闭自动求导，节省性能
    output = model(image)
print(output)