import torch
import torchvision
# 保存方式1对应的加载模型结构 + 参数方式
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)

# 保存方式2：模型参数，保存成字典的形式（官方推荐，存储空间小一点）
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
