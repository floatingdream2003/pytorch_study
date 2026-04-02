import torchvision
import torch
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1--保存模型结构及模型参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2--仅保存模型参数存为字典，不保存模型结构（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
