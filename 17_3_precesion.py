# 计算概率
import torch

# 输出结果中，每个结果各个标签的概率
outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])
# 计算结果标签
print(outputs.argmax(1))
"""
argmax参数=1：横着看（0.1和0.2为一个结果的俩标签概率）
           0：竖着看（0.1和0.3为一个结果的两标签概率
"""
preds = outputs.argmax(1)
targets = torch.tensor([0,1])
print(((preds==targets).sum()/2).item())