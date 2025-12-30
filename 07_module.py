import torch
from torch import nn

class Module_study(nn.Module):
    def __init__(self):
        super(Module_study,self).__init__()

    def forward(self, input):
        output = input + 1
        return output

mod_stu = Module_study()
x = torch.tensor(1.0)
output = mod_stu(x)
print(output)
