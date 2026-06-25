# Pytorch 学习笔记

[PyTorch 官方文档](https://docs.pytorch.ac.cn/docs/stable/index.html)

## 1. Pytorch介绍

**Pytorch基础功能**

- 张量
- 自动求导
- 反向传播
- 神经网络 nn.Module
- 前向传播与损失计算
- 优化器 optim.Adam

**PyTorch 生态系统**

| torchvision | torchtext | torchaudio | 其他专业库 |
| ----------- | --------- | ---------- | ---------- |

**PyTorch 核心**

| torch.nn（神经网络）  | torch.optim（优化器） | torch.utils（工具函数）      |
| --------------------- | --------------------- | ---------------------------- |
| torch核心（张量计算） | autograd（自动微分）  | torch.utils.data（数据加载） |

## 2. tensorboard的使用
- 作用：用于在训练中对训练过程进行可视化

```python
# 使用示例
writer = SummaryWriter("logs")
writer.add_image("train",image_array,1,dataformats='HWC')
writer.close()
```

- 三行代码讲解：

  - 创建一个名为**logs**的目录
  - 在界面中加入图像展示，设置展示窗口名称等
  - 设置结束行代码

- **使用方法：**

  - 运行代码后，会根据路径生成logs文件夹及文件
  - 终端运行：tensorboard --logdir=logs 进入终端中的端口即可

  - 设置指定端口（防止多人使用服务器导致端口冲突）：
                tensorboard --logdir=logs --port = xxxx

## 3. transforms

transforms是一个数据处理/数据增强的库。

- 面向Tensor / PIL图像：两格式可互相转换
- 组合使用：Compose

```python
from torchvision import transforms
```

## 4. 常用的transformer类讲解

| Totensor       | 转换为tensor格式 |
| -------------- | ---------------- |
| **Normalize**  | **标准化**       |
| **Resize**     | **改变图片大小** |
| **Compose**    | **组合多个类**   |
| **RandomCrop** | **随即裁剪**     |

- 这里只放了一些常见的，还有很多类。
  - 使用时注意输入输出
  - 结合**pytorch官方文档**
  - 关注方法需要什么样的参数

## 5. torchvision.datasets
- 一个库，可以直接下载常用数据集

```python
import torchvision
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
```

- 参数设置
  - root：数据集的下载/读取路径

  - train
    - True：参数为数据集中的训练集数据
    - False：参数为数据集中的测试集数据

  - download
    - True：若没有数据集，会自动下载；若有数据集的压缩包，会将其解压
    - False：仅从root读取数据集，若不存在则报错

  - transform：对数据进行函数操作

- Ctrl+P查看函数的参数

## 6. dataloader

数据加载器，从datasets中取数据加载到神经网络中

```python
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)
```

一些常见的参数

- dataset：数据集对象
- batch_size：每次从数据集中拿取的数据数量
- shuffle
  - True，每轮epoch进行随机拿取数据
  - False，每轮epoch顺序拿取数据（每轮同一批次拿取的数据一致）
- num_workers：指定数据加载的工作进程数：默认 0（主进程加载），设为大于 0 的数值时，会创建对应数量子进程并行加载数据，加速预处理。
  - 所有数据加载相关代码**必须放在 `if __name__ == "__main__":` 内**，这是 **Windows 多进程数据加载的强制要求**（Linux/macOS 无此限制）

- drop_last：如果最后一批数据不够数量，舍弃最后一批数据。

**这部分如果没完全理解，参考06代码，调整运行看看变化。**

## 7. nn.Module 神经网络基本骨架

Pytorch中构建自定义神经网络必须继承Module类

- 必须实现的核心方法 : **forward( )**，前向传播，需手动实现（手动定义前向传播过程）

代码示例

```python
import torch
from torch import nn

class Module_study(nn.Module):
    def __init__(self):
        super(Module_study,self).__init__()

    def forward(self, input):
        output = input + 1
        return output

mod_stu = Module_study() # 调用时直接调用类名即可，无需调用forward
x = torch.tensor(1.0)
output = mod_stu(x)
print(output)
```

## 8.1 卷积操作

- 卷积操作：
  - 卷积核，一个小矩阵
  - Stride：卷积核移动步长**（默认值是1）**

卷积计算：用卷积核对输入图像局部区域进行乘积核计算

![1](pngs\1.png)

- 对输入的矩阵和卷积核进行格式转换

[PyTorch 文档](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)中定义的卷积输入和卷积核矩阵（权重矩阵）**shape有4个参数**

![3](pngs\3.png)

所以需要使用**torch.reshape**对这两个矩阵进行形状转换。

```python
import torch

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
```



- 卷积函数+stride参数的使用

```python
# 卷积函数的使用 (stride)
import torch
import torch.nn.functional as F
# 定义input,kernel矩阵...

output = F.conv2d(input,kernel,stride=1)
output2 = F.conv2d(input,kernel,stride=2)
```



- padding——对原矩阵周围的扩展
  - padding = 1时，周围填充一圈0。且从填充后的最左上角开始滑动。




![2](pngs\2.png)

例子说明：原图像大小32，stride=1,若padding=0，输出大小为30。使用padding，输出大小为32

```python
# padding的使用
import torch
import torch.nn.functional as F
# 定义input,kernel矩阵...

output3 = F.conv2d(input,kernel,stride=1,padding=1)
```

**注：8.0的目的是讲解卷积的原理，AI项目使用时应使用8.1的方法。**

## 8.2 神经网络-卷积层

- 三种卷积函数
  - nn.Conv1d 一维卷积
  - nn.Conv2d 二维卷积（图片）
  - nn.Conv3d 三维卷积


- 参数介绍

  ```python
  import torch
  from torch import nn
  from torch.nn import Conv2d
  
  class Run(nn.Module):
      def __init__(self):
          super(Run,self).__init__()
          self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
  
      def forward(self,x):
          x = self.conv1(x)
          return x
  ```

  - **in_channels=3**：输入通道数（适配 CIFAR10数据集 的 3 通道彩色图片）；
  - **out_channels=6**：输出通道数（卷积核的数量，最终输出 6 个特征图）；
  - **kernel_size=3**：卷积核大小（3×3，只需传整数，默认正方形）；
  - **stride=1**：步长 （默认值为1）
  - **padding=0**：周围区域填充（和之前的一样）。

## 9. 最大池化

**池化层的目的：**对图片进行压缩，减小训练时的数据量，提高速度

一般都会在卷积层后加入一层池化层

- Pytorch池化函数的主要参数

| kernel_size   | 池化核大小                                   |
| ------------- | -------------------------------------------- |
| **stride**    | 步长（默认值 = kernel_size)                  |
| **padding**   | 周围区域填充                                 |
| **ceil_mode** | 对边缘小于池化核大小的区域是否取舍（见下图） |

​	**ceil_mode**：若为True（即ceil），则保留不完整窗口，对小于kernel_size的区域也进行最大池化

floor：向下取整； ceil：向上取整

![5](pngs\5.png)

![6](pngs\6.png)

- 池化层计算方法

![4](pngs\4.png)

类似卷积，池化核在输入图像上扫。以上图为例，在输入图像一个3×3的区域内，取最大的数作为该区域的池化结果

## 10. 激活函数

- ReLU

![7](pngs\7.png)

```python
input = -1
Relu(input, inplace=True)
# 结果 input = 0

input = -1
output = Relu(input, inplace=False)
"""
结果
input = -1
output = 0
"""
```

- Sigmoid

![8](pngs\8.png)

## 11. 线形层

- 就是全连接层

- 作用：**将输入特征映射到更高维或更低维的特征空间**

- 核心：对输入进行y = wx+b 操作

```python
import torch.nn as nn
# 展平操作
flatten = nn.Flatten()
# 线性层定义
linear_hidden = nn.Linear(in_features=3072, out_features=1024)

img = torch.randn(1, 3, 32, 32)
img_flat = flatten(img)  # 形状变为 [1, 3072]
output_hidden = linear_hidden(img_flat)  # 形状变为 [1, 1024]
```

```python
# 接上面的隐藏层输出 [1, 1024]
linear_output = nn.Linear(in_features=1024, out_features=10)
output_final = linear_output(output_hidden)  # 形状变为 [1, 10]
```

这个 `[1, 10]` 的输出向量，每个元素对应一个类别的**原始得分（logit）**，再经过 Softmax 函数，就可以转换为每个类别的概率值。

## 12 小实战+Sequential

Sequential的作用：Sequential函数可以直接把神经网络的每一层套进去，变成一个变量，这样只需要对这一个变量实例化即可。

```python
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
```

上述例子也就搭建了一个简单的神经网络

## 13. 损失函数与反向传播

- 损失函数：计算模型推理结果与实际结果的差异，类似考试的分数。
  - 损失值越高，差异越大。
- 反向传播：计算梯度，相当于告诉模型他需要调整的方向。
  - 因为有损失，所以要像靠近推理结果的方向调整。
- 总结：损失函数是告诉模型与目标差异多大，反向传播是告诉模型改进的方向。

```python
# 代码示例
from torch import nn
loss = nn.CrossEntropyLoss()# 损失函数
for data in dataloader:
    imgs, targets = data
    outputs = run(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()# 反向传播
```

## 14. 优化器

优化器核心作用：调整模型参数（权重，偏置等），让loss不断下降。

- 定义（以SGD为例，随机梯度下降）
  - 这里的lr是学习率，run是实例化的模型类（参考上面）

```python
# 1. 定义优化器
optim = torch.optim.SGD(run.parameters(),lr=0.01)
```

- PyTorch 训练模型的 “固定套路”，缺一不可：

```python
# 第一步：梯度清零（必须！）
optim.zero_grad()
# 原因：PyTorch中梯度会累加，如果不清零，本轮的梯度会和上一轮叠加，导致参数更新错误。

# 第二步：反向传播计算梯度
result_loss.backward()
# 作用：从损失函数出发，反向计算每个参数的梯度（即：参数往哪个方向改，能让损失降低）。

# 第三步：更新参数
optim.step()
# 作用：优化器根据计算出的梯度，按照SGD的规则更新模型参数（核心操作）。
```

## 15. 现有网络模型的使用和修改

- 在网络后添加层

```python
model1.add_module('add_linear',nn.Linear(10,4))
```

- 修改网络层

```python
model2 = copy.deepcopy(model1)
model2[8] = nn.Linear(64,4)
```

- 深拷贝

```python
model2 = copy.deepcopy(model1) # 完全复制model1的结构
```

## 16. 模型的保存与加载

即将训练好的模型保存为pth权重文件

- 模型保存的两种方式

```python
import torchvision
import torch
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1--保存模型结构及模型参数
torch.save(vgg16,"vgg16_method1.pth")

# 保存方式2--仅保存模型参数存为字典，不保存模型结构（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")
```

- 模型加载

```python
import torch
import torchvision
# 保存方式1对应的加载模型结构 + 参数方式
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)

# 保存方式2：模型参数，保存成字典的形式（官方推荐，存储空间小一点）
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
```

## 17. 完整的模型训练套路

1. 准备并加载数据集

   ```python
   # 准备数据集
   train_data = torchvision.datasets.CIFAR10(
       root="./data",
       train=True,
       transform=torchvision.transforms.ToTensor(),
       download=True
   )
   
   test_data = torchvision.datasets.CIFAR10(
       root="./data",
       train=False,
       transform=torchvision.transforms.ToTensor(),
       download=True
   )
   # length 长度
   train_data_size = len(train_data)
   test_data_size = len(test_data)
   # 利用 DataLoader 来加载数据集
   train_dataloader = DataLoader(train_data, batch_size=64)
   test_dataloader = DataLoader(test_data, batch_size=64)
   ```

2. 创建网络模型

   ```python
   # 这个calss一般放在单独的model文件中
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
   
   # 实例化
   # 创建网络模型
   run = Run()
   ```

3. 设置损失函数和优化器

   ```python
   # 损失函数
   loss_fn = nn.CrossEntropyLoss()
   
   # 优化器
   learning_rate = 1e-2
   optimizer = torch.optim.SGD(run.parameters(), lr=learning_rate)
   ```

4. log，训练轮数设置

   ```python
   # 设置训练网络的一些参数
   # 记录训练的次数
   total_train_step = 0
   # 记录测试的次数
   total_test_step = 0
   # 训练的轮数
   epoch = 10
   ```

5. 模型训练

   ```python
   for i in range(epoch):
       print("--------第 {} 轮训练开始--------".format(i + 1))
   
       # 训练步骤开始
       for data in train_dataloader:
           imgs, targets = data
           outputs = run(imgs)
           loss = loss_fn(outputs, targets)
   
           # 优化器优化模型
           optimizer.zero_grad() #梯度清零
           loss.backward() #反向传播
           optimizer.step() #优化
   
           total_train_step = total_train_step + 1
           if total_train_step % 100 == 0:
               print("训练次数：{}, Loss：{}".format(total_train_step, loss))
               writer.add_scalar("train_loss",loss.item(),total_train_step)
   ```

6. 模型测试

   ```python
   # 也在上面训练的epoch循环里
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
   ```

最后选择一种模型保存策略，保存效果较好的模型	

## 18. 使用GPU训练

**方法1**：定义device，用to()函数**（常用）**

```python
# 定义一个device训练设备
device = torch.device("cpu")
device = torch.device("cuda")

# 模型
run = Run()
run = run.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 数据
imgs, targets = data
imgs = imgs.to(device)
targets = targets.to(device)
```

- device的几种写法：

  ```python
  device = torch.device("cuda") # 使用默认显卡
  device = torch.device("cuda:0") # 指定了显卡序列号，如果单显卡设别，和上面没区别
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 有gpu就用gpu，没有则用cpu（常用）
  ```

**方法2**：使用cuda( )函数（不常用）

- cuda( )函数的使用对象：
  - 网络模型
  - 数据（输入，标注）
  - 损失函数

使用cuda( )函数时，为了防止设备没有GPU而报错，可以加一个if条件判断。

```python
# Run是模型的实例化类
run = Run()
run = Run()
if torch.cuda.is_available():
    run = run.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 数据
for data in train_dataloader:
    imgs, targets = data
    if torch.cuda.is_available():
        imgs = imgs.cuda()
        targets = targets.cuda()
```

## 19. 完整的模型验证套路

- 保存训练后的模型
- 实例化**保存的模型文件**
- model输入

注意，output时加入2行：

```python
model.eval() # 模型切换到评估模式
with torch.no_grad(): # 关闭自动求导，节省性能
	output = model(image)
```

- 如何查看训练模型的分类种类（分类模型）
  - 在训练中加载测试集的位置加断点，调试模式，查看变量可看到标签。

![9](C:\Users\91274\Desktop\Pytorch_Stu\pytorch_stu_Note\pngs\9.png)





















