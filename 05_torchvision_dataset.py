import torchvision
# torchvision里有很多数据集，可以通过下面的方式下载。(CTRL+P可以查看参数)
# 也可以用网盘下载后运行这行，会自动解压
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)

print(test_set[0])
print(test_set.classes)

img,target= test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()


















