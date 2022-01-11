import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

mnist = MNIST(root="./data",train=True,download=False,transform=None)
ret = transforms.ToTensor()(mnist[0][0])
print(ret.size())