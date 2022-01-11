import torch
import torch.nn as nn
from torch.optim import SGD

#准备数据

x = torch.rand([500,1])
y_true = 3*x+0.8

#定义模型
class MyLinear(nn.Module):
    def __init__(self):
        # 继承父类init
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        out = self.linear(x)
        return out


#实例化模型，优化器类实例化， loss实例化
my_linear = MyLinear()
optimizer = SGD(my_linear.parameters(),0.001)
loss_fn = nn.MSELoss()

#循环，进行梯度下降，参数的更新
for i in range(50000):
    y_predict = my_linear(x)
    loss = loss_fn(y_predict, y_true)

    #梯度置0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #参数更新
    optimizer.step()
    if i % 1000 == 0:
        print(loss.item(),list(my_linear.parameters()))
