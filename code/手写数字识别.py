import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


#准备数据集
def get_dataloader(train=True, batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,),std=(0.3081,))    #mean和std的形状和通道数量
    ])
    dataset = MNIST(root="./data",train=train,download=False,transform=transform_fn)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader

#构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)

    def forward(self, input):
        # 修改形状
        x = input.view([input.size(0),1*28*28])
        # 全链接操作
        x = self.fc1(x)
        # 激活函数处理，形状没有变化
        x = F.relu(x)
        # 输出层
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)

model = MnistModel()
optimizer = Adam(model.parameters(),lr=0.01)
if os.path.exists("./model/model.pkl") and os.path.exists("./model/optimizer.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))

def train(epoch):
    #实现训练的过程
    data_loader = get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        # 调用模型得到预测值
        output = model(input)
        # 得到损失
        loss = F.nll_loss(output,target)
        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()
        if idx%10 == 0:
            print(epoch,idx,loss.item())

        #模型保存
        if idx%100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        # 不对计算进行追踪
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss)
            #计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("acc,loss:",np.mean(acc_list), np.mean(loss_list))


if __name__ == '__main__':
    # for i in range(3):
    #     train(i)
    test()