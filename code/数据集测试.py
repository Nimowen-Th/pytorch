import torch
from torch.utils.data import Dataset,DataLoader

data_path = "/Users/yuhangzhang/Desktop/SMSSpamCollection"

#完成数据集类
class MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path).readlines()

    def __getitem__(self, index):
        #获取索引对应位置的一条数据
        cur_line = self.lines[index].strip()
        lable = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return lable, content

    def __len__(self):
        #返回数据的总数量
        return len(self.lines)

my_dataset = MyDataset()
data_loader = DataLoader(dataset=my_dataset,batch_size=7,shuffle=True)

if __name__ == '__main__':
    # my_dataset = MyDataset()
    # print(my_dataset[1000])
    # print(len(my_dataset))
    for i in data_loader:
        print(i)
        break
    print(len(my_dataset))
    print(len(data_loader))