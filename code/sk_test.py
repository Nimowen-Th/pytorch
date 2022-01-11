import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import time
import torch

# for i in tqdm(range(1000)):
#      #do something
#     time.sleep(0.1)

if __name__ == '__main__':
    x = torch.ones([2888])
    print(x)
    print(x.view(-1))

#
# sex = pd.Series(["male", "female", "female", "male", np.nan])
# sex.fillna("unknown", inplace=True)
# le = preprocessing.LabelEncoder()    #获取一个LabelEncoder
# print(sorted(set(sex)))
# le = le.fit(["male", "female", "unknown", "a"])      #训练LabelEncoder, 把male编码为0，female编码为1
# sex = le.transform(sex)                #使用训练好的LabelEncoder对原数据进行编码
# print(sorted({"male", "female", "unknown", "a"}))
# print(sex)
# print(le.inverse_transform([0, 1, 2, 3]))
