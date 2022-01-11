import torch
import matplotlib.pyplot as plot

learning_rate = 0.01

# 准备数据
# y = 3x + 0.8

x = torch.rand([500, 1])
y_ture = x * 3 + 0.8

# 通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor([0], requires_grad=True, dtype=torch.float32)

# 通过循环，反向传播，更新参数
for i in range(5000):
    # 计算loss
    y_predict = torch.matmul(x, w) + b
    loss = (y_ture - y_predict).pow(2).mean()

    if w.grad is not None:
        # 表示原地修改
        w.data.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if i % 50 == 0:
        print("w, b, loss", w.item(), b.item(), loss.data)

plot.figure(figsize=(20, 8))
plot.scatter(x.numpy().reshape(-1), y_ture.numpy().reshape(-1))
plot.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1))
plot.show()
