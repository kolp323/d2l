import torch
from torch import nn
from d2l import torch as d2l

"""
在深度学习中，模型在训练阶段和推理/评估阶段的行为通常是不同的。
有些层和操作在训练时有特定的行为，但在评估时需要关闭或改变其行为，以确保结果的稳定性和可复现性。
如：关闭Dropout、改变BatchNorm（BatchNorm 层不再使用当前小批量的均值和方差，而是使用在训练阶段累积的全局均值和方差来进行归一化）
"""
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval() # 将神经网络模型设置为评估（evaluation）模式
        if not device:
            device = next(iter(net.parameters())).device # 数据存哪就在哪跑
    metric = d2l.Accumulator(2)  # 做一个累加器
    for X,y in data_iter:
        if isinstance(X, list): # list需要分多次移完
            X = [x.to(device) for x in X]
        else: # tensor 可以一次移完
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X),y),y.numel()) # y.numel() 为y元素个数 
    return metric[0]/metric[1]


def my_train(net, train_iter, test_iter, num_epochs, lr, device, stratergy = "SGD"):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # _表示改变变量
    net.apply(init_weights)
    print("training on", device)
    net.to(device)
    torch.cuda.empty_cache() 
    if stratergy == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs],
                            legend=["train_loss", "train_acc", "test_acc"])
    timer, num_batches = d2l.Timer(), len(train_iter) # 计时器，批数

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train() # * 将神经网络模型设置为训练（training）模式

        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

        with torch.no_grad():
            # 所有批次的累计损失，正确数，批量数
            metric.add(l * X.shape[0], d2l.accuracy(y_hat,y),X.shape[0])

        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

        print(f'epoch:{epoch}, train_loss:{train_l:.3f},train_acc:{train_acc:.3f}')

        # 定时作画，但保证每轮至少最后一批时会画一次
        if(i+1) % (num_batches//5) == 0 or i == num_batches - 1:
            animator.add(epoch + (i+1) / num_batches,
                        (train_l, train_acc, None))
                
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        
    print(f'loss {train_l:.3f},train acc {train_acc:.3f},'
         f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec'
         f'on{str(device)}')