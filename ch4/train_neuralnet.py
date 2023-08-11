# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) #导入数据

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)  #  TwolayerNet  的实例  network

iters_num = 10000  # 适当设定循环的次数
train_size = x_train.shape[0]          #训练集大小
batch_size = 100                        #批大小
learning_rate =1             #学习率

train_loss_list = []                #记录数据
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)                  #一个人epoch；遍历一次所有数据，称为一个epoch
                                                                 #平均每个epoch需要循环多少次 60000/100=600


for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)       #选取mini_batch  花式索引batch_mask
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]


                    #随机（mini_batch）梯度下降法SGD更新参数
    # 计算梯度                                                   1，为中心差分求导数    2.反向传播法 速度快！
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]          #key=W1,b1,W2, b2   循环4次  更新参数的值
    
    loss = network.loss(x_batch, t_batch)                   #输入数据x_batch,标签数据t_batch求损失函数
    train_loss_list.append(loss)                    #记录损失函数值
    
    if i % iter_per_epoch == 0:      #每经过一个epoch,便更新一次数据    10000/600=16.6 所以更新数据17次
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))





# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))        #x为一个数组，
plt.plot(x, train_acc_list, label=
 'train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

lo=np.arange(iters_num)
plt.plot(lo,train_loss_list,label='loss')
plt.show()