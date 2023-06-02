import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net1 import TwoLayerNet
from ch6 import optimizer

#读入数据
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

#实例
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
optimizer=class_SGD.SGD(lr=0.1)

#定义超参数
iters_num=10000   #循环次数
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iter_per_epoch=max(train_size/batch_size,1)
#print(iter_per_epoch)
#平均每个epoch循环6次，即6次循环所有训练数据被训练一次

for i in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]     #mini-batch学习:从训练数据中选取一部分（成为mini-batch）,再以mini-batch为对象，使用梯度法更新参数

    #通过误差反向传播求梯度
    grad=network.gradient(x_batch,t_batch)
    #返回字典grads{}




    #更新参数
    """
    for key in ('W1','b1','W2','b2'):                              #
        #network.params[key]=network.params[key]-learning_rate*grad[key]
        network.params[key] -= learning_rate * grad[key]    

    """
    optimizer.update(network.params,grad)




    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:                 #每个epoch输出一次训练精度以及识别精度
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(train_acc,test_acc)
