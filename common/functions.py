# coding: utf-8
import numpy as np

                                                 #激活函数
def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


                                                #损失函数

#均方误差
def mean_squared_error(y, t):      #y神经网络的输出  t监督数据  k数据的维度
    return 0.5 * np.sum((y-t)**2)

#交叉熵误差的mini-batch版的实现
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        '''
        假设数据有n个，算出的交叉熵误差求和后要除以n,平均化（与训练数据的数量无关）
        判断维数，将单个数据和批量数据处理成同样的形式，
        方便后面shape[0]取出batch_size  
        如果y.ndim等于1，说明是单个数据的情况，此时batch_size应该为1，但是batch_size = y.shape[0]，
        y.shape[0]得到的值是输出神经元的个数而不是1，因此需要特殊处理一下
        '''

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引，此情况下t是一个数组（正确解的标签为1，别的全为0）
    if t.size == y.size:
        t = t.argmax(axis=1)     #t为一个数组，把one-hot改为非one-hot标签

    batch_size = y.shape[0]
    temple=np.arange(batch_size)   #temple为 0 - batch_size-1 的数组
    return -np.sum(np.log(y[temple, t] + 1e-7)) / batch_size     #1e-7  防止log（0）的情况
    # y[temple,t]  花式索引

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
