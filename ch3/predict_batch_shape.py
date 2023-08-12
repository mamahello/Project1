import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

x, t = get_data()   #训练集10000帐 x训练图像  t训练标签
network = init_network()

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']


print(x.shape,"x\n",  t.shape,"t\n",  w1.shape,"w1\n",  w2.shape,"w2\n",  w3.shape,"w3\n",   b1.shape,"b1\n",b2.shape,"b2\n",b3.shape,"b3\n")

a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1)
a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)
a3 = np.dot(z2, w3) + b3
y = softmax(a3)


print(a1.shape,"a1\n",  a2.shape,"a2\n",  a3.shape,"a3\n",  y.shape,"y\n")


