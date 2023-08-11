import sys,os
sys.path.append("C:\deeplearning")
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid,softmax


def get_data():       #(训练图像，训练标签),(测试图像，测试标签)
    (x_train,t_train),(x_text,t_text)=load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_text,t_text


#init_network() 会读入保存在pickle文件sample_weight.pkl中学到的权重参数，这个文件中以字典变量的形式保存了权重和偏置参数
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network=pickle.load(f)       #network为一个字典
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()

print(len(x))   #共一万张图片

network = init_network()  #network 字典
accuracy_cnt = 0
for i in range(len(x)):   #range函数返回一个列表，其实就相当于遍历列表
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引   np.argmax(x)将获取赋予参数x的数组的最大值的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))