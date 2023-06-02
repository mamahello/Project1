import sys, os
sys.path.append("C:\deeplearning\myproject")  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)     #高斯分布初始化

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)

        return loss




#测试
print("测试")

net=simpleNet()         #simpleNet的实例net
print(net.W)

print("\n")

x=np.array([0.6,0.9])
p=net.predict(x)
print(p)

print("\n")

o=np.argmax(p)      #np.argmax()  获取最大值的索引
print(o)

t=np.array([0,0,1])
l=net.loss(x,t)
print(l)