import sys,os
sys.path.append(os.pardir)
import numpy as np
from ch5.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict    #有序字典


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):

        #初始化权重
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params["b1"]=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

        #生成层
        self.layers=OrderedDict()           #有序字典 可以记住字典添加的顺序
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])

        self.lastlayer=SoftmaxWithLoss()        #正向推理时用不到softmax层 故分开

    #前向传播
    def pridict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x

    """
        layer的值依次为
                     Affine1 ， Relu1，Affine2
            layer.forward（x)完成前向传播                  self.layers是有序字典————关键！！
    """

    #损失函数
    #  x输入数据  y监督数据
    def loss(self,x,t):
        y=self.pridict(x)
        return self.lastlayer.forward(y,t)


    #计算识别精度
    def accuracy(self,x,t):
        y=self.pridict(x)
        y=np.argmax(y,axis=1)
        if t.ndim != 1 :t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

        # 细节 注意思考！！
        # a[i][j] aixs=0 i变j不变对比最大值 共j个     aixs=1   i不变j变对比的最大值 共i个  每个批一个

    #数值微分法计算偏导数
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)              #loss_W为函数

        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads   #梯度保存在字典中


    #反向传播法计算偏导数
    def gradient(self,x,t):
        #forward
        self.loss(x,t)

        #backward
        dout=1
        dout=self.lastlayer.backward(dout)

        layers=list(self.layers.values())       #list  内置函数 ，将目标转换为列表
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        grads={}    #字典存储结果
        grads['W1']=self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads        #返回一个字典，记录权重

