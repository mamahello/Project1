# 实现乘法层的类

import numpy as np
from common.functions import *
from common.util import im2col, col2im              #????稍后了解

class Mullayer:
    def __int__(self):
        self.x=None
        self.y=None
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y

        return out

    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x

        return dx,dy


 #实现加法层的类
class Addlayer:
    def __int__(self):
        pass

    def forward(self, x, y):
        out=x+y
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy


 #实现RelU层的类
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)            #x为一维数组，（x<=0）生成一个bool型数组
        out=x.copy()
        out[self.mask]=0            #bool数组型索引

        return out

    def backward(self,dout):
        dout[self.mask]=0
        dx=dout

        return dx


#实现Sigmoid层的类
class Sigmoid:
    def __init__(self):
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out

        return out

    def backward(self,dout):
        dx=dout*(1-self.out)*self.out

        return dx



#仿射变换层
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None

    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b

        return out

    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        return dx


#计算图较复杂 见纸质推导

#softmax with loss层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  #损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx