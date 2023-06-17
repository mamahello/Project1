#池化层
from im2col import im2col
import numpy as np
class pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        N,C,H,W=x.shape
        out_h=int(1+(H-self.pool_h+2*self.pad)/self.stride)
        out_w=int(1+(W-self.pool_w+2*self.pad)/self.stride)

        #展开
        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col= col.reshape(-1,self.pool_h*self.pool_w)

        #取最大值
        out=np.max(col,axis=1)

        #转换格式
        out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out