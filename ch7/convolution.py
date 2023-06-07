import numpy as np

from im2col import im2col

#把卷积层实现为名convolution的类
class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        out_h=int((H+2*self.pad-FH)/self.stride+1)
        out_w=int((W+2*self.pad-FW)/self.stride+1)

        col=im2col(x,FH,FW,self.stride,self.pad)        #将输入数据展开为二维数组
        col_W=self.W.reshape(FN,-1)         #滤波器展开为  注意rshape -1
                                            #函数会自动计算该维度（-1代表的维度）的的元素个数
        out=np.dot(col,col_W)+self.b

        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        #transpose函数会更改多维数组的轴的顺序    （N,H,W,C）->（N,C,H,W）

        return out
