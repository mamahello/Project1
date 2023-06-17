from im2col import im2col
import numpy as np

#卷积层
class convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int((N+2*self.pad-FN)/self.stride+1)
        out_w = int((W+2*self.pad-FW)/self.stride+1)

        col = im2col(x,FH,FW,self.stride,self.pad)
        col_W = self.W.reshape(FN,-1)
        out = np.dot(col,col_W)+self.b

        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        return out