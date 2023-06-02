import numpy as np


class Sigmoid:            #y=1/(1+exp(-x))
    def __init__(self):
        self.y=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.y=out

        return out


    def backward(self,dout):
        dx=dout*(1-self.y)*self.y

        return dx

