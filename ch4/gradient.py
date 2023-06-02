import numpy as np


def numerical_gradient(f,x):          #x为一个数组 如x1,x2,x3,x4 多个自变量
    h=1e-4
    grad=np.zeros_like()

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val

    return x




