import numpy as np

def numerical_gradient(f,x):          #x为一个数组 如x1,x2,x3,x4 多个自变量
    h=1e-4
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val

    return grad


def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad

    return x


#梯度法求 f(x0,x1)=x[2]**2+x[1]**2 的极小值
def function_2(x):
    return x[0]**2+x[1]**2

init_x=np.array([-3.0,4.0])
x=gradient_descent(function_2,init_x,lr=0.1,step_num=100)
print(x)