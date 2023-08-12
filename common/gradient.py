# coding: utf-8
import numpy as np

#中心差分（减少误差)
def _numerical_gradient_1d(f, x):
    h = 1e-4 # 微小值0.0001
    grad = np.zeros_like(x)      #np.zeros_like(x) 生成一个形状与x相同，所有元素都为0的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):                          #enumerate()是python的内置函数
            grad[idx] = _numerical_gradient_1d(f, x)         #enumerate在字典上是枚举、列举的意思
        return grad                                      #对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串）,
                                                        # enumerate将其组成一个索引序列，利用它可以同时获得索引和值

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    # np.nditer 迭代器 迭代访问数组
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad