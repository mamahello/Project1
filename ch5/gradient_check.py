import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net1 import TwoLayerNet

#梯度确认
#检查反向传法与数值微分法的结果是否一致（严格讲为相近）

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#实例化一个两层神经网络的类
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#抽取训练数据的一部分进行检验
x_batch = x_train[:3]
t_batch = t_train[:3]

#分别用两种方法求参数梯度  结果返回为字典
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)



for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))