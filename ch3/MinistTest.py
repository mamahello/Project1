import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train,t_train),(x_text,t_text)=load_mnist(flatten=True,normalize=False)


print(x_train.shape)
# print(t_train.shape)
# print(x_text.shape)
# print(t_text.shape)