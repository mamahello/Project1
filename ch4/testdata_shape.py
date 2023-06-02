import sys,os
sys.path.append("C:\deeplearning")
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image


(x_train,t_train),(x_text,t_text)=load_mnist(normalize=True,one_hot_label=True)   #normalize=True时无法正确显示图像

#还有第二个可选参数flatte=True时数据为一维数组

print(x_train.shape)
print(x_train.shape[0])
print(t_train.shape,"\n")

print(x_text.shape)
print(t_text.shape,"\n")

img=x_train[98]
print(img.shape)
img=img.reshape(28,28)
print(img.shape)
print(x_train[98].size)
print("\n",t_train[98])    #one hot label为False,则为3


def imgshow(imf):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

imgshow(img)