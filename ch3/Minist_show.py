import sys,os
sys.path.append("C:\deeplearning")
from dataset.mnist import load_mnist
from PIL import Image  #图像显示使用PIL模块 （python image library）
import numpy as np

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))    #image.fromarray()  把保存为numpy数组的图像数据转换为PIL用的数据对象 unit8 无符号
    pil_img.show()


#load_mnist函数以（训练图像，训练标签），（测试图像，测试标签）的形式返回读入的MNIST数据集

(x_train,t_train),(x_text,t_text)=load_mnist(flatten=True,normalize=False)
img=x_train[60]
#print(img.shape)
#label=t_train[2]
#print(label)

img=img.reshape(28,28)

img_show(img)