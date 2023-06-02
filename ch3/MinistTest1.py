import sys,os
sys.path.append("C:\deeplearning")
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train),(x_text,t_text)=load_mnist(flatten=True,normalize=False)
img=x_train[8]
label=t_train[8]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)