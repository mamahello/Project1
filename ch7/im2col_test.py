import sys,os
sys.path.append(os.path)
from im2col import im2col
import numpy as np
x1=np.random.rand(1,3,7,7)
x2=np.random.rand(10,3,7,7)
print(x1.shape)

col1=im2col(x1,5,5,stride=1,pad=0)  #卷积核大小5*5
col2=im2col(x2,5,5,stride=1,pad=0)

print(col1.shape)
print(col2.shape)