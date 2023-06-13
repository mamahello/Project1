import sys,os
sys.path.append(os.pardir)
from im2col import im2col
import  numpy as np


x1=np.random.randn(1,3,7,7)
print(x1.shape)
col1=im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)