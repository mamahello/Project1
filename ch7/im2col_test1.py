import sys,os
sys.path.append(os.path)
from im2col import im2col
import numpy as np

x1=np.arange(96).reshape(3,2,4,4)
col1=im2col(x1,2,2,stride=1,pad=0)
print(x1)
print('_______________________________')
print(col1)