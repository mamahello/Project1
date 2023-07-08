import sys,os
sys.path.append(os.path)
from im2col import im2col
import numpy as np
from col2im import col2im

x1=np.arange(96).reshape(3,2,4,4)
col1=im2col(x1,2,2,stride=1,pad=0)
print(x1)
print('_______________________________')
print(col1)
print("\n",col1.shape)    #2*2*2 输出的大小 最后一维

x2=col2im(col1,(3,2,4,4),2,2,stride=1,pad=0)
print("\n",x2.shape,"\n")
print(x2)


#col1im 反向传播时使用  将数据还原为