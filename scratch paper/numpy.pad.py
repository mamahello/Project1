import numpy as np
a=np.arange(16).reshape(2,2,2,2)
print(a,"\n\n\n")

pad=2
a=np.pad(a, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
#对数组a进行填充，轴0，1前后填充宽度为0，轴2，3方向前后填充宽度为pad，填充值为恒值，不指定默认默认为0

print(a)

#在数组A的边缘填充constant_values指定的数值
#（3,2）表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
#（2,3）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充2个宽度的0，后面填充3个宽度的0

#np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))

# #constant_values表示填充值，且(before，after)的填充值等于（0,0）