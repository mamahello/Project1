import numpy as np
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

#后面确定batch的大小时，用的是batch_size = y.shape[0]，若y是一维数组,则是计算单个数据的交叉熵误差，此时batch_size应该为1，但是y.shape[0]是元素个数
#当输入为mini-batch时，要用batch的个数进行正规话 ，对计算单个数据的交叉熵误差

    '''判断维数，将单个数据和批量数据处理成同样的形式，
    方便后面shape[0]取出batch_size'''

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    #one-hot中t为0的

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    temple=np.arange(batch_size)        #temple为 0 - batch_size-1 的数组
    return -np.sum(np.log(y[temple, t] + 1e-7)) / batch_size
    #y[temple,t]  花式索引




#一维处理实例

t = [[1, 2, 1], [3, 4, 2], [5, 6, 6]]
print(np.array(t).shape[0])

y = [1, 2, 3]
y1 = np.array(y)
print(y1.shape[0])

y2 = y1.reshape(1, y1.size)
print(y2.shape[0])