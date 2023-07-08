import numpy as np
'''计算卷积输出数据尺寸'''
def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size+2 * pad - filter_size) / stride + 1


#im2col会考虑滤波器的大小 步幅 填充 将输入数据展开为2维数组
def im2col(input_data, filter_h, filter_w, stride = 1, pad = 0):
    N, C, H, W=input_data.shape  #输入数据尺寸为四维数组 数据量、通道、高、长
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1      # //整除的意思

    #输入数据格式 N C H W对维度3，4进行填充即可  前后分别填充pad,pad
    img = np.pad(input_data, [(0, 0), (0, 0),(pad, pad), (pad, pad)],'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # col为六维数组

    for y in range(filter_h):
        y_max = y + stride*out_h    #获取纵轴方向的最大取值
        for x in range(filter_w):
            x_max = x + stride*out_w    #获取横轴方向的最大取值
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride,x:x_max:stride] #col6维数组，固定3、4维，剩余维与img四维对应

    """
    回到col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]， col数组为六维，img数组为四维，固定col数组的第三维为y，
    第四维为x； img数组的四维与col数组的第1维，第2维，第5维，第6维是相对应的，
    其中y:y_max:stride的长度为(y_max-y)/stride,也就等于out_h;x:x_max:stride的长度为(x_max-x)/stride,也就等于out_w，
    所以img的第3维与col的第5维长度一直，img的第6维与col的第6维长度一致

    所以col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]赋值的意思是：依次把输入数据按照滤波器的尺寸进行分割，
    并存储到对应的位置。
    """
    col=col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col



"""
np.pad 的用法
方法参数：pad(array, pad_width, mode, **kwargs)
方法返回：填充后的数组
参数解释：
array：表示需要填充的数组；
pad_width：表示每个轴（axis）边缘需要填充的数值数目。
参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。
mode：表示填充的方式（常见的有constant、edge等）
"""