import numpy as np
'''计算卷积输出数据尺寸'''
def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size+2 * pad - filter_size) / stride + 1

#im2col会考虑滤波器的大小 步幅 填充 将输入数据展开为2维数组
def im2col(input_data: object, filter_h: object, filter_w: object, stride: object = 1, pad: object = 0) -> object:
    N, C, H, W=input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1      # //整除的意思

    #输入数据格式 N C H W对维度3，4进行填充即可  前后分别填充pad,pad
    img = np.pad(input_data, [(0, 0), (0, 0),(pad, pad), (pad, pad)],'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # col为六维数组

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max= x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride,x:x_max:stride]

    col=col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    return col
