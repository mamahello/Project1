import numpy as np
'''计算卷积输出数据尺寸'''
def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size+2 * pad - filter_size) / stride + 1

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W=input_data.shape
    out_h = (H + 2 * pad - filter_h) / stride + 1
    out_w = (W + 2 * pad - filter_w) / stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0),(pad, pad), (pad, pad)],"constant")