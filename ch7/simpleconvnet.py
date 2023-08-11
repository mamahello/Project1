class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},hidden_size=100,output_size=10,
                 weight_init_std=0.01):
        filter_num=conv_param['filter_num']
        filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']
        filter_stride=conv_param['stride']

        input_size=input_dim[1]  #获取输入数据的长度（宽度），计算输出数据的长（宽）

        conv_output_size=(input_size+2*filter_pad-filter_size)/filter_stride+1
        pool_output_size=int(filter)