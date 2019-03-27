import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class _ConvNd(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d_part(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, thresh_factor = -1, comp_channels =  2 ,thresh_slope = 0.5 ,dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        padding = nn.modules.utils._pair(padding)
        dilation = nn.modules.utils._pair(dilation)
        super(Conv2d_part, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, nn.modules.utils._pair(0), groups, bias)
        self.thresh = nn.Parameter(thresh_factor*(torch.ones(out_channels,1,1)))
        self.thresh_slope = thresh_slope
        self.comp_channels = comp_channels

    def forward(self, input):
        compare = F.conv2d(input[:,:self.in_channels//self.comp_channels],self.weight[:,:self.in_channels//2,:,:],self.bias,
                        self.stride,self.padding,self.dilation,self.groups)
        compare = compare.detach()
        # check_sparse = compare.clone()
        temp = F.relu(-compare+(self.thresh+0.5/self.thresh_slope))*self.thresh_slope
        compare = F.relu(1.0-temp)
        # check_sparse[check_sparse>=self.thresh] = 1.0
        # check_sparse[check_sparse<self.thresh] = 0.0
        # print(check_sparse)
        # i_s = compare.shape
        # print(1-torch.sum(check_sparse,[0,1,2,3])/(i_s[0]*i_s[1]*i_s[2]*i_s[3]))
        layer = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return layer*compare