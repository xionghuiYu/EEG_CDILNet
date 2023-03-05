
import torch 
import torch.nn as nn 
from torch.nn.utils import weight_norm

class CDIL_Block(nn.Module):
    def __init__(self, c_in, c_out, ks, pad, dil):
        super(CDIL_Block, self).__init__()
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode='circular'))
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.normal_(0, 0.01)

        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        res = x if self.res is None else self.res(x)
        return self.nonlinear(out + res) 

class CDIL_ConvPart(nn.Module): 
    def __init__(self, dim_in, hidden_channels, ks=3): 
        super(CDIL_ConvPart, self).__init__() 
        layers = [] 
        num_layer = len(hidden_channels) 
        for i in range(num_layer): 
            this_in = dim_in if i == 0 else hidden_channels[i - 1] 
            this_out = hidden_channels[i] 
            this_dilation = 2 ** i 
            this_padding = int(this_dilation * (ks - 1) / 2) 
            layers += [CDIL_Block(this_in, this_out, ks, this_padding, this_dilation)] 
            self.conv_net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.conv_net(x)

if __name__ == "__main__":
    model = CDIL_ConvPart(16,[22,16,8,4,1])
    input = torch.randn(32,16,1125//64)
    output = model(input)
    print(output.shape)

    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))