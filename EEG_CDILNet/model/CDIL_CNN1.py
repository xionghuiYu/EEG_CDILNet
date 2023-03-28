#padding = circular   can try padding  = reflective
from pickle import FALSE
import torch 
import torch.nn as nn 
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class CDIL_Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,padding, dilation,  dropout=0.1,se = True):
        super(CDIL_Block, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride = stride,padding=padding, dilation=dilation, padding_mode='circular'))
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(n_outputs)
        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,stride = stride,padding=padding, dilation=dilation, padding_mode='circular'))
        # self.relu2 = nn.ELU()
        # self.dropout2 = nn.Dropout(dropout)

        #self.net = nn.Sequential(self.conv1, self.batchnorm, self.relu1, self.dropout1,
        #                         self.conv2, self.batchnorm, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.batchnorm, self.relu1, self.dropout1)
        # self.net = nn.Sequential(self.conv1,self.relu1, self.dropout1,
        #                          self.conv2,self.relu2, self.dropout2)
 
        #self.batchnorm = nn.BatchNorm1d()
        #self.net = nn.Sequential(self.conv1, self.batchnorm, self.relu1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

        # SE layers
        self.se = se
        if self.se:
            self.plane = n_outputs
            self.fc1 = nn.Conv1d(self.plane, self.plane//8, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
            self.fc2 = nn.Conv1d(self.plane//8, self.plane, kernel_size=1)

    def init_weights(self):
        """
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.se:
            # Squeeze
            w = F.avg_pool1d(out, out.size(2))
            w = F.relu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))

            # Excitation
            out = out * w  # New broadcasting feature from v0.2!

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) 

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
            layers += [CDIL_Block(this_in, this_out, ks,1, this_padding, this_dilation)] 
        self.conv_net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.conv_net(x)

if __name__ == "__main__":

    model = CDIL_ConvPart(22,[22,22,22,22])
    input = torch.randn(32,22,22)
    output = model(input)
    print(model)

    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    from torchstat import stat
    stat(model, (22,1125,1))