#DSCNN 
# A lightweight and accurate double-branch neural network for four-class motor imagery classification
import torch
import torch.nn as nn
from braindecode.models.modules import Expression, Ensure4d


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class DSCNN(nn.Module):
    def __init__(self, channel=22) -> None:
        super(DSCNN, self).__init__()
        
        F1 = 48
        F3 = 64
        self.channel = channel
        #[b,1,c,t]
        self.branch_time = nn.Sequential(
            nn.ZeroPad2d((12, 12, 0, 0)),
            nn.Conv2d(1, F1, (1,25), bias=False),
            nn.Conv2d(F1, F3, (self.channel,1), bias=False),
            

        )  
        C = 22 
        F2 = 32
        #[b,c,1,t]
        self.branch_spatial = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(C, F2, (1,16), bias=False),
            nn.ZeroPad2d((12, 12, 0, 0)),
            nn.Conv2d(F2, F3, (1,25), bias=False),
            nn.AvgPool2d((1,30))
        )
        self.golbalAvePool = nn.AvgPool2d((1,15))
        self.out = nn.Linear(4928,4)
        #Brandecode 
        self.en = Ensure4d()
        self.shuffle = Expression(_transpose_to_b_1_c_0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, input):
        input = self.shuffle(self.en(input))
        out_time = self.branch_time(input)
        out_sptail = self.branch_spatial(input.permute(0,2,1,3))
        out = torch.cat([out_time,out_sptail], dim=3)
        out = self.golbalAvePool(out)
        
        out = out.reshape(input.size(0),-1)
        out = self.out(out)
        out = self.LogSoftmax(out)
        return out


if __name__ == '__main__':

    input = torch.randn(64,22,1125,1)
    model = DSCNN()
    output = model(input)
    print(output.shape)
    #params:150964.0
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    