#该网络为MI-EEGNet
#论文名:MI-EEGNET: A novel convolutional neural network for motor imagery classification

import torch
import torch.nn as nn
from braindecode.models.modules import Expression, Ensure4d


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class MIEEGNet(nn.Module):
    def __init__(self,channel = 22) -> None:
        super(MIEEGNet, self).__init__()

        F1 = 64
        D = 4
        F2 = F1 * D
        self.drop_out = 0.5
        self.eegnet = nn.Sequential(
            #[b,1,c,t]  
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, F1, (1, 16),bias=False),#   [b,64,c,t]
            nn.BatchNorm2d(F1), 
            #Depthwise Convolution
            nn.Conv2d(F1, F2, (channel, 1), groups=F1, bias=False),#   [b,64*4,1,t]
            nn.Conv2d(F2, F2, (1,1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,2)),     #   [b,256,1,t//2]
            nn.Dropout(self.drop_out)
        )

        self.inception1 = nn.Sequential(
            nn.Conv2d(F2, F1, (1,1), bias=False),
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(F1, F1, (1, 7), groups=F1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.Dropout(self.drop_out),
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(F1, F1, (1, 7), groups=F1, bias=False),
            nn.AvgPool2d((1,2))
        )
        self.inception2 = nn.Sequential(
            nn.Conv2d(F2, F1, (1,1), bias=False),
            nn.ZeroPad2d((4, 4, 0, 0)),
            nn.Conv2d(F1, F1, (1, 9), groups=F1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.Dropout(self.drop_out),
            nn.ZeroPad2d((4, 4, 0, 0)),
            nn.Conv2d(F1, F1, (1, 9), groups=F1, bias=False),
            nn.AvgPool2d((1,2))
        )
        self.inception3 = nn.Sequential(
            nn.AvgPool2d((1,2)),
            nn.Conv2d(F2, F1, (1,1),bias=False)
        )
        self.inception4 = nn.Sequential(
            nn.Conv2d(F2, F1, (1,1),stride=(1,2),bias=False)
        )

        self.linner = nn.Sequential(
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(F1, F2, (1, 5), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(self.drop_out),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.out = nn.Linear(256, 4)
        #Brandecode 要求
        self.en = Ensure4d()
        self.shuffle = Expression(_transpose_to_b_1_c_0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = self.shuffle(self.en(input))
        out_eegnet = self.eegnet(input)

        out_1 = self.inception1(out_eegnet)
        out_2 = self.inception2(out_eegnet)
        out_3 = self.inception3(out_eegnet)
        out_4 = self.inception4(out_eegnet)
        out_inception = torch.cat([out_1,out_2,out_3,out_4],dim=3)  #[b,64,1,t//4]
        out = self.linner(out_inception)#[b,256,1,1]

        out = out.reshape(input.size(0),-1)
        out = self.out(out)
        out = self.LogSoftmax(out)
        return out
if __name__ == '__main__':

    input = torch.randn(64,22,1125,1)
    model = MIEEGNet(22)
    output = model(input)
    #params:143620.0
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数：{}'.format(total_params))

    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('需训练参数个数：{}'.format(total_trainable_parameters))
    from torchstat import stat
    stat(model, (22,1125,1))