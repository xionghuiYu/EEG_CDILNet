#该网络为论文 TCnet Fusion 网络
#Electroencephalography-based motor imagery classification using temporal convolutional network fusion
#code by Yuxionghui 

import torch
import torch.nn as nn
import torch.nn.functional as F
from TCN import TemporalConvNet
from braindecode.models.modules import Expression, Ensure4d

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class TCNet(nn.Module):
    def __init__(self,channel =22) -> None:
        super(TCNet, self).__init__()

        F1 = 8
        KE = 32
        D = 2
        pe = 0.2
        F2 = F1 * D
        self.eegnet = nn.Sequential(
            #[b,1,c,t]  
            nn.ZeroPad2d((KE//2-1, KE//2, 0, 0)),
            nn.Conv2d(1, F1, (1, KE),bias=False),#   [b,24,c,t]
            nn.BatchNorm2d(F1), 
            #Depthwise Convolution
            nn.Conv2d(F1, F2, (channel, 1), groups=F1, bias=False),#   [b,48,1,t]
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,8)),     #   [b,48,1,t//8]
            nn.Dropout(pe),
            #Separable Convolution
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F2, F2, (1, 16), groups=F2, bias=False),  #   [b,48,1,t//8]
            #Pointwise Convolution
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)), #   [b,48,1,t//64]
            nn.Dropout(pe)
        )
        FT = 12
        KT = 4 
        pt = 0.3
        L = 4
        inchannnels = [FT for _ in range(L)]
        self.tcn = TemporalConvNet(F2, inchannnels, kernel_size=KT,dropout=pt)
        self.out = nn.Linear(FT * 17,4)
        #Brandecode 要求
        self.en = Ensure4d()
        self.shuffle = Expression(_transpose_to_b_1_c_0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input):
        batchSize = input.size(0)
        input = self.shuffle(self.en(input))
        out_eegnet = self.eegnet(input) #[b,48,1,t//64]
        in_tcn = torch.squeeze(out_eegnet, dim=2)#[b,48,t//64]
        out_tcn = self.tcn(in_tcn)#[b,12,t//64]
        out = out_tcn.view(out_tcn.size(0),-1)

        out = self.out(out)
        out = self.LogSoftmax(out)
        return out


if __name__ == '__main__':

    input = torch.randn(64,22,1125,1)
    model = TCNet(22)
    output = model(input)
    #params:26128.0
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数：{}'.format(total_params))

    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('需训练参数个数：{}'.format(total_trainable_parameters))

    print("%s | Params: %.4fM | FLOPs: %.4fM" % ("Informer", params / (1000 ** 2), flops / (1000 ** 3)))
    # from torchsummary import summary
    # summary(model.cuda(),input_size=(22,1125,1),batch_size=64)