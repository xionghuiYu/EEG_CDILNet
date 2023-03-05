#该网络为AMSI-EEGNeT
#A novel multi-scale convolutional neural network for motor imagery classification


import torch
import torch.nn as nn
from braindecode.models.modules import Expression, Ensure4d

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class AMSINet(nn.Module):
    def __init__(self) -> None:
        super(AMSINet, self).__init__()
    
        self.pool1 = nn.AvgPool2d((1,2))
        self.pool2 = nn.AvgPool2d((1,2))
        
        F1 = 64
        D = 3
        F2 = F1*D
        self.step1 = nn.Sequential(
            nn.ZeroPad2d((2, 3, 0, 0)),
            nn.Conv2d(D * F1, F2, (1,7), groups=D*F1, bias=False),
            nn.Conv2d(F2, F2, (1,1), bias=False),#存在疑惑
            nn.AvgPool2d((1,2))
            )
        self.R0 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, F1, (1,16), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (22,1), groups=F1, bias=False),
            nn.Conv2d(D * F1, D * F1, (1,1), bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.Dropout(0.5),
            #后面的一部分 step1
            self.step1
        )
        self.R1 = nn.Sequential(
            self.pool1,
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(1, F1, (1,8), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (22,1), groups=F1, bias=False),
            nn.Conv2d(D * F1, D * F1, (1,1), bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.Dropout(0.25)
        )
        self.R2 = nn.Sequential(
            self.pool1,
            self.pool2,
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(1, F1, (1,4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (22,1), groups=F1, bias=False),
            nn.Conv2d(D * F1, D * F1, (1,1), bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.Dropout(0.125)
        )
        
        self.step2 = nn.Sequential(
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(0.125),
            nn.ZeroPad2d((2, 3, 0, 0)),
            nn.Conv2d(F2, F2, (1,6), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1,1), bias=False),#存在疑惑
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.ZeroPad2d((2, 3, 0, 0)),
            nn.Conv2d(F2, F2, (1,6), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1,1), bias=False),#存在疑惑
            nn.AvgPool2d((1,2))
        )
        self.step3 = nn.Sequential(
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d(1)
        )
        self.out = nn.Linear(F2,4)
        #Brandecode 要求
        self.en = Ensure4d()
        self.shuffle = Expression(_transpose_to_b_1_c_0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = self.shuffle(self.en(input))

        out_R0 = self.R0(input)
        out_R1 = self.R1(input)
        out_R2 = self.R2(input)

        out_step1 = torch.add(out_R0,out_R1)
        temp = self.step2(out_step1)
        out_step2 = torch.add(temp, out_R2)
        out_step3 = self.step3(out_step2)

        out = out_step3.reshape(input.size(0),-1)
        out = self.out(out)
        out = self.LogSoftmax(out)
        return out

if __name__ == '__main__':

    input = torch.randn(64,22,1125,1)
    model = AMSINet()
    output = model(input)
    #params:280964.0
    from thop import profile
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数：{}'.format(total_params))

    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('需训练参数个数：{}'.format(total_trainable_parameters))


    # from torchsummary import summary
    # summary(model.cuda(),input_size=(22,1125,1),batch_size=-1)





        