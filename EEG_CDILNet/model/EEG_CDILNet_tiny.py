#EEG-CDILNet
#
import torch
import torch.nn as nn
from braindecode.models.modules import Expression, Ensure4d
from .CDIL_CNN import CDIL_ConvPart
from braindecode.models import to_dense_prediction_model, get_output_shape

def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

class EEG_CDILNet(nn.Module):
    
    def __init__(self,inChannel =22,outClass=4, F1=24, KE = 64, D = 2, pe = 0.2, hiden=24, layer = 2, ks =3,pool = 8) -> None:
        super(EEG_CDILNet, self).__init__()
        #
        F2 = F1 * D
        self.eegnet = nn.Sequential(
            #[b,1,c,t]  
            nn.ZeroPad2d((KE//2-1, KE//2, 0, 0)),
            nn.Conv2d(1, F1, (1, KE),bias=False),#   [b,24,c,t]
            nn.BatchNorm2d(F1), 
            #Depthwise Convolution
            nn.Conv2d(F1, F2, (inChannel, 1), groups=F1, bias=False),#   [b,48,1,t]
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,pool)),     #   [b,48,1,t//8]
            nn.Dropout(pe),
            #Separable Convolution
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F2, F2, (1, 16), groups=F2, bias=False),  #   [b,48,1,t//8]
            #Pointwise Convolution
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool)), #   [b,48,1,t//64]
            nn.Dropout(pe)
        )

        self.hiden = hiden
        self.layer = layer
        self.ks = ks
        
        self.conv = CDIL_ConvPart(F2, self.layer*[self.hiden], ks=self.ks)
        self.out = nn.Linear(self.hiden + F2, outClass)

        self.en = Ensure4d()
        self.shuffle = Expression(_transpose_to_b_1_c_0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input):
        input = self.shuffle(self.en(input))
        out_eegnet = self.eegnet(input) #[b,48,1,t//64]

        self.eegFeature = torch.flatten(out_eegnet,1)
        #print(self.eegFeature.shape)

        in_cdil = torch.squeeze(out_eegnet, dim=2)#[b,48,t//64]
        out_cdil = self.conv(in_cdil)

        self.cdilFeature = torch.flatten(out_cdil,1)
        #print(self.cdilFeature.shape)

        con = torch.cat([in_cdil, out_cdil], dim=1)
        out = torch.mean(con, dim=2)    #[b,48,1]

        self.catFeature = out#[b,72]

        out = self.out(out)
        out = self.LogSoftmax(out)
        return out

    def get_fea(self):
        return self.eegFeature,self.cdilFeature,self.catFeature


if __name__ == '__main__':


    dataChoose = 0
    if dataChoose == 0:
        inChannel = 22
        Parms = [24,48, 2, 0.5, 24, 2, 3, 8]
    elif dataChoose == 1:
        Parms = [24, 48, 2, 0.5, 24, 2, 3, 8]
        inChannel = 44
    elif dataChoose == 2:
        Parms = [12, 64, 2, 0.5, 24, 2, 3, 8]
        inChannel = 3
    input = torch.randn(64,inChannel,1125,1)
    outclass = 4 if dataChoose == 0 or dataChoose == 1 else 2
    model = EEG_CDILNet(inChannel=inChannel, outClass=outclass,F1=Parms[0], KE = Parms[1], D = Parms[2], pe = Parms[3], hiden=int(Parms[4]), layer = int(Parms[5]), ks =int(Parms[6]), pool = int(Parms[7]))
    output = model(input)

    print(model.get_fea())
    #params:
    if True:
        from thop import profile
        flops, params = profile(model, inputs=(input, ))
        print('flops:{}'.format(flops))
        print('params:{}'.format(params))


    total_params = sum(p.numel() for p in model.parameters())
    print('total_params{}'.format(total_params))

    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_trainable_parameters{}'.format(total_trainable_parameters))
    if False:
        from torchstat import stat
        stat(model, (22,1125,1))

