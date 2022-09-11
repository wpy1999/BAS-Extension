import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *

copy_list = ['Mixed_7b' , 'Mixed_7c']
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, args.num_classes, kernel_size=1, padding=0),
            #nn.ReLU(inplace=True),
        )

        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(768, args.num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self._initialize_weights()
        

    def forward(self, x, label=None, topk=1, evaluate=False):
        #self.weight_deepcopy()
        batch = x.size(0)

        x = self.Conv2d_1a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2a_3x3(x)
        # 112 x 112 x 32
        x = self.Conv2d_2b_3x3(x)
        # 112 x 112 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 56 x 56 x 64
        x = self.Conv2d_3b_1x1(x)
        # 56 x 56 x 64
        x = self.Conv2d_4a_3x3(x)
        # 56 x 56 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # 28 x 28 x 192
        x = self.Mixed_5b(x)
        # 28 x 28 x 192
        x = self.Mixed_5c(x)
        # 28 x 28 x 192
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        # 14 x 14 x 768
        x = self.Mixed_6b(x)
        # 14 x 14 x 768
        x = self.Mixed_6c(x)
        # 14 x 14 x 768
        x = self.Mixed_6d(x)
        # 14 x 14 x 768
        x = self.Mixed_6e(x)
        # 14 x 14 x 768
        x_3 = x.clone()

        x = self.Mixed_7a(x)
        # 14 x 14 x 768
        x = self.Mixed_7b(x)
        # 14 x 14 x 768
        x = self.Mixed_7c(x)
        x = self.classifier_cls(x)
        self.feature_map = x

##  score
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_1 = x
##  p_label 
        if topk == 1:
            p_label = label.unsqueeze(-1)
        else:
            _, p_label = self.score_1.topk(topk, 1, True, True)
##  S   
        self.S = torch.zeros(batch).cuda()
        for i in range(batch):
            self.S[i] = self.score_1[i][label[i]]

##  M_f    
        M = self.classifier_loc(x_3)
        M_fg =  torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            M_fg[i][0] = M[i][p_label[i]].mean(0)
        self.M_fg = M_fg

        if evaluate:
            return self.score_1, None, None, self.M_fg

##  weight_copy
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)
        Mixed_7a_copy = copy.deepcopy(self.Mixed_7a)
        Mixed_7b_copy = copy.deepcopy(self.Mixed_7b)
        Mixed_7c_copy = copy.deepcopy(self.Mixed_7c)
        
## erase
        x_erase = x_3.clone().detach() * (1 - M_fg)
        x_erase = Mixed_7a_copy(x_erase)
        x_erase = Mixed_7b_copy(x_erase)
        x_erase = Mixed_7c_copy(x_erase)
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1)

## S_bg
        self.S_bg = torch.zeros(batch).cuda()
        for i in range(batch):
            self.S_bg[i] = x_erase[i][label[i]]
        
##  score_2 
        x = self.feature_map * nn.AvgPool2d(2)(self.M_fg)
        self.score_2 = self.avg_pool(x).squeeze(-1).squeeze(-1)          

##  bas
        S = nn.ReLU()(self.S.clone().detach()) 
        S_bg = nn.ReLU()(self.S_bg).clone()
        bas = S_bg / (S+1e-8)
        bas[S_bg>S] = 1
##  area
        area = self.M_fg.clone().view(batch, -1)
        area = area.mean(1) 

        return self.score_1, self.score_2, bas, area, self.M_fg

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2,padding=1)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2,padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2,padding=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def model(args, pretrained=True):
    # base and dilation can be modified according to your needs
    model = Model(args)
    if pretrained:
        pretrained_dict = torch.load('pretrained_weight/inception_v3_google-1a9a5a14.pth')
        model_dict = model.state_dict() 
        model_conv_name = []

        for i, (k, v) in enumerate(model_dict.items()):
            if 'tracked' in k[-7:]:
                continue
            model_conv_name.append(k)
        ii = -1
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if k.split('.')[0] == 'AuxLogits' or k.split('.')[0] == 'fc':
                continue
            else:
                ii +=1
            model_dict[model_conv_name[ii]] = v 
        model.load_state_dict(model_dict)
        print("pretrained weight load complete..")
    return model