import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from utils.util import *
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(nn.Module):
    def __init__(self, block, layers, args, num_classes=1000,
                 large_feature_map=True):
        super(ResNetCam, self).__init__()
        self.args = args
        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, args.num_classes, kernel_size=1, padding=0),
        )

        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(1024, args.num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )         
        initialize_weights(self.modules(), init_mode='xavier')
        
    def forward(self, x, label=None, topk=1, evaluate=False ): 
        batch = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x_3 = x.clone()
        
        x = F.max_pool2d(x,kernel_size=2)
        x = self.layer4(x)
        
        x = self.classifier_cls(x)
        self.feature_map = x

        self.score_1 = self.avg_pool(x).squeeze(-1).squeeze(-1)

# p_label 
        if topk == 1:
            p_label = label.unsqueeze(-1)
        else:
            _, p_label = self.score_1.topk(topk, 1, True, True)
# x_sum   
        self.S = torch.zeros(batch).cuda()
        for i in range(batch):
            self.S[i] = self.score_1[i][label[i]]
            
## M_fg    
        M = self.classifier_loc(x_3)
        M_fg =  torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            M_fg[i][0] = M[i][p_label[i]].mean(0)
        self.M_fg = M_fg

        if evaluate:
            return self.score_1, None, None, self.M_fg

## weight_copy
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)  
        layer4_copy = copy.deepcopy(self.layer4)   

##  erase
        x_erase = x_3.clone().detach() * (1-M_fg)
        x_erase = F.max_pool2d(x_erase, kernel_size=2 )
        x_erase = layer4_copy(x_erase)
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1)

## S_bg
        self.S_bg = torch.zeros(batch).cuda()
        for i in range(batch):
            self.S_bg[i] = x_erase[i][label[i]]

##  score_2 
        x = self.feature_map  * nn.AvgPool2d(2)(self.M_fg)
        self.score_2 = self.avg_pool(x).squeeze(-1).squeeze(-1)   

##  bas 
        batch = self.S.size(0)
        S = (nn.ReLU()(self.S.clone().detach())).view(batch,-1).mean(1)
        S_bg = (nn.ReLU()(self.S_bg)).view(batch,-1).mean(1)
        bas = S_bg / (S+1e-8)
        bas[S_bg>S] = 1

## area
        area = self.M_fg.clone().view(batch, -1).mean(1) 

        return self.score_1, self.score_2, bas, area, self.M_fg


    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers
        
    
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

def load_pretrained_model(model):
    strict_rule = True

    state_dict = torch.load('pretrained_weight/resnet50-19c8e357.pth')

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def model(args, pretrained=True):
    model = ResNetCam(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        model = load_pretrained_model(model)
    return model
