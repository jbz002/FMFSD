import torch.nn as nn
import torch
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math

class spatial_weight2(nn.Module):
    def __init__(self, H, W, M):
        super(spatial_weight2, self).__init__()
        self.M = M
        self.H = H
        self.W = W
        self.groups= 8
        self.Spatialweight1 = Spatialweight(8)
        self.Spatialweight2 = Spatialweight(8)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, output):
        #x:[b,c,h,w]
        b, c, h, w = output[0].shape
        f1 = self.Spatialweight1(output[0])
        f2 = self.Spatialweight2(output[1])
        f_en = (f1+f2)/2 #[b*group,m,h,w]
        output[0] = output[0].view(b * self.groups, -1, h, w) 
        output[1] = output[1].view(b * self.groups, -1, h, w) 
        v = (output[0]+output[1])*f_en/2
        v = v.view(b, c, h, w)
        return v

class spatial_weight(nn.Module):
    def __init__(self, H, W, M):
        super(spatial_weight, self).__init__()
        self.M = M
        self.H = H
        self.W = W
        self.groups= 8
        self.Spatialweight1 = Spatialweight(8)
        self.Spatialweight2 = Spatialweight(8)
        self.Spatialweight3 = Spatialweight(8)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, output):
        #x:[b,c,h,w]
        b, c, h, w = output[0].shape
        f1 = self.Spatialweight1(output[0])
        f2 = self.Spatialweight2(output[1])
        f3 = self.Spatialweight3(output[2])
        f_en = (f1+f2+f3)/3 #[b*group,m,h,w]
        output[0] = output[0].view(b * self.groups, -1, h, w) 
        output[1] = output[1].view(b * self.groups, -1, h, w) 
        output[2] = output[2].view(b * self.groups, -1, h, w) 
        v = (output[0]+output[1]+output[2])*f_en/3
        v = v.view(b, c, h, w)
        return v


class Spatialweight(nn.Module):
    def __init__(self, groups = 1):
        super(Spatialweight, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        t = self.sig(t)
        return t

class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 8):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            SpatialGroupEnhance(8),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 1),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1)
        )

        self.bneck2 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1)
        )

        self.bneck3 = nn.Sequential(
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1)
        )

        self.bneck4 = nn.Sequential(
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        
        self.module = spatial_weight(H=8, W=8, M=3)
        self.module2 = spatial_weight2(H=4, W=4, M=2)
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=24,
                channel_out=40
            ),
            SpatialGroupEnhance(8),
            SepConv(
                channel_in=40,
                channel_out=80
            ),
        )
        
        self.scala1_ = nn.Sequential(
            SpatialGroupEnhance(8),
            SepConv(
                channel_in=80,
                channel_out=160
            ),
            SpatialGroupEnhance(8),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=40,
                channel_out=80,
            ),
        )
        
        self.scala2_ = nn.Sequential(
            SpatialGroupEnhance(8),
            SepConv(
                channel_in=80,
                channel_out=160,
            ),
            SpatialGroupEnhance(8),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=80,
                channel_out=160,
            ),
            SpatialGroupEnhance(8),
            
        )
        self.apool = nn.AdaptiveAvgPool2d((1,1))
        self.scala4 = nn.AdaptiveAvgPool2d((1,1))
        
        self.sge1 = SpatialGroupEnhance(8)
        self.sge2 = SpatialGroupEnhance(8)
        self.sge3 = SpatialGroupEnhance(8)

        self.fc1 = nn.Linear(160, num_classes)
        self.fc2 = nn.Linear(160, num_classes)
        self.fc3 = nn.Linear(160, num_classes)
        self.fc4 = nn.Linear(160, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        feature_list = []
        b, _, _, _ = x.shape
        out = self.hs1(self.bn1(self.conv1(x)))
        
        out = self.bneck1(out)
        fea1 = self.sge1(out)
        feature_list.append(out)
        
        out = self.bneck2(out)
        fea2 = self.sge2(out)
        feature_list.append(out)
        
        out = self.bneck3(out)
        fea3 = self.sge3(out)
        feature_list.append(out)
        
        
        out1_fea = self.scala1(feature_list[0])
        out1_feature = self.scala1_(out1_fea).view(x.size(0), -1)
        
        out2_fea = self.scala2(feature_list[1])
        out2_feature = self.scala2_(out2_fea).view(x.size(0), -1)
        
        out3_fea = feature_list[2]
        out5_fea = self.module([out1_fea, out2_fea, out3_fea])
        
        out3_fea = self.scala3(out5_fea)
        
        out = self.bneck4(out)
        feature_list.append(out)
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)
        
        out = self.hs2(self.bn2(self.conv2(out)))
        act = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.hs3(self.bn3(self.linear3(out)))
        out4 = self.linear4(out)
        
#         print(out3_fea.shape)
#         print(feature_list[3].shape)
        out5_fea = self.module2([out3_fea, feature_list[3]])
        out3_feature = self.apool(out5_fea).view(x.size(0), -1)
        
        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]