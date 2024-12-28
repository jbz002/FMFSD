"""squeezenet in pytorch



[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

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
        self.groups= 32
        self.Spatialweight1 = Spatialweight(32)
        self.Spatialweight2 = Spatialweight(32)
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
        self.groups= 32
        self.Spatialweight1 = Spatialweight(32)
        self.Spatialweight2 = Spatialweight(32)
        self.Spatialweight3 = Spatialweight(32)
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
    def __init__(self, groups = 32):
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
    def __init__(self, groups = 16):
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
            SpatialGroupEnhance(32),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)
        
        self.module = spatial_weight(H=8, W=8, M=3)
        self.module2 = spatial_weight2(H=4, W=4, M=2)
        
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=128,
                channel_out=256
            ),
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=256,
                channel_out=384
            ),
        )

        self.scala1_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=384,
                channel_out=512
            ),
            SpatialGroupEnhance(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=256,
                channel_out=384,
            ),
        )
        
        self.scala2_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=384,
                channel_out=512,
            ),
            SpatialGroupEnhance(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=384,
                channel_out=512,
            ),
            SpatialGroupEnhance(32),
        )
        self.apool2 = nn.AdaptiveAvgPool2d((8,8))
        self.apool3 = nn.AdaptiveAvgPool2d((1,1))
        self.scala4 = nn.AdaptiveAvgPool2d((1,1))

        self.sge1 = SpatialGroupEnhance(32)
        self.sge2 = SpatialGroupEnhance(32)
        self.sge3 = SpatialGroupEnhance(32)

        self.fc1 = nn.Linear(512, class_num)
        self.fc2 = nn.Linear(512, class_num)
        self.fc3 = nn.Linear(512, class_num)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feature_list=[]
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2

        fea1 = self.sge1(f3)
        feature_list.append(fea1)
        
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)
        f5 = self.fire5(f4) + f4

        fea2 = self.sge2(f5)
        feature_list.append(fea2)
        
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        
        fea3 = self.sge3(f7)
        feature_list.append(fea3)
        
        out1_fea = self.scala1(feature_list[0])
        out1_feature = self.scala1_(out1_fea).view(f7.size(0), -1)
        
        out2_fea = self.scala2(feature_list[1])
        out2_feature = self.scala2_(out2_fea).view(f7.size(0), -1)
        
        out3_fea = feature_list[2]
        out3_fea = self.apool2(out3_fea)

        out5_fea = self.module([out1_fea, out2_fea, out3_fea])
        
        f8 = self.fire8(out5_fea)
        f8 = self.maxpool(f8)
        f9 = self.fire9(f8)
        feature_list.append(f9)
        
        
        out3_fea = self.scala3(out5_fea)
        out5_fea = self.module2([out3_fea, feature_list[3]])
        out3_feature = self.apool3(out5_fea).view(x.size(0), -1)
        out4_feature = self.scala4(feature_list[3]).view(f9.size(0), -1)

        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        
        c10 = self.conv10(f9)
        x = self.avg(c10)
        out4 = x.view(x.size(0), -1)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

def squeezenet(class_num=200):
    return SqueezeNet(class_num=class_num)