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
    def __init__(self, groups = 32):
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
    
class MobileNet(nn.Module):
    def __init__(self, n_class=100):
        super(MobileNet, self).__init__()
        self.nclass = n_class

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 1),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),

        )
        self.stage4 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(1024, self.nclass)
        
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
                channel_out=512
            ),
        )
        
        self.scala1_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=512,
                channel_out=1024
            ),
            SpatialGroupEnhance(32),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=256,
                channel_out=512,
            ),
        )
        
        self.scala2_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=512,
                channel_out=1024,
            ),
            SpatialGroupEnhance(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=512,
                channel_out=1024,
            ),
            SpatialGroupEnhance(32),
            
        )
        self.apool = nn.AdaptiveAvgPool2d((1,1))
        self.scala4 = nn.AdaptiveAvgPool2d((1,1))
        
        self.sge1 = SpatialGroupEnhance(32)
        self.sge2 = SpatialGroupEnhance(32)
        self.sge3 = SpatialGroupEnhance(32)

        self.fc1 = nn.Linear(1024, self.nclass)
        self.fc2 = nn.Linear(1024, self.nclass)
        self.fc3 = nn.Linear(1024, self.nclass)
        self.fc4 = nn.Linear(1024, self.nclass)
#         self.fc5 = nn.Linear(1024, self.nclass)

    def forward(self, x):
        feature_list = []
        x = self.stage1(x)
        fea1 = self.sge1(x)
        feature_list.append(fea1)
        
        x = self.stage2(x)
        fea2 = self.sge2(x)
        feature_list.append(fea2)
        
        x = self.stage3(x)
        fea3 = self.sge3(x)
        feature_list.append(fea3)
        
        out1_fea = self.scala1(feature_list[0])
        out1_feature = self.scala1_(out1_fea).view(x.size(0), -1)
        
        out2_fea = self.scala2(feature_list[1])
        out2_feature = self.scala2_(out2_fea).view(x.size(0), -1)
        
        out3_fea = feature_list[2]
        out5_fea = self.module([out1_fea, out2_fea, out3_fea])
        
        out3_fea = self.scala3(out5_fea)
        
        x = self.stage4(x)
        feature_list.append(x)
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)
        
        out5_fea = self.module2([out3_fea, feature_list[3]])
        out3_feature = self.apool(out5_fea).view(x.size(0), -1)
        
        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]