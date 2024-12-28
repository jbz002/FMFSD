'''
VGG for CIFAR100

'''
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
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=200):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
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
                channel_out=512
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
                channel_out=512,
            ),
            SpatialGroupEnhance(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=512,
                channel_out=512,
            ),
            SpatialGroupEnhance(32),
        )
        self.apool2 = nn.AdaptiveAvgPool2d((2,2))
        self.apool3 = nn.AdaptiveAvgPool2d((1,1))
        self.scala4 = nn.AdaptiveAvgPool2d((1,1))

        self.sge1 = SpatialGroupEnhance(32)
        self.sge2 = SpatialGroupEnhance(32)
        self.sge3 = SpatialGroupEnhance(32)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
        self._initialize_weights()


    def forward(self, x, is_feat=False, preact=False):
        
        feature_list=[]

        x = F.relu(self.block0(x))
        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)

        fea1 = self.sge1(x)
        feature_list.append(fea1)
        
        x = self.pool1(x)
        x = self.block2(x)

        fea2 = self.sge2(x)
        feature_list.append(fea2)
        
        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)

        fea3 = self.sge3(x)
        feature_list.append(fea3)
        
        out1_fea = self.scala1(feature_list[0])
        out1_feature = self.scala1_(out1_fea).view(x.size(0), -1)
        
        out2_fea = self.scala2(feature_list[1])
        out2_feature = self.scala2_(out2_fea).view(x.size(0), -1)
        
        out3_fea = feature_list[2]
        out5_fea = self.module([out1_fea, out2_fea, out3_fea])
        
        x = self.pool3(out5_fea)
        x = self.block4(x)
        x = F.relu(x)
        x = self.pool4(x)
        feature_list.append(x)
        x = x.view(x.size()[0], -1)
        f5 = x
#         out4 = self.classifier(x)
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)
        
        out3_fea = self.scala3(out5_fea)
        out3_fea = self.apool2(out3_fea)
        out5_fea = self.module2([out3_fea, feature_list[3]])
        out3_feature = self.apool3(out5_fea).view(x.size(0), -1)
        
        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model