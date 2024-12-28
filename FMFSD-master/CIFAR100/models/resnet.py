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
    
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.module = spatial_weight(H=8, W=8, M=3)
        self.module2 = spatial_weight2(H=4, W=4, M=2)
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
        )
        
        self.scala1_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            SpatialGroupEnhance(32),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
        )
        
        self.scala2_ = nn.Sequential(
            SpatialGroupEnhance(32),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            SpatialGroupEnhance(32),
            nn.AvgPool2d(4, 4)
        )
        
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            SpatialGroupEnhance(32),
            
        )
        self.apool = nn.AvgPool2d(4, 4)
        self.scala4 = nn.AvgPool2d(4, 4)
        
        self.sge1 = SpatialGroupEnhance(32)
        self.sge2 = SpatialGroupEnhance(32)
        self.sge3 = SpatialGroupEnhance(32)

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)

        fea1 = self.sge1(x)
        feature_list.append(fea1)

        x = self.layer2(x)

        fea2 = self.sge2(x)
        feature_list.append(fea2)

        x = self.layer3(x)
        fea3 = self.sge3(x)
        feature_list.append(fea3)

        out1_fea = self.scala1(feature_list[0])
        out1_feature = self.scala1_(out1_fea).view(x.size(0), -1)
        
        out2_fea = self.scala2(feature_list[1])
        out2_feature = self.scala2_(out2_fea).view(x.size(0), -1)
        
        out3_fea = feature_list[2]
        out5_fea = self.module([out1_fea, out2_fea, out3_fea])
        
        out3_fea = self.scala3(out5_fea)
        
        
        x = self.layer4(out5_fea)
        feature_list.append(x)
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)
        
        out5_fea = self.module2([out3_fea, feature_list[3]])
        out3_feature = self.apool(out5_fea).view(x.size(0), -1)
        
        out1 = self.fc1(out1_feature)
        out2 = self.fc2(out2_feature)
        out3 = self.fc3(out3_feature)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)