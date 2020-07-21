import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import engine.FRN as frn


__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')  # 选择将'fan_in'
        #  保留前向传递中权重差异的大小。选择'fan_out'保留反向传递的幅度。
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.channel_in = in_dim
        self.num_classes = out_dim
        self.bn_att = nn.BatchNorm2d(self.channel_in)
        self.att_conv = nn.Conv2d(self.channel_in, self.num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, padding=0, bias=False)
        self.bn_att2 = nn.BatchNorm2d(self.num_classes)
        self.att_conv3 = nn.Conv2d(self.num_classes, 1, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(14)  # 14
        self.sigmoid = nn.Sigmoid()

        # self.spatial_attn = SpatialAttn()

    def forward(self, x):

        ax = self.bn_att(x)
        # y_spatial = self.spatial_attn(x)
        ax = self.relu(self.bn_att2(self.att_conv(ax)))

        # y = y_spatial * ax
        self.att = self.sigmoid(self.bn_att3(self.att_conv3(ax)))

        return self.att

        # ax = self.att_conv2(ax)
        # ax = self.att_gap(ax)
        # ax = ax.view(ax.size(0), -1)

        # return self.att, ax
class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(SELayer, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

        self.conv3 = ConvBlock(in_channels, in_channels, 1)
        self.spatial_attn = SpatialAttn()

    def forward(self, x):
        b, c, h, w = x.size()
        # squeeze operation (global average pooling)
        y = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        y = self.conv1(y)
        y = self.conv2(y)


        y_spatial = self.spatial_attn(x)

        y_final = y_spatial * y
        y_final = self.conv3(y_final)
        # # Multi-scale information fusion
        y_final = torch.sigmoid(y_final)
        return x * y_final.expand_as(x)

class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)
        self.bn_mask = nn.BatchNorm2d(1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        # x =F.upsample(
        #     x, (x.size(2) * 2, x.size(3) * 2),
        #     mode='bilinear',
        #     align_corners=True
        # )
        # scaling conv
        x = self.bn_mask(self.conv2(x))
        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_in = channel

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.att_conv = nn.Conv2d(self.channel_in, 1, kernel_size=1, padding=1, bias=False)
        self.conv2 = ConvBlock(self.channel_in, self.channel_in, 1)
        self.spatial_attn = SpatialAttn()

        self.conv_mask = nn.Conv2d(self.channel_in, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.bn_mask = nn.BatchNorm2d(1)

    def forward(self, x):

        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        # y = self.spatial_pool(x)
        # input_x = x
        # # [N, C, H * W]
        # input_x = input_x.view(b, c, h * w)
        # # [N, 1, H, W]
        context = self.bn_mask(self.conv_mask(x))
        context = self.spatial_pool(context)
        # context_mask = context_mask.view(b, 1, h * w)
        context_mask = self.softmax(context)
        # # print(context_mask.size())
        # context = x * context_mask
        # context = context.view(b,c,1,1)
        y = context
        # print(y.size())
        # Two different branches of ECA module

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = y * context_mask
        # y = self.conv2(y)
        # y = self.sigmoid(y)
        x_mask = x + y.expand_as(x)
        # x_mask = y.expand_as(x)

        return x_mask
        # y_spatial = self.spatial_attn(x)
        #
        # y_final = y_spatial * y
        # y_final = self.conv2(y_final)
        # # # Multi-scale information fusion
        # y_final = self.sigmoid(y_final)
        #
        # return x * y_final.expand_as(x)

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4  #输出的通道维度为输入维度的四倍，为了保持维度一致

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None,branch=[1,2,3,4]):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
            # self.bn1 = frn.FRN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            # self.bn1 = frn.FRN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = frn.FRN(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # self.bn3 = frn.FRN(planes * self.expansion)


        self.eca = eca_layer(planes * self.expansion, k_size=3)
        self.branch = branch
        # self.eca = SELayer(planes * self.expansion)
        # self.eca = CAM_Module(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.branch:
            out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self, block, layers, branch = True):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0],branch=True)
        self.layer2 = self._make_layer(block, scale*2, layers[1], branch=True,stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], branch=True,stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], branch=False, stride=1)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)
        self.part_maxpool = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed_x3 = nn.Linear(1792, 1792, bias=False)
        self.bn_x3 = nn.BatchNorm2d(1792)
        self.prelu_x3 = nn.PReLU()
        # 对卷积和与BN层初始化，
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, branch , stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample,branch))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn,branch = branch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        att1 = self.part_maxpool(x)
        att1 = att1.view(att1.size(0), -1)
        x = self.layer2(x)
        att2 = self.part_maxpool(x)
        att2 = att2.view(att2.size(0), -1)
        x = self.layer3(x)
        att3 = self.part_maxpool(x)
        att3 = att3.view(att3.size(0), -1)
        x = self.layer4(x)

        att_feat = torch.cat([att1, att2, att3], 1)


        return x, att_feat

    def load_param(self, model_path):
        model_weight = torch.load(model_path)
        param_dict = model_weight['state_dict']
        new_state_dict = OrderedDict()

        for k, v in param_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        for i in new_state_dict:
            if 'fc' in i or 'bn3' in i:
                continue
            self.state_dict()[i].copy_(new_state_dict[i])

def resnet50_ibn_a(branch = True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(Bottleneck_IBN, [3, 4, 6, 3], branch)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_ibn_a( pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(Bottleneck_IBN, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model