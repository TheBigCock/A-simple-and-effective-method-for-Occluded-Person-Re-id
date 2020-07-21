# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import torch
from torch import nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet import resnet18, resnet50, resnet34,resnet101
import torch.nn.functional as F
from .backbones.resnet_ibn_a import resnet50_ibn_a
from torch.backends import cudnn

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


def weights_init_kaiming2(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def weights_init_classifier2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), mode='bilinear', align_corners=True)


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(ASPP_Module, self).__init__()
        # In our re-implementation of ASPP module,
        # we follow the original paper but change the output channel
        # from 256 to 512 in all of four branches.
        out_channels = in_channels // 4

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return y


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

class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F'
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # CCB: Class Center Block end...

        # CAB: Class Attention Block start...
        # transpose ->(B, C', N)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ACFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFModule, self).__init__()

        # self.conva = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

        self.acf = acf_Module(in_channels, out_channels)

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1, bias=False),
        #     InPlaceABNSync(256),
        #     nn.Dropout2d(0.1,False),
        #     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        # feat_cat = torch.cat([class_output, output],dim=1)
        return class_output


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

class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim
        # self.cam = CAM_Module(self.channel_in )
        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size() #([64, 2048, 16, 8])
        print(x.size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        print(proj_query.size())
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        print(proj_key.size())
        # energy = proj_query * proj_key
        # print(energy.size())
        energy = torch.matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self\
            .value_conv(x)\
            .view(m_batchsize, -1, width * height)

        out = proj_value * attention.permute(0, 2, 1)

        # out = torch.bmm(
        #     proj_value,
        #     attention.permute(0, 2, 1)
        # )
        attention_mask = out.view(m_batchsize, C, height, width)
        # out = self.cam * attention_mask
        # out = out + x
        out = self.gamma * attention_mask + x
        return out

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming2)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier2)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,  model_name, pretrain_choice,breach):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = resnet18(last_stride, False)
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = resnet34(last_stride, False)
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = resnet101(last_stride, False)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(num_classes=num_classes, branch=True)
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        # self.layer4 = self.base.layer4
        # self.layer4.load_state_dict(self.base.layer4.state_dict())
        # self.layer4.load_param(self.base.layer4.state_dict())
########################################################################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # 使得模型对于物体姿势或位置的变化具有一定的不变性,
        # STN作为一种新的学习模块, 具有以下特点:
        # (1)为每一个输入提供一种对应的空间变换方式(如仿射变换)
        # (2)变换作用于整个特征输入
        # (3)变换的方式包括缩放、剪切、旋转、空间扭曲等等
        # Spatial transformer network forward function
#################################################################
        # 全局平均池化 self.gap

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.breach = breach

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shxift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_att = nn.Linear(1792, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier_att.apply(weights_init_classifier)


##############################注意力机制###################################
        # self.cam = CAM_Module(2560, self.num_classes)
        # self.pool = nn.MaxPool2d(kernel_size=(1, 1))
        # self.f_value = nn.Conv2d(in_channels=self.in_planes, out_channels= self.num_classes,
        #                          kernel_size=1, stride=1, padding=0)
        # self.f_key = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
        #               kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.key_channels),
        #     nn.ReLU()
        # )
        # self.f_query = self.f_key
        # self.W = nn.Conv2d(in_channels=self.num_classes, out_channels = self.num_classes,
        #                    kernel_size=1, stride=1, padding=0)
        #
        # self.psp_size = (1, 3, 6, 8)
        # self.psp = PSPModule(self.psp_size)
        # self.pam = PAM_Module(self.in_planes)
        # self.middle_dim = 256  # middle layer dimension
        # # self.classifier = nn.Linear(self.feat_dim, num_classes)
        # self.res_part2 = Bottleneck(2048, 512)
        # self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # self.attention_conv = nn.Sequential(
        #     nn.Conv2d(2048, self.middle_dim, [16,8]),
        #     nn.ReLU()
        # )
        # self.attention_conv.apply(weights_init_kaiming)
        #
        # self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        # self.attention_tconv.apply(weights_init_kaiming)

        # self.pam = PAM_Module(self.in_planes)

        # self.att_conv2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, padding=0, bias=False)
        # self.att_gap = nn.AvgPool2d(14)  # 14
        def stn(self, x):
            xs = self.localization(x)  # 特征提取
            xs = xs.view(-1, 10 * 60 * 28)  # resize 到对应维度
            theta = self.fc_loc(xs)  # 回归theta
            theta = theta.view(-1, 2, 3)  # theta resize到对应维度(仿射变换)

            grid = F.affine_grid(theta, x.size())  # 对theta计算对应到原来的位置
            x = F.grid_sample(x, grid)  # 对原图进行sample得到目标输出

            return x

        self.aspp = ASPP_Module(in_channels=2048, atrous_rates=(6, 12, 18), norm_layer=nn.BatchNorm2d)
        self.dsn = nn.Sequential(
            nn.Conv2d(2560, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.acfhead = ACFModule(2560, 512)

        self.bottlen = nn.Sequential(
            nn.Conv2d(4608, 1024, kernel_size=3, padding=1, dilation=1, bias=False),

            nn.Dropout2d(0.1, False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
#############################局部特征表示###################################

        self.part = 2
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, self.num_classes, True, 256))

        self.part_maxpool = nn.AdaptiveMaxPool2d(1)
        self.embed_x3 = nn.Linear(1792, 1792, bias=False)
        self.bn_x3 = nn.BatchNorm1d(1792)
        self.prelu = nn.PReLU()

    def forward(self, x):
        if self.breach == 'yes'and self.training:
            x = self.stn(x)

        x, att_f = self.base(x)
        # print(att_f.size())
        # print(x.size())
        #  基础网络分支
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (64, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        att_feat = self.embed_x3(att_f)
        att_feat = self.bn_x3(att_feat)
        att_feat = self.prelu(att_feat)
        # att_feat = self.drop(att_feat)
        # att_f = self.bottleneck(att_f)

        #注意力网络分支
        # feat_aspp = self.aspp(x)
        # # print(feat_aspp.size())
        #
        # coarse_x = self.dsn(feat_aspp)
        # acf_out = self.acfhead(feat_aspp, coarse_x)
        #
        # global_feat = self.gap(coarse_x)  # (b, 2048, 1, 1)
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (64, 2048)
        # feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # feat_cat = torch.cat([x, feat_aspp], dim=1)
        # print(feat_cat.size())
        # pre_out = self.bottlen(feat_cat)
        # print(pre_out.size())
        # print(rx1.size())
        # query = self.f_query(x).view(x.size(0), x.size(1), -1)
        # value = self.psp(self.f_value(x))
        # key = self.psp(self.f_key(x))
        # value = value.permute(0, 2, 1)

        # rx = value * x


        # x = self.cam(x)
        # rx = x * self.att
        # rx = rx + x
        # print(self.layer4(x).size())
        # a = self.layer4(x)
        # self.att, ax = self.cam(coarse_x)
        # print(self.att.size())
        # rx1 = x * self.att
        # print(rx1.size())
        # # ax = self.att_gap(rx1)
        # # # ax = self.reduction(ax)
        # # ax = ax.view(ax.size(0), -1)

        # rx = rx1 + x

        # rx = self.pam(a) + self.cam(a)

        # print(rx.size())  #([64, 2048, 16, 8])
        # # a = self.part_maxpool(a).squeeze()
        # # a = F.softmax(a, dim=1)

        # a = a.view(b, self.middle_dim,-1)
        # a = F.relu(self.attention_tconv(a))
        # print(a.size())
        # a = a.expand(b, self.in_planes)


        #
        #
        # //////////////////////////////////////////////////////////////////////  att_x = torch.mul(global_feat, a)
        #     att_x = torch.sum(att_x, 1)
        #     f = att_x.view(b, self.in_planes)
        #     feat = self.bottleneck(f)  # normalize for angular softmax

########################################################################
        # a = self.pam(x)
        # print(a.size())
        # rx = self.layer4(x)
        # rx = rx * self.pam(x)


        # global_feat = self.gap(x)  # (b, 2048, 1, 1)
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (64, 2048)
        # feat = self.bottleneck(global_feat)  # normalize for angular softmax


        # feat = feat * self.pam(x)

        #局部网络分支

        # px = self.avgpool(x)
        # # x = self.dropout(x)
        # part = {}
        # predict = {}
        # for i in range(self.part):
        #     part[i] = px[:, :, i, :]
        #     part[i] = torch.unsqueeze(part[i], 3)
        #     # print part[i].shape
        #     predict[i] = self.classifiers[i](part[i])
        # y = []  # part feat
        # g = []  # local feat
        # for i in range(self.part):
        #     y.append(predict[i])
        # g.append(feat)
        # g.append(y)
        # p = torch.cat(y, dim=1)
        # f = p.view(p.size(0), -1)

        if self.training:
            cls_score = self.classifier(feat)
            ax = self.classifier_att(att_feat)
            # return  softmax_features,triplet_features # feat for ranked loss
            return cls_score, feat, ax
        else:
            return feat
            # return torch.cat(predict, 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i not in self.state_dict() or 'classifier' in i:
                 continue
            self.state_dict()[i].copy_(param_dict[i])
