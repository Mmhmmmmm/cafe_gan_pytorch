# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Network components."""

import torch.nn as nn
from switchable_norm import SwitchNorm1d, SwitchNorm2d


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class make_mtl_block(nn.Module):

    def __init__(self, block, layers, num_tasks):
        self.num_tasks = num_tasks
        super(make_mtl_block, self).__init__()
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.sigmoid = nn.Sigmoid()

        output = [nn.Linear(512 * block.expansion, 1) for _ in range(self.num_tasks)]
        # att_conv = [nn.Conv2d(512 * block.expansion, 1, kernel_size=1, padding=0, bias=True) for _ in range(num_tasks)]
        # att_bn = [nn.BatchNorm2d(1) for _ in range(num_tasks)]
        self.output = nn.ModuleList(output)
        # self.att_conv = nn.ModuleList(att_conv)
        # self.att_bn = nn.ModuleList(att_bn)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        # self.inplanes = 1024
        self.inplanes = 256 * block.expansion

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x, att_elem):
        pred = []
        attention = []
        for i in range(self.num_tasks):
            bs, cs, ys, xs = att_elem.shape
            # item_att_elem = att_elem # [:, i].view(bs, 1, ys, xs)
            item_att = att_elem[:, i].view(bs, 1, ys, xs)
            # item_att = self.att_conv[i](item_att_elem)
            # item_att = self.sigmoid(self.att_bn[i](item_att))
            # item_att = self.sigmoid(item_att)
            attention.append(item_att)

            sh = item_att * x
            sh += x
            sh = self.layer4(sh)
            sh = self.avgpool(sh)
            sh = sh.view(sh.size(0), -1)
            # sh = self.sigmoid(self.output[i](sh))
            sh = self.output[i](sh)
            pred.append(sh)

        return pred, attention

class ABN_Block(nn.Module):
    def __init__(self, in_planes, num_classes=13):
        super(ABN_Block, self).__init__()
        self.att_conv = nn.Conv2d(in_planes, num_classes, kernel_size=1, padding=0,
                                  bias=False)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1,
                                   bias=False)
        self.att_gap = nn.MaxPool2d(8)
        self.sigmoid = nn.Sigmoid()
        self.depth_conv = nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ax):

        ax = self.att_conv(ax)
        ax = self.depth_conv(ax)
        att = self.sigmoid(ax)

        ax = self.att_gap(ax)
        # ax = self.sigmoid(ax)
        ax = ax.view(ax.size(0), -1)

        return ax, att

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=13):
        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], down_size=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, down_size=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        self.att_layer4 = self._make_layer(block, 512, layers[3], stride=1, down_size=False)

        self.abn = ABN_Block(512*block.expansion, num_classes)
        self.cabn = ABN_Block(512*block.expansion, num_classes)

        self.cls1 = make_mtl_block(block, layers, num_classes)
        self.cls2 = make_mtl_block(block, layers, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x, type=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if type is None:
            ax = self.att_layer4(x)

            abn_ax, abn_att = self.abn(ax)
            cabn_ax, cabn_att = self.cabn(ax)

            cls1_rx, cls1_attention = self.cls1(x, abn_att)
            cls2_rx, cls2_attention = self.cls2(x, cabn_att)

            return x, abn_ax, cabn_ax, cls1_rx, cls2_rx, cls1_attention, cls2_attention
        elif type == 'adv':
            return x
        elif type == 'abn':
            ax = self.att_layer4(x)

            _, abn_att = self.abn(ax)
            _, cabn_att = self.cabn(ax)

            _, cls1_attention = self.cls1(x, abn_att)
            _, cls2_attention = self.cls2(x, cabn_att)

            return cls1_attention, cls2_attention