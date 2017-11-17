'''
Modified from https://github.com/pytorch/vision.git
'''
from .layers import KernelLayer


import math

import torch.nn as nn
import torch.nn.init as init


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, kernel_type=None, n_features=None, gamma=None):
        super(VGG, self).__init__()
        self.features = features
        self.gamma = gamma

        if kernel_type is not None and n_features is None:
            raise ValueError('Undefined number of kernel features!')

        if kernel_type is not None and kernel_type.lower() not in ['rbf', 'arccos']:
            raise ValueError('Unknown kernel type!')


        # if kernel_type is not None:
        #     self.classifier = nn.Sequential(#nn.Dropout(),
        #                                     KernelLayer(512, n_features, kernel_type),
        #                                     nn.Linear(n_features, 10)
        #                                    )
        # else:
        #     self.classifier = nn.Sequential(nn.Dropout(),
        #                                     nn.Linear(512, 512),
        #                                     nn.ReLU(True),
        #                                     nn.Dropout(),
        #                                     nn.Linear(512, 512),
        #                                     nn.ReLU(True),
        #                                     nn.Linear(512, 10)
        #                                     )

        if kernel_type is not None:
            self.classifier = KernelLayer(512, n_features, kernel_type, self.gamma)
            self.last_layer = nn.Linear(n_features, 10)
        else:
            self.classifier = nn.Sequential(nn.Dropout(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(True)
                                           )
            self.last_layer = nn.Linear(512, 10)


        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
