from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn
import math


class VGG(nn.Module):

    def __init__(self, name, args):
        super(VGG, self).__init__()
        self.name = name
        self.num_classes = args.num_classes
        if args.dataset == "mnist" or args.dataset == "fashion_mnist":
            self.in_channels = 1
        else:
            self.in_channels = 3
        if args.architecture == 'VGG3':
            self.cfg = [64, 'M', 128, 128, 'M']
        elif args.architecture == 'VGG5':
            self.cfg = [64, 'M', 128, 'M', 256, 'M', 512]
        elif args.architecture == 'VGG7':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
        elif args.architecture == 'VGG9':
            self.cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M']
        elif args.architecture == 'VGG11':
            self.cfg = [
                64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
            ]
        elif args.architecture == 'VGG13':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M'
            ]
        elif args.architecture == 'VGG16':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M'
            ]
        elif args.architecture == 'VGG19':
            self.cfg = [
                64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M'
            ]
        else:
            raise Exception("VGG architecture not known")
        self.features = self.make_layers()
        self.classifier = nn.Linear(
            128 if args.architecture == 'VGG3' else 512, self.num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def make_layers(self):
        layers = []
        in_channels = self.in_channels
        for c in self.cfg:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv = nn.Conv2d(in_channels,
                                 c,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)
                layers += [conv, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                in_channels = c
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
