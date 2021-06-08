import torch.nn as nn
import functools


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from .resnet import resnet_block, GlobalAvgPool2d


class ResNetDiscriminator(nn.Module):
    def __init__(self, input_channel, aus_channel):
        super(ResNetDiscriminator, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        self.resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
        self.resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
        self.resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
        self.resnet.add_module("global_avg_pool", GlobalAvgPool2d())
        
        self.dis_top = nn.Conv2d(512, 1, kernel_size=(1, 1))
        self.aus_top = nn.Conv2d(512, aus_channel, kernel_size=(1, 1))

    def forward(self, x):
        hidden = self.resnet(x)
        pred_map = self.dis_top(hidden)
        pred_aus = self.aus_top(hidden)
        return pred_map.squeeze(), pred_aus.squeeze()
        


class Discriminator(nn.Module):
    def __init__(self, input_nc, aus_nc, image_size=128, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]

        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)
        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
        
        k_size = int(image_size / (2 ** n_layers))
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)


    def forward(self, img):
        embed_features = self.model(img)
        pred_map = self.dis_top(embed_features)
        pred_aus = self.aus_top(embed_features)
        return pred_map.squeeze(), pred_aus.squeeze()



class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    print("not here!")
    pass