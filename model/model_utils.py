import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from networks.generator import Generator
from networks.discriminator import Discriminator
from networks.discriminator import ResNetDiscriminator


def init_net(net, init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def get_norm_layer():
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return norm_layer

def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network!')
    net.apply(init_func)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = target_is_real
        return self.loss(input, target_tensor)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def define_G(img_nc, aus_nc, ngf, use_dropout=False, init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer()
    net_img_au = Generator(img_nc, aus_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    return init_net(net_img_au, init_gain, gpu_ids)


def define_D(input_nc, aus_nc, image_size, ndf, init_gain=0.02, gpu_ids=[]):
    net_dis_aus = ResNetDiscriminator(input_nc, aus_nc)
    norm_layer = get_norm_layer()
    
    net_dis_aus = Discriminator(input_nc, aus_nc, image_size, ndf, n_layers=6, norm_layer=norm_layer)
    
    return init_net(net_dis_aus, init_gain, gpu_ids)


if __name__ == '__main__':
    print("not here!")
    pass