import numpy as np
from .imageTransform import *

class Visualizer(object):

    def __init__(self):
        super(Visualizer, self).__init__()

    def initialize(self, opt):
        self.opt = opt

        self.display_id = self.opt.visdom_display_id
        if self.display_id > 0:
            import visdom
            self.ncols = 8
            self.vis = visdom.Visdom(server="http://localhost", port=self.opt.visdom_port, env=self.opt.visdom_env)

    def throw_visdom_connection_error(self):
        print('\nVisdom server error!\n')
        exit(1)

    def display_current_losses_1(self, epoch, counter_ratio, losses_dict):
        if not hasattr(self, 'plot_data_1'):
            self.plot_data_1 = {'X': [], 'Y': [], 'legend': list(losses_dict.keys())}
        self.plot_data_1['X'].append(epoch + counter_ratio)
        self.plot_data_1['Y'].append([losses_dict[k] for k in self.plot_data_1['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data_1['X'])] * len(self.plot_data_1['legend']), 1),
                Y=np.array(self.plot_data_1['Y']),
                opts={
                    'title': self.opt.name + ' loss over time' + ' 1',
                    'legend': self.plot_data_1['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def print_losses_info(self, info_dict):
        msg = '[{}][Epoch: {:0>3}/{:0>3}; Images: {:0>4}/{:0>4}; Time: {:.3f}s/Batch({}); LR: {:.7f}] '.format(
            self.opt.name, info_dict['epoch'], info_dict['epoch_len'],
            info_dict['epoch_steps'], info_dict['epoch_steps_len'],
            info_dict['step_time'], self.opt.batch_size, info_dict['cur_lr'])
        for k, v in info_dict['losses'].items():
            msg += '| {}: {:.4f} '.format(k, v)
        msg += '|'
        print(msg)
        with open(info_dict['log_path'], 'a+') as f:
            f.write(msg + '\n')

    def display_current_losses_2(self, epoch, counter_ratio, losses_dict):
        if not hasattr(self, 'plot_data_2'):
            self.plot_data_2 = {'X': [], 'Y': [], 'legend': list(losses_dict.keys())}
        self.plot_data_2['X'].append(epoch + counter_ratio)
        self.plot_data_2['Y'].append([losses_dict[k] for k in self.plot_data_2['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data_2['X'])] * len(self.plot_data_2['legend']), 1),
                Y=np.array(self.plot_data_2['Y']),
                opts={
                    'title': self.opt.name + ' loss over time' + ' 2',
                    'legend': self.plot_data_2['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id+2)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_current_losses_3(self, epoch, counter_ratio, losses_dict):
        if not hasattr(self, 'plot_data_3'):
            self.plot_data_3 = {'X': [], 'Y': [], 'legend': list(losses_dict.keys())}
        self.plot_data_3['X'].append(epoch + counter_ratio)
        self.plot_data_3['Y'].append([losses_dict[k] for k in self.plot_data_3['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data_3['X'])] * len(self.plot_data_3['legend']), 1),
                Y=np.array(self.plot_data_3['Y']),
                opts={
                    'title': self.opt.name + ' loss over time' + ' 3',
                    'legend': self.plot_data_3['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id+3)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def display_online_results(self, visuals, epoch):
        win_id = self.display_id + 24
        images = []
        labels = []
        for label, image in visuals.items():
            if 'mask' in label: 
                image = (image - 0.5) / 0.5  
            image_np = tensor2im(image)
            images.append(image_np.transpose([2, 0, 1]))
            labels.append(label)
        try:
            title = ' || '.join(labels)
            self.vis.images(images, nrow=self.ncols, win=win_id,
                            padding=5, opts=dict(title=title))
        except ConnectionError:
            self.throw_visdom_connection_error()
