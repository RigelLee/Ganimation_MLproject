from dataLoader import create_dataloader
from model import create_model
from utils.visualizer import Visualizer
from utils.imageTransform import *
import time
import os
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict


class App(object):

    def __init__(self, opt):
        super(App, self).__init__()
        self.opt = opt
        self.visual = Visualizer()
        self.visual.initialize(self.opt)

    def run(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks()

    def train_networks(self):
        self.init_train_setting()

        for epoch in range(1, self.epoch_len + 1):
            self.train_epoch(epoch)
            self.cur_lr = self.train_model.update_learning_rate()
            if epoch % self.opt.save_epoch_freq == 0:
                self.train_model.save_ckpt(epoch)

        self.train_model.save_ckpt(self.epoch_len)

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        self.train_model = create_model(self.opt)

        self.train_total_steps = 0
        self.epoch_len = self.opt.niter + self.opt.niter_decay
        self.cur_lr = self.opt.lr

    def train_epoch(self, epoch):
        epoch_steps = 0

        last_print_step_t = time.time()
        for idx, batch in enumerate(self.train_dataset):

            self.train_total_steps += self.opt.batch_size
            epoch_steps += self.opt.batch_size

            self.train_model.feed_batch(batch)
            self.train_model.optimize_paras(train_gen=(idx % self.opt.train_gen_iter == 0))

            if self.train_total_steps % self.opt.print_losses_freq == 0:
                cur_losses = self.train_model.get_latest_losses()
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq
                last_print_step_t = time.time()

                info_dict = {'epoch': epoch, 'epoch_len': self.epoch_len,
                             'epoch_steps': idx * self.opt.batch_size, 'epoch_steps_len': len(self.train_dataset),
                             'step_time': avg_step_t, 'cur_lr': self.cur_lr,
                             'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                             'losses': cur_losses
                             }
                self.visual.print_losses_info(info_dict)

            if self.train_total_steps % self.opt.plot_losses_freq == 0 and self.visual.display_id > 0:
                cur_losses = self.train_model.get_latest_losses()
                epoch_steps = idx * self.opt.batch_size
                cur_losses_1 = OrderedDict()
                cur_losses_2 = OrderedDict()
                cur_losses_3 = OrderedDict()
                cur_losses_1['dis_fake'] = cur_losses['dis_fake']
                cur_losses_1['dis_real'] = cur_losses['dis_real']
                cur_losses_2['dis_real_aus_1'] = cur_losses['dis_real_aus']
                cur_losses_2['dis_real_aus_2'] = cur_losses['dis_real_aus']
                cur_losses_3['gen_rec_1'] = cur_losses['gen_rec']
                cur_losses_3['gen_rec_2'] = cur_losses['gen_rec']
                self.visual.display_current_losses_1(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses_1)
                self.visual.display_current_losses_2(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses_2)
                self.visual.display_current_losses_3(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses_3)

            if self.train_total_steps % self.opt.sample_img_freq == 0 and self.visual.display_id > 0:
                cur_vis = self.train_model.get_latest_visuals()
                self.visual.display_online_results(cur_vis, epoch)

    def test_networks(self):
        self.init_test_setting()
        self.test_ops()

    def init_test_setting(self):
        self.test_dataset = create_dataloader(self.opt)
        self.test_model = create_model(self.opt)

    def test_ops(self):
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():
                faces_list = [batch['src_img'].float().numpy()]
                paths_list = [batch['src_path'], batch['tar_path']]
                for idx in range(self.opt.interpolate_len):
                    cur_alpha = (idx + 1.) / float(self.opt.interpolate_len)
                    cur_tar_aus = cur_alpha * batch['tar_aus'] + (1 - cur_alpha) * batch['src_aus']
                    test_batch = {'src_img': batch['src_img'], 'tar_aus': cur_tar_aus, 'src_aus': batch['src_aus'],
                                  'tar_img': batch['tar_img']}

                    self.test_model.feed_batch(test_batch)
                    self.test_model.forward()

                    cur_gen_faces = self.test_model.fake_img.cpu().float().numpy()
                    faces_list.append(cur_gen_faces)
                faces_list.append(batch['tar_img'].float().numpy())
            self.test_save_imgs(faces_list, paths_list)

    def test_save_imgs(self, faces_list, paths_list):
        for idx in range(len(paths_list[0])):
            src_name = os.path.splitext(os.path.basename(paths_list[0][idx]))[0]
            tar_name = os.path.splitext(os.path.basename(paths_list[1][idx]))[0]

            concate_img = np.array(np2im(faces_list[0][idx]))
            for face_idx in range(1, len(faces_list)):
                concate_img = np.concatenate(
                    (concate_img, np.array(np2im(faces_list[face_idx][idx]))), axis=1)
            concate_img = Image.fromarray(concate_img)
            saved_path = os.path.join(self.opt.results, "%s_%s.jpg" % (src_name, tar_name))
            concate_img.save(saved_path)

            print("[Success] Saved images to %s" % saved_path)


if __name__ == '__main__':
    print("not here!")
    pass