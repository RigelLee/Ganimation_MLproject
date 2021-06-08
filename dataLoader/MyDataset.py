from .base_dataset import BaseDataset
import os
import random
import numpy as np


class MyDataset(BaseDataset):

    def __init__(self):
        super(MyDataset, self).__init__()

    def initialize(self, opt):
        super(MyDataset, self).initialize(opt)

    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0

    def make_dataset(self):
        imgs_path = []
        assert os.path.isfile(self.imgs_name_file), "%s does not exist." % self.imgs_name_file
        with open(self.imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs_path = [os.path.join(self.imgs_dir, line.strip()) for line in lines]
            imgs_path = sorted(imgs_path)

        return imgs_path

    def __getitem__(self, index):
        
        img_path = self.imgs_path[index]

        src_img = self.get_img_by_path(img_path)
        src_img_tensor = self.img2tensor(src_img)
        src_aus = self.get_aus_by_path(img_path)

        
        target_names = [    '000367.jpg', '001494.jpg', '001654.jpg', '003285.jpg', '003396.jpg',
                            '000123.jpg', '000332.jpg', '003368.jpg', '132510.jpg', '132626.jpg',
                            '000338.jpg', '003477.jpg', '132406.jpg', '132508.jpg', '176517.jpg',
                            '000001.jpg', '000072.jpg', '000110.jpg', '001740.jpg', '000643.jpg',
                            '001669.jpg', '002649.jpg', '002638.jpg', '002684.jpg', '132617.jpg',
                            '001040.jpg', '001076.jpg', '002500.jpg', '132858.jpg', '176302.jpg',
                            '000025.jpg', '000026.jpg', '000755.jpg', '000774.jpg', '000939.jpg']
        target_paths =  [os.path.join(self.imgs_dir, path) for path in target_names]
        target_paths = sorted(target_paths)
        tar_img_path = random.choice(self.imgs_path)
        tar_img = self.get_img_by_path(tar_img_path)
        tar_img_tensor = self.img2tensor(tar_img)
        tar_aus = self.get_aus_by_path(tar_img_path)
        if self.is_train and not self.opt.no_aus_noise:
            tar_aus = tar_aus + np.random.uniform(-0.1, 0.1, tar_aus.shape)

        data_dict = {'src_img': src_img_tensor, 'src_aus': src_aus, 'tar_img': tar_img_tensor, 'tar_aus': tar_aus, \
                     'src_path': img_path, 'tar_path': tar_img_path}

        return data_dict
