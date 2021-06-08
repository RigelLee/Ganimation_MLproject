import torch
import os
from PIL import Image
import pickle
import torchvision.transforms as transforms

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)
        self.is_train = self.opt.mode == "train"

        filename = self.opt.train_csv if self.is_train else self.opt.test_csv
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)
        self.imgs_path = self.make_dataset()

        aus_pkl = os.path.join(self.opt.data_root, self.opt.aus_pkl)
        self.aus_dict = self.load_dict(aus_pkl)

        self.img2tensor = self.img_transformer()

    def make_dataset(self):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def get_aus_by_path(self, img_path):
        return None

    def img_transformer(self):
        transform_list = []
        transform_list.append(transforms.Lambda(lambda image: image))

        if self.is_train and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        img2tensor = transforms.Compose(transform_list)

        return img2tensor

    def __len__(self):
        return len(self.imgs_path)
