from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        if imgs_dir != '' and masks_dir != '':
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                        if not file.startswith('.')]
            logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        # print(newW, newH)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img).astype('float32')

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # print('37 img_nd', img_nd.shape)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print('41 img_nd', img_trans.shape)
        # normalize gray to [0, 1]
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def mask_preprocess(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img).astype('float32')
        # one-hot coding
        img_res = np.zeros((newH, newW, 4))
        img_res[:, :, 0] = 1
        img_res[:, :, 1] = np.where(img_nd == 1, 1, 0)
        img_res[:, :, 2] = np.where(img_nd == 2, 1, 0)
        img_res[:, :, 3] = np.where(img_nd == 3, 1, 0)
        img_nd = img_res

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print('67', img_trans.shape)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        # mask = np.array(mask)
        # mask[mask > 1] = 0
        mask = self.mask_preprocess(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
