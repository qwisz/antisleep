import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from PIL import Image


class Helen(data.Dataset):
    def __init__(self, root, transforms=None, mode='train'):
        super(Helen, self).__init__()

        self.data = []
        self.root_path = Path(root) # ../data/
        self.mode = mode
        self.transforms = transforms

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info = self.data[index]
        img = Image.open(img_info['img_path']).convert('RGB')

        if self.transforms is not None:
            img, img_info['landmarks'] = self.transforms(img, img_info['landmarks'])

        return img, img_info['landmarks']

    def get_img_path(self, index):
        return self.data[index]['img_path']

    def load_data(self):

        path_to_txt = self.root_path / 'trainimages.txt'

        with open(path_to_txt) as f:
            files = f.readlines()

        images_names = set(map(lambda x: x.strip(), files))

        path_to_images = self.root_path / self.mode
        path_to_images_annotations = self.root_path / '{}_annotation'.format(self.mode)

        if self.mode in ('train', 'val'):

            self.data = []

            for image_name in images_names:
                landmarks = []
                path_to_image = path_to_images / image_name
                path_to_image_annotation = path_to_images_annotations / '{}.txt'.format(image_name[:-4])

                with open(path_to_image_annotation) as f:
                    points = f.readlines()
                    for point in points:
                        xy = point.strip().split(',')
                        for p in xy:
                            landmarks.append(float(p.strip()))

                img_info = {
                    'img_path': path_to_image,
                    'landmarks': landmarks
                }

                self.data.append(img_info)

        # elif self.mode == 'test':
        #     self.data = [{'img_path': line.strip() for line in lines}]
