from pathlib import Path

import torch.utils.data as data
from PIL import Image
from os import listdir


class Eyes(data.Dataset):
    def __init__(self, root, size=None, transforms=None, mode='train'):
        super(Eyes, self).__init__()

        self.data = []
        self.root_path = Path(root)  # ../eyes/
        self.mode = mode
        self.transforms = transforms
        self.size = size

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info = self.data[index]
        img = Image.open(img_info['img_path']).convert('RGB')

        if self.transforms is not None:
            img, img_info['target'] = self.transforms(img, img_info['target'])

        if self.size is not None:
            img = img.resize(self.size)

        return img, img_info['target']

    def get_img_path(self, index):
        return self.data[index]['img_path']

    def load_data(self):

        self.data = []

        for eyes_type in ['open', 'close']:
            path_to_eyes = self.root_path / self.mode / eyes_type
            image_names = listdir(path_to_eyes)
            for image_name in image_names:
                img_info = {'img_path': path_to_eyes / image_name}
                if eyes_type == 'open':
                    img_info['target'] = 0
                else:
                    img_info['target'] = 1

                self.data.append(img_info)
