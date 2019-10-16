import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
from pathlib import Path
from PIL import Image


class Wider(data.Dataset):
    def __init__(self, root, path, transforms=None, mode='train'):
        super(Wider, self).__init__()

        self.data = []
        self.root_path = Path(root)
        self.path = Path(path)  # path to file in wider_face_split
        self.mode = mode
        self.transforms = transforms

        self.load_data(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_info = self.data[index]
        image_root = self.root_path / 'WIDER_{0}'.format(self.mode)
        image_path = image_root / "images" / img_info['img_path']
        img = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            img, img_info = self.transforms(img, img_info)

        return img, img_info

    def load_data(self, path):

        with open(path) as f:
            lines = f.readlines()

        if self.mode in ('train', 'val'):

            self.data = []
            idx = 0

            while idx < len(lines):
                img_path = lines[idx].strip()
                idx += 1
                num_faces = int(lines[idx].strip())
                idx += 1

                bboxes = np.empty((num_faces, 10))

                if num_faces != 0:
                    for j in range(num_faces):
                        bboxes[j, :] = [float(x) for x in lines[idx].strip().split()]
                        idx += 1
                else:
                    idx += 1

                invalid_examples_mask = np.where(np.logical_or(bboxes[:, 2] <= 0, bboxes[:, 3] <= 0))
                bboxes = np.delete(bboxes, invalid_examples_mask, axis=0)

                # xmin, ymin, width, height -> xmin, ymin, xmax, ymax
                bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

                img_info = {
                    'img_path': img_path,
                    'bboxes': bboxes[:, 0:4],
                    'blur': bboxes[:, 4],
                    'expression': bboxes[:, 5],
                    'illumination': bboxes[:, 6],
                    'invalid': bboxes[:, 7],
                    'occlusion': bboxes[:, 8],
                    'pose': bboxes[:, 9]
                }

                self.data.append(img_info)

        elif self.mode == 'test':
            self.data = [{'img_path': line.strip() for line in lines}]
