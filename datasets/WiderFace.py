import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from PIL import Image
import cv2 as cv

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
        # img = Image.open(image_path).convert('RGB')
        img = cv.imread(str(image_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.transforms is not None:
            img, img_info['target'] = self.transforms(img, img_info['target'])

        return img, img_info['target']

    def get_img_path(self, index):
        return self.data[index]['img_path']

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
                if bboxes.shape[0] > 0:
                    # xmin, ymin, width, height -> xmin, ymin, xmax, ymax
                    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

                    bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                    labels = torch.ones((num_faces,), dtype=torch.int64)


                    target = {
                        'boxes': bboxes[:, 0:4],
                        'labels': labels,
                        # 'blur': bboxes[:, 4],
                        # 'expression': bboxes[:, 5],
                        # 'illumination': bboxes[:, 6],
                        # 'invalid': bboxes[:, 7],
                        # 'occlusion': bboxes[:, 8],
                        # 'pose': bboxes[:, 9]
                    }

                    img_info = {
                        'img_path': img_path,
                        'target': target
                    }

                    self.data.append(img_info)

        elif self.mode == 'test':
            self.data = [{'img_path': line.strip() for line in lines}]
