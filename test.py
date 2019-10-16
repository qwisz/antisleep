import datasets.WiderFace as wider

train_dataset = wider.Wider('/Users/antonbarybin/Yandex.Disk.localized/Learning/csc/antisleep/wider/',
                    '/Users/antonbarybin/Yandex.Disk.localized/Learning/csc/antisleep/wider/wider_face_split/wider_face_train_bbx_gt.txt',
                            mode='train')

print(train_dataset[0])