import os
import cv2
import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    Dataset structure:
    ├─A/          # 时相1图像
    ├─B/          # 时相2图像
    ├─label/      # 标签
    └─list/
       ├─train.txt
       ├─val.txt
       └─test.txt
    '''
    def __init__(self, dataset, file_root='data/', split='train', transform=None):
        """
        Args:
            dataset: dataset name (e.g., 'LEVIR', 'SYSU', 'WHU', 'GZ-CD')
            file_root: root directory of datasets
            split: 'train', 'val', or 'test'
            transform: data augmentation transforms
        """
        self.dataset_path = os.path.join(file_root, dataset)
        self.split = split
        
        # Load image name list from list folder
        list_file = os.path.join(self.dataset_path, 'list', split + '.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(
                f"Cannot find list file: {list_file}\n"
                f"Please check your dataset structure."
            )
        
        self.file_list = open(list_file).read().splitlines()
        
        # Build image paths
        self.img_dir_a = os.path.join(self.dataset_path, 'A')
        self.img_dir_b = os.path.join(self.dataset_path, 'B')
        self.label_dir = os.path.join(self.dataset_path, 'label')
        
        self.pre_images = [os.path.join(self.img_dir_a, x) for x in self.file_list]
        self.post_images = [os.path.join(self.img_dir_b, x) for x in self.file_list]
        self.gts = [os.path.join(self.label_dir, x) for x in self.file_list]
        
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]
        
        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)
        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            [img, label] = self.transform(img, label)
        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
