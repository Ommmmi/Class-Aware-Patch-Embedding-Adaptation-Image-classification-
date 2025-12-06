import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

IMAGE_PATH = r'C:\Users\ritul\class aware patch embedding adaptation project\Kvasir dataset\images_by_label'
SPLIT_PATH = r'C:\Users\ritul\class aware patch embedding adaptation project\split_data_for_kvasir'


class MiniImageNet(Dataset):
    def __init__(self, setname, args=None,
                 minority_classes=None,
                 base_transform=None,
                 minority_transform=None):
        # Load the CSV for train/val/test
        csv_path = osp.join(SPLIT_PATH, f'{setname}.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]  # Skip header

        data, label = [], []
        class_to_label = {}
        lb = -1
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, wnid, name)
            if wnid not in class_to_label:
                lb += 1
                class_to_label[wnid] = lb
            data.append(path)
            label.append(class_to_label[wnid])

        self.data = data
        self.label = label
        self.setname = setname

        # Default transforms
        image_size = 224
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
        ])
        self.transform_val_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.setname == 'train':
            image = self.transform_train(Image.open(path).convert('RGB'))
        else:
            image = self.transform_val_test(Image.open(path).convert('RGB'))
        return image, label