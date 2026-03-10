import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset

class CrackDataset(Dataset):

    def __init__(self, image_dir, gt_dir, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform

        all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        self.image_files = []
        for img_name in all_images:
            gt_name = img_name.replace('.jpg', '.mat')
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path):
                self.image_files.append(img_name)

        print(f"找到 {len(self.image_files)} 张有标注的图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        gt_name = image_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)

        gt_data = scipy.io.loadmat(gt_path)['groundTruth']

        gt = gt_data[0,0]['Segmentation']

        has_crack = 1 if (gt > 0).any() else 0

        if self.transform:
            image = self.transform(image)

        return image, has_crack