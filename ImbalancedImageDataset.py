import numpy as np
from torch.utils.data import Dataset
import AwareAugmentation
from PIL import Image

class ImbalancedImageDataset(Dataset):
    def __init__(self, image_paths, labels, class_counts,
                transform=None, use_class_aware_aug=False):
        self.image_paths = image_paths
        self.labels = labels
        self.class_counts = class_counts
        self.transform = transform
        self.use_class_aware_aug = use_class_aware_aug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # # Class-aware augmentation
        # if self.use_class_aware_aug:
        #     aug = AwareAugmentation(
        #         label=label,
        #         class_counts=self.class_counts
        #     )
        #     img = aug(img)
        if self.transform:
            img = self.transform(img)

        return img, label