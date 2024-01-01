from .classes import IMAGENET2012_CLASSES
import os
from PIL import Image
import numpy as np
import torch
import h5py

class ImageNet():
    """ImageNet-dataset loader with 1000 classess."""
    def __init__(self, data_folder_path, num_samples) -> None:
        self.data_dir = data_folder_path
        self.num_samples = num_samples
        self.dataset = self.data_generator()

    def data_generator(self):
        class_list = os.listdir(self.data_dir)
        for _ in range(self.num_samples):
            cls = class_list[np.random.randint(0,len(class_list))]
            img_list = os.listdir(f"{self.data_dir}/{cls}")
            img = img_list[np.random.randint(0,len(img_list))]
            im = Image.open(f"{self.data_dir}/{cls}/{img}")
            im = im.resize((384,384))
            normalized_im = (np.array(im) / 128) - 1
            im_tensor = torch.from_numpy(np.array(normalized_im)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            
            cls_id = cls.split("_")[0]
            label = IMAGENET2012_CLASSES[cls_id]

            yield im, im_tensor, label

class Dataloader():
    """Data loader for single image."""
    def __init__(
        self, 
        img_path,
        test_dataset=None, 
        data_type=None,
        cls_dir=None,
        transform=None
    ) -> None:
        self.data = self.data_generator(test_dataset, data_type, cls_dir, img_path, transform)

    def data_generator(self, test_dataset, data_type, cls_dir, img_path, transform):
        img = None
        if data_type == 'h5':
            path = img_path.split('-')[0]
            idx = img_path.split('-')[1]
            with h5py.File(path, 'r') as f:
                images = f['x'][:]
                img = Image.fromarray(images[int(idx)])
        else:
            img = Image.open(f"./{img_path}")

        if transform is None:
            img = img.resize((384,384))
            normalized_img = (np.array(img) / 128) - 1
            img_tensor = torch.from_numpy(np.array(normalized_img)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        else:
            img_tensor = transform(img).unsqueeze(0)
        img = img.resize((384,384))

        classes = None
        if test_dataset is not None:
            if hasattr(test_dataset, 'classes'):
                classes = test_dataset.classes
        elif cls_dir is not None:
            with open(cls_dir) as f:
                data = f.readlines()
                classes = [cls.strip() for cls in data]

        yield img, img_tensor, classes