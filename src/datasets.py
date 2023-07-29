import os
import skia
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class GraphemesDataset(Dataset):
    def __init__(self, root_dir, train=True, test_size=0.2, random_state=42, by_letter=None, transform=None):
        self.by_letter = by_letter
        if by_letter:
            self.root_dir = os.path.join(root_dir, by_letter)
        else:
            self.root_dir = root_dir
        self.classes = sorted(listdir_nohidden(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.images = self.load_images()

        if train:
            self.images, _ = train_test_split(self.images, test_size=test_size, random_state=random_state)
        else:
            _, self.images = train_test_split(self.images, test_size=test_size, random_state=random_state)

        self.transform = transform

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if self.by_letter:
                for image_file in listdir_nohidden(class_dir):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(class_dir, image_file)
                        images.append((image_path, self.class_to_idx[class_name]))
            else:
                for writing_system in listdir_nohidden(class_dir):
                    writing_system_dir = os.path.join(class_dir, writing_system)
                    for image_file in listdir_nohidden(writing_system_dir):
                        if image_file.endswith('.png'):
                            image_path = os.path.join(writing_system_dir, image_file)
                            images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = skia.Image.open(image_path).toarray()[..., :1]
        # Preprocess the image (e.g., resize, normalize) if needed
        # image = your_preprocessing_function(image)

        if self.transform is not None:
            image = self.transform(image)
    
        tensor_image = torch.Tensor(image)  # Convert Skia image to tensor
        return tensor_image, label