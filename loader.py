import os
import skia
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
class GraphemesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(listdir_nohidden(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for writing_system in listdir_nohidden(class_dir):
                writing_system_dir = os.path.join(class_dir, writing_system)
                grapheme_images = listdir_nohidden(writing_system_dir)
                for image_file in grapheme_images:
                    if image_file.endswith('.png'):
                        image_path = os.path.join(writing_system_dir, image_file)
                        images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = skia.Image.open(image_path).toarray()[..., :3]
        # Preprocess the image (e.g., resize, normalize) if needed
        # image = your_preprocessing_function(image)
        tensor_image = torch.Tensor(image)  # Convert Skia image to tensor
        return tensor_image, label


# plot a sample image
image = skia.Image.open("dataset_skia/aleph/Hebrew/aleph_16_Hebrew_Cardob101_0_0.png").toarray()

# Example usage:
root_dir = "dataset_skia"
dataset = GraphemesDataset(root_dir)

# Accessing dataset samples
image, label = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")

# plot the first 10 images in the dataset
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    image, label = dataset[i]
    ax.imshow(image[:, :, 0])
    ax.set_title(dataset.classes[label])
    ax.axis('off')
plt.show()