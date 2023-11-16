import glob

import numpy as np
from PIL import Image
import torch
import torchvision

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    image_paths = [str(path) for path in glob.glob("/Volumes/western_digital_4tb/train/*/*/*/*.png")]
    depth_paths = [path.replace(".png", "_depth.npy") for path in image_paths]
    self.data_paths = list(zip(image_paths, depth_paths))

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, idx):
    image_path, depth_path = self.data_paths[idx]

    image = Image.open(image_path)
    depth = Image.fromarray(np.load(depth_path).squeeze(axis=-1))

    image, depth = self.transform(image, depth)

    return image, depth

  def transform(self, image, depth):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(224, 224))
    image = torchvision.transforms.functional.crop(image, i, j, h, w)
    depth = torchvision.transforms.functional.crop(depth, i, j, h, w)

    image = torchvision.transforms.ToTensor()(image)
    depth = torchvision.transforms.ToTensor()(depth)

    image = image / 255.0
    depth = depth / 255.0

    return image, depth