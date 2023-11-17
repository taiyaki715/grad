import glob

import numpy as np
from PIL import Image
import torch
import torchvision

class Dataset(torch.utils.data.Dataset):
  def __init__(self, train=True):
    train_path = "/Volumes/western_digital_4tb/train"
    test_path = "/Volumes/western_digital_4tb/val"

    image_paths = [str(path) for path in glob.glob(f"{train_path if train else test_path}/*/*/*/*.png", recursive=True)]
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
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size=(512, 512))
    image = torchvision.transforms.functional.crop(image, i, j, h, w)
    depth = torchvision.transforms.functional.crop(depth, i, j, h, w)

    image = torchvision.transforms.ToTensor()(image)
    depth = torchvision.transforms.ToTensor()(depth) / 350.0

    return image, depth