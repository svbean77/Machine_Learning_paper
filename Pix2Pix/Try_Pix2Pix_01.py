from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

#이미지 하나 출력해봄(어떻게 생겼는지)
img = Image.open('D:/programming/week6/jupyter_notebook/facades/train/1.jpg')
plt.imshow(img)
plt.show()

trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#이미지 로드 및 2개인 이미지 하나로 자르기(실제와 segmentation을 나누기)
class Load_data(Dataset):
    def __init__(self, root, trans = None, mode = "train"):
        super(Load_data, self).__init__()
        self.transform = trans
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "val") + "/*.jpg")))

    def __getitem__(self, index):
        img = Image.open(self.files[index  % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

train_dataset = Load_data("D:/programming/week6/jupyter_notebook/facades", trans = trans)
test_dataset = Load_data("D:/programming/week6/jupyter_notebook/facades", trans = trans, mode = "test")
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

for i, img in enumerate(train_loader):
    real_A = img["A"]
    real_B = img["B"]


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm = True, dropout = 0.0):
        super(Down, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.0):
        super(Up, self).__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace = True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), 1)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.0):
        super(Discriminator, self).__init__()

        layers = []

        layers.append(nn.LeakyReLU(out_channels))