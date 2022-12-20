'''
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
위 사이트를 참고하여 바꿔봄! 모델 훈련은 나동빈님 코드 사용
24000에서 collapsion이 일어났다.. 그 전까지 거의 완벽히 모방하던 이미지가 갑자기 붕괴되는 현상 발생
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_dataset = datasets.MNIST(root="D:/programming/week6/data", train=True, download=True, transform=trans)
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
'''
import matplotlib.pyplot as plt
import numpy as np

def load_image(img, label = ""):
    if img.size(0) == 1:
        plt.imshow(img.squeeze(), cmap = "gray")
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))

    plt.title(label)
    plt.axis('off')

img, label = next(iter(train_loader))

plt.figure(figsize = (4, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    load_image(img[i]/2+0.5, train_dataset.classes[label[i].numpy()])
    plt.axis('off')
plt.show()
'''
latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

G = Generator().cuda()
D = Discriminator().cuda()
criterion = nn.BCELoss().cuda()

optim_G = optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optim_D = optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))

num_epochs = 100
sample_interval = 2000

for epoch in range(num_epochs):
    for i, (img, _) in enumerate(train_loader):
        real = torch.ones(size = (img.size(0), 1, 1, 1)).cuda()
        fake = torch.zeros(size = (img.size(0), 1, 1, 1)).cuda()

        real_img = img.cuda()

        optim_G.zero_grad()
        z = torch.normal(mean = 0, std = 1, size = (img.shape[0], latent_dim, 1, 1)).cuda()
        fake_img = G(z)

        G_loss = criterion(D(fake_img), real)

        G_loss.backward()
        optim_G.step()

        optim_D.zero_grad()

        real_loss = criterion(D(real_img), real)
        fake_loss = criterion(D(fake_img.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        optim_D.step()

        save = epoch * len(train_loader) + i
        if save % sample_interval == 0:
            save_image(fake_img[:25], f"mnist - {save}.png", nrow=5, normalize=True)

    print(f"Epoch {epoch + 1} - G_loss: {G_loss.item():.6f}, D_loss: {D_loss.item():.6f}")
