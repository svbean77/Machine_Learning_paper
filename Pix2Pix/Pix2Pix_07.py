import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

print("학습 데이터셋 A와 B의 개수:", len(next(os.walk('D:/programming/week6/jupyter_notebook/facades/train/'))[2]))
print("평가 데이터셋 A와 B의 개수:", len(next(os.walk('D:/programming/week6/jupyter_notebook/facades/val/'))[2]))
print("테스트 데이터셋 A와 B의 개수:", len(next(os.walk('D:/programming/week6/jupyter_notebook/facades/test/'))[2]))
'''
next(os.walk): [0] - 파일 경로, [1] - 빈 리스트, [2] - 폴더 내 파일들
'''

image = Image.open('D:/programming/week6/jupyter_notebook/facades/train/1.jpg')
print("이미지 크기:", image.size)
plt.imshow(image)
plt.show()
'''
이미지: 256x256 크기의 이미지 2개를 이어 붙인 형태 => 512x256
왼쪽: 정답 이미지(건물), 오른쪽: 조건 이미지(segmentaion)
'''

'''
학습 데이터 불러오기
'''
class ImageDataset(Dataset):
    def __init__(self, root, transforms = None, mode = "train"):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))
        '''
        glob: 파일 이름을 매개변수 형태*로 줄일 수 있는 라이브러리
        root(D:~/) + mode(train) + /*(기타 경로들).jpg의 사진들을 다 불러와서 정렬함
        os.path.join이기 때문에 두 문자열 사이 /가 자동으로 들어감(폴더처리)
        '''
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "val") + "/*.jpg")))
            '''
            학습 데이터의 수가 적기 때문에 test 데이터도 훈련에 사용
            = train mode이면 test도 데이터에 추가
            '''

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        '''
        img_A: 이미지의 왼쪽 절반 = 정답 이미지(실제 사진)
        (0, 0, 256, 256) - 왼쪽시작(가로), 위쪽시작(세로), 오른쪽끝(가로), 아래끝(세로)
        img_B: 이미지의 오른쪽 절반 = 조건 이미지(네모 사진)
        (256, 0, 512, 256) - 왼쪽시작(가로), 위쪽시작(세로), 오른쪽끝(가로), 아래끝(세로)
        '''

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            '''
            데이터 수를 늘리기 위해 랜덤 이미지를 좌우 반전 = horizontal flip
            [상하반전, 좌우반전, RGB반전]
            '''

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
'''
transforms.Resize((256, 256), Image.BICUBIC),
BICUBIC: 이미지 resize시 보간법 - 사이의 변수값에 대한 함수 값을 구하는 근사 계산법 - 오류 생겨서 뺌
'''

train_dataset = ImageDataset("D:/programming/week6/jupyter_notebook/facades", transforms = transforms)
test_dataset = ImageDataset("D:/programming/week6/jupyter_notebook/facades", transforms = transforms, mode = "test")

train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)

'''
Generator 모델 생성
'''
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize = True, dropout = 0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.0):
        super(UNetUp, self).__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace = True))

        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        '''
        (행,열) 중 열에 붙임 -> 열이 증가
        '''
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)  #이미지 2개를 붙이니까 다음 입력이 출력의 2배가 되는 것! 512+512 = 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),  #두 이미지를 붙여 보내주기 때문에 in_channels에 2를 곱해야 함!
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

generator = GeneratorUNet()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

criterion_GAN.cuda()
criterion_pixelwise.cuda()

lr = 0.0002

optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5, 0.999))

import time

num_epochs = 100
sample_interval = 200
lambda_pixel = 100

start_time = time.time()

for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        '''
        A: 실제 건물 이미지
        B: 건물 네모 이미지
        '''
        real_A = batch["B"].cuda()  #네모 이미지
        real_B = batch["A"].cuda()  #건물 이미지

        real = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0)
        fake = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(0.0)
        '''
        [1, 1, 16, 16]
        '''

        optimizer_G.zero_grad()

        fake_B = generator(real_A)
        '''
        네모 이미지를 generator에 넣어 건물 이미지 생성
        '''

        loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        loss_real = criterion_GAN(discriminator(real_B, real_A), real)
        loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        done = epoch * len(train_loader) + i
        if done % sample_interval == 0:
            imgs = next(iter(test_loader))  #batch_size개의 이미지 추출해 생성
            real_A = imgs["B"].cuda()
            real_B = imgs["A"].cuda()
            fake_B = generator(real_A)

            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
            save_image(img_sample, f"{done}.png", normalize = True)

    print(f"[Epoch {epoch + 1}/{num_epochs}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {(time.time() - start_time):.2f}s]")
