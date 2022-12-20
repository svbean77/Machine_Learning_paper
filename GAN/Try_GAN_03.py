#transforms에 Normalize 추가
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

latent_dim = 100  #차원수

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_dim, out_dim, norm = True):
            layers = [nn.Linear(in_dim, out_dim)]  #FC, MLP니까 linear로
            if norm:
                layers.append(nn.BatchNorm1d(out_dim))  #mnist는 흑백 이미지니까 2차원, 3차원인 1d로 사용
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, norm = False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),  #이미지 크기는 28*28, 이 이미지를 flatten하게 펼치기
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)  #noise를 모델에 넣어 이미지 생성
        img = img.view(img.size(0), 1, 28, 28)  #flatten하게 펼친 이미지를 다시 28*28 크기로
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),  #입력 형태는 다시 flatten하게 펼치기
            nn.LeakyReLU(0.2, inplace = True),  #inplace = 즉시 파라미터 수정(인 것 같아)
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1),  #D는 T, F를 판별하기 때문에 최종 출력은 1개
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)  #받은 생성 이미지를 다시 flatten하게 펼침
        out = self.model(flattened)
        return out

trans = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root="D:/programming/week6/data", train=True, download=True, transform=trans)
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

G = Generator().cuda()
D = Discriminator().cuda()
criterion = nn.BCELoss().cuda()

optim_G = optim.Adam(G.parameters(), lr = 0.0002, betas=(0.5, 0.999))
optim_D = optim.Adam(D.parameters(), lr = 0.0002, betas=(0.5, 0.999))

num_epochs = 100
sample_interval = 2000  #2000번 batch마다 그림 저장

for epoch in range(num_epochs):
    for i, (img, _) in enumerate(train_loader):
        #train_loader의 label은 사용하지 않기 때문에 _로 버림
        real = torch.ones(size = (img.size(0), 1)).cuda()  #모두 1인 텐서
        fake = torch.zeros(size = (img.size(0), 1)).cuda()  #모두 0인 텐서
        real_img = img.cuda()

        #G 학습 시작
        optim_G.zero_grad()
        z = torch.normal(mean = 0, std = 1, size = (img.shape[0], latent_dim)).cuda() #noise는 평균이 0, 표준편차가 1인 분포
        fake_img = G(z)  #가짜 데이터 생성

        G_loss = criterion(D(fake_img), real)  #G의 목표는 생성 데이터가 진짜로 판별되는 것 -> label로 real을 넣어 훈련

        G_loss.backward()
        optim_G.step()

        #D 학습 시작
        optim_D.zero_grad()
        real_loss = criterion(D(real_img), real)  #D의 목표는 진짜와 가짜를 완벽히 구분하는 것 -> real에 real, fake에 fake를 label로 넣어 훈련
        fake_loss = criterion(D(fake_img.detach()), fake)  #fake_img의 복사본을 넣음
        D_loss = (real_loss + fake_loss) / 2  #D의 손실값은 두 loss의 평균

        D_loss.backward()
        optim_D.step()

        #2000 batch마다 사진 저장
        save = epoch * len(train_loader) + i
        if save % sample_interval == 0:
            save_image(fake_img[:25], f"mnist - {save}.png", nrow = 5, normalize = True)

    print(f"Epoch {epoch + 1} - G_loss: {G_loss.item():.6f}, D_loss: {D_loss.item():.6f}")