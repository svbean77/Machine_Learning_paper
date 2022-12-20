import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
'''
데이터셋 로드
'''
train_trans = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_trans = transforms.ToTensor()
train_dataset = torchvision.datasets.CIFAR10(root = "D:/programming/week7/data", download = True, train = True, transform = train_trans)
test_dataset = torchvision.datasets.CIFAR10(root = "D:/programming/week7/data", download = True, train = False, transform = test_trans)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, shuffle = True)

'''
데이터 출력
'''
class_names = ("plane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
num = np.random.randint(0, len(test_dataset) + 1, 16)
for i, idx in enumerate(num, 1):
    plt.subplot(4, 4, i)
    img = test_dataset[idx][0]
    label = test_dataset[idx][1]
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.title(f"{class_names[label]}", fontsize = 8)
    plt.axis("off")
plt.show()
'''
모델 구현
'''
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes = 10):
        super(ResNet, self).__init__()

        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(self.in_channels)
        )
        self.layer1 = self.make_layer(block, 16, num_block, 1)
        self.layer2 = self.make_layer(block, 32, num_block, 2)
        self.layer3 = self.make_layer(block, 64, num_block, 2)
        self.avg_pool = nn.AvgPool2d(1, 1)
        self.linear = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def ResNet20():
    return ResNet(Residual_Block, 3)
def ResNet32():
    return ResNet(Residual_Block, 5)
def ResNet44():
    return ResNet(Residual_Block, 7)
def ResNet56():
    return ResNet(Residual_Block, 9)
'''
모델 및 파라미터 설정
'''

model = ResNet20().cuda()
lr = 0.1
optim = optim.SGD(model.parameters(), lr = lr, weight_decay = 0.0001, momentum = 0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 100


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    cnt = 0

    for idx, (img, lbl) in enumerate(train_loader):
        img = img.cuda()
        lbl = lbl.cuda()

        optim.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optim.step()

        print(out)