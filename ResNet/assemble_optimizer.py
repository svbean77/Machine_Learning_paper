import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

trans = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.ImageFolder('D:/SynapseImiging/HTCC_Crack_modify/Train', transform = trans)
train_loader = DataLoader(trainset, batch_size = 1, shuffle = True)
testset = torchvision.datasets.ImageFolder('D:/SynapseImiging/HTCC_Crack_modify/Test', transform = trans)
test_loader = DataLoader(testset, shuffle = True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, stride = 1, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()

        if (stride != 1) or (in_channels != out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride = stride, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride = 1, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, stride = 1, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()

        if (stride != 1) or (in_channels != out_channels * self.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride = stride, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes = 2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, stride = 2, kernel_size = 7, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 3, padding = 1)
        )
        self.conv2 = self.make_layer(block, 64, num_block[0], 1)
        self.conv3 = self.make_layer(block, 128, num_block[1], 2)
        self.conv4 = self.make_layer(block, 256, num_block[2], 2)
        self.conv5 = self.make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])
def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])
def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet34().to(device)

num_epochs = 50
learning_rate = 0.01

criterion = nn.CrossEntropyLoss()

def Train_Test(name, optimizer, num_epochs = num_epochs, learning_rate = learning_rate):
    loss_value = {'Train': [], 'Test': []}
    acc_value = {'Train': [], 'Test': []}
    time_value = {'Train': [], 'Test': []}

    print(f'***** {name} Train-Test Start! *****')
    for epoch in range(num_epochs):
        model.train()
        train_start = time.time()
        train_loss = 0.0
        cnt = 0
        size = 0

        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 훈련을 진행
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            # loss, 전체 크기, 맞은 개수 구함
            _, predict = torch.max(out, dim=1)
            train_loss += loss.item()
            size += labels.size(0)
            cnt += torch.sum(labels == predict).item()

        # 25번째 epoch마다 learning_rate를 줄임
        if (epoch + 1) % 25 == 0:
            learning_rate /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # 시간, loss, 정확도를 모두 리스트에 추가
        train_time = time.time() - train_start
        train_acc = cnt / size * 100
        loss_value['Train'].append(train_loss)
        acc_value['Train'].append(train_acc)
        time_value['Train'].append(train_time)

        # train loss, accuracy, time 출력
        print(f'[Epoch {epoch + 1}]')
        print('Train - loss: %f, accuracy: %.3f%%, Time: %dmin %.5fsec' % (
            train_loss, train_acc, (train_time // 60), (train_time % 60)))

        # test 시작
        model.eval()
        test_start = time.time()
        test_loss = 0.0
        cnt = 0
        size = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                out = model(images)
                loss = criterion(out, labels)

                _, predict = torch.max(out, dim=1)
                test_loss += loss.item()
                size += labels.size(0)
                cnt += torch.sum(labels == predict).item()

            test_time = time.time() - test_start
            test_acc = cnt / size * 100
            loss_value['Test'].append(test_loss)
            acc_value['Test'].append(test_acc)
            time_value['Test'].append(test_time)

            print('Test - loss: %f, accuracy: %.3f%%, Time: %dmin %.5fsec' % (
                test_loss, test_acc, (test_time // 60), (test_time % 60)))

    # train 시간과 test 시간을 모두 더해 epoch의 총 소요시간 구함
    total_time = np.sum(time_value['Train']) + np.sum(time_value['Test'])
    print('%s optimizer - Epoch %d회를 돌리는 데 걸리는 시간: %d분 %.5f초\n' % (name, num_epochs, (total_time // 60), (total_time % 60)))

    #모델의 loss, acc, time과 총 시간을 리턴
    values = {
    'Loss':loss_value,
    'Accuracy':acc_value,
    'Time':time_value
    }
    return values, total_time

SGD, SGD_time = Train_Test('SGD', optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.00000001))
Adagrad, Adagrad_time = Train_Test('Adagrad', optim.Adagrad(model.parameters(), lr = learning_rate, weight_decay = 0.00000001))
RMSprop, RMS_time = Train_Test('RMSprop', optim.RMSprop(model.parameters(), lr = learning_rate, weight_decay = 0.00000001))
Adam, Adam_time = Train_Test('Adam', optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.00000001))
AdamW, AdamW_time = Train_Test('AdamW', optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 0.00000001))

#모든 optimizer의 loss, acc, time을 딕셔너리 형태로 만듦
all_values = {
    'SGD': SGD,
    'Adagrad': Adagrad,
    'RMSprop': RMSprop,
    'Adam': Adam,
    'AdamW': AdamW
}

#모든 optimizer의 총 시간을 리스트에 저장
all_times = list([SGD_time, Adam_time, AdamW_time, Adagrad_time, RMS_time])

#optimizer 이름, 변수 이름을 리스트로 만듦
optim_name = list(all_values.keys())
val_kind = list(SGD.keys())

#plt에 들어갈 라벨 이름 지어줌
legend = {}
for val in val_kind:
    legend[val] = {}
    for name in optim_name:
        legend[val][name] = name + ' ' + val
        #legend[val][name] = [name + ' - ' + val, name + ' - ' + val]

#optimizer별로 같은 색의 그래프를 그릴 수 있도록
color = {
    'SGD': '#1f77b4',
    'Adagrad': '#ff7f0e',
    'RMSprop': '#2ca02c',
    'Adam': '#d62728',
    'AdamW': '#9467bd'
}

x = np.linspace(1, num_epochs, num_epochs)

#각 optimizer별로 subplot으로 loss끼리, acc끼리 그림
for name in optim_name:
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.plot(x, all_values[name]['Loss']['Train'], linestyle='solid', label=legend['Loss'][name])
    plt.plot(x, all_values[name]['Loss']['Test'], linestyle='dashed', label=legend['Loss'][name])
    plt.title(name + "'s Train, Test Loss", fontsize = 18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')

    plt.subplot(2, 1, 2)
    plt.plot(x, all_values[name]['Accuracy']['Train'], linestyle='solid', label=legend['Accuracy'][name])
    plt.plot(x, all_values[name]['Accuracy']['Test'], linestyle='dashed', label=legend['Accuracy'][name])
    plt.title(name + "'s Train, Test Accuracy", fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')

    plt.show()

#train끼리만 optimizer별로
for name in optim_name:
    fig, ax1 = plt.subplots(figsize = (10,5))
    ax2 = ax1.twinx()
    loss = ax1.plot(x, all_values[name]['Loss']['Train'], linestyle='solid', label=legend['Loss'][name])
    acc = ax2.plot(x, all_values[name]['Accuracy']['Train'], linestyle='solid', label=legend['Accuracy'][name], color = '#ff7f0e')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')

    tmp = loss + acc
    label = [g.get_label() for g in tmp]
    ax1.legend(tmp, label, loc = 2, fontsize = 'small')
    plt.title(name + " optimizer's Train", fontsize=18)

    plt.show()

#test끼리만 optimizer별로
for name in optim_name:
    fig, ax1 = plt.subplots(figsize = (10,5))
    ax2 = ax1.twinx()
    loss = ax1.plot(x, all_values[name]['Loss']['Test'], linestyle='solid', label=legend['Loss'][name])
    acc = ax2.plot(x, all_values[name]['Accuracy']['Test'], linestyle='solid', label=legend['Accuracy'][name], color = '#ff7f0e')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')

    tmp = loss + acc
    label = [g.get_label() for g in tmp]
    ax1.legend(tmp, label, loc = 2, fontsize = 'small')
    plt.title(name + " optimizer's Test", fontsize=18)

    plt.show()

#각 optimizer별로 loss와 acc를 같이 그림
for name in optim_name:
    fig, ax1 = plt.subplots(figsize = (10,5))
    ax2 = ax1.twinx()
    loss_a = ax1.plot(x, all_values[name]['Loss']['Train'], color = color['SGD'], linestyle = 'solid', label = legend['Loss'][name] + '(Train)')
    loss_b = ax1.plot(x, all_values[name]['Loss']['Test'], color=color['Adagrad'], linestyle='solid', label=legend['Loss'][name] + '(Test)')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    acc_a = ax2.plot(x, all_values[name]['Accuracy']['Train'], color=color['SGD'], linestyle='dashed', label=legend['Accuracy'][name] + '(Train)')
    acc_b = ax2.plot(x, all_values[name]['Accuracy']['Test'], color=color['Adagrad'], linestyle='dashed', label=legend['Accuracy'][name] + '(Test)')
    ax2.set_ylabel('Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')

    tmp = loss_a + loss_b + acc_a + acc_b
    label = [g.get_label() for g in tmp]
    ax1.legend(tmp, label, loc = 2, fontsize = 'small')
    plt.title(name + " optimizer's Loss and Accuracy", fontsize=18)

    plt.show()

#train, test로 구분하여 loss optimizer를 합침
plt.figure(figsize=(10, 7))
plt.subplots_adjust(hspace=0.4)
for name in optim_name:
    plt.subplot(2, 1, 1)
    plt.plot(x, all_values[name]['Loss']['Train'], linestyle='solid', label=legend['Loss'][name])
    plt.title('Train Loss', fontsize = 18)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')

    plt.subplot(2, 1, 2)
    plt.plot(x, all_values[name]['Loss']['Test'], linestyle='solid', label=legend['Loss'][name])
    plt.title('Test Loss', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')
plt.show()

#train, test로 구분하여 Accuracy optimizer를 합침
plt.figure(figsize=(10, 7))
plt.subplots_adjust(hspace=0.4)
for name in optim_name:
    plt.subplot(2, 1, 1)
    plt.plot(x, all_values[name]['Accuracy']['Train'], linestyle='solid', label=legend['Accuracy'][name])
    plt.title('Train Accuracy', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')

    plt.subplot(2, 1, 2)
    plt.plot(x, all_values[name]['Accuracy']['Test'], linestyle='solid', label=legend['Accuracy'][name])
    plt.title('Test Accuracy', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.axvline(x=25, color='red', linestyle = 'dashed')
    plt.legend(loc=2, fontsize = 'small')
plt.show()

#train, test로 구분하여 time optimizer를 합침
plt.figure(figsize=(10, 7))
plt.subplots_adjust(hspace=0.4)
for name in optim_name:
    plt.subplot(2, 1, 1)
    plt.plot(x, all_values[name]['Time']['Train'], linestyle='solid', label=legend['Time'][name])
    plt.title('Train Time', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Train Time')
    plt.legend(loc=2, fontsize = 'small')

    plt.subplot(2, 1, 2)
    plt.plot(x, all_values[name]['Time']['Test'], linestyle='solid', label=legend['Time'][name])
    plt.title('Test Time', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Test Time')
    plt.legend(loc=2, fontsize = 'small')
plt.show()

x2 = np.linspace(1, len(optim_name), len(optim_name))
#총 total time 막대그래프로
plt.bar(x2, all_times, color = list(color.values()))
plt.title('Total Time', fontsize = 18)
plt.xticks(x2, optim_name)
plt.ylabel('Total Time')
plt.show()