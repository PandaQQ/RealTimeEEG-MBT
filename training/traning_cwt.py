"""
Created on Fri Nov 29 17:06:43 2024
@author: guang
This script implements a deep learning pipeline for EEG data analysis using continuous wavelet transforms (CWT). It includes:
- Loading EEG data from a .mat file.
- Computing the CWT for each signal segment.
- Preparing data for training/testing with PyTorch.
- Defining a CNN model and training/testing it.
- Saving the trained model to disk.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from scipy import signal
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pywt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device = 'mps'

batch_size = 512
temp = loadmat('./training/EEG_SPA_001.mat')
# temp = loadmat('EEG_001_SPA_MATLAB.mat')
# temp = loadmat('EEG_001_SPA_DS.mat')

data = temp['data']
label = temp['labels']
del temp

CHANNELS = 24
# TOTAL_TIME_FRAME = 3251
# 1800 for training, 1451 for testing
TOTAL_TRAIN_SET = 48*30
TOTAL_TIME_FRAME = 2639
FREQUENCY = 125

data_cwt1 = np.empty((CHANNELS, 20, FREQUENCY, TOTAL_TIME_FRAME))
scale = np.array([1., 1.25, 1.5625, 1.953125, 2.44140625,
                  3.05175781, 3.81469727, 4.76837158, 5.96046448, 7.4505806,
                  9.31322575, 11.64153218, 14.55191523, 18.18989404, 22.73736754,
                  28.42170943, 35.52713679, 44.40892099, 55.51115123, 69.38893904])

for j in range(data.shape[0]):
    if j % 100 == 0:
        print(j)
    for k in range(data.shape[1]):
        temp_seg = data[j, k, :]
        # coef, freqs = pywt.cwt(temp_seg, np.arange(1, 50), 'cmor1-1')
        coef, freqs = pywt.cwt(temp_seg, scale, 'cmor1-1')
        freqs = freqs * data.shape[2]
        data_cwt1[k, :, :, j] = abs(coef)

data = data_cwt1
print(data.shape)

data = torch.tensor(data, device=device, dtype=torch.float)
data = torch.permute(data, (3, 0, 1, 2))
label = torch.tensor(label, device=device, dtype=torch.long)

# temp = torch.randperm(1800)
# data = data[torch.cat((temp[:1800],torch.arange(1800,3600))),:,:,:]
# label = label[torch.cat((temp[:1800],torch.arange(1800,3600)))]

# temp = torch.randperm(3600)
# data_train = data[temp[:1800],:,:,:]
# data_test = data[temp[1800:],:,:,:]
# label_train = label[temp[:1800]]
# label_test = label[temp[1800:]]

TRAIN_SET_SIZE = TOTAL_TIME_FRAME - TOTAL_TRAIN_SET

temp = torch.randperm(TOTAL_TRAIN_SET)
data = data[torch.cat((temp[:TRAIN_SET_SIZE], torch.arange(TOTAL_TRAIN_SET, TOTAL_TIME_FRAME))), :, :, :]
label = label[torch.cat((temp[:TRAIN_SET_SIZE], torch.arange(TOTAL_TRAIN_SET, TOTAL_TIME_FRAME)))]

temp = torch.randperm(TRAIN_SET_SIZE * 2)
data_train = data[temp[:TOTAL_TRAIN_SET], :, :, :]
data_test = data[temp[TOTAL_TRAIN_SET:], :, :, :]
label_train = label[temp[:TOTAL_TRAIN_SET]]
label_test = label[temp[TOTAL_TRAIN_SET:]]


# conv1 = nn.Conv2d(10, 10, (46,1))
# pool1 = nn.MaxPool2d((1,1))
# conv2 = nn.Conv2d(10, 10, (1,10))
# pool2 = nn.MaxPool2d((1,10))
# conv3 = nn.Conv2d(20, 1, (1,1))
# pool3 = nn.MaxPool2d((1,1))


# temp1 = conv1(data_train)
# temp2 = pool1(temp1)
# temp3 = conv2(temp2)
# temp4 = pool2(temp3)
# temp5 = conv3(temp4)
# temp6 = pool3(temp5)


class my_Dataset(Dataset):
    def __init__(self, data, label):
        self.label = label
        self.data = data.to(device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


my_data_train = my_Dataset(data_train, label_train)
my_data_test = my_Dataset(data_test, label_test)
train_dataloader = DataLoader(my_data_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(my_data_test, batch_size=batch_size, shuffle=True)

# print kernel size (data.shape[2], 1)
print(data.shape[2])


class my_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 50, (data.shape[2], 1))
        self.bn1 = nn.BatchNorm2d(50, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(1, affine=False)
        self.pool1 = nn.MaxPool2d((1, 20))
        self.conv2 = nn.Conv2d(10, 10, (1, 10))
        self.conv3 = nn.Conv2d(20, 1, (1, 1))
        self.pool2 = nn.MaxPool2d((1, 20))
        self.pool3 = nn.MaxPool2d((1, 1))
        self.fc1 = nn.Linear(50 * (FREQUENCY // 20), 32)
        self.fc2 = nn.Linear(32, 2)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = my_cnn()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')


def train_loop(data_loader, model, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        logits = model(x)
        logits1 = F.softmax(logits, 1)
        loss = F.cross_entropy(logits1, y[:, 0])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(data_loader, model):
    model.eval()
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    correct = 0

    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)
            logits1 = F.softmax(logits, 1)
            correct += (logits1.argmax(1) == y[:, 0]).type(torch.float).sum().item()
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%")
    return correct


epoch = 50
accu_record = []
for t in range(epoch):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, optimizer)
    temp = test_loop(test_dataloader, model)
    accu_record.append(temp)

#  Save Model
print("Saving model...")
torch.save(model.state_dict(), "./models/eeg_cnn_model.pth")
print("Model saved as eeg_cnn_model.pth")

# plt.plot(accu_record)

plt.plot(accu_record)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.show()
