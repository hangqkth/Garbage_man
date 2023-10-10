import torch.nn as nn
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from load_data import GarbageData, build_dataset


class GarbageNet(nn.Module):
    def __init__(self, c_in=3):
        super(GarbageNet, self).__init__()
        self.deep_feature = ((224//(2**4))**2)*256  # 1536
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128))
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256))
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256))
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.deep_feature, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.fc3 = nn.Linear(256, 12)

    def forward(self, x_in):
        x = self.cnn1(x_in)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # test_data, test_label = build_dataset('./garbage_classification/test.txt')
    train_data, train_label = build_dataset('./garbage_classification/train.txt')
    # val_data, val_label = build_dataset('./garbage_classification/val.txt')
    val_dataset = GarbageData(train_data, train_label)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    net = GarbageNet()
    for img_data, label in val_loader:
        pred = net(img_data.to(dtype=torch.float32).permute(0, 3, 1, 2))
