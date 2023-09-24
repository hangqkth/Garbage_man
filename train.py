import torchvision.models as models
from load_data import GarbageData
import torch.nn as nn
import torch
import torch.utils.data as data
import time
import numpy as np

resnet18 = models.resnet18(pretrained=True)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_and_test(train_loader, val_loader, lr, model, epochs, criterion, device):
    best_loss_train, best_loss_test = float("inf"), float("inf")
    best_train_result, best_test_result = [], []

    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate, 0.001
        batch = 0
        "start training"
        for data, label in train_loader:
            batch += 1
            model.train()  # train mode
            data = data.to(device=device, dtype=torch.float32)  # numpy to tensor
            a_pred = model(data)  # forward propagation
            optimizer.zero_grad()  # initialize optimizer
            loss = criterion(a_pred, label.to(device=device))
            loss.backward()  # backward propagation
            optimizer.step()  # update model parameters
            # if loss < best_loss_train:
            #     best_loss_train = loss
            #     a_pred = a_pred.cpu().detach().numpy()
            #     best_train_result = [a_pred, label.cpu().detach().numpy(), data.cpu().detach().numpy()]
            if batch % 10 == 0:
                print("\rTrain Epoch: {:d} | Train loss: {:.4f} | Batch : {}/{}".format(epoch + 1, loss, batch, len(train_loader)))

            "start testing"
            if batch % 30 == 0:
                loss_sum = 0
                for data, label in val_loader:
                    model.eval()  # evaluating mode
                    with torch.no_grad():  # no gradient
                        data = data.to(device=device, dtype=torch.float32)
                        a_pred = model(data)
                        loss_test = criterion(a_pred, label.to(device=device))
                        loss_sum += loss_test
                loss_avg = loss_sum / len(val_loader)
                if loss_avg <= best_loss_test:
                    best_loss_test = loss_avg
                    # torch.save(model.state_dict(), './model/model_saved/'+model_name+'.pth')
                print(
                    "\rTest Epoch: {:d} | Test loss: {:.4f} | Best evaluation loss: {:.6f}".format(epoch + 1, loss_avg,
                                                                                                   best_loss_test))
                time.sleep(0.1)


if __name__ == "__main__":
    # 冻结参数的梯度
    feature_extract = True
    model = models.resnet18(pretrained=True)
    # set_parameter_requires_grad(model, feature_extract)
    train_data = np.load('processed_data/train_data.npy')
    train_label = np.load('processed_data/train_label.npy')
    val_data = np.load('processed_data/val_data.npy')
    val_label = np.load('processed_data/val_label.npy')
    train_loader = data.DataLoader(dataset=GarbageData(train_data, train_label), batch_size=32, shuffle=True)
    val_loader = data.DataLoader(dataset=GarbageData(val_data, val_label), batch_size=32, shuffle=True)
    num_feature = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_feature, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 12))
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_and_test(train_loader, val_loader, 0.001, model, 10, criterion, device)





