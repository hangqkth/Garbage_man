import torchvision.models as models
from load_data import GarbageData
import torch.nn as nn
import torch
import torch.utils.data as data
import time
import numpy as np
from sklearn.metrics import accuracy_score

resnet18 = models.resnet18(pretrained=True)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_and_val(train_loader, val_loader, lr, model, epochs, criterion, device):
    best_loss_test = float("inf")

    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate, 0.001
        batch = 0
        "start training"
        for data, label in train_loader:
            batch += 1
            model.train()  # train mode
            data = data.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)  # numpy to tensor
            pred = model(data)  # forward propagation
            optimizer.zero_grad()  # initialize optimizer
            loss = criterion(pred, label.to(device=device, dtype=torch.long))
            loss.backward()  # backward propagation
            optimizer.step()  # update model parameters
            if batch % 10 == 0:
                print("\rTrain Epoch: {:d} | Train loss: {:.4f} | Batch : {}/{}".format(epoch + 1, loss, batch, len(train_loader)))

            "start testing"
            if batch % 30 == 0:
                loss_sum = 0
                pred_list, true_list = [], []
                for data, label in val_loader:
                    model.eval()  # evaluating mode
                    with torch.no_grad():  # no gradient
                        data = data.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
                        pred = model(data)
                        loss_test = criterion(pred, label.to(device=device, dtype=torch.long))
                        loss_sum += loss_test
                        pred = pred.cpu().detach().numpy()
                        pred = np.argmax(pred, axis=1)
                        pred_list += pred.tolist()
                        true_list += label.numpy().tolist()
                loss_avg = loss_sum / len(val_loader)
                acc = accuracy_score(true_list, pred_list)
                if loss_avg <= best_loss_test:
                    best_loss_test = loss_avg
                    torch.save(model.state_dict(), './saved_model/garbage_net.pth')
                print("\rTest Epoch: {:d} | Test loss: {:.4f} | Test Accuracy: {:.4%} | Best evaluation loss: {:.6f}".format(epoch + 1, loss_avg, acc, best_loss_test))
                time.sleep(0.1)


if __name__ == "__main__":
    # freeze resnet parameter or not
    # feature_extract = True
    # model = models.resnet18(pretrained=True)
    model = models.resnet34(pretrained=True)
    num_feature = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_feature, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 12))

    # set_parameter_requires_grad(model, feature_extract)
    train_data = np.load('processed_data/train_data.npy')
    train_label = np.load('processed_data/train_label.npy')
    val_data = np.load('processed_data/val_data.npy')
    val_label = np.load('processed_data/val_label.npy')
    train_loader = data.DataLoader(dataset=GarbageData(train_data, train_label), batch_size=64, shuffle=True)
    val_loader = data.DataLoader(dataset=GarbageData(val_data, val_label), batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_and_val(train_loader, val_loader, 1e-4, model, 10, criterion, device)




