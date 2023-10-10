from load_data import GarbageData, build_dataset
import torch.nn as nn
import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from model import GarbageNet
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = GarbageNet()
    num_feature = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_feature, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 12))

    test_data, test_label = build_dataset('./garbage_classification/test.txt')

    test_loader = data.DataLoader(dataset=GarbageData(test_data, test_label), batch_size=32, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_param = torch.load('./saved_model/garbage_net.pth')
    model.load_state_dict(model_param)
    model.to(device)
    pred_list, true_list = [], []
    for data, label in tqdm(test_loader):
        model.eval()  # evaluating mode
        with torch.no_grad():  # no gradient
            data = data.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            pred = model(data)
            pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
            pred_list += pred.tolist()
            true_list += label.numpy().tolist()

    acc = accuracy_score(y_true=true_list, y_pred=pred_list)
    print("\rTest finished, Test Accuracy: {:.4%} ".format(acc))

    str_labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                  'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    disp = ConfusionMatrixDisplay.from_predictions(y_true=true_list, y_pred=pred_list, cmap=plt.cm.Blues,
                                                   xticks_rotation="vertical", display_labels=str_labels)
    plt.title("Test Accuracy: {:.4%} ".format(acc))
    plt.show()


