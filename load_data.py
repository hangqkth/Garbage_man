import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data


def build_label_transform_dict():
    """
    Match every string label to a number between 0 and 11
    :return: a dictionary, whose keys are string label and the corresponding value are number label
    """
    str_labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                  'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    label_trans_dict = {}
    for i in range(len(str_labels)):
        label_trans_dict[str_labels[i]] = i
    return label_trans_dict


def build_dataset(txt_file):
    """
    Processing data according to the txt file
    :param txt_file: 'test.txt', 'train.txt' or 'val.txt'
    :return: save processed data and label into .npy file
    """
    with open(txt_file, "r") as f:
        file_list = [line.strip('\n') for line in f.readlines()]
    label_list, data_list = [], []
    label_trans_dict = build_label_transform_dict()
    for f in range(len(file_list)):
        idx_r = file_list[f].find('/')+1
        idx_l = file_list[f].find('/', idx_r)
        label_list.append(label_trans_dict[file_list[f][idx_r:idx_l]])
        img_data = np.asarray(Image.open(file_list[f]).resize((224, 224)))
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        if img_data.shape[-1] != 3:
            img_data = np.stack([img_data, img_data, img_data], axis=-1)
        data_list.append(img_data)
    return np.stack(data_list, axis=0), np.stack(label_list, axis=0)


class GarbageData(data.Dataset):
    def __init__(self, data_array, label_array):
        self.data = data_array
        self.label = label_array

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.data[item, ], self.label[item, ]


if __name__ == "__main__":
    test_data, test_label = build_dataset('./garbage_classification/test.txt')
    # train_data, train_label = build_dataset('./garbage_classification/train.txt')
    # val_data, val_label = build_dataset('./garbage_classification/val.txt')
    test_dataset = GarbageData(test_data, test_label)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    for img_data, label in test_loader:
        print(img_data.shape, label.shape)
