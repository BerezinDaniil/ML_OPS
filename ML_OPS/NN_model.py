import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class my_model(nn.Module):
    def __init__(self, features, num_classes):
        super(my_model, self).__init__()
        self.linear_1 = nn.Linear(features, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.linear_4 = nn.Linear(64, num_classes)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        self.act_4 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm1d(256)
        self.batchnorm_2 = nn.BatchNorm1d(128)
        self.batchnorm_3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.batchnorm_1(x)
        x = self.act_1(x)

        x = self.linear_2(x)
        x = self.batchnorm_2(x)
        x = self.act_2(x)

        x = self.linear_3(x)
        x = self.batchnorm_3(x)
        x = self.act_3(x)

        x = self.linear_4(x)
        x = self.act_4(x)
        return F.softmax(x, dim=1)

    def save(self):
        state = {"model_state_dict": self.state_dict()}
        torch.save(state, "Model.pth")
        print("Model saved.")

    def load(self, path):
        state = torch.load(path)
        model_state_dict = state["model_state_dict"]
        self.load_state_dict(model_state_dict)
        print("Model loaded.")
