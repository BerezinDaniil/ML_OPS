import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ML_OPS.NN_model import my_model

path = "../data/dataset.csv"
full_data = pd.read_csv(path)
X = full_data.drop(columns=["round_winner"])
y = full_data["round_winner"]
_, X_test, _, y_test = train_test_split(
    X, y, random_state=42, train_size=0.85, stratify=y
)

BATCH_SIZE = 1024
NUM_FEATURES = X.shape[-1]
NUM_CLASSES = len(np.unique(y))

X_test, y_test = np.array(X_test, dtype=np.float32), y_test.values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = my_model(NUM_FEATURES, NUM_CLASSES).to(device)
path = "Model.pth"
model.load(path)
model.eval()
outputs = model(torch.from_numpy(X_test).float().to(device))
_, predict = torch.max(outputs.data, 1)
labels = torch.from_numpy(y_test).float().to(device)
correct = (predict == labels).sum().item()
total = labels.size(0)
acc = correct / total
print(f"accuracy: {acc:.3f}")
