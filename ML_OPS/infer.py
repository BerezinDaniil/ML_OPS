import hydra
import numpy as np
import pandas as pd
import torch
from dvc.api import DVCFileSystem
from NN_model import my_model
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    fs = DVCFileSystem()
    fs.get_file(f"data/{cfg.test.dataset}", f"data/{cfg.test.dataset}")

    test = pd.read_csv(f"data/{cfg.train.dataset}")
    X_test = np.array(test.drop(columns=cfg.train.target), dtype=np.float32)
    y_test = test[cfg.train.target].values
    NUM_FEATURES = X_test.shape[-1]
    NUM_CLASSES = len(np.unique(y_test))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_model(NUM_FEATURES, NUM_CLASSES).to(device)
    path = "models/Model.onnx"
    model.load(path)
    model.eval()
    outputs = model(torch.from_numpy(X_test).float().to(device))
    _, predict = torch.max(outputs.data, 1)
    labels = torch.from_numpy(y_test).float().to(device)
    correct = (predict == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    print(f"accuracy: {acc:.3f}")


if __name__ == "__main__":
    infer()
